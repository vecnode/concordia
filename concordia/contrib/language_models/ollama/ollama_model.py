# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ollama Language Model, a wrapper for models running on the local machine."""

from collections.abc import Collection, Sequence
import json
import threading
from typing import override

from concordia.language_model import language_model
from concordia.utils import measurements as measurements_lib
from concordia.utils import sampling
import ollama


class GenerationCancelledError(RuntimeError):
  """Raised by `OllamaLanguageModel.sample_text`/`sample_choice` when the
  call's `cancel_event` (see `OllamaLanguageModel.__init__`) was set while
  a generation was in flight.

  This is what makes real mid-call cancellation possible for this
  provider: without it, a blocking `ollama.Client().generate()` call has
  no way to be interrupted once sent, since Python threads can't be
  preempted from the outside and Concordia's `StepController`/`pause()`
  is only ever checked between simulation steps, not inside a call
  already in flight (see the "No true mid-run cancellation" limitation in
  agora's ARCHITECTURE.md, which this addresses for the Ollama backend
  specifically -- other providers would need the same streaming-plus-
  cancellation-check treatment individually).
  """


def _field(obj, key: str, default=None):
  """Reads `key` off `obj`, whether it's a plain dict or an attribute-style
  object (e.g. ollama-python's `GenerateResponse`, a pydantic model) --
  shared by `_extract_token_usage` and `OllamaLanguageModel._generate`
  below so both tolerate either shape the same way.
  """
  get = obj.get if hasattr(obj, 'get') else lambda k, d=None: getattr(obj, k, d)
  return get(key, default)


def _extract_token_usage(response) -> dict[str, int]:
  """Real prompt/completion token counts from an Ollama `generate()`
  response, when the server reports them.

  Ollama's `/api/generate` includes `prompt_eval_count`/`eval_count` on
  most backends, but not guaranteed on every server version/model
  combination -- so this only adds keys that were actually present,
  rather than defaulting missing ones to 0 and silently claiming a real
  count that isn't one.
  """
  prompt_tokens = _field(response, 'prompt_eval_count')
  completion_tokens = _field(response, 'eval_count')
  usage = {}
  if prompt_tokens is not None:
    usage['prompt_tokens'] = prompt_tokens
  if completion_tokens is not None:
    usage['completion_tokens'] = completion_tokens
  if prompt_tokens is not None and completion_tokens is not None:
    usage['total_tokens'] = prompt_tokens + completion_tokens
  return usage


_MAX_MULTIPLE_CHOICE_ATTEMPTS = 20
_DEFAULT_TEMPERATURE = 0.5
_DEFAULT_TERMINATORS = ()
_DEFAULT_SYSTEM_MESSAGE = (
    "Continue the user's sentences. Never repeat their starts. For example, "
    "when you see 'Bob is', you should continue the sentence after "
    "the word 'is'. Here are some more examples: 'Question: Is Jake a "
    "turtle?\nAnswer: Jake is ' should be completed as 'not a turtle.' and "
    "'Question: What is Priya doing right now?\nAnswer: Priya is currently ' "
    "should be completed as 'working on repairing the sink.'. Notice that "
    "it is OK to be creative with how you finish the user's sentences. The "
    'most important thing is to always continue in the same style as the user.'
)


class OllamaLanguageModel(language_model.LanguageModel):
  """Language Model that uses Ollama LLM models."""

  def __init__(
      self,
      model_name: str,
      *,
      system_message: str = _DEFAULT_SYSTEM_MESSAGE,
      measurements: measurements_lib.Measurements | None = None,
      channel: str = language_model.DEFAULT_STATS_CHANNEL,
      seed: int | None = None,
      timeout: float | None = None,
      cancel_event: threading.Event | None = None,
  ) -> None:
    """Initializes the instance.

    Args:
        model_name: The language model to use. For more details, see
          https://github.com/ollama/ollama.
        system_message: System message to prefix to requests when prompting the
          model.
        measurements: The measurements object to log usage statistics to.
        channel: The channel to write the statistics to.
        seed: Fixed generation seed passed to Ollama's API on every call, for
          run-to-run reproducibility. Note this is a single seed for the
          model instance's lifetime (nothing upstream in Concordia's engine
          currently passes a per-call seed to sample_text/sample_choice), so
          it makes an entire simulation run reproducible given an identical
          sequence of prompts, not each individual call independently.
        timeout: Per-request timeout in seconds, forwarded to the underlying
          HTTP client. Without this, a request has no deadline: if the
          Ollama server itself stalls (e.g. under memory pressure from a
          large model), the call blocks indefinitely with no way to recover
          short of killing the process. None preserves the previous
          (unbounded) behavior. `cancel_event` below is the complementary,
          proactive mechanism -- this is a passive worst-case backstop.
        cancel_event: Shared with whatever external code wants to interrupt
          a call already in flight (e.g. a `StepController`-driven pause/
          stop, wired up one layer above this class). Every `sample_text`/
          `sample_choice` call streams the response and checks this event
          between chunks; setting it from another thread makes the current
          call raise `GenerationCancelledError` promptly instead of running
          to completion. `None` (the default): calls are never cancelled
          mid-flight, the same as before this parameter existed. One
          `Event` is shared across every call this instance makes for its
          lifetime -- construct a fresh `OllamaLanguageModel` (or clear the
          event with `.clear()` after handling the cancellation) to make
          further calls possible again.
    """
    self._model_name = model_name
    self._client = (
        ollama.Client(timeout=timeout) if timeout is not None
        else ollama.Client()
    )
    self._system_message = system_message
    self._terminators = []
    self._seed = seed
    self._cancel_event = cancel_event

    self._measurements = measurements
    self._channel = channel

  def _generate(self, *, prompt: str, options: dict, response_format=None):
    """Streams one `generate()` call chunk by chunk, concatenating the
    response text and checking `self._cancel_event` between chunks --
    the mechanism that makes `GenerationCancelledError` possible (see its
    docstring). Returns the final chunk (a `GenerateResponse`) with its
    `response` field replaced by the full concatenated text, so callers
    can keep treating the result exactly like a non-streaming `generate()`
    response (`result['response']`, `_extract_token_usage(result)`, ...).

    Raises:
      GenerationCancelledError: if `self._cancel_event` was already set
        before this call started (in which case the request is never even
        sent), or became set partway through the stream (in which case
        whatever text had already arrived is discarded and the underlying
        HTTP stream is closed, not just abandoned -- `chunks.close()`
        triggers a `GeneratorExit` inside ollama-python's own generator,
        which exits its `with self._client.stream(...) as r:` block and
        actually releases the connection, rather than leaving a request
        the server is still generating against dangling).
    """
    if self._cancel_event is not None and self._cancel_event.is_set():
      raise GenerationCancelledError(
          'Ollama generation cancelled before it started (cancel_event was'
          ' already set).'
      )
    kwargs = {'format': response_format} if response_format else {}
    chunks = self._client.generate(
        model=self._model_name,
        prompt=prompt,
        think=False,
        options=options,
        keep_alive='10m',
        stream=True,
        **kwargs,
    )
    text_parts = []
    final_chunk = None
    for chunk in chunks:
      text_parts.append(_field(chunk, 'response', '') or '')
      final_chunk = chunk
      if self._cancel_event is not None and self._cancel_event.is_set():
        close = getattr(chunks, 'close', None)
        if close is not None:
          close()
        raise GenerationCancelledError(
            'Ollama generation cancelled mid-stream (cancel_event was set).'
        )
    if final_chunk is None:
      # An empty stream (zero chunks) shouldn't happen against a real
      # Ollama server, but a test double or a genuinely empty response
      # could produce one -- fall back to an empty dict rather than
      # raising an unrelated UnboundLocalError-style failure below.
      final_chunk = {}
    final_chunk['response'] = ''.join(text_parts)
    return final_chunk

  @override
  def sample_text(
      self,
      prompt: str,
      *,
      max_tokens: int = language_model.DEFAULT_MAX_TOKENS,
      terminators: Collection[str] = _DEFAULT_TERMINATORS,
      temperature: float = _DEFAULT_TEMPERATURE,
      top_p: float = language_model.DEFAULT_TOP_P,
      top_k: int = language_model.DEFAULT_TOP_K,
      timeout: float = -1,
      seed: int | None = None,
  ) -> str:
    del max_tokens, timeout, seed  # Unused (no per-call seed is passed in).

    prompt_with_system_message = f'{self._system_message}\n\n{prompt}'

    terminators = self._terminators + list(terminators)

    options = {
        'stop': terminators,
        'temperature': temperature,
        'top_p': top_p,
        'top_k': top_k,
    }
    if self._seed is not None:
      options['seed'] = self._seed

    response = self._generate(prompt=prompt_with_system_message, options=options)
    result = response['response']

    if self._measurements is not None:
      self._measurements.publish_datum(
          self._channel,
          {'raw_text_length': len(result), **_extract_token_usage(response)},
      )

    return result

  @override
  def sample_choice(
      self,
      prompt: str,
      responses: Sequence[str],
      *,
      seed: int | None = None,
  ) -> tuple[int, str, dict[str, float]]:
    del seed  # Unused (no per-call seed is passed in).
    prompt_with_system_message = f'{self._system_message}\n\n{prompt}'
    template = {'choice': '', 'single sentence explanation': ''}
    sample = ''
    answer = ''
    for attempts in range(_MAX_MULTIPLE_CHOICE_ATTEMPTS):
      # Increase temperature after the first failed attempt.
      temperature = sampling.dynamically_adjust_temperature(
          attempts, _MAX_MULTIPLE_CHOICE_ATTEMPTS
      )

      options = {'stop': (), 'temperature': temperature}
      if self._seed is not None:
        # Offset by attempt number so retries don't repeat the exact same
        # (seed, temperature) pair and get stuck resampling the same output.
        options['seed'] = self._seed + attempts

      response = self._generate(
          prompt=(
              f'{prompt_with_system_message}.\n'
              # `prompt` already renders the options as e.g. "(a) Danny\n(b)
              # Sage\n...\nAnswer: (", a continuation-style cue for the
              # letter key -- but forcing JSON output here breaks that
              # continuation framing, so without this explicit instruction
              # the model tends to answer with an option's full text (e.g.
              # "Danny") instead of its key, which then can never match
              # `responses` (always the short keys, e.g. ('a', 'b')).
              'Respond with exactly one of the following options: '
              f'{", ".join(responses)}.\n'
              f'Use the following json template: {json.dumps(template)}.'
          ),
          options=options,
          response_format='json',
      )
      try:
        json_data_response = json.loads(response['response'])
      except json.JSONDecodeError:
        continue
      sample_or_none = json_data_response.get('choice', None)
      if sample_or_none is None:
        if isinstance(json_data_response, dict) and json_data_response:
          sample = next(iter(json_data_response.values()))
        elif isinstance(json_data_response, str) and json_data_response:
          # pyrefly: ignore [missing-attribute]
          sample = sample_or_none.strip()
        else:
          continue
      else:
        sample = sample_or_none
        if isinstance(sample, str) and sample:
          sample = sample.strip()

      if not isinstance(sample, str):
        # Some models (e.g. deepseek-r1) occasionally nest the choice in a
        # non-string JSON value (a dict or list) instead of returning it as
        # a plain string. extract_choice_response only handles strings, so
        # treat this as a failed attempt and retry rather than crash.
        continue

      # Prefer an exact match against the valid options first -- `responses`
      # is normally the short option keys interactive_document.py's
      # multiple_choice_question assigns (e.g. ('a', 'b')), which the model
      # sometimes returns bare/unparenthesized ("a" rather than "(a)"), a
      # case extract_choice_response's parenthesized-only fallback below
      # would otherwise miss.
      answer = sample if sample in responses else sampling.extract_choice_response(
          sample
      )
      try:
        idx = responses.index(answer)
      except ValueError:
        continue
      else:
        token_usage = _extract_token_usage(response)
        if self._measurements is not None:
          self._measurements.publish_datum(
              self._channel, {'choices_calls': attempts, **token_usage}
          )
        return idx, responses[idx], token_usage

    raise language_model.InvalidResponseError((
        f'Too many multiple choice attempts.\nLast attempt: {sample}, '
        + f'extracted: {answer}'
    ))
