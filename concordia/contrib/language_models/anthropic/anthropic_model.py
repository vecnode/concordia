# Copyright 2026 DeepMind Technologies Limited.
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

"""Language Model wrapper for Anthropic's Claude models, via the Messages API."""

from collections.abc import Collection, Sequence
import os
from typing import Any, override

import anthropic
from concordia.language_model import language_model
from concordia.utils import measurements as measurements_lib


def _extract_token_usage(response) -> dict[str, int]:
  """Real input/output token counts from a Messages API response, keyed to
  match the other providers' `prompt_tokens`/`completion_tokens`/
  `total_tokens` (Anthropic's own `usage.input_tokens`/`output_tokens`
  naming otherwise wouldn't line up with them)."""
  usage = getattr(response, 'usage', None)
  if usage is None:
    return {}
  result = {}
  if usage.input_tokens is not None:
    result['prompt_tokens'] = usage.input_tokens
  if usage.output_tokens is not None:
    result['completion_tokens'] = usage.output_tokens
  if usage.input_tokens is not None and usage.output_tokens is not None:
    result['total_tokens'] = usage.input_tokens + usage.output_tokens
  return result


_MAX_MULTIPLE_CHOICE_ATTEMPTS = 20

# Anthropic's Messages API takes the system prompt as a top-level `system`
# argument, not a `{"role": "system", ...}` message -- unlike the OpenAI/Groq
# style wrappers this one is otherwise modeled on.
_SYSTEM_PROMPT = (
    "You always continue sentences provided by the user and you never repeat"
    " what the user already said."
)


class AnthropicLanguageModel(language_model.LanguageModel):
  """Language Model that uses the Anthropic API."""

  def __init__(
      self,
      model_name: str,
      *,
      api_key: str | None = None,
      measurements: measurements_lib.Measurements | None = None,
      channel: str = language_model.DEFAULT_STATS_CHANNEL,
      timeout: float | None = None,
  ) -> None:
    """Initializes the instance.

    Args:
      model_name: The Claude model to use, e.g. "claude-sonnet-4-5".
      api_key: The API key to use, if None will use the ANTHROPIC_API_KEY
        environment variable.
      measurements: The measurements object to log usage statistics to.
      channel: The channel to write the statistics to.
      timeout: Per-request timeout in seconds, forwarded to the underlying
        HTTP client. None uses the SDK's own default rather than leaving
        requests unbounded.
    """
    if api_key is None:
      api_key = os.getenv('ANTHROPIC_API_KEY')
      if not api_key:
        raise ValueError(
            'ANTHROPIC_API_KEY not found. Please provide it via the api_key'
            ' parameter or set the ANTHROPIC_API_KEY environment variable.'
        )
    self._model_name = model_name
    self._client = anthropic.Anthropic(
        api_key=api_key,
        **({'timeout': timeout} if timeout is not None else {}),
    )
    self._measurements = measurements
    self._channel = channel

  def _sample_text(
      self,
      prompt: str,
      *,
      max_tokens: int = language_model.DEFAULT_MAX_TOKENS,
      terminators: Collection[str] = language_model.DEFAULT_TERMINATORS,
      temperature: float = language_model.DEFAULT_TEMPERATURE,
      top_p: float = language_model.DEFAULT_TOP_P,
  ) -> str:
    response = self._client.messages.create(
        model=self._model_name,
        system=_SYSTEM_PROMPT,
        messages=[{'role': 'user', 'content': prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop_sequences=list(terminators) if terminators else None,
    )

    text = ''.join(
        block.text for block in response.content if block.type == 'text'
    )

    if self._measurements is not None:
      self._measurements.publish_datum(
          self._channel,
          {'raw_text_length': len(text), **_extract_token_usage(response)},
      )

    return text

  @override
  def sample_text(
      self,
      prompt: str,
      *,
      max_tokens: int = language_model.DEFAULT_MAX_TOKENS,
      terminators: Collection[str] = language_model.DEFAULT_TERMINATORS,
      temperature: float = language_model.DEFAULT_TEMPERATURE,
      top_p: float = language_model.DEFAULT_TOP_P,
      top_k: int = language_model.DEFAULT_TOP_K,
      timeout: float = language_model.DEFAULT_TIMEOUT_SECONDS,
      seed: int | None = None,
  ) -> str:
    del top_k, timeout, seed  # top_k/seed unsupported by the Messages API;
    # timeout is set once at construction, not per call.

    return self._sample_text(
        prompt=prompt,
        max_tokens=max_tokens,
        terminators=terminators,
        temperature=temperature,
        top_p=top_p,
    )

  @override
  def sample_choice(
      self,
      prompt: str,
      responses: Sequence[str],
      *,
      seed: int | None = None,
  ) -> tuple[int, str, dict[str, Any]]:
    del seed  # unsupported by the Messages API.

    prompt = (
        prompt
        + '\nRespond EXACTLY with one of the following strings:\n'
        + '\n'.join(responses)
        + '.'
    )

    answer = ''
    for _ in range(_MAX_MULTIPLE_CHOICE_ATTEMPTS):
      answer = self._sample_text(
          prompt, temperature=language_model.DEFAULT_TEMPERATURE
      ).strip()

      try:
        idx = responses.index(answer)
      except ValueError:
        continue
      else:
        return idx, responses[idx], {}

    raise language_model.InvalidResponseError(
        f'Too many multiple choice attempts. Last answer: {answer}'
    )
