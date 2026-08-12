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

"""Tests for real token-usage capture, and for mid-stream cancellation.

Token-usage tests guard that the wrapper reads and republishes Ollama's
real `prompt_eval_count`/`eval_count` (as `prompt_tokens`/
`completion_tokens`/`total_tokens`, matching the other providers) instead
of silently dropping them, the way the pre-existing `raw_text_length`-only
measurement did.

Cancellation tests guard `OllamaLanguageModel`'s `cancel_event` mechanism
(see `ollama_model.GenerationCancelledError`'s docstring): every call now
streams the response chunk by chunk and checks a shared `threading.Event`
between chunks, so an in-flight call can actually be interrupted instead
of always running to completion -- the fix for Concordia's "no true
mid-run cancellation" limitation, for the Ollama backend specifically.
"""

import threading
from unittest import mock

from absl.testing import absltest
from concordia.contrib.language_models.ollama import ollama_model
from concordia.utils import measurements as measurements_lib


def _fake_stream(chunks):
  """A generator standing in for `ollama.Client.generate(..., stream=True)`
  -- yields `chunks`, and returns (via the paired `Event`) whether `.close()`
  was ever called on it, the same signal `OllamaLanguageModel._generate`
  relies on to actually abort ollama-python's own underlying HTTP stream on
  cancellation.

  Returns:
    `(generator, closed_event)` -- `closed_event` is set iff `.close()` was
    called while the generator was still suspended mid-iteration (that's
    what raises `GeneratorExit` inside it); running it to natural
    exhaustion never sets it.
  """
  closed = threading.Event()

  def gen():
    try:
      yield from chunks
    except GeneratorExit:
      closed.set()
      raise

  return gen(), closed


class ExtractTokenUsageTest(absltest.TestCase):

  def test_extracts_both_counts_when_present(self):
    response = {'response': 'hi', 'prompt_eval_count': 12, 'eval_count': 34}
    self.assertEqual(
        ollama_model._extract_token_usage(response),
        {'prompt_tokens': 12, 'completion_tokens': 34, 'total_tokens': 46},
    )

  def test_omits_keys_when_server_does_not_report_them(self):
    # Not every Ollama server/model combination reports these fields --
    # absence should mean "no data", never a fabricated 0.
    response = {'response': 'hi'}
    self.assertEqual(ollama_model._extract_token_usage(response), {})

  def test_partial_counts_do_not_produce_a_total(self):
    response = {'response': 'hi', 'prompt_eval_count': 12}
    self.assertEqual(
        ollama_model._extract_token_usage(response),
        {'prompt_tokens': 12},
    )

  def test_works_with_attribute_style_response_objects(self):
    # ollama-python's GenerateResponse supports attribute access; the
    # extractor shouldn't assume a plain dict.
    response = mock.Mock(spec=['prompt_eval_count', 'eval_count'])
    response.prompt_eval_count = 5
    response.eval_count = 7
    self.assertEqual(
        ollama_model._extract_token_usage(response),
        {'prompt_tokens': 5, 'completion_tokens': 7, 'total_tokens': 12},
    )


class SampleTextPublishesTokenUsageTest(absltest.TestCase):

  def _model_with_fake_client(self, chunks, cancel_event=None):
    """`chunks`: the list of dicts `generate(..., stream=True)` would
    yield -- a single-element list behaves like one non-streaming
    response for tests that don't care about multi-chunk concatenation.
    """
    model = ollama_model.OllamaLanguageModel.__new__(
        ollama_model.OllamaLanguageModel
    )
    model._model_name = 'llama3.1'
    model._client = mock.Mock()
    model._client.generate.return_value = _fake_stream(chunks)[0]
    model._system_message = 'system'
    model._terminators = []
    model._seed = None
    model._cancel_event = cancel_event
    model._measurements = measurements_lib.Measurements()
    model._channel = 'language_model_stats'
    return model

  def test_sample_text_publishes_real_token_counts(self):
    model = self._model_with_fake_client([{
        'response': 'a continuation',
        'prompt_eval_count': 100,
        'eval_count': 20,
    }])

    model.sample_text('prompt')

    datum = model._measurements.get_last_datum('language_model_stats')
    self.assertEqual(datum['prompt_tokens'], 100)
    self.assertEqual(datum['completion_tokens'], 20)
    self.assertEqual(datum['total_tokens'], 120)

  def test_sample_text_still_publishes_when_server_omits_counts(self):
    model = self._model_with_fake_client([{'response': 'a continuation'}])

    model.sample_text('prompt')

    datum = model._measurements.get_last_datum('language_model_stats')
    self.assertNotIn('prompt_tokens', datum)
    self.assertIn('raw_text_length', datum)  # pre-existing field untouched

  def test_sample_text_concatenates_multiple_chunks(self):
    # Streaming means a real response usually arrives as many small
    # chunks, not one -- the final result must be their concatenation,
    # not just the last chunk's (typically empty, once `done: true`) text.
    model = self._model_with_fake_client([
        {'response': 'The quick '},
        {'response': 'brown fox '},
        {'response': 'jumps.', 'prompt_eval_count': 10, 'eval_count': 3},
    ])

    result = model.sample_text('prompt')

    self.assertEqual(result, 'The quick brown fox jumps.')


class CancellationTest(absltest.TestCase):
  """Covers `cancel_event`/`GenerationCancelledError` -- the mid-stream
  cancellation mechanism this module adds on top of Ollama's plain
  blocking `generate()` call (see `GenerationCancelledError`'s docstring
  for why this exists)."""

  def _model_with_fake_client(self, chunks, cancel_event):
    model = ollama_model.OllamaLanguageModel.__new__(
        ollama_model.OllamaLanguageModel
    )
    model._model_name = 'llama3.1'
    model._client = mock.Mock()
    generator, closed_event = _fake_stream(chunks)
    model._client.generate.return_value = generator
    self.closed_event = closed_event
    model._system_message = 'system'
    model._terminators = []
    model._seed = None
    model._cancel_event = cancel_event
    model._measurements = None
    model._channel = 'language_model_stats'
    return model

  def test_uncancelled_call_completes_normally(self):
    cancel_event = threading.Event()  # never set
    model = self._model_with_fake_client(
        [{'response': 'all good'}], cancel_event
    )

    self.assertEqual(model.sample_text('prompt'), 'all good')
    self.assertFalse(self.closed_event.is_set())

  def test_cancel_event_set_before_the_call_raises_immediately(self):
    cancel_event = threading.Event()
    cancel_event.set()  # already set before sample_text is even called
    model = self._model_with_fake_client(
        [{'response': 'should never be read'}], cancel_event
    )

    with self.assertRaises(ollama_model.GenerationCancelledError):
      model.sample_text('prompt')
    # The request is never even sent in this case -- nothing to close.
    model._client.generate.assert_not_called()

  def test_cancel_event_set_mid_stream_stops_before_later_chunks(self):
    # cancel_event.set() here is a side effect of *consuming* the first
    # chunk (not of building this generator -- it's lazy), standing in for
    # an external stop() firing while a call is already in flight.
    cancel_event = threading.Event()

    def chunks():
      yield {'response': 'first '}
      cancel_event.set()
      yield {'response': 'second '}
      yield {'response': 'third'}  # must never be pulled from the stream

    model = self._model_with_fake_client(chunks(), cancel_event)

    with self.assertRaises(ollama_model.GenerationCancelledError):
      model.sample_text('prompt')
    # Cancellation is only noticed after 'second ' is already pulled (it
    # was produced as a side effect of resuming the generator, before the
    # check below it runs) -- but the stream is closed right there,
    # before 'third' is ever requested.
    self.assertTrue(self.closed_event.is_set())

  def test_sample_choice_is_also_cancellable(self):
    cancel_event = threading.Event()
    cancel_event.set()
    model = self._model_with_fake_client(
        [{'response': '{"choice": "a"}'}], cancel_event
    )

    with self.assertRaises(ollama_model.GenerationCancelledError):
      model.sample_choice('prompt', ('a', 'b'))


if __name__ == '__main__':
  absltest.main()
