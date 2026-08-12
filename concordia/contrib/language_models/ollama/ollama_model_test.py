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

"""Tests for real token-usage capture -- Ollama actually reports
`prompt_eval_count`/`eval_count` on its generate() response; these tests
guard that the wrapper reads and republishes them (as
`prompt_tokens`/`completion_tokens`/`total_tokens`, matching the other
providers) instead of silently dropping them, the way the pre-existing
`raw_text_length`-only measurement did.
"""

from unittest import mock

from absl.testing import absltest
from concordia.contrib.language_models.ollama import ollama_model
from concordia.utils import measurements as measurements_lib


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

  def _model_with_fake_client(self, generate_return):
    model = ollama_model.OllamaLanguageModel.__new__(
        ollama_model.OllamaLanguageModel
    )
    model._model_name = 'llama3.1'
    model._client = mock.Mock()
    model._client.generate.return_value = generate_return
    model._system_message = 'system'
    model._terminators = []
    model._seed = None
    model._measurements = measurements_lib.Measurements()
    model._channel = 'language_model_stats'
    return model

  def test_sample_text_publishes_real_token_counts(self):
    model = self._model_with_fake_client({
        'response': 'a continuation',
        'prompt_eval_count': 100,
        'eval_count': 20,
    })

    model.sample_text('prompt')

    datum = model._measurements.get_last_datum('language_model_stats')
    self.assertEqual(datum['prompt_tokens'], 100)
    self.assertEqual(datum['completion_tokens'], 20)
    self.assertEqual(datum['total_tokens'], 120)

  def test_sample_text_still_publishes_when_server_omits_counts(self):
    model = self._model_with_fake_client({'response': 'a continuation'})

    model.sample_text('prompt')

    datum = model._measurements.get_last_datum('language_model_stats')
    self.assertNotIn('prompt_tokens', datum)
    self.assertIn('raw_text_length', datum)  # pre-existing field untouched


if __name__ == '__main__':
  absltest.main()
