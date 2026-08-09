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

"""Tests for language_model_setup's opt-in kwarg forwarding (seed, timeout).

Both `seed` and `timeout` are forwarded to a backend's constructor only if
that backend's __init__ actually declares the parameter (checked via
introspection), so unsupported backends silently ignore it rather than
crashing. This is exercised here against fake backend classes rather than
real provider SDKs, so it runs with no network/API dependency.
"""

from absl.testing import absltest
from concordia.contrib import language_models
from concordia.language_model import language_model


class _SupportsTimeoutAndSeed(language_model.LanguageModel):
  """Fake backend whose __init__ declares both seed and timeout."""

  def __init__(self, model_name, seed=None, timeout=None):
    self.model_name = model_name
    self.seed = seed
    self.timeout = timeout

  def sample_text(self, *args, **kwargs):
    raise NotImplementedError

  def sample_choice(self, *args, **kwargs):
    raise NotImplementedError


class _DoesNotSupportTimeoutOrSeed(language_model.LanguageModel):
  """Fake backend whose __init__ declares neither seed nor timeout."""

  def __init__(self, model_name):
    self.model_name = model_name

  def sample_text(self, *args, **kwargs):
    raise NotImplementedError

  def sample_choice(self, *args, **kwargs):
    raise NotImplementedError


class LanguageModelSetupTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._original_registry = dict(language_models._REGISTRY)  # pylint: disable=protected-access
    self._original_import_model = language_models._import_model  # pylint: disable=protected-access

  def tearDown(self):
    language_models._REGISTRY = self._original_registry  # pylint: disable=protected-access
    language_models._import_model = self._original_import_model  # pylint: disable=protected-access
    super().tearDown()

  def _register_fake(self, api_type: str, cls: type) -> None:
    language_models._REGISTRY = {  # pylint: disable=protected-access
        **language_models._REGISTRY,  # pylint: disable=protected-access
        api_type: 'fake.path',
    }
    language_models._import_model = lambda _: cls  # pylint: disable=protected-access

  def test_forwards_timeout_when_backend_supports_it(self):
    self._register_fake('fake_with_support', _SupportsTimeoutAndSeed)
    model = language_models.language_model_setup(
        api_type='fake_with_support',
        model_name='m',
        seed=42,
        timeout=90.0,
    )
    self.assertEqual(model.seed, 42)
    self.assertEqual(model.timeout, 90.0)

  def test_omits_timeout_when_backend_does_not_support_it(self):
    self._register_fake('fake_without_support', _DoesNotSupportTimeoutOrSeed)
    # Must not raise (e.g. TypeError: unexpected keyword argument 'timeout')
    # even though timeout/seed are requested -- the backend just doesn't get
    # them.
    model = language_models.language_model_setup(
        api_type='fake_without_support',
        model_name='m',
        seed=42,
        timeout=90.0,
    )
    self.assertEqual(model.model_name, 'm')

  def test_timeout_none_is_not_forwarded_even_if_supported(self):
    self._register_fake('fake_with_support2', _SupportsTimeoutAndSeed)
    model = language_models.language_model_setup(
        api_type='fake_with_support2',
        model_name='m',
    )
    self.assertIsNone(model.timeout)


if __name__ == '__main__':
  absltest.main()
