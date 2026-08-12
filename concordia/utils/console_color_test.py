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

import io
from unittest import mock

from absl.testing import absltest
from concordia.utils import console_color


class ColoredTest(absltest.TestCase):

  def test_returns_plain_text_when_stdout_is_not_a_tty(self):
    # A plain io.StringIO (what a captured/piped stdout looks like) has no
    # isatty() at all -- exactly the case that used to leak ANSI escape
    # codes into a subprocess-captured transcript.
    with mock.patch('sys.stdout', io.StringIO()):
      self.assertEqual(console_color.colored('hello', 'red'), 'hello')

  def test_returns_colored_text_when_stdout_is_a_tty(self):
    fake_stdout = mock.Mock()
    fake_stdout.isatty.return_value = True
    # termcolor.colored does its own (redundant, from here on) tty check
    # via os.isatty(sys.stdout.fileno()) -- give it a real fd number and
    # patch os.isatty too, so a fake stdout's non-real fileno() doesn't
    # make that second check blow up.
    fake_stdout.fileno.return_value = 1
    with (
        mock.patch('sys.stdout', fake_stdout),
        mock.patch('os.isatty', return_value=True),
    ):
      result = console_color.colored('hello', 'red')
    self.assertIn('hello', result)
    self.assertNotEqual(result, 'hello')  # escape codes were added

  def test_returns_plain_text_when_isatty_returns_false(self):
    fake_stdout = mock.Mock()
    fake_stdout.isatty.return_value = False
    with mock.patch('sys.stdout', fake_stdout):
      self.assertEqual(console_color.colored('hello', 'red'), 'hello')


if __name__ == '__main__':
  absltest.main()
