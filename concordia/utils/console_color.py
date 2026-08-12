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


"""Terminal color helper that respects non-interactive stdout.

`termcolor.colored` colors unconditionally by default; library code that
prints simulation transcripts (the engines' own verbose `run_loop`
output, several Game Master components' debug prints) wants automatic
detection instead -- a caller piping stdout (a subprocess harness, a
dashboard capturing output for a UI, `python foo.py > log.txt`) should
get plain text, the same way any well-behaved CLI tool does when it
isn't attached to a terminal. Without this, escape codes like
`\x1b[36m...\x1b[0m` show up as literal text wherever that stdout ends
up rendered.

`termcolor.colored` itself grew equivalent auto-detection later
(`can_colorize`, checking `NO_COLOR`/`FORCE_COLOR`/tty), but leaning on
it here would mean every future call site has to remember not to pass
`force_color=True`, and its decision is cached process-wide at first
call via `functools.cache` -- a long-running process whose stdout
changes destination mid-run (redirected, then reattached) wouldn't be
picked up. Checking `sys.stdout.isatty()` fresh on every call is simpler
to reason about and only costs anything on the already-slow, already-
verbose print path.
"""

from __future__ import annotations

import sys
from typing import Any

import termcolor


def colored(text: Any, *args: Any, **kwargs: Any) -> str:
  """`termcolor.colored`, but a no-op (returns `str(text)`) unless
  `sys.stdout` is a real terminal.

  Same signature as `termcolor.colored` -- drop-in replacement for
  `termcolor.colored(text, color, on_color, attrs)` call sites.
  """
  if not _stdout_is_a_tty():
    return str(text)
  return termcolor.colored(text, *args, **kwargs)


def _stdout_is_a_tty() -> bool:
  isatty = getattr(sys.stdout, 'isatty', None)
  return bool(isatty and isatty())
