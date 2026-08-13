# Copyright 2023 DeepMind Technologies Limited.
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

"""A modular entity agent that supports logging from components."""

from collections.abc import Mapping
import types
from typing import Any

from concordia.agents import entity_agent
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component
from concordia.utils import measurements as measurements_lib


class _LoggingChannelPublisher:
  """Picklable replacement for the closure `EntityAgentWithLogging.__init__`
  used to wire a component's logging channel to `self._component_logging`.

  A `lambda` capturing `self` (as the old wiring did) can never be pickled
  -- `pickle` has no way to serialize a function defined inside another
  function's scope, regardless of what it references (`TypeError: cannot
  pickle 'function' object` / `AttributeError: Can't get local object
  '...<locals>.<lambda>'`, depending on Python version). A plain,
  module-level callable object has no such restriction: as long as the
  agent it references is itself picklable, so is this. The reference back
  to `agent` does create a cycle (agent -> component -> this -> agent),
  but `pickle` (and `copy`) handle reference cycles natively via
  memoization, so that's not a problem.
  """

  def __init__(self, agent: 'EntityAgentWithLogging', channel_name: str):
    self._agent = agent
    self._channel_name = channel_name

  def __call__(self, datum: Any) -> None:
    self._agent._component_logging.publish_datum(  # pylint: disable=protected-access
        self._channel_name,
        datum,
        capture_key=self._agent._active_capture_key,  # pylint: disable=protected-access
    )


class EntityAgentWithLogging(entity_agent.EntityAgent,
                             entity_lib.EntityWithLogging):
  """An agent that exposes the latest information of each component."""

  def __init__(
      self,
      agent_name: str,
      act_component: entity_component.ActingComponent,
      context_components: Mapping[str, entity_component.ContextComponent] = (
          types.MappingProxyType({})
      ),
      measurements: measurements_lib.Measurements | None = None,
  ):
    """Initializes the agent.

    The passed components will be owned by this entity agent (i.e. their
    `set_entity` method will be called with this entity as the argument).

    Whenever `get_last_log` is called, the latest values published in all the
    channels in the given measurements object will be returned as a mapping of
    channel name to value.

    Args:
      agent_name: The name of the agent.
      act_component: The component that will be used to act.
      context_components: The ContextComponents that will be used by the agent.
      measurements: Optional measurements instance to use for logging. Defaults
        to a standard Measurements().
    """
    super().__init__(
        agent_name=agent_name,
        act_component=act_component,
        context_components=context_components,
    )
    self._component_logging = (
        measurements
        if measurements is not None
        else measurements_lib.Measurements()
    )

    for component_name, component in self._context_components.items():
      if isinstance(component, entity_component.ComponentWithLogging):
        component.set_logging_channel(
            _LoggingChannelPublisher(self, component_name)
        )
    if isinstance(act_component, entity_component.ComponentWithLogging):
      act_component.set_logging_channel(
          _LoggingChannelPublisher(self, '__act__')
      )

  # Per-thread capture key routing for async log isolation. The async
  # engine registers entity_name -> thread_id mappings so that when
  # multiple entity threads share a game master, each thread's
  # game_master.act() publishes log data with the calling entity's name
  # as the capture_key, not the game master's name.

  def set_capture_key_for_thread(
      self, thread_id: int, entity_name: str
  ) -> None:
    """Register a per-thread capture key for async log isolation."""
    if not hasattr(self, '_capture_key_by_thread'):
      self._capture_key_by_thread = {}
    self._capture_key_by_thread[thread_id] = entity_name

  def clear_capture_key_for_thread(self, thread_id: int) -> None:
    """Remove the per-thread capture key when the entity loop exits."""
    if hasattr(self, '_capture_key_by_thread'):
      self._capture_key_by_thread.pop(thread_id, None)

  @property
  def measurements(self) -> measurements_lib.Measurements:
    return self._component_logging

  def get_all_logs(self):
    return self._component_logging.get_all_channels()

  def get_last_log(self):
    log: dict[str, Any] = {}
    for channel_name in sorted(self._component_logging.available_channels()):
      log[channel_name] = self._component_logging.get_last_datum(channel_name)
    return log
