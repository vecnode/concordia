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

"""Village commons scenario: a norm-drift extension of the pasture CPR.

Reuses ``examples.resource_dilemma.scenarios.pasture`` for config-building
and the harvesting/discussion Game Master engine unchanged — this module
only adds a two-phase runner that introduces a single mid-run perturbation
(a rule change or a newcomer agent) and stitches the two phases' resource
level and step logs together so norm-drift metrics can be computed over the
full timeline.

Design note / known limitation: Phase A and Phase B are run as two separate
``Simulation.play()`` calls, not a single continuous engine run. The
resource level is carried over manually between phases, but agents'
in-context memory of Phase A is only available to Phase B agents through the
shared-memory pre-load (i.e. there is no live conversational continuity
across the phase boundary). This is a genuine phase boundary, not a fully
continuous simulation — see the README for why this is an acceptable
approximation for the paper's purposes, and what a continuous-engine version
would require.

Default cycle budget rationale: empirically, per-cycle Gini stabilizes
within 2-3 cycles, so a 3-cycle baseline (DEFAULT_PERTURBATION_ROUND=4) is
enough to establish it. The bulk of the cycle budget
(DEFAULT_NUM_CYCLES - DEFAULT_PERTURBATION_ROUND) should go to Phase B --
a short baseline with a short post-perturbation window produces recovery
verdicts based on only 1-2 observed cycles after the shock, which is not
enough to distinguish "recovered" from "the run ended before drift had a
chance to show up" (see analysis.MIN_POST_PERTURBATION_CYCLES_FOR_MEANING).
"""

import dataclasses

from concordia.typing import prefab as prefab_lib
from examples.norm_drift.personas import village_personas
from examples.resource_dilemma import resource_logger
from examples.resource_dilemma import simulation_state as sim_state_lib
from examples.resource_dilemma.scenarios import pasture
from concordia.prefabs.simulation import generic as simulation
from concordia.environment.engines import simultaneous

DEFAULT_NUM_CYCLES = 16
DEFAULT_PERTURBATION_ROUND = 4
DEFAULT_CAPACITY = 100

PERTURBATION_TYPES = ('rule_change', 'newcomer', 'none')


def _rule_change_notice(round_number: int) -> str:
  return f"""\
  ============================================================
  VILLAGE ANNOUNCEMENT — Cycle {round_number}
  ============================================================
  The village council has just announced a change to the informal grazing
  customs: herders are now strongly encouraged to keep their grazing to no
  more than 12 hectares per cycle, down from the previous informal norm of
  up to 20 hectares, in order to protect the commons for the community.
  There is no formal enforcement — this is a norm shift, not a binding
  rule — but the council expects herders to adjust their behaviour
  accordingly starting this cycle.
"""


def _newcomer_notice(newcomer_name: str, round_number: int) -> str:
  return f"""\
  ============================================================
  VILLAGE ANNOUNCEMENT — Cycle {round_number}
  ============================================================
  A newcomer, {newcomer_name}, has just been granted grazing rights on the
  commons starting this cycle and will graze alongside the existing
  herders from now on. {newcomer_name} is not yet familiar with the
  village's established grazing customs.
"""


def build_phase_config(
    player_configs,
    num_cycles: int,
    embedder=None,
    extra_premise: str = '',
):
  """Build a Config for one phase of the village commons run.

  Thin wrapper around ``pasture.build_config`` (standard mode only — the
  norm-drift experiment does not use elections) that optionally appends a
  perturbation notice to the premise.

  Args:
    player_configs: Dict of villager persona configs for this phase.
    num_cycles: Number of cycles in this phase.
    embedder: Sentence embedder for pre-loading memories.
    extra_premise: Text appended to the phase's premise (e.g. a perturbation
      announcement). Empty string for no perturbation.

  Returns:
    A prefab_lib.Config ready for Simulation.
  """
  config = pasture.build_config(
      player_configs=player_configs,
      num_cycles=num_cycles,
      mode='standard',
      embedder=embedder,
  )
  if extra_premise:
    config = dataclasses.replace(
        config, default_premise=config.default_premise + '\n\n' + extra_premise
    )
  return config


def _run_phase(
    config,
    model,
    embedder,
    initial_resources: float,
    num_cycles: int,
    html_output_path: str | None,
    cycle_offset: int,
):
  """Run one phase and return (final_resource_level, step_logs)."""
  sim_state = sim_state_lib.ResourceSimulationState(
      initial_resources=initial_resources, num_cycles=num_cycles
  )
  player_names = [
      inst.params['name']
      for inst in config.instances
      if inst.role == prefab_lib.Role.ENTITY
  ]
  logger_state = resource_logger.ResourceLoggerState(
      initial_resources=initial_resources,
      num_cycles=num_cycles,
      html_output_path=html_output_path,
      max_steps=config.default_max_steps,
      player_names=player_names,
      sim_state=sim_state,
  )
  for instance in config.instances:
    if instance.role == prefab_lib.Role.GAME_MASTER:
      instance.params['logger_state'] = logger_state
      instance.params['sim_state'] = sim_state

  engine = simultaneous.Simultaneous()
  sim = simulation.Simulation(
      config=config, model=model, embedder=embedder, engine=engine
  )
  sim.play()

  # Re-tag step logs onto the global cycle timeline for cross-phase metrics.
  offset_logs = []
  for log in logger_state.step_logs:
    new_log = dict(log)
    if 'cycle' in new_log and isinstance(new_log['cycle'], int):
      new_log['cycle'] = new_log['cycle'] + cycle_offset
    offset_logs.append(new_log)

  return sim_state.resource_level, offset_logs


def run_two_phase_simulation(
    model,
    embedder,
    num_cycles: int = DEFAULT_NUM_CYCLES,
    perturbation_round: int = DEFAULT_PERTURBATION_ROUND,
    perturbation_type: str = 'rule_change',
    html_output_dir: str | None = None,
):
  """Run the norm-drift village commons experiment.

  Phase A runs cycles [1, perturbation_round - 1] with the unperturbed
  village. At the perturbation round, either a rule-change notice is
  injected into the premise, or a newcomer agent is added to the roster.
  Phase B then runs the remaining cycles [perturbation_round, num_cycles]
  with the resource level carried over from the end of Phase A.

  Args:
    model: Language model (or mock, if disabled) shared by both phases.
    embedder: Sentence embedder shared by both phases.
    num_cycles: Total cycles across both phases.
    perturbation_round: The cycle (1-indexed) at which the perturbation is
      introduced. Must satisfy 1 < perturbation_round <= num_cycles.
    perturbation_type: One of 'rule_change', 'newcomer', or 'none' (the
      'none' baseline runs the full simulation with no shock, for
      comparison).
    html_output_dir: Optional directory for phase HTML logs.

  Returns:
    A dict with keys 'step_logs' (combined, cycle-continuous list of step
    log dicts across both phases) and 'perturbation_round'.
  """
  if perturbation_type not in PERTURBATION_TYPES:
    raise ValueError(
        f'Unknown perturbation_type={perturbation_type!r}; expected one of'
        f' {PERTURBATION_TYPES}.'
    )
  if not (1 < perturbation_round <= num_cycles):
    raise ValueError(
        'perturbation_round must satisfy 1 < perturbation_round <='
        f' num_cycles (got perturbation_round={perturbation_round},'
        f' num_cycles={num_cycles}).'
    )

  phase_a_cycles = perturbation_round - 1
  phase_b_cycles = num_cycles - phase_a_cycles

  phase_a_html = (
      f'{html_output_dir}/village_commons_phase_a.html'
      if html_output_dir
      else None
  )
  phase_b_html = (
      f'{html_output_dir}/village_commons_phase_b.html'
      if html_output_dir
      else None
  )

  # --- Phase A: unperturbed baseline ---
  phase_a_config = build_phase_config(
      player_configs=village_personas.VILLAGERS,
      num_cycles=phase_a_cycles,
      embedder=embedder,
  )
  resource_level, logs_a = _run_phase(
      config=phase_a_config,
      model=model,
      embedder=embedder,
      initial_resources=DEFAULT_CAPACITY,
      num_cycles=phase_a_cycles,
      html_output_path=phase_a_html,
      cycle_offset=0,
  )

  # --- Perturbation ---
  phase_b_player_configs = dict(village_personas.VILLAGERS)
  extra_premise = ''
  if perturbation_type == 'rule_change':
    extra_premise = _rule_change_notice(perturbation_round)
  elif perturbation_type == 'newcomer':
    newcomer_name = next(iter(village_personas.NEWCOMER.values()))['Name']
    phase_b_player_configs.update(village_personas.NEWCOMER)
    extra_premise = _newcomer_notice(newcomer_name, perturbation_round)
  # perturbation_type == 'none': no change, used as a control run.

  # --- Phase B: post-perturbation ---
  phase_b_config = build_phase_config(
      player_configs=phase_b_player_configs,
      num_cycles=phase_b_cycles,
      embedder=embedder,
      extra_premise=extra_premise,
  )
  _, logs_b = _run_phase(
      config=phase_b_config,
      model=model,
      embedder=embedder,
      initial_resources=resource_level,
      num_cycles=phase_b_cycles,
      html_output_path=phase_b_html,
      cycle_offset=phase_a_cycles,
  )

  return {
      'step_logs': logs_a + logs_b,
      'perturbation_round': perturbation_round,
      'phase_a_cycles': phase_a_cycles,
      'phase_b_cycles': phase_b_cycles,
      'perturbation_type': perturbation_type,
  }
