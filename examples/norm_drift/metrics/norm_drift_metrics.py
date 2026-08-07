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

"""Norm-drift metrics computed from resource_dilemma step logs.

Pure-Python, no LLM or Concordia dependency — operates on the
``step_logs`` list produced by
``examples.resource_dilemma.resource_logger.ResourceLoggerState`` (or the
cycle-continuous merge of two phases produced by
``examples.norm_drift.scenarios.village_commons.run_two_phase_simulation``).

Three metrics:
  - ``gini_coefficient``: inequality of a single cycle's harvest
    distribution across agents.
  - ``per_cycle_gini``: the above, computed for every cycle in a run.
  - ``convergence_variance``: how much the per-cycle Gini series has
    settled down over a trailing window (low variance == converged norm).
  - ``perturbation_recovery``: whether/when the harvest-inequality norm
    recovers to within tolerance of its pre-perturbation baseline after a
    mid-run shock.
"""

import dataclasses
import statistics


def gini_coefficient(values: list[float]) -> float:
  """Compute the Gini coefficient of a list of non-negative values.

  Returns 0.0 for a fully equal distribution and approaches 1.0 for a
  maximally unequal one. Returns 0.0 for fewer than two values or when all
  values are zero (no meaningful inequality to measure).

  Args:
    values: Non-negative values (e.g. one cycle's harvest amount per agent).
  """
  n = len(values)
  if n < 2:
    return 0.0
  total = sum(values)
  if total <= 0:
    return 0.0
  sorted_values = sorted(values)
  cumulative = sum(
      (i + 1) * value for i, value in enumerate(sorted_values)
  )
  return (2.0 * cumulative) / (n * total) - (n + 1.0) / n


def per_cycle_gini(step_logs: list[dict]) -> dict[int, float]:
  """Compute the Gini coefficient of harvest distribution for each cycle.

  Reads the 'summary' phase step logs (one per cycle, produced by
  ``ResourceStepLoggerComponent._log_cycle_summary``), each of which carries
  an 'action'.'harvests' dict of agent_name -> hectares grazed that cycle.

  Args:
    step_logs: The combined step log list from one or more simulation runs.

  Returns:
    Dict mapping cycle number to that cycle's Gini coefficient.
  """
  result = {}
  for log in step_logs:
    if log.get('phase') != 'summary':
      continue
    cycle = log.get('cycle')
    harvests = log.get('action', {}).get('harvests', {})
    if cycle is None or not harvests:
      continue
    result[cycle] = gini_coefficient(list(harvests.values()))
  return result


def convergence_variance(
    gini_series: dict[int, float], window: int = 3
) -> float | None:
  """Variance of the Gini series over its trailing window of cycles.

  A small value indicates the harvest-inequality norm has stabilised; a
  large value indicates the population is still shifting its behaviour.

  Args:
    gini_series: Dict mapping cycle number to Gini coefficient, as returned
      by ``per_cycle_gini``.
    window: Number of trailing cycles (by cycle number, not list position)
      to compute variance over.

  Returns:
    The sample variance of the trailing window's Gini values, or None if
    there are fewer than two cycles available in the window.
  """
  if not gini_series:
    return None
  cycles = sorted(gini_series.keys())
  trailing_cycles = cycles[-window:]
  trailing_values = [gini_series[c] for c in trailing_cycles]
  if len(trailing_values) < 2:
    return None
  return statistics.variance(trailing_values)


@dataclasses.dataclass
class RecoveryResult:
  """Result of a perturbation-recovery analysis.

  Attributes:
    baseline_gini: Mean Gini over the pre-perturbation cycles used as the
      reference point.
    tolerance_band: The absolute Gini distance from baseline_gini within
      which a cycle counts as "recovered".
    recovered: Whether the series ever returned within tolerance_band of
      baseline_gini after the perturbation.
    recovery_cycle: The first post-perturbation cycle number at which the
      series was within tolerance_band, or None if it never recovered.
    cycles_to_recovery: recovery_cycle - perturbation_round, or None.
    verdict: One of 'stable' (never left the tolerance band), 'drifted'
      (left and came back to a nearby-but-different level, or never
      returned), or 'no_recovery' (left the band and never returned within
      the observed cycles).
  """

  baseline_gini: float
  tolerance_band: float
  recovered: bool
  recovery_cycle: int | None
  cycles_to_recovery: int | None
  verdict: str


def perturbation_recovery(
    gini_series: dict[int, float],
    perturbation_round: int,
    relative_tolerance: float = 0.15,
    min_tolerance: float = 0.03,
    baseline_window: int = 3,
) -> RecoveryResult:
  """Detect whether/when harvest inequality recovers after a perturbation.

  The tolerance band is ``max(relative_tolerance * baseline_gini,
  min_tolerance)``. A purely relative band collapses toward zero whenever
  the pre-perturbation population happens to converge on a near-perfectly
  equal split (baseline Gini near 0) — in that regime a fractional-only
  band is narrower than the run-to-run noise in the Gini estimate itself,
  so "recovery" could never be detected even when the population visibly
  settles back down. The absolute floor (``min_tolerance``) keeps the band
  meaningful in that near-zero-baseline case, without materially loosening
  it when the baseline is already large.

  Args:
    gini_series: Dict mapping cycle number to Gini coefficient, as returned
      by ``per_cycle_gini``. Must include cycles before and after
      ``perturbation_round``.
    perturbation_round: The cycle (1-indexed) at which the perturbation was
      introduced. Cycles < perturbation_round are treated as baseline;
      cycles >= perturbation_round are checked for recovery.
    relative_tolerance: Fractional tolerance band relative to baseline_gini.
    min_tolerance: Absolute floor for the tolerance band, in Gini units.
    baseline_window: Number of cycles immediately before perturbation_round
      to average for the baseline.

  Returns:
    A RecoveryResult summarising the outcome.
  """
  cycles = sorted(gini_series.keys())
  baseline_cycles = [
      c for c in cycles if c < perturbation_round
  ][-baseline_window:]
  post_cycles = [c for c in cycles if c >= perturbation_round]

  if not baseline_cycles or not post_cycles:
    raise ValueError(
        'gini_series must include at least one cycle before and one cycle'
        f' at/after perturbation_round={perturbation_round}; got'
        f' cycles={cycles}.'
    )

  baseline_gini = statistics.mean(gini_series[c] for c in baseline_cycles)
  tolerance_band = max(relative_tolerance * baseline_gini, min_tolerance)

  recovery_cycle = None
  ever_left_band = False
  for cycle in post_cycles:
    within_band = abs(gini_series[cycle] - baseline_gini) <= tolerance_band
    if not within_band:
      ever_left_band = True
    elif ever_left_band and recovery_cycle is None:
      recovery_cycle = cycle

  # If the series never left the band, it is trivially "recovered" at the
  # perturbation round itself (nothing to recover from).
  if not ever_left_band:
    recovered = True
    recovery_cycle = perturbation_round
    verdict = 'stable'
  elif recovery_cycle is not None:
    recovered = True
    verdict = (
        'stable'
        if abs(gini_series[cycles[-1]] - baseline_gini) <= tolerance_band
        else 'drifted'
    )
  else:
    recovered = False
    verdict = 'no_recovery'

  cycles_to_recovery = (
      recovery_cycle - perturbation_round
      if recovery_cycle is not None
      else None
  )

  return RecoveryResult(
      baseline_gini=baseline_gini,
      tolerance_band=tolerance_band,
      recovered=recovered,
      recovery_cycle=recovery_cycle,
      cycles_to_recovery=cycles_to_recovery,
      verdict=verdict,
  )
