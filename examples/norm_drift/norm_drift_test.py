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

"""Smoke tests for the norm-drift village commons experiment.

Mirrors examples.resource_dilemma.resource_dilemma_test: runs the two-phase
simulation with a mock language model and dummy embedder to verify that
config-building, the perturbation wiring (rule_change / newcomer / none),
and the engine run complete without errors. No LLM is required.

Note: with the mock model, agent responses don't contain parseable
"GRAZE X" text, so per-cycle Gini series come back empty — these tests
verify the metrics functions handle that gracefully (no crash), not that
they produce meaningful values. Meaningful values require a real model;
see README.md for how to run against one.
"""

from absl.testing import absltest
from absl.testing import parameterized
from examples.norm_drift.metrics import norm_drift_metrics
from examples.norm_drift.scenarios import village_commons
from concordia.language_model import no_language_model
import numpy as np


def _mock_embedder(text: str) -> np.ndarray:
  del text
  return np.ones(384)


class NormDriftTest(parameterized.TestCase):
  """Smoke tests for the norm-drift village commons scenario."""

  @parameterized.named_parameters(
      dict(testcase_name=name, perturbation_type=name)
      for name in village_commons.PERTURBATION_TYPES
  )
  def test_two_phase_simulation_runs_to_completion(self, perturbation_type):
    """Verifies each perturbation type runs both phases without errors."""
    model = no_language_model.NoLanguageModel()
    result = village_commons.run_two_phase_simulation(
        model=model,
        embedder=_mock_embedder,
        num_cycles=2,
        perturbation_round=2,
        perturbation_type=perturbation_type,
    )
    self.assertIn('step_logs', result)
    self.assertEqual(result['perturbation_type'], perturbation_type)
    self.assertEqual(result['phase_a_cycles'], 1)
    self.assertEqual(result['phase_b_cycles'], 1)

  def test_newcomer_only_appears_in_phase_b(self):
    """Verifies the newcomer persona is absent from Phase A step logs."""
    model = no_language_model.NoLanguageModel()
    result = village_commons.run_two_phase_simulation(
        model=model,
        embedder=_mock_embedder,
        num_cycles=2,
        perturbation_round=2,
        perturbation_type='newcomer',
    )
    newcomer_name = next(iter(village_commons.village_personas.NEWCOMER.values()))[
        'Name'
    ]
    phase_a_logs = [
        log for log in result['step_logs']
        if log.get('cycle', 0) < result['phase_a_cycles'] + 1
    ]
    for log in phase_a_logs:
      self.assertNotIn(newcomer_name, str(log))

  def test_invalid_perturbation_round_raises(self):
    model = no_language_model.NoLanguageModel()
    with self.assertRaises(ValueError):
      village_commons.run_two_phase_simulation(
          model=model,
          embedder=_mock_embedder,
          num_cycles=4,
          perturbation_round=1,  # Must be > 1.
          perturbation_type='rule_change',
      )
    with self.assertRaises(ValueError):
      village_commons.run_two_phase_simulation(
          model=model,
          embedder=_mock_embedder,
          num_cycles=4,
          perturbation_round=5,  # Must be <= num_cycles.
          perturbation_type='rule_change',
      )

  def test_invalid_perturbation_type_raises(self):
    model = no_language_model.NoLanguageModel()
    with self.assertRaises(ValueError):
      village_commons.run_two_phase_simulation(
          model=model,
          embedder=_mock_embedder,
          num_cycles=4,
          perturbation_round=2,
          perturbation_type='not_a_real_type',
      )

  def test_metrics_pipeline_handles_mock_model_output_gracefully(self):
    """With the mock model, harvests won't parse; metrics must not crash."""
    model = no_language_model.NoLanguageModel()
    result = village_commons.run_two_phase_simulation(
        model=model,
        embedder=_mock_embedder,
        num_cycles=2,
        perturbation_round=2,
        perturbation_type='rule_change',
    )
    gini_series = norm_drift_metrics.per_cycle_gini(result['step_logs'])
    self.assertIsInstance(gini_series, dict)
    variance = norm_drift_metrics.convergence_variance(gini_series)
    self.assertIsNone(variance) if not gini_series else None


class NormDriftMetricsTest(absltest.TestCase):
  """Pure-Python unit tests for norm_drift_metrics (no simulation needed)."""

  def test_gini_coefficient_equal_distribution_is_zero(self):
    self.assertAlmostEqual(
        norm_drift_metrics.gini_coefficient([10.0, 10.0, 10.0, 10.0]), 0.0
    )

  def test_gini_coefficient_maximal_inequality_approaches_one(self):
    gini = norm_drift_metrics.gini_coefficient([0.0, 0.0, 0.0, 100.0])
    self.assertGreater(gini, 0.7)

  def test_gini_coefficient_handles_all_zero(self):
    self.assertEqual(norm_drift_metrics.gini_coefficient([0.0, 0.0, 0.0]), 0.0)

  def test_gini_coefficient_handles_single_value(self):
    self.assertEqual(norm_drift_metrics.gini_coefficient([42.0]), 0.0)

  def test_perturbation_recovery_near_zero_baseline_uses_absolute_floor(self):
    # Regression test: a purely relative tolerance band collapses to near
    # nothing when the pre-perturbation population happens to converge on
    # a near-perfectly equal split, making recovery undetectable. The
    # min_tolerance floor must keep the band meaningful in that regime.
    gini_series = {
        1: 0.001, 2: 0.001, 3: 0.001,  # Baseline: near-zero inequality.
        4: 0.25,  # Perturbation shock.
        5: 0.1, 6: 0.03, 7: 0.01, 8: 0.005,  # Gradual recovery.
    }
    result = norm_drift_metrics.perturbation_recovery(
        gini_series, perturbation_round=4
    )
    self.assertGreater(result.tolerance_band, 0.001 * 0.15)
    self.assertTrue(result.recovered)
    self.assertIsNotNone(result.recovery_cycle)

  def test_perturbation_recovery_raises_without_cycles_on_both_sides(self):
    gini_series = {1: 0.1, 2: 0.1, 3: 0.1}
    with self.assertRaises(ValueError):
      norm_drift_metrics.perturbation_recovery(
          gini_series, perturbation_round=5
      )


if __name__ == '__main__':
  absltest.main()
