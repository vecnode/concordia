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

"""Tests for norm_drift.analysis. Pure Python, no LLM or simulation run."""

import json
import pathlib
import tempfile

from absl.testing import absltest
from examples.norm_drift import analysis


def _write_run(
    sweep_dir: pathlib.Path,
    run_id: str,
    perturbation_round: int,
    gini_series: dict[int, float],
    verdict: str,
    cycles_to_recovery,
    convergence_variance: float = 0.001,
) -> None:
  run_dir = sweep_dir / run_id
  run_dir.mkdir(parents=True)
  data = {
      'perturbation_round': perturbation_round,
      'perturbation_type': run_id.rsplit('_seed', 1)[0],
      'gini_series': {str(k): v for k, v in gini_series.items()},
      'convergence_variance': convergence_variance,
      'recovery': {
          'baseline_gini': 0.15,
          'tolerance_band': 0.03,
          'recovered': verdict != 'no_recovery',
          'recovery_cycle': None if cycles_to_recovery is None else (
              perturbation_round + cycles_to_recovery
          ),
          'cycles_to_recovery': cycles_to_recovery,
          'verdict': verdict,
      },
  }
  (run_dir / 'norm_drift_metrics.json').write_text(json.dumps(data))


class AnalysisTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._tmpdir = tempfile.TemporaryDirectory()
    self.sweep_dir = pathlib.Path(self._tmpdir.name)
    self.addCleanup(self._tmpdir.cleanup)

  def test_load_run_parses_perturbation_type_and_seed(self):
    _write_run(
        self.sweep_dir, 'rule_change_seed3',
        perturbation_round=4,
        gini_series={1: 0.1, 2: 0.12, 3: 0.11, 4: 0.2, 5: 0.15, 6: 0.13},
        verdict='stable', cycles_to_recovery=2,
    )
    run = analysis.load_run(self.sweep_dir / 'rule_change_seed3')
    self.assertIsNotNone(run)
    self.assertEqual(run.perturbation_type, 'rule_change')
    self.assertEqual(run.seed, 3)
    self.assertEqual(run.max_cycle, 6)
    self.assertEqual(run.n_post_perturbation_cycles, 3)  # cycles 4, 5, 6

  def test_load_run_missing_metrics_file_returns_none(self):
    empty_dir = self.sweep_dir / 'newcomer_seed9'
    empty_dir.mkdir(parents=True)
    self.assertIsNone(analysis.load_run(empty_dir))

  def test_verdict_meaningful_respects_threshold(self):
    _write_run(
        self.sweep_dir, 'rule_change_seed1', perturbation_round=6,
        gini_series={1: 0.1, 2: 0.1, 6: 0.2}, verdict='no_recovery',
        cycles_to_recovery=None,
    )
    run = analysis.load_run(self.sweep_dir / 'rule_change_seed1')
    # Only cycle 6 is >= perturbation_round=6: 1 post-perturbation cycle.
    self.assertEqual(run.n_post_perturbation_cycles, 1)
    self.assertLess(
        run.n_post_perturbation_cycles,
        analysis.MIN_POST_PERTURBATION_CYCLES_FOR_MEANING,
    )
    self.assertFalse(run.verdict_meaningful)

  def test_load_sweep_sorts_and_skips_incomplete_runs(self):
    _write_run(
        self.sweep_dir, 'rule_change_seed2', perturbation_round=4,
        gini_series={1: 0.1, 4: 0.2, 5: 0.15, 6: 0.13, 7: 0.14},
        verdict='stable', cycles_to_recovery=1,
    )
    _write_run(
        self.sweep_dir, 'rule_change_seed1', perturbation_round=4,
        gini_series={1: 0.1, 4: 0.2, 5: 0.15, 6: 0.13, 7: 0.14},
        verdict='stable', cycles_to_recovery=1,
    )
    (self.sweep_dir / 'rule_change_seed3').mkdir()  # No metrics file.
    runs = analysis.load_sweep(self.sweep_dir)
    self.assertEqual([r.run_id for r in runs],
                      ['rule_change_seed1', 'rule_change_seed2'])

  def test_summarize_by_condition_excludes_unmeaningful_from_verdict_counts(
      self,
  ):
    # Meaningful run: 4 post-perturbation cycles.
    _write_run(
        self.sweep_dir, 'rule_change_seed1', perturbation_round=4,
        gini_series={1: 0.1, 2: 0.1, 3: 0.1, 4: 0.2, 5: 0.15, 6: 0.13,
                     7: 0.14},
        verdict='stable', cycles_to_recovery=1,
    )
    # Not meaningful: only 1 post-perturbation cycle.
    _write_run(
        self.sweep_dir, 'rule_change_seed2', perturbation_round=6,
        gini_series={1: 0.1, 2: 0.1, 6: 0.5}, verdict='stable',
        cycles_to_recovery=0,
    )
    runs = analysis.load_sweep(self.sweep_dir)
    summary = analysis.summarize_by_condition(runs)
    cond = summary['rule_change']
    self.assertEqual(cond['n'], 2)
    self.assertEqual(cond['n_meaningful'], 1)
    # Only the meaningful run's verdict should be counted.
    self.assertEqual(cond['verdict_counts'], {'stable': 1})

  def test_write_summary_csv_roundtrip(self):
    _write_run(
        self.sweep_dir, 'newcomer_seed1', perturbation_round=4,
        gini_series={1: 0.1, 4: 0.2, 5: 0.15, 6: 0.13, 7: 0.14},
        verdict='drifted', cycles_to_recovery=None,
    )
    runs = analysis.load_sweep(self.sweep_dir)
    out_path = self.sweep_dir / 'summary.csv'
    analysis.write_summary_csv(runs, out_path)
    content = out_path.read_text()
    self.assertIn('newcomer_seed1', content)
    self.assertIn('drifted', content)

  def test_write_long_csv_has_one_row_per_cycle(self):
    _write_run(
        self.sweep_dir, 'newcomer_seed1', perturbation_round=4,
        gini_series={1: 0.1, 2: 0.2, 3: 0.3}, verdict='stable',
        cycles_to_recovery=0,
    )
    runs = analysis.load_sweep(self.sweep_dir)
    out_path = self.sweep_dir / 'long.csv'
    analysis.write_long_csv(runs, out_path)
    rows = out_path.read_text().strip().split('\n')
    self.assertLen(rows, 1 + 3)  # header + 3 cycles

  def test_format_report_flags_low_data_runs(self):
    _write_run(
        self.sweep_dir, 'rule_change_seed1', perturbation_round=6,
        gini_series={1: 0.1, 6: 0.2}, verdict='no_recovery',
        cycles_to_recovery=None,
    )
    runs = analysis.load_sweep(self.sweep_dir)
    report = analysis.format_report(runs)
    self.assertIn('WARNING', report)
    self.assertIn('NOT MEANINGFUL', report)


if __name__ == '__main__':
  absltest.main()
