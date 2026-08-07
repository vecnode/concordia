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

"""Aggregation and reporting for norm_drift multi-seed sweeps.

Reads the norm_drift_metrics.json files produced by run.py across many
seeds/perturbation-type combinations (as written by sweep.py) and builds a
per-run summary table, a condition-level aggregate, and export files
(CSV, JSON) for external analysis.

Pure Python, no LLM or Concordia dependency -- operates entirely on the
already-computed metrics JSON files.
"""

import csv
import dataclasses
import json
import pathlib

# A run's post-perturbation recovery verdict is only interpretable if the
# simulation actually continued for a meaningful number of cycles after the
# perturbation. A verdict computed from 1 post-perturbation cycle is not
# evidence of "recovery" or "no recovery" -- it just means the run ended
# before drift (or its absence) had a chance to show up.
MIN_POST_PERTURBATION_CYCLES_FOR_MEANING = 3


@dataclasses.dataclass
class RunSummary:
  """Summary of a single norm_drift run, extracted from its metrics JSON."""

  run_id: str
  perturbation_type: str
  seed: int
  max_cycle: int | None
  n_cycles_observed: int
  n_post_perturbation_cycles: int
  baseline_gini: float | None
  tolerance_band: float | None
  verdict: str | None
  recovered: bool | None
  cycles_to_recovery: int | None
  convergence_variance: float | None
  gini_series: dict[int, float]

  @property
  def verdict_meaningful(self) -> bool:
    """Whether n_post_perturbation_cycles is enough to trust the verdict."""
    return (
        self.n_post_perturbation_cycles
        >= MIN_POST_PERTURBATION_CYCLES_FOR_MEANING
    )


def load_run(run_dir: pathlib.Path) -> RunSummary | None:
  """Load one run's summary from <run_dir>/norm_drift_metrics.json.

  Args:
    run_dir: Directory named "<perturbation_type>_seed<seed>" containing a
      norm_drift_metrics.json (as written by examples.norm_drift.run).

  Returns:
    A RunSummary, or None if no metrics file is present (e.g. the run
    crashed before writing output).
  """
  metrics_path = run_dir / 'norm_drift_metrics.json'
  if not metrics_path.exists():
    return None
  data = json.loads(metrics_path.read_text())

  pert_type, _, seed_str = run_dir.name.rpartition('_seed')
  gini = {int(k): v for k, v in data.get('gini_series', {}).items()}
  cycles = sorted(gini.keys())
  pert_round = data['perturbation_round']
  post_cycles = [c for c in cycles if c >= pert_round]
  rec = data.get('recovery') or {}

  return RunSummary(
      run_id=run_dir.name,
      perturbation_type=pert_type,
      seed=int(seed_str) if seed_str.isdigit() else -1,
      max_cycle=max(cycles) if cycles else None,
      n_cycles_observed=len(cycles),
      n_post_perturbation_cycles=len(post_cycles),
      baseline_gini=rec.get('baseline_gini'),
      tolerance_band=rec.get('tolerance_band'),
      verdict=rec.get('verdict'),
      recovered=rec.get('recovered'),
      cycles_to_recovery=rec.get('cycles_to_recovery'),
      convergence_variance=data.get('convergence_variance'),
      gini_series=gini,
  )


def load_sweep(sweep_dir: pathlib.Path) -> list[RunSummary]:
  """Load every run summary found under sweep_dir.

  Args:
    sweep_dir: Directory containing one subdirectory per run.

  Returns:
    RunSummary list, sorted by (perturbation_type, seed). Directories with
    no metrics file (crashed runs) are silently skipped -- check the
    expected run count against len(result) to detect this.
  """
  runs = []
  for child in sorted(sweep_dir.iterdir()):
    if child.is_dir():
      run = load_run(child)
      if run is not None:
        runs.append(run)
  runs.sort(key=lambda r: (r.perturbation_type, r.seed))
  return runs


def summarize_by_condition(runs: list[RunSummary]) -> dict[str, dict]:
  """Aggregate run summaries into per-perturbation_type statistics.

  Args:
    runs: Run summaries, as returned by load_sweep.

  Returns:
    Dict mapping perturbation_type to a dict with: 'n' (total runs),
    'n_meaningful' (runs with enough post-perturbation data to trust),
    'verdict_counts' (dict of verdict -> count, restricted to meaningful
    runs), and 'avg_post_perturbation_cycles'.
  """
  summary: dict[str, dict] = {}
  for run in runs:
    cond = summary.setdefault(
        run.perturbation_type,
        {
            'n': 0,
            'n_meaningful': 0,
            'verdict_counts': {},
            'avg_post_perturbation_cycles': 0.0,
        },
    )
    cond['n'] += 1
    cond['avg_post_perturbation_cycles'] += run.n_post_perturbation_cycles
    if run.verdict_meaningful:
      cond['n_meaningful'] += 1
      cond['verdict_counts'][run.verdict] = (
          cond['verdict_counts'].get(run.verdict, 0) + 1
      )
  for cond in summary.values():
    if cond['n']:
      cond['avg_post_perturbation_cycles'] /= cond['n']
  return summary


_CSV_FIELDNAMES = [
    'run_id',
    'perturbation_type',
    'seed',
    'max_cycle',
    'n_cycles_observed',
    'n_post_perturbation_cycles',
    'verdict_meaningful',
    'baseline_gini',
    'tolerance_band',
    'verdict',
    'recovered',
    'cycles_to_recovery',
    'convergence_variance',
]


def write_summary_csv(runs: list[RunSummary], path: pathlib.Path) -> None:
  """Write one row per run: the wide summary table (no per-cycle detail)."""
  with path.open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=_CSV_FIELDNAMES)
    writer.writeheader()
    for run in runs:
      row = {name: getattr(run, name) for name in _CSV_FIELDNAMES}
      writer.writerow(row)


def write_long_csv(runs: list[RunSummary], path: pathlib.Path) -> None:
  """Write long-format (perturbation_type, seed, cycle, gini) rows.

  One row per (run, cycle) pair -- the shape most plotting libraries want
  for a faceted line chart.
  """
  with path.open('w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['perturbation_type', 'seed', 'cycle', 'gini'])
    for run in runs:
      for cycle in sorted(run.gini_series.keys()):
        writer.writerow(
            [run.perturbation_type, run.seed, cycle, run.gini_series[cycle]]
        )


def write_json(runs: list[RunSummary], path: pathlib.Path) -> None:
  """Write the full run summary list (including per-cycle Gini) as JSON."""
  path.write_text(
      json.dumps([dataclasses.asdict(r) for r in runs], indent=2)
  )


def format_report(runs: list[RunSummary]) -> str:
  """Human-readable text report: per-condition aggregate + per-run detail."""
  lines = []
  by_condition = summarize_by_condition(runs)
  for pert_type, stats in by_condition.items():
    lines.append(f'\n{pert_type} (N={stats["n"]}):')
    if stats['n_meaningful'] < stats['n']:
      lines.append(
          f'  WARNING: only {stats["n_meaningful"]}/{stats["n"]} runs have'
          f' >= {MIN_POST_PERTURBATION_CYCLES_FOR_MEANING} post-perturbation'
          ' cycles -- verdicts from the rest are not meaningful (the run'
          ' ended too soon after the perturbation to show drift or its'
          ' absence).'
      )
    lines.append(f'  verdict counts (meaningful runs only): '
                  f'{stats["verdict_counts"]}')
    lines.append(
        '  avg post-perturbation cycles observed:'
        f' {stats["avg_post_perturbation_cycles"]:.1f}'
    )
    for run in runs:
      if run.perturbation_type != pert_type:
        continue
      flag = '' if run.verdict_meaningful else '  [NOT MEANINGFUL]'
      lines.append(
          f'    seed{run.seed}: verdict={run.verdict},'
          f' post_cycles={run.n_post_perturbation_cycles},'
          f' cycles_to_recovery={run.cycles_to_recovery}{flag}'
      )
  return '\n'.join(lines)


def main() -> None:
  import argparse  # pylint: disable=g-import-not-at-top

  parser = argparse.ArgumentParser(
      description='Aggregate a norm_drift multi-seed sweep into CSV/JSON.'
  )
  parser.add_argument(
      'sweep_dir', type=pathlib.Path, help='Directory containing run subdirs.'
  )
  args = parser.parse_args()

  runs = load_sweep(args.sweep_dir)
  if not runs:
    raise SystemExit(f'No run metrics found under {args.sweep_dir}')

  write_summary_csv(runs, args.sweep_dir / 'sweep_summary.csv')
  write_long_csv(runs, args.sweep_dir / 'sweep_gini_trajectories.csv')
  write_json(runs, args.sweep_dir / 'sweep_aggregate.json')

  print(format_report(runs))
  print(f'\nWrote sweep_summary.csv, sweep_gini_trajectories.csv,'
        f' sweep_aggregate.json to {args.sweep_dir}')


if __name__ == '__main__':
  main()
