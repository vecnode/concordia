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

"""Figure generation for norm_drift sweeps.

Separated from analysis.py so that the core aggregation logic (tested in
analysis_test.py) has no matplotlib dependency -- only this module needs
it, and only when a figure is actually requested.
"""

import pathlib

from examples.norm_drift import analysis


def plot_gini_trajectories(
    runs: list[analysis.RunSummary],
    output_path: pathlib.Path,
    model_name: str = '',
) -> None:
  """Render per-cycle Gini trajectories, faceted by perturbation_type.

  One subplot per perturbation_type found in runs, one line per seed within
  each subplot, with a vertical marker at the perturbation round.

  Args:
    runs: Run summaries, as returned by analysis.load_sweep.
    output_path: Where to write the PNG.
    model_name: Optional model name to include in the figure title.
  """
  import matplotlib  # pylint: disable=g-import-not-at-top
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top

  conditions = sorted(set(r.perturbation_type for r in runs))
  if not conditions:
    raise ValueError('No runs to plot.')

  fig, axes = plt.subplots(
      1, len(conditions), figsize=(6.5 * len(conditions), 5.5), sharey=True
  )
  if len(conditions) == 1:
    axes = [axes]
  colors = plt.cm.tab10.colors

  # Perturbation round is the same across a sweep by construction (sweep.py
  # always passes one --perturbation_round for the whole sweep); read it
  # from the first run that has cycle data spanning it, for the vline.
  pert_round = None
  for run in runs:
    if run.n_post_perturbation_cycles > 0 and run.max_cycle is not None:
      pert_round = run.max_cycle - run.n_post_perturbation_cycles + 1
      break

  for ax, cond in zip(axes, conditions):
    subset = [r for r in runs if r.perturbation_type == cond]
    for i, run in enumerate(subset):
      cycles = sorted(run.gini_series.keys())
      values = [run.gini_series[c] for c in cycles]
      label = f'seed {run.seed} ({run.verdict or "n/a"})'
      if not run.verdict_meaningful:
        label += ' *'
      ax.plot(
          cycles,
          values,
          marker='o',
          markersize=4,
          linewidth=1.6,
          color=colors[i % len(colors)],
          label=label,
      )
    if pert_round is not None:
      ax.axvline(x=pert_round, color='black', linestyle='--',
                 linewidth=1, alpha=0.6)
    ax.set_title(f'{cond} perturbation', fontsize=12)
    ax.set_xlabel('Cycle')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc='upper left')

  axes[0].set_ylabel('Gini coefficient (harvest inequality)')
  subtitle = f' ({model_name})' if model_name else ''
  fig.suptitle(
      f'norm_drift: per-cycle Gini trajectories across seeds{subtitle}\n'
      '(* = verdict not meaningful: too few post-perturbation cycles)',
      fontsize=11,
  )
  fig.tight_layout(rect=(0, 0, 1, 0.92))
  fig.savefig(output_path, dpi=150)
  plt.close(fig)


def main() -> None:
  import argparse  # pylint: disable=g-import-not-at-top

  parser = argparse.ArgumentParser(
      description='Render the Gini-trajectory figure for a norm_drift sweep.'
  )
  parser.add_argument('sweep_dir', type=pathlib.Path)
  parser.add_argument('--model_name', type=str, default='')
  parser.add_argument(
      '--output', type=pathlib.Path, default=None,
      help='Defaults to <sweep_dir>/gini_trajectories.png.',
  )
  args = parser.parse_args()

  runs = analysis.load_sweep(args.sweep_dir)
  if not runs:
    raise SystemExit(f'No run metrics found under {args.sweep_dir}')

  output_path = args.output or (args.sweep_dir / 'gini_trajectories.png')
  plot_gini_trajectories(runs, output_path, model_name=args.model_name)
  print(f'Wrote {output_path}')


if __name__ == '__main__':
  main()
