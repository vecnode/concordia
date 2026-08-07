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

r"""Entry point for the norm-drift village commons experiment.

Runs a two-phase common pool resource (CPR) simulation: an unperturbed
baseline phase, a single mid-run perturbation (a grazing-norm rule change or
a newcomer agent joining the commons), and a post-perturbation phase. Then
computes and prints per-cycle Gini coefficient of harvest inequality,
trailing convergence variance, and a perturbation-recovery verdict.

Usage:
  python -m examples.norm_drift.run \
    --api_type=openai \
    --model_name=gpt-4o \
    --num_cycles=12 \
    --perturbation_round=6 \
    --perturbation_type=rule_change

  # Mock-model smoke test, no API key needed:
  python -m examples.norm_drift.run \
    --disable_language_model --num_cycles=10 --perturbation_round=6
"""

import argparse
import dataclasses
import json
import logging
import os
import random

from concordia.contrib.language_models import language_model_setup  # pyrefly: ignore[missing-import]
from examples.norm_drift import analysis
from examples.norm_drift.metrics import norm_drift_metrics
from examples.norm_drift.scenarios import village_commons
import numpy as np


def main() -> None:
  parser = argparse.ArgumentParser(
      description='Run the norm-drift village commons CPR experiment.',
  )
  parser.add_argument(
      '--api_type', type=str, default='openai',
      help='Type of API to use for the language model.',
  )
  parser.add_argument(
      '--model_name', type=str, default='gpt-4o',
      help='Name of the language model to use.',
  )
  parser.add_argument(
      '--api_key', type=str, default=None,
      help='API key for the language model provider.',
  )
  parser.add_argument(
      '--disable_language_model', action='store_true',
      help='Run with a mock language model (for testing).',
  )
  parser.add_argument(
      '--num_cycles', type=int, default=village_commons.DEFAULT_NUM_CYCLES,
      help='Total number of cycles across both phases.',
  )
  parser.add_argument(
      '--perturbation_round', type=int,
      default=village_commons.DEFAULT_PERTURBATION_ROUND,
      help='Cycle (1-indexed) at which the perturbation is introduced.',
  )
  parser.add_argument(
      '--perturbation_type', type=str, default='rule_change',
      choices=list(village_commons.PERTURBATION_TYPES),
      help=(
          "Perturbation to apply: 'rule_change' (norm shift), 'newcomer'"
          " (agent injection), or 'none' (control run, no shock)."
      ),
  )
  parser.add_argument(
      '--output_dir', type=str, default='/tmp/norm_drift_results',
      help='Directory to save HTML logs and the metrics summary JSON.',
  )
  parser.add_argument(
      '--seed', type=int, default=42,
      help='Random seed for reproducibility.',
  )
  parser.add_argument(
      '--use_dummy_embedder', action='store_true',
      help='Use a zero-vector embedder instead of sentence-transformers.',
  )
  args = parser.parse_args()

  logging.basicConfig(level=logging.INFO)

  # Concordia's engine uses Python's global `random` module directly in some
  # components (e.g. next_acting.py's random.choice for speaker selection),
  # not just the language model's own sampling. Seeding it here is necessary
  # for run-to-run reproducibility on top of the model-level seed passed to
  # language_model_setup below.
  random.seed(args.seed)
  np.random.seed(args.seed)

  model = language_model_setup(
      api_type=args.api_type,
      model_name=args.model_name,
      api_key=args.api_key,
      disable_language_model=args.disable_language_model,
      seed=args.seed,
  )

  if args.use_dummy_embedder or args.disable_language_model:
    embedder = lambda _: np.ones(384)  # 384-dimensional dummy embedder
  else:
    import sentence_transformers  # pylint: disable=g-import-not-at-top
    st_model = sentence_transformers.SentenceTransformer('all-mpnet-base-v2')
    embedder = lambda x: st_model.encode(x, show_progress_bar=False)

  print(
      'Starting norm-drift experiment: perturbation_type='
      f'{args.perturbation_type}, num_cycles={args.num_cycles},'
      f' perturbation_round={args.perturbation_round}, seed={args.seed}'
  )

  os.makedirs(args.output_dir, exist_ok=True)

  run_result = village_commons.run_two_phase_simulation(
      model=model,
      embedder=embedder,
      num_cycles=args.num_cycles,
      perturbation_round=args.perturbation_round,
      perturbation_type=args.perturbation_type,
      html_output_dir=args.output_dir,
  )

  step_logs = run_result['step_logs']
  gini_series = norm_drift_metrics.per_cycle_gini(step_logs)
  variance = norm_drift_metrics.convergence_variance(gini_series)

  # With --disable_language_model, the mock model doesn't produce parseable
  # "GRAZE X" responses, so no cycle will have numeric harvests to compute a
  # Gini coefficient from. In that case skip the recovery analysis rather
  # than fail — the smoke test's purpose is to exercise the plumbing, not to
  # produce meaningful metric values from unparseable mock output.
  has_baseline_and_post_cycles = any(
      c < args.perturbation_round for c in gini_series
  ) and any(c >= args.perturbation_round for c in gini_series)

  recovery = None
  if args.perturbation_type != 'none' and has_baseline_and_post_cycles:
    recovery = norm_drift_metrics.perturbation_recovery(
        gini_series, perturbation_round=args.perturbation_round
    )
  elif args.perturbation_type != 'none':
    print(
        '\nSkipping perturbation-recovery analysis: not enough cycles with'
        ' parseable numeric harvests on both sides of the perturbation'
        ' round (expected with --disable_language_model, since the mock'
        ' model does not produce parseable "GRAZE X" responses).'
    )

  n_post_perturbation_cycles = sum(
      1 for c in gini_series if c >= args.perturbation_round
  )
  verdict_meaningful = (
      n_post_perturbation_cycles
      >= analysis.MIN_POST_PERTURBATION_CYCLES_FOR_MEANING
  )

  print('\n=== Norm-Drift Metrics Summary ===')
  print(f'Per-cycle Gini: {json.dumps(gini_series, indent=2)}')
  print(f'Trailing convergence variance: {variance}')
  if recovery is not None:
    print(
        'Perturbation recovery: verdict='
        f'{recovery.verdict}, recovered={recovery.recovered},'
        f' cycles_to_recovery={recovery.cycles_to_recovery},'
        f' baseline_gini={recovery.baseline_gini:.4f},'
        f' tolerance_band={recovery.tolerance_band:.4f}'
    )
    if not verdict_meaningful:
      print(
          f'WARNING: only {n_post_perturbation_cycles} post-perturbation'
          f' cycle(s) observed (< '
          f'{analysis.MIN_POST_PERTURBATION_CYCLES_FOR_MEANING} threshold).'
          ' This recovery verdict reflects the run ending shortly after the'
          ' perturbation, not necessarily whether the population would have'
          ' recovered or drifted given more time -- treat it as'
          ' inconclusive, not as evidence either way.'
      )

  summary_path = os.path.join(args.output_dir, 'norm_drift_metrics.json')
  with open(summary_path, 'w', encoding='utf-8') as f:
    json.dump(
        {
            'perturbation_round': run_result['perturbation_round'],
            'perturbation_type': run_result['perturbation_type'],
            'gini_series': gini_series,
            'convergence_variance': variance,
            'n_post_perturbation_cycles': n_post_perturbation_cycles,
            'verdict_meaningful': verdict_meaningful,
            'recovery': (
                dataclasses.asdict(recovery) if recovery is not None else None
            ),
        },
        f,
        indent=2,
        default=str,
    )

  print(f'\nExperiment finished. Metrics summary written to {summary_path}')


if __name__ == '__main__':
  main()
