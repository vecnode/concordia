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

r"""Multi-seed sweep runner for the norm_drift experiment.

Runs run.py once per (perturbation_type, seed) combination, writing each
run's output to its own subdirectory of --sweep_dir, then aggregates the
results via analysis.py.

Each run is spawned as its own OS process (not an in-process thread): the
underlying simulation seeds Python's global `random` module and passes a
seed to the language model, and running multiple simulations as threads in
one process would have them contend for that same global RNG state,
silently breaking the seeding. Separate processes keep each run's random
state fully independent.

--max_parallel controls how many runs execute concurrently. This is a
real wall-clock win only if the language model backend actually serves
concurrent requests in parallel rather than queueing them -- verify this
against your Ollama server before assuming a speedup (e.g. compare the
wall time of two concurrent `curl .../api/generate` calls against two
sequential ones).

Usage:
  python -m examples.norm_drift.sweep \
    --sweep_dir=/tmp/norm_drift_sweep \
    --perturbation_types=rule_change,newcomer \
    --seeds=1,2,3,4,5 \
    --api_type=ollama --model_name=llama3.2:3b \
    --num_cycles=16 --perturbation_round=4 \
    --max_parallel=2 --use_dummy_embedder
"""

import argparse
import concurrent.futures
import pathlib
import subprocess
import sys

from examples.norm_drift import analysis

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _run_single(
    run_id: str,
    perturbation_type: str,
    seed: int,
    sweep_dir: pathlib.Path,
    common_args: list[str],
) -> tuple[str, int]:
  """Run one (perturbation_type, seed) config as a subprocess.

  Returns:
    (run_id, exit_code).
  """
  out_dir = sweep_dir / run_id
  log_path = sweep_dir / f'{run_id}.log'
  cmd = [
      sys.executable,
      '-m',
      'examples.norm_drift.run',
      *common_args,
      f'--perturbation_type={perturbation_type}',
      f'--seed={seed}',
      f'--output_dir={out_dir}',
  ]
  with log_path.open('w') as log_f:
    proc = subprocess.run(
        cmd, cwd=_REPO_ROOT, stdout=log_f, stderr=subprocess.STDOUT,
        check=False,
    )
  return run_id, proc.returncode


def run_sweep(
    sweep_dir: pathlib.Path,
    perturbation_types: list[str],
    seeds: list[int],
    common_args: list[str],
    max_parallel: int = 2,
) -> list[analysis.RunSummary]:
  """Run the full (perturbation_types x seeds) sweep and return summaries.

  Args:
    sweep_dir: Output directory; one subdirectory per run is created here.
    perturbation_types: e.g. ['rule_change', 'newcomer'].
    seeds: e.g. [1, 2, 3, 4, 5].
    common_args: Extra run.py CLI flags shared by every run (api_type,
      model_name, num_cycles, perturbation_round, embedder flags, etc.) --
      NOT including --perturbation_type, --seed, or --output_dir, which
      this function sets per-run.
    max_parallel: Max number of run.py subprocesses running concurrently.

  Returns:
    RunSummary list for every run that produced a metrics file. Runs that
    crashed (nonzero exit code, no metrics file) are reported to stderr
    and omitted -- check len(result) against
    len(perturbation_types) * len(seeds) to detect this.
  """
  sweep_dir.mkdir(parents=True, exist_ok=True)
  jobs = [
      (f'{pert}_seed{seed}', pert, seed)
      for pert in perturbation_types
      for seed in seeds
  ]

  failures = []
  with concurrent.futures.ThreadPoolExecutor(
      max_workers=max_parallel
  ) as executor:
    futures = {
        executor.submit(
            _run_single, run_id, pert, seed, sweep_dir, common_args
        ): run_id
        for run_id, pert, seed in jobs
    }
    for future in concurrent.futures.as_completed(futures):
      run_id, exit_code = future.result()
      if exit_code == 0:
        print(f'{run_id}: OK')
      else:
        print(f'{run_id}: FAILED (exit={exit_code}) -- see'
              f' {sweep_dir / (run_id + ".log")}')
        failures.append(run_id)

  if failures:
    print(f'\n{len(failures)}/{len(jobs)} runs failed: {failures}')

  return analysis.load_sweep(sweep_dir)


def main() -> None:
  parser = argparse.ArgumentParser(
      description='Run a multi-seed norm_drift sweep.'
  )
  parser.add_argument('--sweep_dir', type=pathlib.Path, required=True)
  parser.add_argument(
      '--perturbation_types', type=str, default='rule_change,newcomer',
      help='Comma-separated list.',
  )
  parser.add_argument(
      '--seeds', type=str, default='1,2,3,4,5', help='Comma-separated list.'
  )
  parser.add_argument('--api_type', type=str, default='ollama')
  parser.add_argument('--model_name', type=str, default='llama3.2:3b')
  parser.add_argument('--api_key', type=str, default=None)
  parser.add_argument('--num_cycles', type=int, default=16)
  parser.add_argument('--perturbation_round', type=int, default=4)
  parser.add_argument('--max_parallel', type=int, default=2)
  parser.add_argument('--use_dummy_embedder', action='store_true')
  parser.add_argument('--disable_language_model', action='store_true')
  args = parser.parse_args()

  perturbation_types = [
      s.strip() for s in args.perturbation_types.split(',') if s.strip()
  ]
  seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]

  common_args = [
      f'--api_type={args.api_type}',
      f'--model_name={args.model_name}',
      f'--num_cycles={args.num_cycles}',
      f'--perturbation_round={args.perturbation_round}',
  ]
  if args.api_key:
    common_args.append(f'--api_key={args.api_key}')
  if args.use_dummy_embedder:
    common_args.append('--use_dummy_embedder')
  if args.disable_language_model:
    common_args.append('--disable_language_model')

  print(
      f'Running {len(perturbation_types)} x {len(seeds)} ='
      f' {len(perturbation_types) * len(seeds)} configs,'
      f' max_parallel={args.max_parallel}, into {args.sweep_dir}'
  )

  runs = run_sweep(
      sweep_dir=args.sweep_dir,
      perturbation_types=perturbation_types,
      seeds=seeds,
      common_args=common_args,
      max_parallel=args.max_parallel,
  )

  if not runs:
    raise SystemExit('No runs produced metrics -- all failed or crashed.')

  analysis.write_summary_csv(runs, args.sweep_dir / 'sweep_summary.csv')
  analysis.write_long_csv(
      runs, args.sweep_dir / 'sweep_gini_trajectories.csv'
  )
  analysis.write_json(runs, args.sweep_dir / 'sweep_aggregate.json')
  print(analysis.format_report(runs))
  print(f'\nAggregate written to {args.sweep_dir}')


if __name__ == '__main__':
  main()
