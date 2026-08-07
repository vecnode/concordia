# Norm-Drift Example

This example studies whether an LLM-agent population's emergent grazing
norm on a common pool resource (CPR) **recovers** after a mid-run
perturbation, and if so, how quickly and to what equilibrium.

It is built directly on top of `examples/resource_dilemma`'s pasture
scenario — the harvesting Game Master, discussion Game Master, resource
components, and shared-memory config-building are reused unchanged. This
example only adds:

1. A **two-phase runner** (`scenarios/village_commons.py`) that runs an
   unperturbed baseline phase, applies a single perturbation, then runs a
   post-perturbation phase with the resource level carried over.
2. **Norm-drift metrics** (`metrics/norm_drift_metrics.py`): per-cycle Gini
   coefficient of harvest inequality, trailing convergence variance, and a
   perturbation-recovery verdict.
3. A **village personas** set (`personas/village_personas.py`): six
   villagers present for the whole run, plus a seventh newcomer persona held
   in reserve for the agent-injection perturbation.

## Perturbations

Two shock types, selected via `--perturbation_type`:

- `rule_change` — at the perturbation round, the village council announces
  an informal grazing-limit reduction (20 → 12 hectares/cycle). This is a
  norm shift, not an enforced rule; whether agents actually adjust their
  behaviour is exactly what the experiment measures.
- `newcomer` — at the perturbation round, a seventh agent unfamiliar with
  the village's grazing customs is added to the roster and grazes alongside
  the existing six from then on.
- `none` — control run, no shock. Used as the baseline-vs-extension
  comparison for the paper (does the metric correctly report "stable" with
  no perturbation?).

## Baseline vs. extension comparison plan

For the paper, the intended comparison is:

1. Run `perturbation_type=none` for `num_cycles` cycles as a control —
   confirm `convergence_variance` settles low and `per_cycle_gini` has no
   discontinuity.
2. Run `perturbation_type=rule_change` and `perturbation_type=newcomer` at
   matched `perturbation_round` and `num_cycles` — compare
   `cycles_to_recovery` and final `verdict` (`stable` / `drifted` /
   `no_recovery`) between shock types.
3. Repeat each condition across multiple seeds/models to report recovery
   time and collapse rate as **distributions**, not single-run point
   estimates. Multi-agent LLM social simulations are inherently stochastic
   (see the seeding caveat below) — treat each seed as an independent
   rollout, not a controlled replicate, consistent with prior work in this
   space (Piatti et al., 2024).

### Cycle budget: give Phase B enough runway

Empirically, per-cycle Gini stabilizes within 2-3 cycles, so a long
pre-perturbation baseline buys little. What matters is Phase B having
enough cycles for drift (or its absence) to actually show up — a run that
terminates 1 cycle after the perturbation cannot distinguish "recovered"
from "ran out of time." `analysis.py` flags this automatically
(`verdict_meaningful = n_post_perturbation_cycles >= 3`); a sweep where
most runs are flagged NOT MEANINGFUL needs a larger `--num_cycles` or
smaller `--perturbation_round`, not a different perturbation type. The
defaults (`num_cycles=16`, `perturbation_round=4`) reflect this: a
3-cycle baseline, 12 cycles of Phase B runway.

### Reproducibility: seeded, not bit-reproducible

`--seed` is passed to both the language model (via `language_model_setup`)
and Python's global `random` module (used by Concordia's `next_acting.py`
for speaker selection). This removes two real, previously-silent sources
of nondeterminism, but does **not** make a full run bit-for-bit
reproducible: Concordia's engine executes concurrent agent actions via a
`ThreadPoolExecutor`, and thread completion order affects shared state in
ways no seed controls. Report `--seed` values as run labels for an
independent stochastic sample, not as guaranteeing a replayable trajectory.

## Known limitation: phase boundary, not a continuous run

Phase A and Phase B are each a separate `Simulation.play()` call — this is
a genuine **phase boundary**, not a single continuous engine run. The
resource level is carried over manually between phases, but agents in
Phase B only inherit Phase A's context through the pre-loaded shared
memories (the pasture config's `SHARED_MEMORIES`), not live conversational
continuity from Phase A's discussion transcript. This is flagged here
explicitly rather than hidden: a fully continuous version would require
extending the harvesting/discussion Game Masters to support injecting a
premise change or a new player mid-`play()`, which `resource_dilemma`
does not currently support. That is future work, not something this
skeleton claims to already do.

## Quick start

### Mock-model smoke test (no API key needed)

Exercises the full plumbing — config build, two-phase engine run, logging,
metrics — without spending any API calls:

```bash
python -m examples.norm_drift.run \
  --disable_language_model --num_cycles=10 --perturbation_round=6
```

### Using OpenAI

```bash
pip install gdm-concordia sentence-transformers

python -m examples.norm_drift.run \
  --api_type=openai \
  --model_name=gpt-4o \
  --num_cycles=12 \
  --perturbation_round=6 \
  --perturbation_type=rule_change
```

### Using Google AI Studio

```bash
export GOOGLE_API_KEY=your_key_here

python -m examples.norm_drift.run \
  --api_type=gemini \
  --model_name=gemini-2.0-flash \
  --perturbation_type=newcomer
```

### Using Ollama (local)

```bash
pip install ollama
ollama pull llama3.2:3b

python -m examples.norm_drift.run \
  --api_type=ollama \
  --model_name=llama3.2:3b \
  --perturbation_type=rule_change \
  --use_dummy_embedder
```

## Key parameters

| Flag                       | Default            | Description                                              |
| --------------------------- | ------------------- | ---------------------------------------------------------- |
| `--num_cycles`              | `16`                | Total cycles across both phases.                          |
| `--perturbation_round`      | `4`                 | Cycle (1-indexed) the perturbation is introduced at.       |
| `--perturbation_type`       | `rule_change`       | `rule_change`, `newcomer`, or `none` (control).            |
| `--api_type`                | `openai`            | Language model provider.                                   |
| `--model_name`              | `gpt-4o`            | Language model name.                                        |
| `--disable_language_model`  | `false`             | Use a mock model for testing.                               |
| `--use_dummy_embedder`      | `false`             | Use a zero-vector embedder instead of sentence-transformers.|
| `--output_dir`              | `/tmp/norm_drift_results` | Directory for HTML logs and the metrics JSON summary. |
| `--seed`                    | `42`                | Fixed generation seed (see reproducibility note above).     |

## Output

Written to `--output_dir`:

- `village_commons_phase_a.html` / `village_commons_phase_b.html` — full
  simulation transcripts for each phase.
- `norm_drift_metrics.json` — per-cycle Gini series, trailing convergence
  variance, `n_post_perturbation_cycles`/`verdict_meaningful`, and the
  perturbation-recovery result.

## Multi-seed sweeps

For a proper comparison you need many independent rollouts per condition,
not one run per perturbation type. `sweep.py` runs a full
(`perturbation_types` × `seeds`) grid, `--max_parallel` runs concurrently
as separate OS processes (real parallelism, verified against the model
backend actually serving concurrent requests — check this against your own
Ollama server before assuming a speedup), and aggregates automatically:

```bash
python -m examples.norm_drift.sweep \
  --sweep_dir=/tmp/norm_drift_sweep \
  --perturbation_types=rule_change,newcomer \
  --seeds=1,2,3,4,5 \
  --api_type=ollama --model_name=llama3.2:3b \
  --num_cycles=16 --perturbation_round=4 \
  --max_parallel=2 --use_dummy_embedder
```

This writes `sweep_summary.csv` (one row per run),
`sweep_gini_trajectories.csv` (long format, one row per run-cycle, for
plotting), and `sweep_aggregate.json` (full detail) to `--sweep_dir`.
`analysis.py` can also be run standalone against an existing sweep
directory to regenerate these without re-running anything:

```bash
python -m examples.norm_drift.analysis /tmp/norm_drift_sweep
```

To render the Gini-trajectory figure (requires matplotlib, kept as a
separate optional dependency from the core aggregation logic):

```bash
python -m examples.norm_drift.plotting /tmp/norm_drift_sweep \
  --model_name=llama3.2:3b
```

## File structure

```
norm_drift/
├── README.md                        # This file
├── run.py                           # Main entry point (single run)
├── sweep.py                         # Multi-seed x condition sweep runner
├── analysis.py                      # Sweep aggregation (CSV/JSON), no LLM dep
├── analysis_test.py                 # Tests for analysis.py, synthetic fixtures
├── plotting.py                      # Gini-trajectory figure (matplotlib)
├── __init__.py
├── personas/
│   ├── __init__.py
│   └── village_personas.py          # Six villagers + one newcomer
├── scenarios/
│   ├── __init__.py
│   └── village_commons.py           # Two-phase config builder + runner
└── metrics/
    ├── __init__.py
    └── norm_drift_metrics.py        # Gini, convergence variance, recovery
```
