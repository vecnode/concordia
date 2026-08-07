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
   time as a distribution, not a single run's point estimate.

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

## Key parameters

| Flag                       | Default            | Description                                              |
| --------------------------- | ------------------- | ---------------------------------------------------------- |
| `--num_cycles`              | `12`                | Total cycles across both phases.                          |
| `--perturbation_round`      | `6`                 | Cycle (1-indexed) the perturbation is introduced at.       |
| `--perturbation_type`       | `rule_change`       | `rule_change`, `newcomer`, or `none` (control).            |
| `--api_type`                | `openai`            | Language model provider.                                   |
| `--model_name`              | `gpt-4o`            | Language model name.                                        |
| `--disable_language_model`  | `false`             | Use a mock model for testing.                               |
| `--use_dummy_embedder`      | `false`             | Use a zero-vector embedder instead of sentence-transformers.|
| `--output_dir`              | `/tmp/norm_drift_results` | Directory for HTML logs and the metrics JSON summary. |
| `--seed`                    | `42`                | Random seed for reproducibility.                            |

## Output

Written to `--output_dir`:

- `village_commons_phase_a.html` / `village_commons_phase_b.html` — full
  simulation transcripts for each phase.
- `norm_drift_metrics.json` — per-cycle Gini series, trailing convergence
  variance, and the perturbation-recovery result.

## File structure

```
norm_drift/
├── README.md                        # This file
├── run.py                           # Main entry point
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
