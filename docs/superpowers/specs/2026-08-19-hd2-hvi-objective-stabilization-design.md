# HD2 HVI Objective-Conflict and Stabilization Design

## Status

- Phase: design only
- Branch: `exp/hd2-hvi-objective-stabilization`
- Parent evidence: committed HD1 diagnostic results
- Hardware target: one Windows laptop with NVIDIA RTX 4070 Laptop GPU, 8 GB VRAM
- Canonical seed: exactly `101`
- Claim status: `diagnostic_only`
- Scope: separate high-dimensional research panel; never numerically rank against M01--M24

## Purpose

HD2 diagnoses and stabilizes the hybrid variational-inequality PINN objective used in HD1. It does not change the HD1 DeepSets architecture. Its purpose is to determine whether the price degradation observed after activating the physics objective is primarily associated with gradient-magnitude dominance, delayed-price convergence, persistent directional conflict, or a material high-dimensional Hutchinson-gradient approximation error.

HD2 is intentionally small. It exists to freeze one defensible HVI training protocol for the subsequent HD3 10D/20D/50D/100D architecture-scaling experiment.

## Scientific Question

Can a controlled reduction in HVI optimization interference retain a material part of HD1's PDE/complementarity benefit without materially degrading the trusted oracle price fit of the supervised reference?

## Fixed Financial Problem

Use the HD1 normalized equal-weight American geometric-basket put:

```text
K = 1.0
T = 1.0
r = 0.05
q = 0.02
sigma = 0.25
rho = 0.30
seed = 101
HD2 diagnostic dimensions = [2, 5]
```

Use the existing HD1 training/evaluation domains and data construction for the stabilization study. Do not introduce the HD3 transverse-state sampling redesign in HD2, because the purpose of HD2 is objective diagnosis rather than a simultaneous data-distribution change.

Use fresh HD2 Sobol streams for supervised training, physics collocation, validation, and final diagnostic evaluation. HD1 test points are legacy evidence and must not be reused as fresh confirmation.

## Fixed Architecture

Use the exact HD1 DeepSets architecture:

```text
shared per-asset encoder: 2 x 64, tanh
mean aggregation
value head: 4 x 128, tanh
hard value transform: v = g + tau * softplus(h)
```

Do not widen, deepen, add attention, add a Transformer, add a separate boundary head, or alter the hard obstacle/terminal representation in HD2.

## Training Objective

Retain the HD1 hybrid objective:

\[
L = L_{sup} + \lambda L_{FB}.
\]

`L_sup` is the normalized trusted CN/PSOR price MSE on the frozen supervised stream.

The log-price VI residual remains

\[
R_\theta
=v_\tau
-\frac12\operatorname{tr}(\Sigma\nabla_x^2v)
-\left(r-q-\frac12\sigma^2\right)\sum_i v_{x_i}
+rv.
\]

With

\[
a=v-g,\qquad b=R_\theta,
\]

use the same smoothed Fischer--Burmeister residual as HD1:

\[
\phi_\varepsilon(a,b)=\sqrt{a^2+b^2+\varepsilon^2}-a-b.
\]

Do not change the historical FB smoothing schedule during the primary screen:

```text
epsilon: 1e-2 -> 1e-4
```

## Phase A: Objective-Gradient Diagnosis

Run the fresh HD1-style HVI anchor at `d=2` and `d=5`:

```text
total steps = 10000
supervised warm start = 2000
physics final weight = 0.1
physics ramp = linear over the remaining 8000 steps
Adam lr = 1e-3
```

At sparse diagnostic checkpoints after physics activation, compute objective-specific parameter gradients on a small fixed diagnostic batch:

\[
g_{sup}=\nabla_\theta L_{sup},
\qquad
g_{phys}=\nabla_\theta L_{FB}.
\]

Record:

\[
\|g_{sup}\|,
\quad
\|g_{phys}\|,
\quad
R=\frac{\|g_{phys}\|}{\|g_{sup}\|},
\]

\[
R_\lambda=\lambda R,
\]

and

\[
C=\frac{g_{sup}^{T}g_{phys}}{\|g_{sup}\|\|g_{phys}\|}.
\]

Also record:

\[
g_{sup}^{T}g_{total},
\qquad
g_{phys}^{T}g_{total},
\]

where `g_total = g_sup + lambda * g_phys`.

### Diagnostic cadence

```text
basic train-loss logging: every 100 optimizer steps
validation metrics: every 500 steps
objective-gradient diagnostics: every 500 steps after physics activation
layerwise gradient norms/cosines: every 1000 steps
```

Use one fixed diagnostic batch per dimension to make temporal measurements comparable. Diagnostic batches are not optimizer batches.

Suggested maximum diagnostic batch sizes:

```text
d=2: 128
d=5: 64
```

If those sizes OOM, use the predeclared diagnostic fallback `64 -> 32 -> 16` and record it. A diagnostic fallback does not change canonical optimizer batch size.

Gradient tensors must be reduced immediately to CPU scalars/compact vectors and detached. Never retain computation graphs in training-history objects.

## Persistent Directional-Conflict Criterion

Classify the anchor as showing persistent directional conflict only if both conditions hold after physics activation:

1. two consecutive 500-step windows contain negative cosine at a majority of recorded checkpoints and median `C <= -0.1`;
2. the same windows show worsening or stalled supervised validation price trajectory relative to its pre-physics trend.

The `-0.1` value is an operational preregistered threshold, not a literature constant.

If the criterion is not met, conflict-aware optimization is not permitted in HD2.

## Phase B: Mandatory Stabilization Screen

Use the unchanged architecture, optimizer, data distribution, total step count, and FB smoothing schedule.

Run exactly:

| ID | Warm start | Final physics weight | Purpose |
|---|---:|---:|---|
| `SUP-HD2` | N/A | 0 | fresh supervised reference |
| `HVI-HD1-anchor` | 2000 | 0.1 | historical anchor |
| `HVI-low` | 2000 | 0.01 | isolate 10x lower physics envelope |
| `HVI-warm` | 5000 | 0.1 | isolate delayed physics activation |
| `HVI-warm-low` | 5000 | 0.01 | combined conservative stabilization |

Run all mandatory variants at `d=2` and `d=5`, seed `101` only.

All runs use 10,000 total Adam steps. The physics ramp begins only after each candidate's warm start and increases linearly to its frozen final coefficient.

Do not add an epsilon sweep, learning-rate sweep, architecture sweep, adaptive sampler, L-BFGS stage, or wider hyperparameter search.

## Conditional ReLoBRaLo Comparator

ReLoBRaLo is not part of the mandatory screen.

It may run once only if all three stabilization variants (`HVI-low`, `HVI-warm`, `HVI-warm-low`) fail the deterministic selection rule and the remaining one-day wall-clock budget permits it.

If enabled, use one frozen published/reference implementation with no validation hyperparameter sweep. Record the exact implementation and parameters in `config_manifest.json`.

## Conditional Conflict-Aware Comparator

A conflict-aware candidate may run only if Phase A satisfies the persistent directional-conflict criterion.

Use exactly one method: `CAGrad`.

Do not also run PCGrad, DCGD, MGDA, or another gradient-surgery tournament.

If the measured pathology is magnitude-dominated (`R_lambda` large while cosine is non-negative or only transiently negative), do not run CAGrad.

## Phase C: d=5 Exact-vs-Hutchinson Gradient Diagnostic

The HD1 d=2 algebraic check does not establish that two fixed Hutchinson probes preserve the training direction at d=5.

At frozen d=5 anchor checkpoints:

```text
step 2000
step 5000
step 10000
```

use a fixed small diagnostic batch of 16 states and compare:

1. exact full-Hessian covariance contraction;
2. two-probe fixed-seed Hutchinson contraction;
3. exact-Hessian physics parameter gradient;
4. Hutchinson physics parameter gradient.

Record:

- contraction relative error;
- gradient norm ratio;
- gradient cosine between exact and Hutchinson physics gradients;
- diagnostic wall time;
- peak CUDA memory.

The default two-probe training estimator is classified as materially poor only if either:

```text
relative contraction error > 5%
OR
exact-vs-Hutchinson physics-gradient cosine < 0.95
```

at any of the frozen diagnostic checkpoints.

These are operational engineering thresholds, not general theorems.

Only if the condition is triggered may one additional controlled `2 probes -> 8 probes` comparison run. Keep probe distribution, seed policy, architecture, data, and all other settings unchanged.

## Deterministic Selection Rule

Because HD2 uses one seed only, this is a diagnostic selection, not statistical promotion.

Selection is lexicographic. A large PDE improvement can never compensate for failed price preservation.

### Gate 1: price preservation

For both `d=2` and `d=5`:

\[
PriceRMSE_{candidate} \le 1.05\,PriceRMSE_{SUP}.
\]

### Gate 2: derivative/boundary non-regression

For both dimensions:

\[
GammaRMSE_{candidate}\le1.10\,GammaRMSE_{SUP},
\]

\[
BoundaryRMSE_{candidate}\le1.10\,BoundaryRMSE_{SUP}.
\]

Exercise F1 and boundary-tube F1 are reported as differences, not ratios.

### Gate 3: retained physics benefit

For both dimensions, require at least a 25% reduction versus SUP in:

- trusted continuation PDE P95;
- unsmoothed FB P95.

### Gate 4: numerical/resource integrity

Require:

- finite required metrics;
- hard obstacle invariant within numerical tolerance;
- no hidden OOM/failure;
- verified artifact hashes;
- peak CUDA allocation below 7.2 GiB.

Among variants passing all gates, choose the one with the strongest joint PDE/FB validation evidence. If multiple candidates are effectively tied, prefer the simpler fixed-weight schedule over an adaptive/conflict-aware method.

## No-Promotion Fallback

HD3 must still be able to run even if HD2 finds no strict passing candidate.

If no HD2 candidate passes all gates, record:

```text
selection_status = no_stabilized_candidate
```

and freeze the predeclared HD3 fallback objective:

```text
HVI-warm-low
warm fraction = 0.5
final physics weight = 0.01
```

The fallback must be marked:

```text
fallback_objective = true
promoted_by_hd2 = false
claim_status = diagnostic_only
```

This prevents post hoc selection while guaranteeing that the high-dimensional scaling study can proceed.

## Detailed Result Capture

Every training job writes crash-tolerant detailed local artifacts.

### Per-job runtime directory

```text
results/hd2_hvi_objective_stabilization/runtime/<job_id>/
    latest_checkpoint.pt
    status.json
    events.jsonl
    training_history.jsonl
    validation_history.jsonl
    gradient_diagnostics.jsonl
```

`latest_checkpoint.pt` is ignored by Git.

### Checkpoint payload

Write an atomic resumable checkpoint every 250 optimizer steps and at every phase boundary. The payload contains:

- model state dict;
- optimizer state dict;
- completed optimizer step;
- current warm-start/physics schedule position;
- current lambda and epsilon;
- Python RNG state;
- NumPy RNG state;
- Torch CPU RNG state;
- all CUDA RNG states;
- Sobol stream identifiers and generated-point counters;
- current canonical/fallback batch size;
- elapsed training seconds;
- implementation commit SHA;
- resolved config SHA-256;
- job ID and dimension/model/candidate identity.

Use write-to-temporary-file plus atomic rename. Never overwrite a valid checkpoint non-atomically.

### Sobol resume rule

Sobol streams are reconstructed from their frozen scramble seed plus generated-point counter and advanced deterministically to the recorded position. A resumed run must not silently begin a new collocation stream.

### Status states

Allowed job states:

```text
pending
running
interrupted
completed
failed_numerical
oom_exhausted
resource_blocked
stale_provenance
```

A completed job is skipped on rerun only when its implementation SHA, config hash, result hashes, and expected metric files all verify.

### Compact final HD2 outputs

Finalization produces:

```text
environment_manifest.json
config_manifest.json
run_ledger.csv
run_metrics.csv
training_history.csv
validation_history.csv
gradient_diagnostics.csv
layerwise_gradient_diagnostics.csv
hutchinson_gradient_validation.csv
conditional_methods.json
selection_decision.json
dimension_summary.csv
artifact_hashes.json
```

The compact CSVs may be regenerated from JSONL runtime logs after interruption. JSONL files are append-and-flush records; malformed/incomplete final lines from an abrupt crash are ignored during recovery and never treated as completed evidence.

## One-Command / Resume Contract

HD2 must be callable both independently and as part of the combined HD2->HD3 runner.

The same command is safe to execute after a reboot or user interruption. Resume is the default; rerunning completed current-provenance jobs requires an explicit force flag.

HD2 phases:

```text
preflight
phase-a
gradient-diagnosis
screen
conditional
select
finalize
all
status
```

`all` executes the phases in dependency order and safely resumes from the latest verified state.

## Stop Rules

- Do not run canonical work when focused tests fail.
- Do not use HD1 legacy test points as fresh HD2 evidence.
- Do not alter architecture.
- Do not access the M01--M24 blind holdout.
- Do not run CAGrad unless the frozen conflict rule triggers.
- Do not change Hutchinson probes unless the frozen d=5 diagnostic triggers.
- Do not run a large adaptive-method sweep.
- Do not tune after final diagnostic evaluation.
- Preserve negative results.

## Expected Implementation Layout

```text
configs/hd2_hvi_objective_stabilization.json
experiments/hd2_hvi_objective_stabilization.py
src/diagnostics/hd2_gradient_diagnostics.py
src/diagnostics/hd2_hutchinson_gradient_validation.py
tests/test_hd2_hvi_stabilization.py
scripts/run_hd2_windows.ps1
scripts/check_hd2_status.ps1
docs/campaign/hd2_windows_execution_instructions.md
results/hd2_hvi_objective_stabilization/
```

Reuse existing HD1 model, oracle, and high-dimensional Black--Scholes utilities where their contracts match. Do not mutate HD1 result artifacts.

## Definition Of Done For HD2 Implementation

Before laptop execution:

- all mandatory/conditional logic is encoded from the frozen config;
- objective-gradient diagnostics are tested for differentiability and no silent detach;
- exact-vs-Hutchinson gradient diagnostic is tested on a tiny deterministic case;
- checkpoint/resume equivalence is tested by comparing uninterrupted and interrupted/resumed short runs;
- stale-provenance status files are rejected;
- all compact output schemas and hashes are tested;
- PowerShell runner is syntactically reviewed;
- no laptop CUDA result is claimed before the user runs it.
