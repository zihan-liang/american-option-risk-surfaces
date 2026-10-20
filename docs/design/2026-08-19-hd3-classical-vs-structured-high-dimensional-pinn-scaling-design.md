# HD3 Classical vs Structured High-Dimensional PINN Scaling Design

## Status

- Phase: design only
- Depends on: HD2 objective selection/fallback
- Hardware target: one Windows laptop with NVIDIA RTX 4070 Laptop GPU, 8 GB VRAM
- Seed: exactly `101`
- Claim status: `diagnostic_only`
- Canonical dimensions: `10, 20, 50`
- Stress dimension: `100`

## Purpose

HD3 is the high-dimensional result-producing experiment. It compares a conventional fully connected HVI PINN with a permutation-invariant DeepSets HVI PINN under the same stabilized/fallback objective frozen by HD2.

The goal is not to prove universal superiority. The goal is to obtain a reproducible, auditable high-dimensional PINN comparison on a controlled American geometric-basket obstacle problem with a trusted reducible CN/PSOR oracle.

## Scientific Question

How do a conventional fully connected PINN and a permutation-invariant structured PINN scale in price accuracy, PDE/complementarity diagnostics, derivatives, structural invariance, training time, and GPU memory as the state dimension increases from 10 to 50 assets, with 100 assets as a hardware stress test?

## Fixed Financial Problem

Use the same normalized equal-weight American geometric-basket put as HD1/HD2:

```text
K = 1.0
T = 1.0
r = 0.05
q = 0.02
sigma = 0.25
rho = 0.30
seed = 101
canonical dimensions = [10, 20, 50]
stress dimensions = [100]
```

For

\[
G(\mathbf S)=\left(\prod_i S_i\right)^{1/d}
\]

use the trusted effective one-dimensional oracle parameters

\[
\sigma_G^2=\sigma^2\left(\rho+\frac{1-\rho}{d}\right),
\qquad
q_G=q+\frac12\left(\sigma^2-\sigma_G^2\right).
\]

## HD2 Objective Handoff

Before any HD3 canonical run, the combined pipeline must contain a verified `selection_decision.json` from HD2.

HD3 uses exactly one objective configuration:

1. the HD2-selected candidate if one passed all frozen HD2 gates; otherwise
2. the predeclared fallback `HVI-warm-low` with warm fraction 0.5 and final physics weight 0.01.

The resolved objective is written once to:

```text
results/hd3_classical_vs_structured_high_dimensional_pinn/resolved_objective.json
```

and hashed before HD3 begins. HD3 refuses to run if this file changes after its first canonical job starts.

## Sampling Design

Independent per-asset uniform sampling makes the aggregate geometric mean concentrate as dimension grows. HD3 instead fixes the aggregate log geometric mean distribution and adds zero-mean transverse deviations.

For each state:

\[
x_i=m+\delta_i,
\qquad
\frac1d\sum_i\delta_i=0.
\]

Use:

```text
m ~ Uniform(log(0.5), log(1.5))
tau ~ Uniform(0.0025, 1.0)
raw u_i ~ Uniform(-1, 1)
delta_i = 0.35 * (u_i - mean(u))
x_i = m + delta_i
```

This keeps `m = mean(x)` fixed across dimensions while exposing both architectures to genuine d-dimensional states.

Training, validation, and test use separate fixed scrambled Sobol streams. The same state stream is used by FC and DeepSets at a given dimension.

The final test stream must not influence any HD3 architecture or objective choice.

## Model A: Conventional FC-HVI PINN

Input:

\[
(x_1,\ldots,x_d,\tau).
\]

Architecture:

```text
6 hidden layers
width 128
activation tanh
scalar raw head
hard value transform v = g + tau * softplus(h)
```

Use the same architecture form at every dimension. Report its dimension-dependent parameter count.

## Model B: Structured DeepSets-HVI PINN

Use the established compact HD1 architecture:

```text
shared per-asset encoder: 2 x 64, tanh
mean aggregation
global head: 4 x 128, tanh
scalar raw head
hard value transform v = g + tau * softplus(h)
```

Do not enlarge the network in HD3 unless a future experiment explicitly studies capacity.

## Matched Training Protocol

FC and DeepSets use the same:

- supervised oracle sample stream;
- physics collocation stream;
- optimizer;
- learning rate;
- total steps;
- warm-start fraction;
- final physics weight;
- physics ramp shape;
- FB epsilon schedule;
- Hessian estimator;
- training precision;
- validation/test states.

The only primary intended difference is architecture/inductive bias.

## Step Budget

Canonical dimensions:

```text
d=10: 6000 steps per model
d=20: 6000 steps per model
d=50: 6000 steps per model
```

Stress dimension:

```text
d=100: 4000 steps per model
```

Scale the HD2-selected warm start by fraction, not absolute step count. For example:

```text
HD2 warm fraction 0.5 -> HD3 canonical warm steps 3000 / 6000
HD2 warm fraction 0.2 -> HD3 canonical warm steps 1200 / 6000
```

The 100D stress uses the same fraction of 4000 total steps.

The physics weight ramp begins after the scaled warm start and ends at the HD2-selected/fallback final coefficient.

## High-Dimensional Second Derivatives

Use fixed-seed Hutchinson Hessian-vector estimation for every HD3 training dimension.

Default:

```text
probe count = 2
probe type = Rademacher
probe seed = 101
```

If and only if HD2's frozen d=5 diagnostic triggered the predeclared 8-probe intervention and that intervention became part of the selected/fallback resolved objective, HD3 may use 8 probes. Otherwise it uses two.

Never materialize a full Hessian during 10D/20D/50D/100D training.

## Canonical Batch Sizes And OOM Fallback

Start each single-process job at:

```text
supervised batch = 512
physics batch = 512
```

Predeclared fallback:

```text
512 -> 256 -> 128
```

On OOM, resume from the most recent atomic checkpoint with the next smaller batch size. Record the OOM event and new batch size. Do not change architecture, dimension, probe count, objective, or step budget in response to OOM.

For concurrent jobs, the runner may start from a smaller fixed concurrent batch size determined in preflight, but both architectures at the same dimension must use the same concurrent batch size.

## Parallelism Preflight

Parallelism is an execution optimization, not part of the scientific comparison.

At 10D run three 200-step smoke measurements from fresh processes:

1. FC alone;
2. DeepSets alone;
3. FC and DeepSets concurrently.

Enable pairwise concurrency for 10D and 20D only if:

```text
concurrent wall time <= 0.90 * (FC-alone wall time + DeepSets-alone wall time)
AND
sampled total device memory < 7.2 GiB
AND
no OOM/non-finite event occurs
```

Otherwise use sequential execution.

50D and 100D run sequentially by default.

## One-Day Budget

The combined HD2+HD3 pipeline has an 18-hour hard wall-clock cap.

HD3 target allocation is approximately:

```text
10D/20D/50D canonical: <= 10 hours after HD2
100D stress: <= 2 hours projected remaining time
finalization/contingency: reserve 1 hour
```

The runner computes seconds/step from smoke jobs and records projected completion before starting canonical work.

If the measured projection indicates the remaining canonical 10D/20D/50D ladder cannot finish inside the hard cap, stop with `resource_blocked` rather than silently reducing dimensions or steps.

The 100D pair starts only if measured elapsed time plus smoke-projected 100D time remains inside the same 18-hour global cap.

## Metrics

Use the same fixed test states for both models at each dimension.

### Accuracy

- normalized price RMSE, MAE, P95, maximum;
- per-asset Delta vector RMSE and maximum;
- covariance-weighted Hessian/Gamma-contraction RMSE.

### Physics

- obstacle invariant maximum/rate;
- unsmoothed FB RMS/P95/maximum;
- trusted continuation PDE RMS/P95.

### Structure

- permutation-consistency RMSE;
- transverse-invariance RMSE;
- diagonal geometric-basket boundary RMSE;
- exercise F1 and boundary-tube F1 as secondary diagnostics.

### Scalability

- parameter count;
- training wall time;
- seconds/step;
- derivative seconds;
- inference seconds;
- peak CUDA allocation;
- sampled device-memory peak during concurrency smoke;
- OOM count;
- processed supervised and physics points.

## Transverse-Invariance Diagnostic

For fixed `(m, tau)` generate multiple independent zero-mean transverse deviation vectors. Because the trusted reducible solution depends only on `m` and `tau`, the oracle price is unchanged.

Record each model's prediction standard deviation/RMSE across these transverse perturbations.

This is a primary structural diagnostic for the FC-vs-DeepSets comparison.

## Result Interpretation

HD3 is diagnostic only.

A structured-model advantage may be described only for the tested symmetric geometric-basket problem. It must not be generalized to arbitrary multi-asset American options.

The 100D run is a stress test. If it uses fewer steps than canonical runs, do not directly interpret its accuracy trend as a matched-compute continuation of the 10D/20D/50D accuracy curve.

## Detailed Runtime Artifacts

Every job writes:

```text
results/hd3_classical_vs_structured_high_dimensional_pinn/runtime/<job_id>/
    latest_checkpoint.pt
    status.json
    events.jsonl
    training_history.jsonl
    validation_history.jsonl
```

The checkpoint contract matches HD2: model/optimizer states, step, objective schedule position, RNG states, Sobol counters, batch state, elapsed time, implementation SHA, resolved config hash, and job identity.

Checkpoints are written atomically every 250 steps and at warm-start/physics/final boundaries.

Basic losses are logged every 100 steps; validation/scalability metrics every 500 steps.

## Compact Final Outputs

```text
environment_manifest.json
config_manifest.json
resolved_objective.json
parallelism_benchmark.csv
run_ledger.csv
run_metrics.csv
training_history.csv
validation_history.csv
structural_diagnostics.csv
dimension_summary.csv
decision.json
artifact_hashes.json
```

100D rows include `run_class = stress`.

## Resume Rules

Resume is default.

A completed job is skipped only if:

- status says completed;
- implementation SHA matches;
- config hash matches;
- resolved-objective hash matches;
- compact result hashes verify.

Interrupted jobs resume from `latest_checkpoint.pt`.

Stale provenance causes a hard refusal rather than silent resume.

Rerunning a completed canonical job requires an explicit force option and must never be the default path.

## Stop Rules

- Do not run without a frozen HD2 resolved objective.
- Do not change the selected/fallback objective after the first HD3 canonical run starts.
- Do not access the M01--M24 blind holdout.
- Do not change architecture after seeing HD3 test results.
- Do not drop a dimension because its results look poor.
- Stop the 100D stress cleanly if repeated batch fallback still OOMs; preserve it as a valid hardware-limit result.

## Expected Implementation Layout

```text
configs/hd3_classical_vs_structured_high_dimensional_pinn.json
experiments/hd3_classical_vs_structured_high_dimensional_pinn.py
src/models/hd3_high_dimensional_pinn.py
src/physics/hd3_state_sampling.py
src/diagnostics/hd3_structural_diagnostics.py
tests/test_hd3_high_dimensional_pinn.py
scripts/run_hd3_windows.ps1
scripts/check_hd3_status.ps1
docs/campaign/hd3_windows_execution_instructions.md
results/hd3_classical_vs_structured_high_dimensional_pinn/
```

## Definition Of Done For HD3 Implementation

Before laptop execution:

- FC and DeepSets architectures are tested on 10/20/50/100D shapes;
- structured permutation invariance is tested;
- transverse-state sampler exactly preserves mean log-moneyness within tolerance;
- training uses the same resolved HD2 objective for both architectures;
- no full Hessian is materialized in HD3 training;
- checkpoint/resume equivalence passes a short deterministic interruption test;
- OOM fallback and stale-provenance rejection are tested;
- output schemas and artifact hashing are tested;
- no RTX 4070 result is claimed before the user runs it.
