# HD2 Classical vs Structured High-Dimensional PINN Design

## Status

- Phase: design only
- Branch: `exp/hd2-classical-vs-structured-high-dimensional-pinn`
- Parent experiment: HD1 (`exp/hd1-windows-high-dimensional-pinn-48h`)
- Target hardware: one Windows laptop with NVIDIA RTX 4070 Laptop GPU, 8 GB VRAM
- Seed count: exactly one canonical seed, `101`
- Claim status: every HD2 result is `diagnostic_only`
- Campaign relation: separate high-dimensional panel; never rank numerically against M01--M24

## Purpose

HD2 is a controlled architecture-scaling comparison for the American geometric-basket variational inequality. It asks whether a permutation-invariant structured PINN remains more usable than a conventional fully connected PINN as the state dimension increases to 10, 20, 50, and optionally 100 assets on a single 8 GB laptop GPU.

The goal is not state-of-the-art American-option accuracy and not statistical superiority. The goal is to obtain one auditable, reproducible, genuinely high-dimensional neural-PDE comparison that can be used in the SURF study.

The central comparison is:

1. `FC_PINN`: a conventional fully connected PINN taking all asset log-prices and time to maturity as one vector.
2. `DEEPSETS_PINN`: a permutation-invariant PINN with a shared per-asset encoder, mean aggregation, and a global value head.

Both models solve the same obstacle/variational-inequality problem, use the same collocation states, optimizer, loss definition, training steps, precision, Hessian estimator, and evaluation oracle. No interior CN/PSOR price labels are used during training. CN/PSOR is used only for oracle construction and post-training evaluation.

## Relationship To HD1

HD1 established the following useful facts and infrastructure:

- the equal-weight geometric-basket American put reduces to a one-dimensional trusted oracle in the geometric mean;
- the high-dimensional Black--Scholes operator and covariance-weighted Hessian contraction are implemented;
- fixed-seed Hutchinson Hessian-vector estimation works for the correlated operator;
- the existing DeepSets implementation is permutation invariant;
- one-seed high-dimensional results are appropriately `diagnostic_only`;
- 10D CUDA execution is inexpensive enough that a 50D/100D laptop study is reasonable to attempt.

HD2 does not rewrite or reinterpret HD1 results. It extends the validated high-dimensional infrastructure on a separate branch and uses a new training protocol and new evaluation samples.

## Scientific Question

Primary question:

> Under a matched pure physics-informed American-option VI objective, how does a conventional fully connected PINN compare with a permutation-invariant DeepSets PINN as dimension grows from 10 to 50 assets, with 100 assets as a predeclared hardware stress test?

Secondary questions:

- Does either model remain finite and trainable as dimension increases?
- How do normalized price error, PDE/complementarity diagnostics, derivative error, runtime, and peak CUDA memory scale with dimension?
- Does the structured model preserve permutation and transverse-state invariance more reliably than the conventional model?
- Is two-process GPU concurrency beneficial on the actual RTX 4070 Laptop, or does second-order autodiff contention make sequential execution faster?

## Claim Boundary

Allowed claims are limited to the fixed geometric-basket contract and the tested laptop/runtime configuration.

HD2 must not claim:

- general high-dimensional American-option superiority;
- statistical superiority from one seed;
- a replacement for CN/PSOR;
- a result for arbitrary non-symmetric baskets;
- a GPU speedup over another hardware platform;
- that 100D stress accuracy is directly comparable with 10D/20D/50D if the stress step budget differs.

A positive result may be described only as evidence that the tested architecture is a usable high-dimensional neural-PDE comparator on this reducible, permutation-symmetric control.

## Frozen Financial Contract

Use the HD1 normalized contract:

```text
K = 1.0
T = 1.0
r = 0.05
q = 0.02
sigma = 0.25
rho = 0.30
seed = 101
canonical dimensions = [10, 20, 50]
stress dimension = [100]
tau evaluation range = [0.0025, 1.0]
```

For equal weights,

\[
G(\mathbf S)=\left(\prod_{i=1}^{d} S_i\right)^{1/d},
\qquad
 g(\mathbf S)=(1-G)^+.
\]

In log-moneyness coordinates

\[
x_i=\log(S_i/K),\qquad m=\frac1d\sum_i x_i=\log G,
\]

and

\[
\Sigma_{ii}=\sigma^2,\qquad
\Sigma_{ij}=\rho\sigma^2\quad(i\ne j).
\]

The effective one-dimensional oracle parameters remain

\[
\sigma_G^2=\sigma^2\left(\rho+\frac{1-\rho}{d}\right),
\qquad
q_G=q+\frac12\left(\sigma^2-\sigma_G^2\right).
\]

## Why The Sampling Distribution Changes From HD1

Uniformly sampling every asset independently from a fixed interval causes the sample geometric mean to concentrate increasingly near the center as dimension grows. That would make a 100D test distribution materially easier and narrower than a 10D test distribution.

HD2 therefore samples the aggregate log geometric mean directly and adds zero-mean transverse deviations:

\[
x_i=m+\delta_i,\qquad \frac1d\sum_i\delta_i=0.
\]

This keeps the geometric mean distribution fixed across dimensions while still presenting a genuinely d-dimensional state vector to the network and PDE operator.

### Core collocation distribution

For every dimension and model:

```text
m ~ Uniform(log(0.25), log(2.0))
tau ~ Uniform(0.0025, 1.0)
raw u_i ~ Uniform(-1, 1)
delta_i = 0.35 * (u_i - mean(u))
x_i = m + delta_i
```

Use scrambled Sobol draws with separate fixed sequence seeds for training, validation, and test. Canonical training resets the generator and model seed independently of all smoke runs.

### Fixed evaluation distribution

The evaluation set uses

```text
m ~ Uniform(log(0.5), log(1.5))
tau ~ Uniform(0.0025, 1.0)
the same zero-mean transverse-deviation construction
```

Use a new fixed test scramble that was not used by HD1. The test set is generated from the frozen config but is not read for model selection or architecture changes.

## Model A: Conventional Fully Connected PINN

Name: `FC_PINN`.

Input:

\[
(x_1,\ldots,x_d,\tau).
\]

Architecture:

```text
input d+1
6 hidden layers
width 192 each
tanh activations
scalar raw head
```

Use one fixed architecture at every dimension; do not resize the network to match parameter counts after seeing results. Exact parameter count is reported per dimension because the first layer grows with `d`.

The physical value transform is shared with the structured model:

\[
v_\theta(\mathbf x,\tau)
=g(\mathbf x)+\tau\,\operatorname{softplus}(h_\theta(\mathbf x,\tau)).
\]

This enforces

\[
v_\theta\ge g
\]

and exactly enforces the terminal payoff at `tau=0`.

## Model B: Structured DeepSets PINN

Name: `DEEPSETS_PINN`.

Shared per-asset encoder:

```text
1 -> 96 -> 96 -> 96
three tanh hidden layers
```

Aggregation:

```text
mean over assets
```

Global head:

```text
pooled 96-vector + tau
6 hidden layers
width 192 each
tanh activations
scalar raw head
```

Use the same hard terminal/obstacle transform as `FC_PINN`.

The architecture is permutation invariant by construction. Its parameter count is nearly independent of the number of assets. This is part of the scalability comparison and must be reported rather than removed by dimension-specific width tuning.

## Pure Physics-Informed Training Objective

No interior oracle prices appear in the training loss.

Define the Black--Scholes VI residual in log coordinates:

\[
R_\theta
=v_\tau
-\frac12\operatorname{tr}(\Sigma\nabla_x^2v)
-\left(r-q-\frac12\sigma^2\right)\sum_i v_{x_i}
+rv.
\]

With

\[
a=v_\theta-g,\qquad b=R_\theta,
\]

use the smoothed Fischer--Burmeister residual

\[
\phi_\varepsilon(a,b)
=\sqrt{a^2+b^2+\varepsilon^2}-a-b.
\]

Core physics loss:

\[
L_{FB}=\operatorname{mean}(\phi_\varepsilon^2).
\]

The smoothing parameter is log-linearly annealed from

```text
epsilon_initial = 1e-2
epsilon_final = 1e-4
```

over the full training run.

### Asymptotic regime losses

Because the Black--Scholes state space is unbounded and HD2 does not use oracle boundary labels during training, add two analytic asymptotic anchors that are applied identically to both models.

Low-geometric-mean exercise regime:

```text
G in [0.15, 0.25]
```

penalize

\[
L_{low}=\operatorname{mean}[(v-g)^2].
\]

High-geometric-mean decay regime:

```text
G in [2.0, 3.0]
```

penalize

\[
L_{high}=\operatorname{mean}[v^2].
\]

Total loss:

\[
L=L_{FB}+L_{low}+L_{high}.
\]

All three terms are means over their own fixed-size batches, so equal weights are frozen and do not depend on raw sample counts. No adaptive loss weighting, PCGrad, NTK weighting, ReLoBRaLo, L-BFGS, or post-test tuning is part of HD2.

The far-field terms are approximate asymptotic conditions, not exact finite-domain Dirichlet data. Their possible truncation bias must be listed as a limitation.

## Batch Construction

For every canonical step:

```text
core FB batch = 256
low-G asymptotic batch = 64
high-G asymptotic batch = 64
```

The canonical batch sizes do not change when the runner chooses concurrent or sequential execution. This prevents GPU scheduling decisions from changing the scientific training protocol.

Predeclared OOM fallbacks apply globally to every run before restart:

```text
core: 256 -> 192 -> 128
asymptotic: 64 -> 48 -> 32
```

A fallback changes batch size only; architecture, loss, seed, steps, dimension, and test distribution remain frozen. Record every OOM attempt.

## High-Dimensional Second Derivatives

Use the existing covariance-weighted Hutchinson Hessian-vector estimator for every HD2 training dimension:

```text
probe count = 2
probe seed = 101
```

With `LL^T = Sigma` and Rademacher probes `xi_k`, let `z_k=L xi_k` and estimate

\[
\operatorname{tr}(\Sigma H)
\approx
\frac1{2}\sum_{k=1}^{2} z_k^T H z_k.
\]

Never materialize a full per-sample Hessian for 10D/20D/50D/100D training.

Preflight must run a small d=2 exact-Hessian comparison and require relative contraction error no greater than 5% before any canonical training starts.

Tests must specifically detect:

- accidental `.detach()` through the HVP path;
- non-finite second derivatives;
- graph reuse across optimizer steps;
- incorrect correlated covariance contraction;
- probe nondeterminism.

## Optimizer And Step Budget

```text
optimizer = Adam
learning rate = 1e-3
canonical steps = 5000 per model for d=10,20,50
100D stress steps = 3000 per model
mixed precision = disabled
training dtype = CUDA float32
oracle/evaluation dtype = CPU float64
```

No supervised warm start is used.

### One-day budget fallback

Before any canonical run, smoke timing projects the complete 10D/20D/50D ladder.

If projected canonical training exceeds 18 wall-clock hours, the sole predeclared fallback changes all six canonical runs from 5000 to 3000 steps before canonical execution begins. Record this in the resolved config manifest.

If the projected canonical ladder still exceeds 18 hours after the fallback, stop with `resource_blocked`; do not silently shrink the networks, dimensions, Hessian probe count, or loss.

The 100D stress pair is started only if measured elapsed time plus its smoke-based projected time remains within the same 18-hour cap.

## Parallelism Policy

Parallelism is an execution optimization, not a scientific variable.

One RTX 4070 Laptop GPU does not guarantee that two CUDA processes are faster than sequential execution. HD2 therefore uses an automatic preflight decision.

### Parallelism smoke

At 10D, benchmark exactly 200 training steps for each architecture:

1. `FC_PINN` alone;
2. `DEEPSETS_PINN` alone;
3. both jobs concurrently after a fresh process restart.

Canonical jobs reset all model and Sobol seeds, so smoke execution cannot alter canonical data order.

Enable two-job concurrency for the 10D and 20D pairs only when all conditions hold:

- both concurrent smoke jobs complete without OOM or non-finite values;
- concurrent pair wall time is at most 90% of the sum of the two sequential smoke wall times;
- sampled total device use remains below 7.2 GiB.

Otherwise use sequential execution.

50D and 100D run sequentially by default regardless of the 10D decision.

The final report records which execution mode was selected and both smoke timings. Do not describe this as a hardware speedup study.

## Canonical Run Order

The logical order is frozen even if a pair is executed concurrently:

```text
d10 FC_PINN
d10 DEEPSETS_PINN
d20 FC_PINN
d20 DEEPSETS_PINN
d50 FC_PINN
d50 DEEPSETS_PINN
```

Then, if the time gate permits:

```text
d100 FC_PINN stress
d100 DEEPSETS_PINN stress
```

A failed architecture at one dimension does not suppress the other architecture's run. Continue to the next canonical dimension unless both models fail numerical correctness or the global resource cap is reached.

## Oracle And Evaluation

For every dimension, generate the reducible one-dimensional CN/PSOR oracle using the effective `(sigma_G, q_G)` parameters.

Use HD1's refinement protocol:

```text
coarse grid = (256, 256)
refined grid = (512, 512)
normalized price refinement RMSE <= 1e-4
```

If refinement fails for any dimension, stop that dimension as `oracle_blocked` rather than relaxing the threshold.

The oracle is not queried for interior training labels.

## Metrics

Use the same fixed test states for both models at a given dimension.

### Accuracy

- normalized price RMSE, MAE, P95, maximum error;
- full-vector per-asset Delta RMSE and maximum error;
- covariance-weighted Hessian/Gamma-contraction RMSE.

### Physics

- obstacle violation maximum and rate;
- unsmoothed Fischer--Burmeister RMS, P95, maximum;
- trusted continuation PDE RMS and P95 using the oracle continuation region.

### Structural diagnostics

- permutation-consistency RMSE: prediction difference after fixed random asset permutations;
- transverse-invariance RMSE: prediction variation across states with identical `m`, `tau`, and different zero-mean deviations;
- diagonal geometric-basket boundary RMSE as a secondary diagnostic only.

The structured model should have permutation consistency at floating-point tolerance by construction. The classical model is not penalized during training for permutation inconsistency; the metric is observational.

### Scalability

- training wall time;
- seconds per optimizer step;
- derivative evaluation time;
- inference time;
- peak CUDA allocation;
- sampled device-memory peak during concurrent smoke;
- parameter count;
- points processed;
- OOM fallback count.

## Success Interpretation

HD2 has no promotion to `candidate_supported`.

A run is `completed` when it has finite outputs, verified artifact hashes, and all required metrics. The experiment as a whole is useful even if one architecture has materially worse accuracy.

For a concise descriptive comparison, report whether each model satisfies these non-promotion diagnostic bands:

```text
normalized price RMSE <= 1e-2
obstacle violation max <= 1e-6
finite FB/PDE metrics
peak CUDA allocation < 7.2 GiB
```

These are usability bands, not claims of superiority and not gates that may be changed after results are visible.

## Stop Rules

Stop before canonical training if:

- CUDA is unavailable;
- focused HD2 tests fail;
- oracle refinement fails;
- d=2 Hutchinson validation exceeds 5% relative error;
- a required metric path is non-finite during smoke;
- the resolved implementation SHA does not match the manifest.

During execution:

- use only the declared batch fallbacks after OOM;
- stop a run if repeated fallback still OOMs;
- stop the experiment at the 18-hour wall-clock cap;
- do not lower dimension because a result looks poor;
- do not add extra seeds;
- do not change architecture, weights, learning rate, loss, Hessian probes, or evaluation data after canonical training begins;
- do not access the M01--M24 blind holdout.

## Expected Repository Layout

HD2 should add isolated files rather than mutate HD1 behavior:

```text
configs/hd2_classical_vs_structured_high_dimensional_pinn.json
experiments/hd2_classical_vs_structured_high_dimensional_pinn.py
src/models/hd2_high_dimensional_pinn.py
src/physics/hd2_high_dimensional_sampling.py
tests/test_hd2_high_dimensional_pinn.py
scripts/run_hd2_windows.ps1
scripts/check_hd2_status.ps1
docs/campaign/hd2_windows_execution_instructions.md
results/hd2_classical_vs_structured_high_dimensional_pinn/
```

Reuse the existing tested Black--Scholes operator/oracle utilities where their contracts match. Do not modify HD1 result files.

## Compact Outputs

The finalizer writes:

```text
environment_manifest.json
config_manifest.json
parallelism_benchmark.csv
hessian_validation.csv
run_metrics.csv
dimension_summary.csv
training_history.csv
decision.json
artifact_hashes.json
```

Per-job status, checkpoints, full logs, oracle arrays, and temporary monitoring traces remain ignored local runtime artifacts.

## Git Protocol

HD2 lives only on:

```text
exp/hd2-classical-vs-structured-high-dimensional-pinn
```

The branch intentionally extends the HD1 result branch so it can reuse the exact tested high-dimensional infrastructure while preserving HD1 evidence unchanged.

Required checkpoints:

1. design/spec commit;
2. implementation commit before any laptop canonical run;
3. separate result commit only after the user's RTX 4070 Laptop run actually completes.

Do not merge into `campaign/independent-methods` during HD2 development.

## Windows Execution Interface

Implementation must expose one PowerShell runner:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_hd2_windows.ps1 -Phase preflight
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_hd2_windows.ps1 -Phase smoke
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_hd2_windows.ps1 -Phase canonical
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_hd2_windows.ps1 -Phase stress
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_hd2_windows.ps1 -Phase all
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_hd2_windows.ps1 -Phase status
```

`all` must be safely resumable and must never rerun a completed current-commit canonical job unless explicitly forced by a separate documented option.

## Testing Strategy

Implementation follows TDD. At minimum, tests must cover:

1. both model output shapes and hard obstacle/terminal guarantees;
2. exact permutation invariance of `DEEPSETS_PINN`;
3. `FC_PINN` accepts all declared dimensions;
4. aggregate-preserving state sampling (`mean(x)==m` within tolerance);
5. identical geometric means under zero-mean transverse perturbations;
6. covariance matrix correctness and HVP differentiability;
7. d=2 full-vs-Hutchinson validation path;
8. no interior oracle labels in the training batch/objective;
9. fixed config dimensions, seed, steps, batch fallbacks, and memory cap;
10. smoke never writes canonical evidence;
11. 100D outputs are marked `stress`;
12. resume rejects stale status files from another implementation SHA;
13. artifact hashes verify;
14. runner never accesses the campaign blind holdout.

## Design Risks

### Pure-PINN convergence

Without interior oracle price labels, both models may converge less accurately than HD1's supervised/hybrid models. This is acceptable because HD2 is an architecture-scaling comparison, not an accuracy-promotion experiment.

### Far-field approximation

The low/high aggregate losses are asymptotic approximations. They can introduce finite-range truncation bias. Oracle evaluation must expose this rather than tuning the boundaries after results are seen.

### 100D autodiff memory

The main implementation risk is differentiable second-order HVP computation at 100D under 8 GB. The 100D pair is therefore a stress test and uses a smaller fixed step budget.

### Single-GPU concurrency

Two CUDA processes may be slower than sequential execution. The smoke benchmark chooses execution mode before canonical training and records the choice.

### Symmetry-aligned benchmark

The geometric-basket problem is intentionally permutation symmetric and therefore well matched to DeepSets. Any structured-model advantage applies to this class of symmetric contracts and must not be generalized to arbitrary multi-asset stopping problems.

## Definition Of Done For The Implementation Phase

Before the user pulls the branch to the laptop, the implementation phase is complete only when:

- the frozen config and all source files exist on the HD2 branch;
- focused unit/contract tests pass in the available non-CUDA environment;
- Python modules compile;
- PowerShell scripts are syntactically reviewed and contain no claimed CUDA results;
- the execution guide contains exact pull, environment, preflight, run, status, resume, and result-commit instructions;
- no canonical RTX 4070 result is claimed until the user actually runs it;
- the implementation commit SHA is recorded for the laptop run.
