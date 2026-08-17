# HD1 Windows High-Dimensional PINN 48-Hour Pilot

## Purpose And Claim Boundary

HD1 is a one-seed feasibility pilot for an RTX 4060 8 GB Windows machine. It
tests whether an obstacle-aware neural PDE surrogate can retain accuracy as the
state input grows from one to ten assets. It is separate from M01--M24 and must
appear in a separate high-dimensional panel in the final report.

This pilot may produce a useful result within 48 hours, but it cannot establish
statistical superiority, general high-dimensional American-option performance,
or GPU speedup. Every result is `diagnostic_only` because only seed `101` is
used.

## Research Basis

The local Deep Research audit identifies direct precedents:

- DGM reported high-dimensional free-boundary PDE examples, including American
  options, up to 200 state dimensions.
- Gatta et al. studied single- and multi-asset American-put PINNs.
- ETCNN studied obstacle-aware terminal structure for single- and multi-asset
  American complementarity problems.
- Whole-domain neural variational-inequality and Deep Penalty work support
  obstacle-aware and high-dimensional optimal-stopping formulations.
- Deep BSDE and recurrent optimal-stopping methods are relevant high-dimensional
  comparators but do not return the same global PINN surface.

The evidence does not establish a mature high-dimensional PINN that jointly
delivers accurate price, all Greeks, complementarity, and a general stopping
surface. Read only:

- `/Users/liangzihan/Documents/SURF_0159/SURF/deep_research/2.md`
- `/Users/liangzihan/Documents/SURF_0159/SURF/deep-research-report_PINNs.md`

## Why The Pilot Uses A Geometric Basket

For equal weights define

\[
G(\mathbf S)=\left(\prod_{i=1}^d S_i\right)^{1/d},
\qquad
g(\mathbf S)=(K-G(\mathbf S))^+.
\]

Under correlated geometric Brownian motion, $G$ is itself a one-dimensional
geometric Brownian motion. With identical asset volatility $\sigma$, dividend
yield $q$, and equicorrelation $\rho$,

\[
\sigma_G^2=\sigma^2\left(\rho+\frac{1-\rho}{d}\right),
\qquad
q_G=q+\frac12\left(\sigma^2-\sigma_G^2\right).
\]

The $d$-asset American geometric-basket put therefore has a trusted
one-dimensional CN/PSOR reference in $G$. This is an intentionally reducible
high-dimensional control: it validates the neural PDE, derivatives, obstacle,
and CUDA pipeline but is not evidence for an arbitrary high-dimensional
stopping boundary.

## Frozen Contract And Dimensions

Use one fixed contract for every dimension:

```text
K = 1.0
T = 1.0
r = 0.05
q = 0.02
sigma = 0.25
rho = 0.30
dimensions = [1, 2, 5, 10]
seed = 101
log-moneyness range = [log(0.5), log(1.5)] for every asset
tau range = [0.0025, 1.0]
```

The log-return covariance is

\[
\Sigma_{ii}=\sigma^2,
\qquad
\Sigma_{ij}=\rho\sigma^2\quad(i\ne j).
\]

Generate an independently converged one-dimensional CN/PSOR oracle for each
effective $(\sigma_G,q_G)$. Use a grid-refinement comparison before accepting
the oracle. The train, validation, and test Sobol sequences must use distinct
scrambles/seeds. The final test sample is fixed before training and never used
for loss selection.

## Models

Train exactly two models at each dimension with identical architecture,
initialization, supervised samples, minibatch order, optimizer, and step count.

### SUP: Supervised DeepSets

- Shared per-asset encoder: two width-64 smooth layers.
- Mean aggregation across assets, preserving permutation invariance.
- Value head: four width-128 smooth layers.
- Activation: `tanh` for every hidden layer.
- Hard terminal/obstacle representation:

\[
v_\theta(\mathbf S,\tau)
=g(\mathbf S)+\tau\,\operatorname{softplus}(h_\theta).
\]

- Objective: normalized CN/PSOR price MSE only.

### HVI: Hybrid Variational-Inequality PINN

Use the identical network and supervised objective, plus a dimensionless
whole-domain Fischer--Burmeister loss. In log-price coordinates
$x_i=\log(S_i/K)$, define

\[
R_\theta
=v_\tau
-\frac12\sum_{i,j}\Sigma_{ij}v_{x_ix_j}
-\sum_i\left(r-q-\frac12\sigma^2\right)v_{x_i}
+rv,
\]

\[
a=v_\theta-g,\qquad b=R_\theta,
\qquad
\phi_\varepsilon(a,b)
=\sqrt{a^2+b^2+\varepsilon^2}-a-b.
\]

Use a 2,000-step supervised warm start. During the remaining 8,000 steps,
linearly ramp the predeclared complementarity weight from zero to `0.1` and
anneal $\varepsilon$ from `1e-2` to `1e-4`. Evaluate the unsmoothed residual.
Do not tune these values after test results are visible.

## High-Dimensional Derivatives

Use full automatic-differentiation Hessian contraction for $d=1,2$. For
$d=5,10$, use a two-probe fixed-seed Hutchinson Hessian-vector estimator for

\[
\operatorname{tr}(\Sigma\nabla_x^2v).
\]

With $LL^\top=\Sigma$ and independent Rademacher probes
$\xi_k$, compute $z_k=L\xi_k$ and

\[
\widehat{\operatorname{tr}(\Sigma H)}
=\frac1{N_p}\sum_{k=1}^{N_p}z_k^\top H z_k,
\qquad N_p=2.
\]

Use Hessian-vector products; do not materialize a full per-sample Hessian for
$d=5,10$.

Before canonical training, compare the estimator with the full contraction on
a small $d=2$ batch and require relative error at most 5%. Record probe seed,
probe count, estimator error, derivative time, and peak memory. Do not call a
diagonal-Hessian approximation the correlated operator.

For oracle Greek checks, if $U=U(G,\tau)$,

\[
\frac{\partial G}{\partial S_i}=\frac{G}{dS_i},
\]

\[
\frac{\partial^2G}{\partial S_i^2}
=\frac{G(1-d)}{d^2S_i^2},
\qquad
\frac{\partial^2G}{\partial S_i\partial S_j}
=\frac{G}{d^2S_iS_j}\quad(i\ne j),
\]

and apply the chain rule to the one-dimensional oracle $U_G,U_{GG}$. Use
the same finite-difference stencil on the refined $G$-grid for every model.

## Data And Training Budget

```text
supervised train points: 20,000 per dimension
validation points: 8,192 per dimension
fixed test points: 16,384 per dimension
physics collocation: generated Sobol batches during HVI training
Adam steps: 10,000 per model
initial batch size: 2,048
predeclared OOM fallback: 1,024, then 512
maximum wall time: 3 hours per training run
maximum CUDA allocation: 7.0 GiB
mixed precision: disabled
canonical dtype: CUDA float32; oracle generation/evaluation: CPU float64
```

Use CN/PSOR grids `(m,n)=(256,256)` and `(512,512)` for the oracle refinement
check. Require normalized price RMSE between the two grids no greater than
`1e-4` on the declared $G$-evaluation interval. Use the refined grid as the
reference. If this fails, stop and report `oracle_blocked` rather than relaxing
the threshold after training begins.

There are eight canonical runs: two models times four dimensions. The maximum
declared training budget is therefore 24 GPU-hours. Run sequentially in this
order:

```text
d1 SUP, d1 HVI, d2 SUP, d2 HVI,
d5 SUP, d5 HVI, d10 SUP, d10 HVI
```

Before them, run only 500 steps for `d=1 HVI` and `d=2 HVI` as implementation
smokes. Smoke outputs are not evidence. If the measured 500-step profile
projects more than 24 total training hours, reduce all canonical steps once to
the predeclared fallback of 6,000 and record the change before any canonical
run starts. Do not reduce steps after observing accuracy.

## 48-Hour Schedule

```text
hours 0--4: environment, CUDA, oracle and focused-test gates
hours 4--10: implementation and derivative validation
hours 10--14: 500-step smokes and runtime projection
hours 14--38: sequential canonical training, capped at 24 GPU-hours
hours 38--46: fixed-test evaluation, hashes and closure
hours 46--48: contingency only; do not begin an extra experiment
```

The schedule is a cap, not a requirement to consume all 48 hours. Stop early
when a correctness gate fails and preserve the negative result.

## Expected But Non-Binding Outcome

- `d=1` and `d=2` should establish whether the implementation is correct.
- Price accuracy may remain usable at `d=5` and `d=10` because the geometric
  basket is reducible and the architecture is permutation invariant.
- HVI may reduce complementarity/PDE diagnostics without improving price.
- Delta and Hessian-contraction accuracy are expected to deteriorate before
  global price RMSE, especially at `d=10`.
- A negative result at `d=5` or `d=10` is scientifically valid and must not
  trigger architecture or loss changes inside the pilot.

## Metrics

Use the same fixed test points for SUP and HVI. Report by dimension:

- normalized price RMSE, MAE, P95 and maximum error;
- per-asset Delta RMSE and maximum error;
- covariance-weighted Hessian/Gamma-contraction RMSE;
- obstacle violation maximum and rate;
- unsmoothed Fischer--Burmeister RMS, P95 and maximum;
- trusted continuation PDE RMS and P95 using the oracle $G$-boundary;
- exercise/continuation F1 and boundary-tube F1;
- boundary RMSE on diagonal $S_1=\cdots=S_d=G$ slices;
- fixed-threshold boundary sensitivity;
- training, derivative, oracle, evaluation and inference seconds;
- peak CPU and CUDA memory;
- parameter count and points processed.

For $d=1$, also emit a bridge table comparing the new SUP/HVI implementation
with committed M02 metrics only when definitions and test distributions match.
Otherwise mark the bridge `not_comparable`. Never place HD1 and M01--M24 in one
numerical ranking because they solve different contracts and dimensions.

## Diagnostic Gates

The one-seed pilot never produces `candidate_supported`. Record pass/fail for:

1. finite metrics and artifact hashes;
2. maximum obstacle violation no greater than `1e-6`;
3. HVI price RMSE no worse than SUP by more than 5%;
4. HVI trusted PDE P95 no worse than SUP by more than 5%;
5. HVI Gamma-contraction RMSE no worse than SUP by more than 10%;
6. boundary RMSE no worse than SUP by more than 10%;
7. CUDA allocation below 7.0 GiB;
8. no failed/OOM run hidden from the summary.

An HVI result is called `pilot_promising` only when price or diagonal-boundary
RMSE improves at least 5% and every non-regression gate passes. This is still a
single-seed diagnostic, not an improvement claim.

## Stop Rules

- Stop before training if CUDA, dataset/oracle generation, or Hessian checks
  fail.
- Stop the dimension ladder if both models fail correctness at a dimension;
  preserve outputs and do not continue to larger $d$.
- An OOM may use only the two predeclared batch fallbacks.
- A three-hour run becomes `resource_blocked`; do not silently extend it.
- Do not add $d=20$, arithmetic baskets, Heston, extra seeds, L-BFGS, local
  adaptive sampling, or another architecture during this 48-hour pilot.
- Do not access the campaign blind holdout.

## Files And Git

Create a separate branch from the latest pushed
`origin/campaign/independent-methods`:

```text
exp/hd1-windows-high-dimensional-pinn-48h
```

Expected source layout:

```text
configs/hd1_windows_high_dimensional_pinn_48h.json
experiments/hd1_windows_high_dimensional_pinn_48h.py
src/models/high_dimensional_pinn.py
src/physics/high_dimensional_black_scholes.py
tests/test_hd1_high_dimensional_pinn.py
results/hd1_windows_high_dimensional_pinn_48h/
docs/campaign/hd1_windows_high_dimensional_pinn_48h_closure.md
```

Checkpoints, full logs, environments and large arrays remain ignored. Compact
outputs must include:

```text
environment_manifest.json
config_manifest.json
run_metrics.csv
dimension_summary.csv
training_history.csv
hessian_validation.csv
decision.json
artifact_hashes.json
```

Make an implementation commit before canonical runs so every manifest records
the exact implementation SHA. After all available runs finish, make a separate
result commit. Push the HD1 branch but do not merge it into the campaign branch.

## Required Completion Report

Return at most 30 lines containing branch and commits, Windows/CUDA versions,
smoke status, completed dimensions, projected and actual GPU-hours, per-model
summary metrics, gates, stopped/blocked runs, artifact paths, test commands,
and explicit limitations. Do not return full logs.
