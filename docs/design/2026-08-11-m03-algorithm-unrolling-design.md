# M03 Algorithm Unrolling Design

**Status:** approved for implementation from the Independent All-Method Experiment Campaign

## Objective

Screen one learned feasible warm start against tuned projected SOR (PSOR) for the single-asset American put. Both methods must solve the same Crank--Nicolson linear complementarity problems to the same infinity-norm natural-residual tolerance. The learned method is an accelerator only: ordinary PSOR cleanup is mandatory, and prediction, projection, fallback, and cleanup all count toward total solve time.

## Considered Approaches

1. **Learned feasible warm start (selected).** Train a float64 MLP to predict the correction from the previous time-step solution to the next LCP solution using local CN/LCP and regime features. Project the prediction onto the obstacle before cleanup. This is trainable, independent of M01's evolving public solver API, and has an auditable correctness boundary.
2. **Trainable fixed-depth projected Jacobi over-relaxation (PJOR) unroll.** Learn one relaxation parameter per parallel projected-Jacobi step, then use ordinary PSOR for the mandatory tolerance-matched cleanup. This candidate was added alongside the learned feasible warm start; it must be reported as PJOR, not SOR, because each unrolled update uses the previous iterate at every coordinate.
3. **Fixed short PSOR prefix.** Rejected because a fixed number of ordinary PSOR iterations is not machine learning and cannot satisfy the approved M03 claim.

## Numerical Contract

For `A x >= b`, `x >= phi`, define the natural-residual vector as

```text
R(x) = x - max(phi, x - (A x - b)).
```

Both baseline and learned-cleanup solves stop only when `||R(x)||_inf <= residual_tolerance`. The implementation lives under `src/models` and `src/campaign`; core solver files remain unchanged. The baseline starts from the projected previous time-step solution. The candidate starts from the projected learned correction blended with that same previous solution, then runs the identical PSOR cleanup routine with the validation-selected omega.

Prediction failures, non-finite outputs, wrong shapes, or cleanup failures trigger and record fallback to the ordinary projected previous solution. A fallback solve still uses the same residual tolerance and all fallback and cleanup work remains timed.

## Learning Contract

The canonical model is a CPU float64 MLP with hidden width 128, depth 3, 30 epochs, batch size 8192, patience 8, and stochastic seeds `[101, 203, 307, 409, 503]`. Features are normalized using training regimes only. Targets are exact next-step corrections produced by a tighter PSOR label solve. Hard projection, not the network, guarantees obstacle feasibility.

Canonical complete surfaces use `m=76`, `n=100`: 20 training regimes produce 150,000 interior-node samples and 10 validation regimes produce 75,000. Regimes are generated deterministically and split before any node expansion. No legacy-test or blind-holdout access is available to the M03 runner.

At most eight configurations are predeclared. The default screen uses four PSOR omega values and candidate blend values that reuse the same trained model. Omega and one shared candidate blend are selected on complete validation regimes; the selected configuration is fixed across all five training seeds.

## Evaluation

Deterministic canonical timing uses five warm-ups and 30 measured repetitions per validation regime. Candidate total time includes feature construction, inference, projection, fallback, and cleanup. Training-label generation, preprocessing, training, selection, timing components, process RSS, and model bytes are reported separately.

Diagnostics compare the tolerance-matched methods with a tighter numerical oracle and include price RMSE, strict-mask Delta RMSE, strike-scaled Gamma RMSE, boundary RMSE, continuation-region RMSE, complementarity error, natural residual, obstacle violation, cleanup iterations, and fallback behavior. Missing required diagnostics block promotion and are represented as `null` plus a reason in method details; JSON never emits NaN or Infinity.

The primary statistic is paired total solve-time difference, resampled by complete validation regime. Promotion requires a gain of at least 2% and a paired 95% interval excluding zero, obstacle violation at most `1e-10`, no more than 2% worsening in price/Delta/`K*Gamma`/boundary/continuation/complementarity diagnostics, and no more than 5% residual worsening.

## Outputs And Claim Policy

`experiments/26_independent_method_campaign.py --method M03 --phase screen [--smoke]` dispatches the runner. The M03 directory contains:

- `config_manifest.json`
- multi-row `run_metrics.csv`
- `training_history.csv`
- `decision.json`
- `artifact_hashes.json`
- `scenario_metrics.csv`
- `selection_metrics.csv`
- `timing_samples.csv`
- `screen_summary.json`

Every output records or inherits branch/commit provenance, actual regime and sample counts, residual tolerance, timing components, and `blind_holdout_accessed=false`. Smoke runs are small enough for CI and are always `diagnostic_only`, `evidence=false`, and non-promoting even when their point estimates look favorable.
