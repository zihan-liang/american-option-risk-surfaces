# M08 Standalone Tensor and Low-Rank Screen Plan

## Objective

Test whether complete CN/PSOR American-put risk surfaces have enough structured low-rank content to achieve at least 10x storage compression without materially degrading price, Delta, Gamma, exercise-boundary, obstacle, or discrete complementarity diagnostics. This is a classical falsification/compression experiment, not a QTT PDE solver or quantum run.

## Mathematical object and oracle

Use the repository CN/PSOR solver as the detailed-model oracle. The sampled object is the normalized one-asset American put surface over state/time and parameter regimes, with strike fixed at K=1 so that the repository's homogeneity convention corresponds to v=V/K. The tensor modes are `(tau, S, sigma, r, q)`; each `(sigma,r,q)` regime is a complete `(tau,S)` surface.

For every reconstructed candidate, evaluate the same Crank-Nicolson obstacle stencil as the oracle:

`R^{n+1} = A V^{n+1} - b(V^n)`.

Report obstacle violation, dual-feasibility violation, complementarity product, and continuation-region residual using the oracle contact mask. Do not apply `max(V,g)` or another hard repair before diagnostics.

## Compared families

Exactly eight predeclared configurations are permitted, two per family:

1. `tt_svd_r4`, `tt_svd_r8`: deterministic TT-SVD oracle-compressibility baselines using the complete tensor.
2. `tt_completion_r4_f10`, `tt_completion_r8_f25`: genuine TT-core completion approximations trained only on complete observed training regimes. These are deliberately named `tt_completion`; they are not described as TT-cross.
3. `pod_r4`, `pod_r8`: matrix POD/SVD oracle-compressibility baselines with `(tau,S)` as the state block and `(sigma,r,q)` as the regime block.
4. `primal_dual_rb_r4`, `primal_dual_rb_r8`: dual-aware reduced-basis compression baselines. The primal snapshots are value surfaces and the dual snapshots are nonnegative CN/LCP residual snapshots. This implementation is not claimed to be a reduced variational-inequality time-stepper.

TT-SVD, POD, and primal-dual RB have full-tensor access because they test oracle compressibility. Only `tt_completion` tests reconstruction from incomplete regime observations.

## Frozen canonical configuration

- CPU float64.
- Base CN/PSOR grid: `m=96`, `n=48`, `T=2`, `Smax=4`, tolerance `1e-10`.
- Refined numerical-floor oracle: `m=192`, `n=96` on validation/test regimes.
- Parameter grid: 7 sigma values in `[0.10,0.60]`, 5 rates in `[0,0.10]`, 5 dividend yields in `[0,0.08]` = 175 complete regimes.
- Regime split: 70% train, 15% validation, 15% test; seed 808.
- TT-completion replicate seeds: `[101,203,307,409,503]` for each of the two completion configurations.
- Completion optimizer: CPU-float64 Adam, 400 steps, batch 2048, learning rate 0.01.
- Timing: five warmups plus 30 measured reconstruction repetitions.
- Paired regime bootstrap: 500 repetitions, seed 808.
- Screen budget: 3600 seconds.

The 175-regime grid is a bounded screen configuration. The larger 819-regime tensor proposed in `deep_research/8.md` is a research recommendation rather than a campaign requirement and is deferred unless the bounded screen establishes sufficient margin.

## Selection protocol

All eight configurations are fit before test evaluation. Selection is validation-only:

- deterministic family score = validation gate ratio;
- TT-completion configuration score = worst validation gate ratio over all five seeds;
- freeze one winner per family;
- evaluate only the four frozen winners on test;
- a frozen TT-completion winner passes only if all five seeds satisfy the test gate.

No blind holdout is accessed.

## Metrics and gates

Record actual stored scalar count, compression ratio, TT rank vectors where applicable, reconstruction time, price RMSE, Delta RMSE, Gamma RMSE, boundary RMSE, obstacle violation, dual violation, complementarity error, continuation residual, and raw shape violations.

The refined CN/PSOR surface supplies a numerical-discretization floor. Degradation is computed relative to the base-CN versus refined-CN error on the same regime, then paired-bootstrap resampled by complete regime.

A configuration passes only if:

- compression ratio >= 10x;
- 97.5% bootstrap upper degradation <= 2% for price, Delta, Gamma, boundary, and obstacle diagnostics;
- 97.5% bootstrap upper degradation <= 5% for dual, complementarity, and continuation-residual diagnostics.

Smoke evidence is always `diagnostic_only` regardless of numerical outcome.

## TDD and verification

Focused tests cover TT reconstruction/storage, POD and primal/dual basis dimensions, complete-regime completion sampling, best-checkpoint retention, exact CN residual reproduction, dual snapshot construction, paired bootstrap, timing counts, status classification, and the 10x/2%/5% gate.

Verification ladder:

```bash
export PYTHONPATH=src:experiments
export MPLBACKEND=Agg
python -m pytest tests/test_m08_tensor_low_rank.py -q
python -m compileall -q src tests experiments
python experiments/standalone/m08_tensor_low_rank.py \
  --config configs/standalone/m08_tensor_low_rank.json --smoke
```

Canonical screen, once the implementation commit is clean and fixed:

```bash
export PYTHONPATH=src:experiments
export MPLBACKEND=Agg
python experiments/standalone/m08_tensor_low_rank.py \
  --config configs/standalone/m08_tensor_low_rank.json
```

The canonical command is run once from the implementation commit. Stop on NaN/nonfinite outputs, CN/PSOR non-convergence, split/config/seed mismatch, blind-holdout access, or one-hour budget exhaustion.

## Standard artifacts and claim limits

Write only the standard evidence files under `results/standalone/M08/`: `config_manifest.json`, `metrics.csv`, `decision.json`, and `artifact_hashes.json`. Preserve negative outcomes. Do not claim TT-cross, a tensorized American PDE solver, quantum advantage, preserved free-boundary accuracy, or preserved complementarity unless the corresponding measured gate passes.
