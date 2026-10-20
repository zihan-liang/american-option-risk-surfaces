# Final Poster Result Claims

Primary source of truth for matched PINN/HD numerical claims: consolidated dataset at commit `ca5e806`, with derived tables under `results/final_poster_summary/`.

Classical poster-headline claims additionally use Ruixi's frozen unified-poster evidence at commit `3aa2d821b9c92d095658782e7bcaf1322a90a7cb`. These classical timing and accuracy-reference scopes must remain separate from Zihan's independent M01 screen.

## Classical numerical results

### Strict LCP solver timing

Frozen unified-poster protocol: 67 held-out test/stress regimes, the same Mac, one CPU thread, float64, the same frozen 121x121 CN-LCP, five warm-ups, and 30 timed repetitions per arm.

- CN+PSOR pooled median: `0.3097067705 s`.
- CN+Policy Iteration pooled median: `0.0159997290 s`.
- CN+Projected LU pooled median: `0.0102586250 s`.
- CN+Projected LU is therefore `30.18989099416345x` faster than CN+PSOR by pooled median latency, which is poster-safe as `30.2x faster`.

  > Note (2026-10-20): the final poster reports a 26.0x median speedup (2.44e-1 s → 9.38e-3 s, same CN–LCP discretisation, 1e-12 residual tolerance). This document predates the final poster.

- Policy Iteration and Projected LU pass strict certification on all 67 regimes. Projected LU's maximum full-trajectory difference from Policy Iteration is approximately `2.62e-14`, with maximum normalized LCP residual approximately `6.30e-16`.
- Penalty/Newton is not a benchmark: it passed only `40/67` common residual gates and remains `FAILED_CORRECTNESS`.

Important scope distinction: Zihan's independent M01 screen used nine preregistered American-put regimes and found CN+Policy median `0.0073127569 s` versus CN+PSOR `0.1406132638 s` (`94.799%` reduction, about `19.23x`). That is independent supporting evidence and must not be substituted for the synchronized 67-regime poster timing result above.

### High-accuracy reference audit

Gamma comparisons are restricted to the validated stable mask and are not runtime claims.

Temporal discretization:

- CN quadratic median Gamma max error: `1.65742e-6`.
- L-stable DIRK quadratic median Gamma max error: `6.94696e-7`.
- Ratio: `2.385820560360215x`; poster-safe wording: `2.39x lower Gamma error` (or `2.4x` when space is tight).

Spatial grid:

- Uniform-grid median Gamma max error: `1.52088e-4`.
- Sinh strike-concentrated grid median Gamma max error: `2.59175e-5`.
- Ratio: `5.868158580109965x`; poster-safe wording: `5.87x lower Gamma error` (or `5.9x` when space is tight).

The selected high-accuracy numerical reference remains `DIRK + Policy Iteration + sinh grid`. It is an internal numerical reference, not an analytic American-option theorem and not an online speed competitor.

## Matched-contract cross-model price comparison

Contract: `put_T100_s060_r005_q010`, seed 101.
Shared comparison domain: `S/K in [0.5, 1.5]`, `tau/T in [0.01, 1.0]`.
Common reference: `DIRK_policy_sinh_M480_N960`.

- Arm C shared-domain DIRK price RMSE: `0.027882283751685957`.
- Arm D shared-domain DIRK price RMSE: `0.001045142110098794`.
- Therefore Arm D has `96.25159072547062%` lower matched-contract shared-domain price RMSE than Arm C, equivalent to a `26.67798329267427x` smaller RMSE.
- This matched-contract statement must not be replaced by the older 67-regime median claim of `98.3%`; they are different scopes.

For the final boundary-complete HD run:

| Method | d=10 | d=20 | d=50 |
| --- | ---: | ---: | ---: |
| FC-HVI shared-domain DIRK RMSE | 0.0028608864 | 0.0025077819 | 0.0023061590 |
| DeepSets-HVI shared-domain DIRK RMSE | 0.0032194287 | 0.0033166360 | 0.0040005197 |

Interpretation allowed for the poster: in this single-seed matched diagnostic, FC-HVI has lower shared-domain price RMSE than DeepSets-HVI at d=10, 20, and 50. Do not claim that DeepSets is more price-accurate.

At the predeclared middle dimension d=20, FC-HVI RMSE is `2.3994649859925334x` Arm D's matched RMSE; DeepSets-HVI is `3.1733828212472472x` Arm D's matched RMSE. These are observed accuracy ratios, not solver speedups.

## HD architecture diagnostics

All final HD runs use seed 101 on the RTX 4070 Laptop GPU environment.

### Parameter count

- FC-HVI: 84,225 (d=10), 85,505 (d=20), 89,345 (d=50).
- DeepSets-HVI: 62,401 at d=10, 20, and 50.
- DeepSets therefore uses 25.91%, 27.02%, and 30.16% fewer parameters than FC-HVI at d=10, 20, and 50 respectively.

### Permutation consistency

- FC-HVI RMSE: `4.48744e-4`, `6.10881e-4`, `1.12627e-3` at d=10, 20, 50.
- DeepSets-HVI RMSE: `2.32939e-8`, `2.35058e-8`, `2.60852e-8`.
- DeepSets has approximately `1.93e4x`, `2.60e4x`, and `4.32e4x` smaller permutation-consistency RMSE at d=10, 20, and 50.

Poster-safe conclusion: DeepSets provides a strong symmetry-consistency and parameter-complexity advantage.

### Training time

- FC-HVI: 46.69 s, 46.35 s, 46.55 s at d=10, 20, 50.
- DeepSets-HVI: 48.54 s, 48.61 s, 55.16 s.
- DeepSets is approximately 3.95%, 4.88%, and 18.51% slower in these three runs.

Poster-safe conclusion: no runtime advantage was observed for DeepSets in this implementation; do not claim a speedup.

## Free-boundary result

For the final boundary-complete HD experiment, all six models (FC-HVI and DeepSets-HVI at d=10, 20, 50) have primary free-boundary coverage `0%` at threshold `1e-6`, with zero predicted boundary points on the 201-point reference grid.

Poster-safe conclusion: reasonable price accuracy does not imply successful recovery of the American stopping geometry.

Do not draw or interpolate a synthetic HD free-boundary curve when the primary extractor reports no boundary.

## Protocol caveats that must accompany cross-model interpretation

- Low-dimensional Arm C/D are physics-only held-out PINN pilots.
- High-dimensional FC-HVI/DeepSets-HVI are oracle-assisted supervised + HVI diagnostic runs.
- The shared-domain DIRK RMSE is suitable for comparing observed predictive accuracy on a common reference/domain, but not for claiming a pure training-method efficiency or runtime speedup across low-D and HD protocols.
- HD results are post-hoc matched-contract, single-seed (101), diagnostic-only; no robustness claim is supported.
- d=20 is the predeclared middle dimension for the main poster slices, not a dimension selected after looking for the best result.

## Recommended poster wording

Classical numerical results:

> CN+Projected LU reduced pooled median solve time by 30.2x relative to CN+PSOR on the synchronized 67-regime benchmark, while passing 67/67 strict certification. In the accuracy-reference audit, L-stable DIRK reduced stable-mask median Gamma max error by 2.39x versus CN, and the sinh grid reduced it by 5.87x versus the uniform grid.

Matched low-dimensional comparison:

> On the matched American-put contract, structure-aware Arm D reduced shared-domain DIRK price RMSE by 96.3% relative to Arm C (0.00105 vs 0.02788; seed 101).

High-dimensional architecture comparison:

> DeepSets enforced near-exact permutation consistency and used 26-30% fewer parameters, but FC-HVI achieved lower price RMSE at all tested dimensions and DeepSets showed no runtime advantage.

Free-boundary conclusion:

> Despite approximately 2-4e-3 shared-domain price RMSE, all final HD models had 0% primary free-boundary coverage at the 1e-6 threshold, separating value accuracy from stopping-boundary fidelity.
