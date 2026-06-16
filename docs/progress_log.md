# Progress Log

Use this file to record dated progress, commands, metrics, and decisions.

## Template

- Date:
- Stage:
- Command:
- Key metrics:
- Decision:
- Failure modes observed:
- Next step:

## 2026-06-13

- Stage: CN/PSOR solver validation audit.
- Commands:
  - `python -m unittest discover -s tests -p test_solver_audit.py`
  - `python experiments/01_solver_validation.py --config configs/solver_validation.yaml`
- Key metrics:
  - Smoke grid 120 European put max error: `2.781746966190046e-4`.
  - Smoke grid 120 no-dividend American call max error: `2.781740085198575e-4`.
  - Refined grid 240 European put max error: `6.937489127123442e-5`.
  - Refined grid 240 no-dividend American call max error: `6.937471924713112e-5`.
  - American put obstacle violation: `0.0`.
  - American put complementarity error: `4.0088005617245605e-12`.
  - Dividend-paying American call obstacle violation: `0.0`.
  - Dividend-paying American call complementarity error: `5.2696412383000194e-12`.
- Decision: solver-validation gate passes for continuing classical risk-surface experiments, but not yet as publication-grade final accuracy evidence.
- Failure modes observed: finite-difference error is still visible at moderate grids; `rannacher_steps` is configured but not implemented; boundary values move in grid increments because boundary extraction uses the current uniform grid.
- Next step: use validated solver settings for boundary/Greek diagnostics and keep grid/domain/runtime evidence attached to any reported surfaces.

## 2026-06-15

- Stage: Boundary extraction, Delta/Gamma, and diagnostics.
- Commands:
  - `python -m unittest discover -s tests`
  - `python -m compileall -q src tests experiments`
  - `python experiments/02_boundary_greeks_diagnostics.py --config configs/base_put.yaml`
  - `python experiments/02_boundary_greeks_diagnostics.py --config configs/base_call_dividend.yaml`
  - `python experiments/02_boundary_greeks_diagnostics.py --both`
- Key metrics:
  - Put obstacle violation: `0.0`.
  - Put complementarity error: `4.0088005617245605e-12`.
  - Put boundary at final time level: `0.8K`, status `boundary_found`.
  - Put strict Greek mask fraction: `0.9669421487603306`.
  - Dividend-call obstacle violation: `0.0`.
  - Dividend-call complementarity error: `5.2696412383000194e-12`.
  - Dividend-call boundary at final time level: `1.3333333333333333K`, status `boundary_found`.
  - Dividend-call strict Greek mask fraction: `0.9670104501058671`.
- Decision: Stage 3 passes for classical boundary and Greek diagnostics, with masked reporting required for Gamma.
- Failure modes observed: Gamma spikes near payoff kink and free boundary; boundary locations remain grid-resolution limited; Rannacher smoothing is still not active.
- Next step: run grid/domain sensitivity and runtime studies for price, boundary, Delta, and Gamma before surrogate dataset construction.

## 2026-06-15

- Stage: Grid/domain sensitivity and runtime for price, boundary, Delta, and Gamma.
- Commands:
  - `python -m unittest discover -s tests`
  - `python -m compileall -q src tests experiments`
  - `python experiments/03_convergence_domain_runtime.py --config configs/convergence.yaml`
- Key metrics:
  - Cases: American put and dividend-paying American call.
  - Grid reference: `M=N=240`, `Smax=4K`.
  - Grid set: `[40, 80, 120, 160, 240]`.
  - Domain set: `[3K, 4K, 5K]`.
  - Maximum obstacle violation across runs: `0.0`.
  - Maximum complementarity error across runs: `6.1288898628566304e-12`.
  - Maximum monotonicity violation across runs: `1.2988727874697525e-11`.
  - Maximum convexity violation across runs: `2.597745574939505e-11`.
  - Put grid-160 price RMSE vs reference: `4.720526685137331e-05`.
  - Put grid-160 boundary MAE vs reference: `0.017552083333333343`.
  - Put grid-160 strict Delta RMSE: `0.00046138667440877613`.
  - Put grid-160 strict Gamma RMSE: `0.05211017561299287`.
  - Dividend-call grid-160 price RMSE vs reference: `3.745913490534023e-05`.
  - Dividend-call grid-160 boundary MAE vs reference: `0.036822916666666674`.
  - Dividend-call grid-160 strict Delta RMSE: `0.0005393221866860245`.
  - Dividend-call grid-160 strict Gamma RMSE: `0.04616593897247874`.
  - Runtime at grid 240: put `0.8492777999999817s`, dividend call `1.1688343000000714s`.
- Decision: Stage 4 passes for continuing to classical application risk-surface experiments, but not for publication-grade convergence claims.
- Failure modes observed: boundary errors remain grid-resolution limited; Gamma convergence is weaker than price and Delta; grid-240 is an internal engineering reference, not an external benchmark; Rannacher smoothing is still inactive.
- Next step: generate American put and dividend-paying American call application risk surfaces with diagnostics attached to every surface.

## 2026-06-15

- Stage: Application risk-surface experiments.
- Commands:
  - `python -m unittest discover -s tests`
  - `python -m compileall -q src tests experiments`
  - `python experiments/04_application_risk_surfaces.py --config configs/application_sweep.yaml`
- Key metrics:
  - Total solved regimes: `288`.
  - Regime counts: put `144`, dividend call `108`, no-dividend call control `36`.
  - Saved representative surfaces: `12`, with four per regime class.
  - Surface run id: `run_20260615_102826`.
  - Maximum obstacle violation: `0.0`.
  - Maximum complementarity error: `1.6812843948429075e-09`.
  - Maximum monotonicity violation: `1.2279622587102262e-10`.
  - Maximum convexity violation: `0.006996739106769435`.
  - Delicate regimes: `32`.
  - Diagnostic failures: `32`.
- Decision: Stage 5 passes as an audited application risk-surface generation stage, but not as an unrestricted surrogate-label gate.
- Failure modes observed: high-volatility, long-maturity regimes trigger convexity warnings; no-dividend call controls correctly report no genuine boundary; strict-mask Gamma remains delicate.
- Next step: construct surrogate datasets only from diagnostic-passing regimes or refined reruns of flagged regimes.

## 2026-06-15

- Stage: Surrogate dataset construction.
- Commands:
  - `python -m unittest discover -s tests`
  - `python -m compileall -q src tests experiments`
  - `python experiments/05_surrogate_dataset.py --config configs/surrogate_dataset.yaml`
- Key metrics:
  - Clean regimes included: `223`.
  - Excluded regimes: `65`.
  - Dataset samples: `3264943`.
  - Feature dimension: `7`.
  - Label dimension: `5`.
  - Train/validation/test regimes: `156/33/34`.
  - Train/validation/test samples: `2283996/483153/497794`.
  - Strict Greek mask fraction: `0.9669724708823401`.
  - Exercise indicator fraction: `0.5410535497863209`.
- Decision: Stage 6 passes for beginning baseline surrogate experiments on clean, diagnostic-passing regimes.
- Failure modes observed: 65 regimes are excluded from clean labels, including no-dividend controls and Stage 5 diagnostic-warning regimes; Gamma remains mask-dependent.
- Next step: train baseline surrogate models and report failures honestly before introducing positive-premium architectures.
