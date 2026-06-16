# Claim Audit

## Allowed Claims

- Validated CN/PSOR benchmark generates American put and dividend-paying American call surfaces.
- Positive continuation-premium surrogate can provide obstacle-consistent price/premium estimates within tested regimes.
- Direct boundary heads provide scoped diagnostic boundary estimates.
- Bounded supervised Delta heads provide Delta diagnostics.

## Rejected Claims

- One neural network solves price, boundary, and Greeks.
- The price surrogate is automatically a Greek engine.
- The surrogate replaces PSOR completely.
- Failed regimes can be hidden.

## Evidence Table

| Claim | Evidence Script | Status | Notes |
| --- | --- | --- | --- |
| Solver validation | `01_solver_validation.py` | Passed for next-stage benchmark use | European put and no-dividend American call errors decrease under grid refinement; obstacle/complementarity diagnostics pass. Not publication-grade final accuracy yet. |
| Boundary and Greek diagnostics | `02_boundary_greeks_diagnostics.py` | Passed for classical diagnostics | Boundaries, Delta, Gamma, named masks, and diagnostics are generated for base put and dividend-paying call. Gamma must be discussed with kink/boundary masks. |
| Grid/domain/runtime stability | `03_convergence_domain_runtime.py` | Passed for engineering-scale classical experiments | Price, boundary, strict-mask Delta/Gamma, runtime, PSOR iterations, and shape diagnostics are generated for both option cases. Grid-240 is an internal reference, not publication-grade final evidence. |
| Application risk surfaces | `04_application_risk_surfaces.py` | Passed with restrictions | 288 regimes solved and reported. 32 high-stress regimes are flagged by diagnostics and must not be hidden or used as clean labels without refinement. |
| Surrogate dataset construction | `05_surrogate_dataset.py` | Passed for clean-label dataset construction | Clean dataset includes 223 diagnostic-passing put/dividend-call regimes. 65 control or flagged regimes are excluded and audited. No model-performance claim yet. |
| Baseline direct-price surrogates | `06_surrogate_baselines.py` | Passed as violation audit only | Plain MLP has best validation RMSE but still violates the obstacle and shape diagnostics. Penalty baselines are not hard constraints and do not justify consistency claims. |
| Literature review integration | `docs/literature_review.md` | Passed as claim-support evidence | Literature supports the classical benchmark, diagnostics, cautious Greek reporting, and modular constraint-aware ML framing. Positive premium support is indirect and must be tested. |
| Price/premium surrogate | `07_positive_premium_tuning.py` | Passed with restrictions | Positive-premium models have zero obstacle violation. Linear sanity baseline has lower aggregate validation RMSE; neural MLP is retained as selected softplus artifact and has better boundary-band RMSE. |
| Boundary diagnostics | `09_boundary_head.py` | Pending run | Scope to tested regimes |
| Delta diagnostics | `12_supervised_delta_head.py` | Pending run | Separate bounded head only |

## Solver Validation Audit - 2026-06-13

Allowed wording:

- The CN/PSOR implementation has passed a solver-validation audit sufficient to proceed to classical risk-surface experiments.
- The finite-difference operator and Crank--Nicolson matrix signs were tested directly.
- PSOR projection enforces the obstacle in the tested American put and dividend-paying American call cases.
- The closed-form European put and no-dividend American call errors decrease as the grid is refined from 80 to 240.

Restricted wording:

- Do not claim publication-grade solver accuracy yet.
- Do not claim Rannacher smoothing is active; `rannacher_steps` is currently a configuration placeholder.
- Do not use neural-surrogate results to support solver reliability.

## Boundary and Greek Diagnostics - 2026-06-15

Allowed wording:

- The validated CN/PSOR solver now produces boundary curves, Delta surfaces, Gamma surfaces, and named Greek masks for the base American put and dividend-paying American call.
- Gamma concentration is reported explicitly near the payoff kink and free boundary instead of being hidden.
- Strict-mask Greek summaries are suitable for later risk-surface diagnostics and surrogate-label preparation.

Restricted wording:

- Do not claim the Greeks are smoothed or publication-grade near maturity.
- Do not claim Gamma is stable at the payoff kink or free boundary.
- Do not use this classical diagnostic stage as evidence for neural surrogate Greek accuracy.

## Grid/Domain/Runtime Stability - 2026-06-15

Allowed wording:

- The CN/PSOR benchmark has passed an engineering-scale grid/domain/runtime audit for the base American put and dividend-paying American call.
- Price and strict-mask Delta errors decrease clearly under grid refinement against the grid-240 internal reference.
- Boundary errors decrease with refinement but remain limited by the uniform grid spacing.
- Gamma is reported with strict masks and remains the most delicate convergence quantity.
- Runtime and PSOR iteration counts are attached to the convergence evidence.

Restricted wording:

- Do not call the grid-240 surface an analytic or publication-grade reference.
- Do not claim unmasked Gamma convergence.
- Do not claim domain sensitivity is resolved for all future regimes; this audit covers the tested base put and dividend-call settings.
- Do not use this stage as neural surrogate evidence.

## Application Risk Surfaces - 2026-06-15

Allowed wording:

- The CN/PSOR solver generated audited American put, dividend-paying American call, and no-dividend call control risk-surface metrics across 288 regimes.
- No-dividend call controls are separated and report no genuine early-exercise boundary.
- Dividend-paying call regimes produce boundary curves in the tested sweep.
- The application sweep retains 32 diagnostic-failure regimes in the reports and flags them explicitly.

Restricted wording:

- Do not claim all regimes are clean surrogate-label candidates.
- Do not hide the convexity-warning regimes.
- Do not use no-dividend call controls as dividend-call exercise evidence.
- Do not make any neural-surrogate claim from this classical application stage.

## Surrogate Dataset Construction - 2026-06-15

Allowed wording:

- A clean supervised dataset has been built from diagnostic-passing CN/PSOR regimes.
- The dataset uses regime-level train/validation/test splits to avoid pointwise leakage.
- Stage 5 control and diagnostic-failure regimes are excluded from clean labels and listed in `excluded_regimes.csv`.
- Delta and Gamma labels are supplied with strict Greek masks and delicate-region masks.

Restricted wording:

- Do not claim any neural surrogate accuracy from this dataset stage.
- Do not claim the dataset covers the excluded high-stress regimes.
- Do not train or evaluate on no-dividend call controls as dividend-call labels.
- Do not report Gamma performance later without strict-mask qualification.

## Baseline Surrogate Models - 2026-06-16

Allowed wording:

- Direct-price baseline models were trained and audited on the clean Stage 6 regime-level splits.
- The plain direct-price MLP achieved the lowest validation RMSE among tested baselines, with validation RMSE about `0.0281` and test RMSE about `0.0282`.
- All direct-price baselines still showed obstacle and shape-diagnostic violations.
- Obstacle-penalty baselines are useful negative evidence but are not hard obstacle constraints.

Restricted wording:

- Do not claim the baseline MLP is obstacle-consistent.
- Do not claim the obstacle penalty guarantees `V >= Phi`.
- Do not claim any baseline estimates free boundaries or Greeks.
- Do not use this stage as evidence for the later positive-premium architecture before that architecture is trained and audited.

## Literature Review Integration - 2026-06-16

Allowed wording:

- The project is grounded in classical VI/LCP/free-boundary American option literature.
- CN/PSOR is used as a classical, auditable benchmark and label generator.
- Obstacle, complementarity, shape, grid, domain, runtime, boundary, and Greek diagnostics are supported by the numerical and scientific-computing literature.
- The positive continuation-premium surrogate is a literature-inspired hard obstacle parameterization that still requires empirical validation.
- Boundary and Delta heads are modular diagnostic components, not automatic byproducts of a price surrogate.

Restricted wording:

- Do not claim the literature establishes the proposed positive-premium neural architecture as the standard solution.
- Do not cite the raw Deep Research file as a final source.
- Do not leave Deep Research internal citation markers in paper-facing text.
- Do not use literature support as a substitute for Stage 8 surrogate evidence.

## Positive Premium Price Surrogate - 2026-06-16

Allowed wording:

- Positive continuation-premium models enforce `price >= payoff` by construction on evaluated nodes.
- The selected neural MLP is an obstacle-consistent price/premium surrogate within the clean Stage 6 regime envelope.
- Boundary-distance diagnostics are reported as price-error profiles only.
- The linear positive-premium sanity baseline has lower aggregate validation RMSE in this run and must be reported.

Restricted wording:

- Do not claim the neural MLP dominates all positive-premium baselines.
- Do not claim this price surrogate is a Greek engine.
- Do not infer early-exercise boundaries from the price surrogate in this stage.
- Do not claim this model replaces PSOR or validates excluded regimes.

## Direct Boundary Diagnostic Head - 2026-06-16

Allowed wording:

- Direct boundary heads provide scoped supervised diagnostic estimates of $B/K$ within the clean Stage 6 regime envelope.
- Boundary labels are taken from validated CN/PSOR dataset artifacts, and finite-label coverage is counted explicitly.
- Put and dividend-call boundary errors are reported separately.
- The pooled MLP improves over the pooled linear sanity baseline in this run.

Restricted wording:

- Do not claim the boundary head solves the free-boundary problem.
- Do not claim the Stage 8 price surrogate implies or recovers the boundary.
- Do not fabricate no-boundary or control cases as positive boundary labels.
- Do not claim the call-only MLP is a repaired universal call-boundary solution; its test RMSE is weaker than the call-only linear baseline in this run.
- Do not use this stage as Delta or Gamma evidence.
