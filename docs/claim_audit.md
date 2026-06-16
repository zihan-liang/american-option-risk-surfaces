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
| Price/premium surrogate | `07_positive_premium_tuning.py` | Pending run | Requires zero obstacle violation |
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
