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
