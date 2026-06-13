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
| Solver validation | `01_solver_validation.py` | Pending run | Requires analytic and theorem checks |
| Price/premium surrogate | `07_positive_premium_tuning.py` | Pending run | Requires zero obstacle violation |
| Boundary diagnostics | `09_boundary_head.py` | Pending run | Scope to tested regimes |
| Delta diagnostics | `12_supervised_delta_head.py` | Pending run | Separate bounded head only |
