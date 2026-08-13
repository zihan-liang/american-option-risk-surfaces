# M02 Obstacle-Aware PINN Closure

## Decision

M02 is closed as `insufficient_diagnostics`. It is retained as a negative
screening result and is not a candidate for confirmation or blind-holdout
evaluation. CN/PSOR remains the benchmark and label generator.

## Accepted Evidence

- Canonical CPU float64 screen using seeds `101`, `203`, `307`, `409`, and
  `503`.
- Four matched-initialization configurations: supervised, masked PDE,
  separate complementarity, and Fischer-Burmeister.
- Runtime `586.7359027499333` seconds, below the 3600-second screen budget.
- No blind-holdout access.
- Correct normalized log-space chain rule:
  `Delta = p_x / s` and `K*Gamma = (p_xx - p_x) / s^2`.
- Compact canonical CSV/JSON summaries under
  `results/independent_method_campaign/M02/`.

## Why The Method Did Not Advance

All 200 validation and legacy-test metric rows reported
`unavailable_no_threshold_crossing` for the boundary diagnostic. The positive
premium architecture ends in Softplus, so its premium is strictly positive;
the predeclared fixed-threshold proxy did not identify a crossing for any
method, including the supervised baseline. Because `boundary_rmse` is a
required M02 promotion diagnostic, every candidate remains
`insufficient_diagnostics`.

The remaining diagnostics do not provide an alternative promotion case. No
candidate passed all price, Greek, complementarity, dual-feasibility, residual,
and obstacle gates. The result therefore does not support a claim that an
obstacle-aware candidate improves the supervised baseline.

## Rejected Follow-Up

Commits `8d56a92` and `97bd3e8` are not integrated. Their contact-boundary
estimator was exploratory and did not pass manager acceptance:

- the first version produced false boundaries for smooth non-contact profiles;
- the revised BIC comparison did not count the searched boundary location as a
  fitted parameter; and
- pointwise canonical persistence was estimated at about 3,000,000 rows and
  461 MiB, which violates the compact-evidence policy.

No further M02 canonical rerun is authorized. A future boundary study belongs
to M04 or a separately approved high-dimensional experiment with an explicit
boundary or exercise-classification representation.

## Provenance Note

The canonical artifacts record code commit `d6c1ff0` and branch
`codex/m02-canonical-rerun`. The integration branch contains the same accepted
M02 implementation through cherry-picked commit `cb04253`; hashes identify the
original generated artifacts and must not be rewritten to the integration SHA.
