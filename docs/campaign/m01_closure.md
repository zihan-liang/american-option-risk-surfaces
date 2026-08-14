# M01 Classical Solver Closure

## Decision

M01 is accepted as a scoped `candidate_supported` result for
`cn_policy_iteration`. CN/PSOR remains the audited benchmark and label
generator. The result supports a faster discrete LCP solve on the tested local
CPU workloads; it does not establish a universal fastest solver or a
continuous-PDE theorem.

## Accepted Evidence

- Canonical screen on nine predeclared American-put regimes.
- Clean committed provenance:
  `campaign/independent-methods` at
  `df21b8a54763666a00c0fd05d4d602d3bab1bd30`.
- Five timing warm-ups and 30 repetitions for every measured workload.
- Total screen time `230.39810937503353` seconds, below the 3600-second budget.
- No blind-holdout access.
- All artifact hashes match.
- European analytic controls, applicable no-dividend American-call controls,
  applicable Brennan--Schwartz checks, the independent integral-boundary
  checks, obstacle checks, and hard LCP convergence gates pass.

## Method Decisions

| Configuration | Decision | Interpretation |
|---|---|---|
| `cn_psor` | `diagnostic_only` | Audited benchmark. |
| `rannacher_cn_psor` | `diagnostic_only` | Accuracy diagnostics pass; the primary timing inference is not supported for this different time discretization. |
| `dirk_psor` | `diagnostic_only` | Accuracy diagnostics pass; the selected SDIRK2 discretization is slower here and the primary timing inference is not supported. |
| `cn_policy_iteration` | `candidate_supported` | Passes no-regression and hard LCP gates with a significant practical timing gain. |
| `cn_penalty_newton` | `failed_metric_gate` | Fast, but its natural residual violates the shared no-regression requirement. |

For `cn_policy_iteration`, the median solve time is
`0.007312756937204136` seconds versus `0.14061326379629058` seconds for
CN/PSOR, a `94.79938325889493%` relative improvement. The paired timing
difference interval is `[-0.24570344548022452, -0.06008740873738943]`
seconds and the Holm-adjusted p-value is `0.030396960303969604`.

The selected candidate's mean price, Delta, and Gamma RMSE values are
`9.034395106995704e-05`, `0.001044099209432042`, and
`0.08697552561365657`. Its maximum natural residual is
`6.06898341976902e-11`, below the shared `1e-10` tolerance.

The penalty/Newton candidate reaches similar price error and timing but has a
maximum natural residual of `5.3324630917685035e-08`; it therefore remains a
failed candidate rather than being promoted on speed alone.

## Non-Finite Boundary Values

The only `NaN` values occur in boundary RMSE/MAE columns for the predeclared
negative-rate regime `put_T1p00_sig0p10_rm0p012_qm0p016`. That regime has no
finite exercise boundary under the applicable extraction convention. The
independent integral-boundary and related theorem controls record it explicitly
as `not_applicable_negative_rate`; these values are not silently treated as
passes or included in applicable boundary averages.

## Reproducibility

Compact canonical evidence is stored under
`results/independent_method_campaign/M01/screen/`. The result was regenerated
after the final solver and gate code was committed because the historical M01
artifacts incorrectly recorded parent commit `bc2b4a0` while using uncommitted
changes. Only the clean-provenance rerun is accepted.
