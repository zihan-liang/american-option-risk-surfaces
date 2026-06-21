# Verified Metric Ledger

This ledger records the paper-facing values checked against the existing
CSV/JSON evidence artifacts. Raw artifacts remain the numerical source of truth;
the rounded values below are for prose and tables.

| Evidence area | Verified paper-facing value | Source artifact |
|---|---:|---|
| European put smoke-grid maximum error | \(2.782\times10^{-4}\) | `results/experiment_01_solver_validation/validation_metrics.csv` |
| No-dividend American call maximum error | \(2.782\times10^{-4}\) | `results/experiment_01_solver_validation/validation_metrics.csv` |
| American put obstacle violation | \(0\) | `results/experiment_01_solver_validation/diagnostics_table.csv` |
| American put complementarity error | \(4.009\times10^{-12}\) | `results/experiment_01_solver_validation/diagnostics_table.csv` |
| Stage 5 application regimes | 288 total: 144 put, 108 dividend-call, 36 no-dividend controls | `results/experiment_04_application_risk_surfaces/application_summary.json` |
| Stage 5 diagnostic-failure regimes | 32, retained in the audit | `results/experiment_04_application_risk_surfaces/regime_flags.csv` |
| Stage 6 clean dataset | 223 regimes; 3,264,943 samples | `results/experiment_05_surrogate_dataset/dataset_manifest.json` |
| Stage 6 excluded regimes | 65 | `results/experiment_05_surrogate_dataset/excluded_regimes.csv` |
| Stage 6 regime split | 156 train, 33 validation, 34 test | `results/experiment_05_surrogate_dataset/split_table.csv` |
| Positive-premium MLP validation RMSE | 0.0200 | `results/experiment_07_positive_premium_tuning/positive_premium_metrics.csv` |
| Positive-premium MLP test RMSE | 0.0218 | `results/experiment_07_positive_premium_tuning/positive_premium_metrics.csv` |
| Positive-premium obstacle violation | 0 on evaluated samples | `results/experiment_07_positive_premium_tuning/positive_premium_violation_table.csv` |
| Pooled boundary MLP test RMSE | 0.3513 | `results/experiment_09_boundary_head/boundary_summary.json` |
| Pooled boundary MLP test MAE | 0.2087 | `results/experiment_09_boundary_head/boundary_summary.json` |
| Price-model autograd Delta test RMSE | 0.0977 | `results/experiment_11_delta_validation/delta_validation_summary.json` |
| Bounded Delta MLP test RMSE | 0.0332 | `results/experiment_12_supervised_delta_head/delta_head_summary.json` |
| Bounded Delta MLP test MAE | 0.0140 | `results/experiment_12_supervised_delta_head/delta_head_summary.json` |
| Bounded Delta maximum bound violation | 0 on evaluated samples | `results/experiment_12_supervised_delta_head/delta_head_violation_table.csv` |
| Integrated fresh PSOR audit | 6 deterministic held-out regimes | `results/experiment_13_integrated_application_workflow/integrated_summary.json` |
| Integrated mean price RMSE | 0.0092 | `results/experiment_13_integrated_application_workflow/integrated_summary.json` |
| Integrated mean boundary MAE | 0.1539 | `results/experiment_13_integrated_application_workflow/integrated_summary.json` |
| Integrated mean strict-mask Delta RMSE | 0.0300 | `results/experiment_13_integrated_application_workflow/integrated_summary.json` |
| Integrated mean speedup | 87.5 | `results/experiment_13_integrated_application_workflow/integrated_summary.json` |
| Integrated obstacle and Delta-bound violations | 0 on the six audited regimes | `results/experiment_13_integrated_application_workflow/integrated_summary.json` |

## Interpretation Rules

- The grid-240 surface is an internal engineering reference, not an analytic or
  universal publication-grade benchmark.
- The positive-premium construction enforces only the payoff obstacle. It does
  not establish boundary, monotonicity, convexity, Delta, or Gamma accuracy.
- Boundary performance is the weakest neural component and remains a scoped
  diagnostic result.
- The bounded supervised Delta head is evaluated under the strict Greek mask.
- The integrated workflow supports scenario screening with PSOR audit only.

No irreconcilable headline-value conflict was found during this audit. The
Stage 6 exclusion count is consistent with overlapping categories: 33
control-only regimes, 29 diagnostic/delicate regimes, and 3 no-dividend controls
that also carry diagnostic flags, for 65 unique exclusions.
