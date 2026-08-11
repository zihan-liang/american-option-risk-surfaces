# M22-M24 Primary-Source Audit Completion Report

## Status

`DONE_WITH_CONCERNS`

## Scope completed

- Branch: `research/m22-m24-primary-audit`
- Worktree: `/Users/liangzihan/Documents/SURF_0159/SURF/.worktrees/american-option-risk-surfaces/m22-m24-primary-audit`
- Research files only; no source code, configuration, experiment, test, or generated binary changes.
- English-first and Chinese-after reports completed for M22, M23, and M24.
- Primary-source audit ledger completed with direct supported-claim and status fields.
- Manager review round 1 incorporated: M23 now follows the fixed 24/24/24 three-policy matrix, and M22-S3 uses the official NeurIPS proceedings page as its primary URL.

## Deliverables

- `docs/research/m22_mechanistic_interaction_audit.md`
- `docs/research/m23_agentic_hyperparameter_search.md`
- `docs/research/m24_formal_verification_certificates.md`
- `docs/research/m22_m24_primary_source_audit.csv`
- `.superpowers/m22-m24-research-report.md`

## Source counts

| Method | Audited primary/authoritative sources | Notes |
|---|---:|---|
| M22 | 7 | Set-function/Mobius interaction sources, explanation sanity checks, and option-property sources. |
| M23 | 9 | Random/grid campaign baselines, Bayesian literature context only, selection leakage, reusable holdout, agent benchmarking, reproducibility, trial systems, and current official tracking docs. |
| M24 | 10 | American-option VI/LCP, PSOR/LCP theory, continuum-convergence boundary, validated numerics, IEEE standards, shape properties, and floating-point analysis. |

## Key conclusions

### M22

Exact `2^7` evaluation can exactly reconstruct only the 128-value coalition game induced by a frozen model, point, baseline, masker, and output. It cannot establish a causal neural mechanism, global financial validity, or a baseline-independent continuous decomposition. In a put-only audit, `is_call` must behave as a dummy. Scale homogeneity is partly imposed by the normalized representation and cannot be claimed as learned without a dimensional pipeline test.

### M23

The approved method matrix fixes three policies: three blind agents with eight trials each, giving 24 agentic evaluations; 24 random-search evaluations; and 24 fixed-grid evaluations. The total is 72 calls, so the one-hour screening budget allows an average of 50 seconds per call and 1,200 seconds per policy. Bayesian optimization is literature context or an optional separately budgeted future diagnostic only; it receives no campaign or holdout slot. Holdout access remains structurally unavailable until the three winners are frozen, followed by one manager-controlled evaluation each and no adaptation.

### M24

A float64 LCP residual check is a diagnostic, not a formal proof. A rigorous finite-dimensional certificate requires a named statement, independently reconstructed equations, outward-rounded enclosures or another validated-numerics argument, and a matrix condition before claiming uniqueness. Passing a discrete certificate does not prove the continuous obstacle problem. `indeterminate` interval results, especially near the free boundary, must remain distinct from pass and fail.

## Unresolved evidence gaps and concerns

1. No source directly validates the proposed M22 baseline panel or numerical stability thresholds for this repository; they are explicitly marked project inferences.
2. There is no primary evidence that Codex agents outperform random or fixed-grid search on American-option surrogates. M23 is intentionally a falsification experiment.
3. The one-hour M23 screen may still be incompatible with 72 canonical neural trials at 50 seconds each. A timing calibration and either manager-approved extra budget, a frozen lower-cost benchmark, or `resource_blocked` status are required.
4. Availability and IEEE-1788 conformance of an interval backend on the Apple M5 environment have not been established. Ordinary `decimal` or high-precision arithmetic is not an acceptable substitute for outward-rounded interval proof.
5. The current CN central-difference matrix cannot be assumed to be an M-matrix for every regime. M24 uniqueness claims require per-regime matrix-condition evidence or must be omitted.
6. The corruption suite and acceptance thresholds are proposed engineering choices, not literature-derived universal standards; they must be committed before detector execution.

## Verification and commit

- `git diff --check`: passed with no output.
- `git diff --cached --check`: passed with no output.
- CSV parse/source counts: passed; 26 complete rows (`M22=7`, `M23=9`, `M24=10`).
- Commit: the scoped completion commit is identified by SHA in the manager notification.
