# M22-M24 Primary-Source Audit Completion Report

## Status

`DONE_WITH_CONCERNS`

## Scope completed

- Branch: `research/m22-m24-primary-audit`
- Worktree: `/Users/liangzihan/Documents/SURF_0159/SURF/.worktrees/american-option-risk-surfaces/m22-m24-primary-audit`
- Research files only; no source code, configuration, experiment, test, or generated binary changes.
- English-first and Chinese-after reports completed for M22, M23, and M24.
- Primary-source audit ledger completed with direct supported-claim and status fields.

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
| M23 | 9 | Random/grid/Bayesian HPO, selection leakage, reusable holdout, agent benchmarking, reproducibility, trial systems, and current official tracking docs. |
| M24 | 10 | American-option VI/LCP, PSOR/LCP theory, continuum-convergence boundary, validated numerics, IEEE standards, shape properties, and floating-point analysis. |

## Key conclusions

### M22

Exact `2^7` evaluation can exactly reconstruct only the 128-value coalition game induced by a frozen model, point, baseline, masker, and output. It cannot establish a causal neural mechanism, global financial validity, or a baseline-independent continuous decomposition. In a put-only audit, `is_call` must behave as a dummy. Scale homogeneity is partly imposed by the normalized representation and cannot be claimed as learned without a dimensional pipeline test.

### M23

Fairness requires equal evaluator calls, fidelity, time limits, search space, seeds, and failure accounting. Treating the three eight-trial agents as replicates gives 24 calls per policy and 96 calls across agentic, random, grid, and Bayesian search. At a one-hour screening budget this leaves 37.5 seconds per call; the canonical CPU-float64 neural workload may therefore be resource-blocked. Holdout access must remain structurally unavailable until every policy winner is frozen, followed by one manager-controlled evaluation and no adaptation.

### M24

A float64 LCP residual check is a diagnostic, not a formal proof. A rigorous finite-dimensional certificate requires a named statement, independently reconstructed equations, outward-rounded enclosures or another validated-numerics argument, and a matrix condition before claiming uniqueness. Passing a discrete certificate does not prove the continuous obstacle problem. `indeterminate` interval results, especially near the free boundary, must remain distinct from pass and fail.

## Unresolved evidence gaps and concerns

1. No source directly validates the proposed M22 baseline panel or numerical stability thresholds for this repository; they are explicitly marked project inferences.
2. There is no primary evidence that Codex agents outperform random, grid, or Bayesian search on American-option surrogates. M23 is intentionally a falsification experiment.
3. The campaign brief does not resolve whether “equal budget” means 24 calls per baseline or 8 calls per baseline. The report recommends 24 per policy; the manager must freeze that interpretation before execution.
4. The one-hour M23 screen may be incompatible with 96 canonical neural trials. A timing calibration and either manager-approved extra budget, a frozen lower-cost benchmark, or `resource_blocked` status are required.
5. Availability and IEEE-1788 conformance of an interval backend on the Apple M5 environment have not been established. Ordinary `decimal` or high-precision arithmetic is not an acceptable substitute for outward-rounded interval proof.
6. The current CN central-difference matrix cannot be assumed to be an M-matrix for every regime. M24 uniqueness claims require per-regime matrix-condition evidence or must be omitted.
7. The corruption suite and acceptance thresholds are proposed engineering choices, not literature-derived universal standards; they must be committed before detector execution.

## Verification and commit

- `git diff --check`: passed with no output.
- `git diff --cached --check`: passed with no output.
- CSV parse/source counts: passed; 26 complete rows (`M22=7`, `M23=9`, `M24=10`).
- Commit: the scoped completion commit is identified by SHA in the manager notification.
