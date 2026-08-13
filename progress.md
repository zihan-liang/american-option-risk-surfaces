# Session Progress

This is the short current-session state file for agent continuity. The long research history remains in `docs/progress_log.md`.

## Current State

- Last updated: 2026-08-11
- Active feature: `independent-all-method-campaign`
- Branch: `campaign/independent-methods`
- Worktree: `/Users/liangzihan/Documents/SURF_0159/SURF/.worktrees/american-option-risk-surfaces/campaign-independent-methods`

## What Is Done

- Round-2 method branches exist in external worktrees.
- Round-2 aggregate evidence found zero candidate methods and no best-combo justification.
- The initial v1 harness has been implemented on `exp/initial-harness-v1`.
- The v1.1 operational harness has been implemented on `exp/harness-v1.1-operational`.
- The v1.2 feature lifecycle helper has been implemented on `exp/harness-v1.2-feature-lifecycle`.
- The reviewed v1, v1.1, and v1.2 harness commits have been imported into the campaign branch.
- The current `stage-13-future-extensions` branch was pushed before campaign setup.
- The blind-holdout SHA-256 commitment was created before method development; the seed is stored outside Git.
- The inherited suite passes after ignored report assets are bootstrapped with `MPLBACKEND=Agg`.
- The campaign registry contains exactly M01--M24 with category, budget, gate, provenance, and multi-record output contracts.
- Scenario-level statistics, paired bootstrap intervals and p-values, Holm correction, promotion rules, and deterministic timing helpers are implemented and tested.
- Development splits cannot access the blind holdout; final generation requires a verified committed seed after configurations are frozen.
- The missing M22--M24 research reports have been primary-source audited, including the corrected 24/24/24 M23 comparison.

## In Progress

- M01 classical oracle corrections after independent numerical review.
- M02 obstacle-aware PINN is closed as `insufficient_diagnostics`; the
  canonical five-seed run completed within budget, but required boundary RMSE
  was unavailable for every method and no candidate passed the remaining gates.

## Next Step

- Correct and re-review the M01 numerical issues before accepting oracle evidence.
- Preserve M02 as a negative result and do not rerun it. Prepare the separate
  HD0 Windows RTX 4060 precision/performance calibration before HD1.

## Blockers / Risks

- The campaign is large and compute-heavy; heavy screening runs must remain serialized.
- Some Round-2 evidence lives in sibling external worktrees, not in this harness branch.
- Fresh worktrees may need ignored figure artifacts regenerated before the legacy full report-link tests pass.
- `scripts/bootstrap_artifacts.sh` can rewrite tracked result summaries through path/timestamp/line-ending churn; keep commits scoped.
- The inherited suite emits numerical overflow warnings in legacy dry-run/extreme-regime paths even when tests pass; campaign diagnostics must reject non-finite metrics explicitly.
- A first baseline test invocation crashed in the macOS interactive Matplotlib backend; all headless commands must set `MPLBACKEND=Agg` and a writable `MPLCONFIGDIR`.
- The first M01 review blocked integration because of negative-rate boundary handling, an incomplete SDIRK2 obstacle reaction, an invalid Brennan--Schwartz control, and inconsistent LCP convergence semantics.
- M23 may be resource-blocked if its calibrated trial time exceeds the predeclared 50-second average ceiling.

## Decisions Made

- Use `AGENTS.md` as the root instruction file.
- Keep `progress.md` short; preserve `docs/progress_log.md` as the full research chronology.
- Keep the v1 harness small and revise it lesson-by-lesson as the user provides more course content.
- Keep `AGENTS.md` as a router; detailed harness rules live under `docs/harness/`.
- Treat `feature_list.json` as the system of record and keep at most one active feature.
- Use `scripts/update_feature_status.py` for routine status transitions.
- Run all 24 methods independently; do not test combinations in this campaign.
- Treat the existing test split as `legacy_test`; use the committed blind holdout only once after freezing configurations.

## Evidence of Completion

- `$PY -m compileall -q src tests experiments` passed.
- `$PY -m pytest tests -q` passed after regenerating ignored canonical figure artifacts required by legacy report-link tests.
- `bash scripts/init_harness.sh` passed.
- `bash scripts/verify_harness.sh` passed.
- `bash scripts/verify_round2_evidence.sh` passed and confirmed the current zero-candidate Round-2 decision.
- `bash scripts/verify_report_artifacts.sh` passed after `bash scripts/bootstrap_artifacts.sh`.
- `$PY experiments/16_release_verification.py` passed with `release_gate: pass`.
- `$PY -m pytest tests/test_feature_lifecycle_tools.py -q` passed.
- `MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/codex_mplconfig $PY -m pytest tests -q` passed on the campaign worktree after artifact bootstrap (150 tests).
- `MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/codex_mplconfig PYTHONPATH=src:experiments $PY -m pytest tests -q` passed after the reviewed foundation and M22--M24 audit (209 tests).
- `bash scripts/verify_harness.sh` passed after the reviewed foundation integration.
- `$PY -m compileall -q src tests experiments` and `git diff --check` passed.
