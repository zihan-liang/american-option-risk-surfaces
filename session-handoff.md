# Session Handoff

## Current Objective

- Goal: execute the approved independent 24-method experiment campaign and produce a bilingual comparison report.
- Current status: Wave 0 campaign foundation complete; M01 is accepted with
  `cn_policy_iteration` as a scoped candidate, M02 is closed as
  `insufficient_diagnostics`, and M03 is closed as `failed_metric_gate`.
  M04 is next; HD0/HD1 remain deferred.
- Branch: `campaign/independent-methods`.
- Worktree: `/Users/liangzihan/Documents/SURF_0159/SURF/.worktrees/american-option-risk-surfaces/campaign-independent-methods`.

## Completed This Session

- [x] Add root harness artifacts.
- [x] Add `docs/harness/` policies.
- [x] Add harness verification scripts.
- [x] Add tests and verification evidence.
- [x] Add feature-list schema validation.
- [x] Add report-artifact verification and bootstrap scripts.
- [x] Add clean-state check, evaluator rubric, quality document, decision log, and harness scoring.
- [x] Add `scripts/update_feature_status.py`.
- [x] Add feature lifecycle documentation and tests.
- [x] Push the four pending visualization commits on `stage-13-future-extensions`.
- [x] Create the campaign integration branch and external worktree.
- [x] Cherry-pick reviewed harness commits `a343c78`, `6c6d4c3`, and `ec3e694`.
- [x] Create the pre-development blind-holdout SHA-256 commitment.
- [x] Bootstrap ignored report assets and establish a passing inherited baseline.
- [x] Implement and review the campaign registry, output contracts, statistics, leakage controls, and CLI.
- [x] Complete and primary-source audit the missing M22--M24 research reports.
- [ ] Implement and screen M01-M24 independently.
- [ ] Freeze configurations and run the blind final evaluation once.
- [ ] Produce and inspect the bilingual final report.

## Verification Evidence

| Check | Command | Result | Notes |
|---|---|---|---|
| Harness contract | `$PY -m pytest tests/test_harness_contract.py tests/test_harness_v11_contract.py -q` | passed | 11 tests passed. |
| Feature lifecycle | `$PY -m pytest tests/test_feature_lifecycle_tools.py -q` | passed | 4 tests passed. |
| Full tests | `$PY -m pytest tests -q` | passed | Required ignored canonical figures were regenerated locally first. |
| Harness startup | `bash scripts/init_harness.sh` | passed | Compileall plus harness verification. |
| Harness scripts | `bash scripts/verify_harness.sh` | passed | Harness contract and v1.1 contract passed. |
| Round-2 evidence | `bash scripts/verify_round2_evidence.sh` | passed | Uses sibling Round-2 worktrees; confirmed no candidates/no combo. |
| Report artifacts | `bash scripts/verify_report_artifacts.sh` | passed | Passed after `bash scripts/bootstrap_artifacts.sh`. |
| Release gate | `$PY experiments/16_release_verification.py` | passed | Reported `release_gate: pass`. |
| Campaign inherited baseline | `MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/codex_mplconfig PYTHONPATH=src:experiments $PY -m pytest tests -q` | passed | 150 tests passed after ignored artifact bootstrap. |
| Reviewed campaign foundation | `MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/codex_mplconfig PYTHONPATH=src:experiments $PY -m pytest tests -q` | passed | 209 tests passed after registry, statistics, holdout, and M22--M24 audit integration. |
| Foundation compilation | `$PY -m compileall -q src tests experiments` | passed | No compilation errors. |
| Foundation diff hygiene | `git diff --check` | passed | No whitespace errors. |
| M03 focused integration | `$PY -m pytest -p no:cacheprovider tests/test_m03_algorithm_unrolling.py tests/test_m03_unrolled.py tests/test_m03_campaign.py tests/test_m01_classical_campaign_outputs.py tests/test_m02_obstacle_pinn_outputs.py tests/test_campaign_outputs.py` | passed | 42 tests passed. |
| M03 canonical screen | `$PY experiments/26_independent_method_campaign.py --method M03 --phase screen --config configs/m03_algorithm_unrolling.yaml --output-root results/independent_method_campaign` | completed | `failed_metric_gate`; 11.85% timing improvement but frozen diagnostic gates failed; blind holdout untouched. |

## Files Changed

- Root harness files and `docs/harness/` policies.
- Harness scripts under `scripts/`.
- Harness contract tests under `tests/`.
- `feature_list.json` schema version 2.
- `scripts/update_feature_status.py`.
- `docs/harness/feature-lifecycle.md`.
- Campaign plan, public holdout commitment, and campaign lifecycle state.

## Decisions Made

- Keep `AGENTS.md` as a router; do not turn it into one giant instruction file.
- Adapt course patterns to this research repo rather than copying the course project verbatim.
- Use the lifecycle helper for routine feature status changes instead of manual edits.
- Do not touch or clean the dirty main checkout.
- Run independent methods only; combined methods are future work.
- Require manager artifact inspection and refinement before every cherry-pick.
- Keep the holdout seed secret until all method configurations are frozen.

## Blockers / Risks

- Round-2 evidence paths are absolute and point to external worktrees.
- Future lesson content may revise file names or lifecycle conventions.
- Full legacy report-link tests require ignored/generated figure artifacts in a fresh worktree; this session regenerated them locally but did not commit ignored binaries.
- Artifact bootstrap may rewrite tracked result summaries; this branch keeps the commit scoped to harness files.
- Headless plotting must use `MPLBACKEND=Agg`; the macOS interactive backend aborts inside the Codex test environment.
- Legacy dry-run paths emit non-finite arithmetic warnings, so campaign schemas and gates must reject non-finite metrics.
- M03's relative diagnostic gates are conservative near the numerical floor.
  They were not weakened after observing the result; M03 remains a negative
  result despite its timing signal.
- M23's approved 72-call comparison implies a 50-second average ceiling and may correctly end as `resource_blocked` after calibration.

## Next Session Startup

1. Read `AGENTS.md`.
2. Read `feature_list.json`, `progress.md`, and this handoff.
3. Run `bash scripts/init_harness.sh`.
4. Run `bash scripts/verify_report_artifacts.sh` before full report/manuscript tests.
5. Read `docs/superpowers/plans/2026-08-11-independent-all-method-campaign.md` and resume from the campaign ledger.

## Recommended Next Step

Do not rerun M01--M03. Start with a focused review of M04 free-boundary
representation and run its canonical screen only if its boundary definition,
sub-grid extraction, split isolation, and comparison gates pass. Keep HD0/HD1
deferred and outside the campaign scoreboard.
