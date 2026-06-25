# Startup Readiness Checklist

Use this checklist at the beginning of a session. It is deliberately separate from `AGENTS.md` so the root instruction file stays short and acts as a router.

## Required first pass

- [ ] Confirm the repository root with `pwd`.
- [ ] Confirm branch/worktree state with `git status --short --branch`.
- [ ] Read `AGENTS.md`.
- [ ] Read `feature_list.json`, `progress.md`, and `session-handoff.md`.
- [ ] Run `bash scripts/init_harness.sh`.
- [ ] Pick one active feature from `feature_list.json`.
- [ ] If starting a new feature, use `scripts/update_feature_status.py` instead of editing the status by hand.

## Research-artifact readiness

Before running full report/manuscript tests:

- [ ] Run `bash scripts/verify_report_artifacts.sh`.
- [ ] If report artifacts are missing, run `bash scripts/bootstrap_artifacts.sh`.
- [ ] Re-run `bash scripts/verify_report_artifacts.sh`.

## Evidence readiness

Before claiming research evidence:

- [ ] Run the method-specific test/experiment command.
- [ ] Validate comparison CSV schema with `scripts/check_result_schema.py`.
- [ ] Check claim language against `docs/harness/claim-safety-rules.md`.
- [ ] Record result paths and risks in `session-handoff.md`.

## Exit readiness

Before handing off:

- [ ] Update `feature_list.json`.
- [ ] Update `progress.md`.
- [ ] Update `session-handoff.md`.
- [ ] Run `bash scripts/check_clean_session.sh`.
