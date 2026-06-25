# Clean-State Checklist

Run this checklist at the end of a session and before handing work to another agent.

## Branch and worktree

- [ ] Work happened in the assigned external worktree.
- [ ] `git status --short --branch` has no unexpected files.
- [ ] Any remaining dirty files are intentionally documented in `session-handoff.md`.
- [ ] The main checkout was not cleaned or modified.

## Harness state

- [ ] `feature_list.json` validates with `scripts/validate_feature_list.py`.
- [ ] At most one feature is `in-progress`.
- [ ] Completed features include verification evidence.
- [ ] `progress.md` records the current state.
- [ ] `session-handoff.md` records branch, commit, verification, risks, and next step.

## Verification

- [ ] Focused tests for the changed area passed.
- [ ] `bash scripts/verify_harness.sh` passed for harness changes.
- [ ] `bash scripts/check_clean_session.sh` passed or an intentional exception is recorded.

## Artifacts

- [ ] Generated figures/models/data were not accidentally staged.
- [ ] If full report tests were run, ignored figure artifacts were verified or regenerated.
- [ ] Result-table schema stayed compatible with `docs/harness/result-schema-policy.md`.
