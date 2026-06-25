# AGENTS.md — American Option Risk Surfaces Harness

This repository studies free-boundary-aware American option risk surfaces. The harness exists to make agent sessions reliable: clear startup, persistent state, isolated scope, runnable verification, and safe handoff.

## Startup Workflow

Before writing code:

1. Confirm the working directory with `pwd`.
2. Read this file completely.
3. Read `feature_list.json`, `progress.md`, and `session-handoff.md`.
4. Review the relevant project docs:
   - `README.md`
   - `docs/reproducibility_checklist.md`
   - `docs/claim_audit.md`
   - `docs/release_handoff.md`
   - `docs/harness/verification-matrix.md`
   - `docs/harness/startup-readiness-checklist.md`
5. Run `bash scripts/init_harness.sh` from the repository root.
6. If you need full report/manuscript verification, run `bash scripts/verify_report_artifacts.sh`; if it reports missing ignored figure artifacts, run `bash scripts/bootstrap_artifacts.sh`.
7. Review recent commits with `git log --oneline -5`.

If baseline verification fails, diagnose that first before adding new scope.

## Working Rules

- Use an external worktree for implementation work. Do not work directly in the dirty main checkout.
- Work on one active feature or experiment at a time unless `docs/harness/subagent-protocol.md` defines separate branch ownership.
- Preserve user changes. Do not clean, reset, or overwrite unrelated dirty/untracked files.
- Keep generated `.npz`, `.pt`, large figures, and transient logs out of Git unless the feature explicitly curates them as evidence.
- Add tests before behavior-changing code when practical.
- Record evidence in `feature_list.json`, `progress.md`, or `session-handoff.md` before ending a session.
- Treat `feature_list.json` as the system of record for active work. Keep at most one feature at `in-progress` unless `docs/harness/subagent-protocol.md` assigns explicit branch ownership.
- Use `scripts/update_feature_status.py` for status transitions when practical; see `docs/harness/feature-lifecycle.md`.
- Run `bash scripts/check_clean_session.sh` before handoff; use `--allow-dirty` only while intentionally working in an uncommitted branch.

## Claim Boundaries

- CN/PSOR is the audited benchmark and label generator within tested regimes.
- Surrogate models provide scoped diagnostic or scenario-screening evidence.
- Boundary heads are supervised diagnostics, not exact free-boundary solvers.
- The bounded Delta head is a separate strict-mask diagnostic head.
- The integrated workflow supports scenario screening with PSOR audit.
- Do not claim that a surrogate model replaces PSOR.
- Do not claim one network solves price, boundary, and Greeks.
- Gamma surrogate work is out of scope unless the user explicitly authorizes it.

## Verification Commands

Use the shared pilot venv when available:

```bash
PY=/Users/liangzihan/Documents/SURF_0159/SURF/.worktrees/american-option-risk-surfaces/.venv-pilot/bin/python
export PYTHONPATH=src:experiments
export PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/american_option_pycache"
```

Core checks:

```bash
$PY -m compileall -q src tests experiments
$PY -m pytest tests -q
bash scripts/verify_harness.sh
bash scripts/verify_round2_evidence.sh
bash scripts/verify_report_artifacts.sh
bash scripts/check_clean_session.sh
$PY experiments/16_release_verification.py
```

Use narrower checks from `docs/harness/verification-matrix.md` for focused edits.

## Definition of Done

A feature is done only when all of these are true:

- Target behavior is implemented in the intended isolated branch.
- Required tests and verification commands ran in the current session.
- Result schemas and claim-safety rules still pass.
- Evidence paths, metrics, commit SHA, and risks are recorded.
- Harness observability has been updated when relevant: `docs/harness/quality-document.md`, `docs/harness/decision-log.md`, or `results/harness/harness_score.json`.
- Worktree status is clean or remaining dirtiness is explicitly explained.
- The next session can restart from `AGENTS.md`, `feature_list.json`, `progress.md`, and `session-handoff.md`.

## Subagents

Use subagents only with explicit branch/worktree ownership. Workers may edit only their assigned worktree. Reviewers are read-only unless explicitly asked to fix a scoped issue. Every worker must return changed files, commands run, result paths, metrics, and risks.

## End of Session

Before ending:

1. Update `progress.md`.
2. Update `feature_list.json` for the active feature status and evidence.
3. Update `session-handoff.md` with branch, commit, verification, risks, and next step.
4. Run `bash scripts/check_clean_session.sh` or record why the session intentionally ends dirty.
5. Commit safe, scoped changes.
6. Leave the worktree restartable via `bash scripts/init_harness.sh`.
