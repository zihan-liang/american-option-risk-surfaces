# AGENTS.md - American Option Risk Surfaces Harness

This repository studies free-boundary-aware American option risk surfaces. The
harness exists to make Codex sessions reliable through clear startup,
persistent state, isolated scope, reproducible verification, and safe handoff.
The detailed token and model policy is in
`docs/campaign/token_budget_policy.md`.

## User-Owned Task Creation

- The project manager must not create Codex tasks, threads, subagents, or other
  delegated workers.
- When specialist work is useful, the manager must stop and give the user a
  task name, recommended model, repository/worktree path, branch, initialization
  prompt, full prompt, stop conditions, and acceptance criteria.
- Only the user creates a task and assigns its model.
- Task creation uses two stages. The user first creates the task with the short
  initialization prompt below. The full task prompt is sent only after the
  manager verifies the created task's ID, worktree, repository root, HEAD, and
  status.
- If the app displays `conversation not found` after initialization, do not
  resend or create a duplicate. Wait briefly, return to the Project or Recent
  list, locate the task, and tell the manager it was created.
- A user-created specialist task may work only in its explicitly assigned
  branch or worktree. Reviewers are read-only unless the user authorizes a
  scoped fix.
- A new task must verify that `git rev-parse --show-toplevel` points to this
  repository before reading or modifying project files. If it cannot access the
  repository, it must stop without creating substitute work elsewhere.

The initialization prompt is:

```text
Initialization only.

Report:
git rev-parse --show-toplevel
git status --short --branch
git rev-parse HEAD

Do not read project files, modify files, run tests, or start the requested work.
Wait for the full task prompt.
```

## Manager Notification

- The full task prompt must include the Project Manager task ID.
- If `send_message_to_thread` is available, the worker must proactively send the
  same compact report to the manager when it reaches a blocker or completes its
  assigned phase. It must still reply in its own task for the user.
- If thread messaging is unavailable, the worker must say so explicitly and ask
  the user to relay the compact report to the manager.
- Do not send routine progress messages to the manager. Notify only on a blocker
  that needs a decision or on completion.
- A notification is a handoff, not acceptance or authorization to continue.

## Startup Workflow

Before writing code:

1. Confirm the working directory with `pwd` and `git rev-parse --show-toplevel`.
2. Read this file completely.
3. Read `feature_list.json`, `progress.md`, and `session-handoff.md`.
4. Review only the project docs relevant to the assigned phase. The standard
   references are `README.md`, `docs/reproducibility_checklist.md`,
   `docs/claim_audit.md`, `docs/release_handoff.md`,
   `docs/harness/verification-matrix.md`, and
   `docs/harness/startup-readiness-checklist.md`.
5. Run `bash scripts/init_harness.sh` from the repository root.
6. If full report verification is in scope, run
   `bash scripts/verify_report_artifacts.sh`; when it reports missing ignored
   figures, run `bash scripts/bootstrap_artifacts.sh`.
7. Review the accepted base and focused candidate commits. Do not scan broad
   history when a small diff is sufficient.

If baseline verification fails, diagnose it before adding scope.

## Token And Scope Control

- One task handles one method and one phase: design, implementation, smoke
  review, canonical run, or acceptance review.
- Use callback-first coordination. After assigning work, wait for the worker's
  blocker or completion message; do not run shell `sleep` commands or poll an
  active task's transcript for routine progress.
- Do not call `read_thread` on an active worker merely to check whether it has
  finished. If a callback is unexpectedly unavailable and a decision cannot
  wait, take at most one compact status snapshot without tool outputs or old
  turns, then return control to the user.
- Read the smallest relevant file set. Do not rescan the repository, all prior
  reports, or long task histories when a focused diff and method packet suffice.
- Bound command output at its source with focused paths, `rg`, line ranges, and
  small output limits. Keep full logs in ignored files and inspect only the
  failing excerpt or final summary.
- Use local programs for training, timing, bootstrap, hashing, and aggregation.
  The model should not repeatedly reason while a deterministic process runs.
- Long numerical jobs run outside an active reasoning loop. Start them from a
  local terminal with a log, PID, and exit-status file; return control to the
  user and inspect artifacts only after the process finishes.
- Run no more than one implementation pass and one focused review pass before
  returning to the user for a decision.
- If diagnostic coordination or repeated review begins to consume experiment-
  scale context, stop, record the exact blocker, and request a narrower next
  phase instead of continuing to accumulate history.
- Do not activate report-writing, visualization, or cleanup work during a
  numerical implementation phase unless the user explicitly requests it.
- Completion messages must be compact: branch and commit, files changed,
  commands run, exact results, known limitations, and decision requested. Keep
  full logs in files.

## Working Rules

- Use an external worktree for implementation. Do not work directly in a dirty
  main checkout.
- Preserve user changes. Do not clean, reset, or overwrite unrelated files.
- Keep generated `.npz`, `.pt`, large figures, and transient logs out of Git
  unless a feature explicitly curates them as compact evidence.
- Add tests before behavior-changing code when practical.
- Treat `feature_list.json` as the active-work system of record. Keep at most
  one feature `in-progress` unless the user assigns separate branch ownership.
- Use `scripts/update_feature_status.py` for status transitions when practical;
  see `docs/harness/feature-lifecycle.md`.
- Run `bash scripts/check_clean_session.sh` before handoff; use `--allow-dirty`
  only while intentionally working on an uncommitted branch.

## Experimental Stop Gates

Do not start or continue a canonical experiment if any of the following holds:

- a focused unit or contract test fails;
- an analytic control or formula check fails;
- a required metric is `NaN`, infinite, null, or unavailable without a declared
  reason accepted by the protocol;
- seeds, splits, budgets, precision, or timing settings differ from the frozen
  protocol;
- artifact provenance does not match `HEAD`;
- the primary metric cannot be computed;
- a virtual environment, secret, large checkpoint, or unrelated generated
  output is staged;
- the blind holdout would be accessed before the method is frozen.

Fix the blocker, rerun focused verification, and obtain manager acceptance
before canonical work. Never remove a required metric or weaken a gate merely
to make a method pass.

## Claim Boundaries

- CN/PSOR is the audited benchmark and label generator within tested regimes.
- Surrogate models provide scoped diagnostic or scenario-screening evidence.
- Boundary heads are supervised diagnostics, not exact free-boundary solvers.
- The bounded Delta head is a separate strict-mask diagnostic head.
- The integrated workflow supports scenario screening with PSOR audit.
- Do not claim that a surrogate model replaces PSOR.
- Do not claim one network solves price, boundary, and Greeks.
- Gamma surrogate work is out of scope unless the user explicitly authorizes it.

## Model Allocation

- Recommend GPT-5.6 Sol for scientific design, difficult mathematics, claim
  review, and final acceptance.
- Recommend GPT-5.6 Terra for normal implementation and focused debugging.
- Recommend GPT-5.6 Luna for narrow, repetitive, rule-governed inventory,
  status, and authorized cleanup.
- DeepSeek models may be recommended only for bounded supporting work. Their
  output is provisional until checked against code, tests, equations, and
  artifacts.
- Model recommendations do not authorize task creation. The user chooses and
  assigns every model.

## DeepSeek Boundary

- Suitable work includes a second derivation, small-diff review, focused unit
  tests from an approved specification, log classification, and summaries.
- DeepSeek must not independently approve scientific claims, declare a method
  better than CN/PSOR, unblind the holdout, merge or push evidence, or make final
  acceptance decisions.
- Limit each DeepSeek task to explicitly permitted files. Do not send API keys,
  credentials, authentication files, unnecessarily broad unpublished material,
  or the whole private repository.
- Treat the local OpenCodex-to-DeepSeek route as a third-party provider path.
  Validate tool use and instruction compliance with a small qualification task
  before repository modifications.

## Verification Commands

Use the shared pilot environment when available:

```bash
PY=/Users/liangzihan/Documents/SURF_0159/SURF/.worktrees/american-option-risk-surfaces/.venv-pilot/bin/python
export PYTHONPATH=src:experiments
export PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/american_option_pycache"
```

For the independent-method campaign, use the pinned Python 3.12 environment:

```bash
PY=/Users/liangzihan/Documents/SURF_0159/SURF/.worktrees/american-option-risk-surfaces/.venv-campaign/bin/python
export PYTHONPATH=src:experiments
export PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/american_option_pycache"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/codex_mplconfig"
```

The exact packages are in `requirements-campaign.lock`; environment and MPS
probe details are in `configs/campaign_environment.yaml`.

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

## Git And Evidence

- Review focused commits or diffs, not whole stale branches. Prefer selective
  cherry-picks when a method branch has an old base.
- Never commit virtual environments. Keep compact result summaries and artifact
  hash manifests; keep large reproducible outputs ignored.
- The manager must inspect every returned result. A worker saying "done" is not
  acceptance.
- Push only reviewed commits at meaningful checkpoints. Do not merge to `main`
  without the user's requested PR and acceptance workflow.

## Definition of Done

A feature is done only when all of these are true:

- Target behavior is implemented in the intended isolated branch.
- Required tests and verification commands ran in the current session.
- Result schemas, experimental gates, and claim-safety rules pass.
- Evidence paths, metrics, commit SHA, and risks are recorded.
- Relevant harness observability is updated in
  `docs/harness/quality-document.md`, `docs/harness/decision-log.md`, or
  `results/harness/harness_score.json`.
- Worktree status is clean or remaining dirtiness is explicitly explained.
- The next session can restart from `AGENTS.md`, `feature_list.json`,
  `progress.md`, and `session-handoff.md`.

## End Of Session

Before ending:

1. Update `progress.md`.
2. Update `feature_list.json` for active status and evidence.
3. Update `session-handoff.md` with branch, commit, verification, risks, and next
   step.
4. Run `bash scripts/check_clean_session.sh` or record why the session ends
   dirty.
5. Commit safe, scoped changes.
6. Leave the worktree restartable with `bash scripts/init_harness.sh`.
