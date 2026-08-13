# Project Agent Rules

These instructions apply to every Codex task working in this repository. Read
this file before inspecting broad project history or editing files. The detailed
campaign policy is in `docs/campaign/token_budget_policy.md`.

## User-Owned Task Creation

- The project manager must not create Codex tasks, threads, subagents, or other
  delegated workers.
- When specialist work is useful, the manager must stop and give the user a
  task name, recommended model, repository/worktree path, branch, permitted
  files, complete prompt, stop conditions, and acceptance criteria.
- Only the user creates a task and assigns its model.
- A newly created task must verify that `git rev-parse --show-toplevel` points
  to this repository before it reads or modifies project files. If it cannot
  access the repository, it must stop without producing substitute work
  elsewhere.

## Token And Scope Control

- One task handles one method and one phase: design, implementation, smoke
  review, canonical run, or acceptance review.
- Read the smallest relevant file set. Do not rescan the repository, all prior
  reports, or long task histories when a focused diff and method packet suffice.
- Use local programs for training, timing, bootstrap, hashing, and aggregation.
  The model should not repeatedly reason while a deterministic process runs.
- Run no more than one implementation pass and one focused review pass before
  returning to the user for a decision.
- Do not activate report-writing, visualization, or cleanup work during a
  numerical implementation phase unless the user explicitly requests it.
- Completion messages must be compact: branch and commit, files changed,
  commands run, exact results, known limitations, and the decision requested.
  Keep full logs in files.

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
before launching canonical work. Never remove a required metric or weaken a
gate merely to make a method pass.

## Model Allocation

- Recommend GPT-5.6 Sol for scientific design, difficult mathematics, claim
  review, and final acceptance.
- Recommend GPT-5.6 Terra for normal implementation and focused debugging.
- Recommend GPT-5.6 Luna for narrow, repetitive, rule-governed inventory,
  status, and authorized cleanup.
- DeepSeek models may be recommended only for bounded supporting work under the
  rules below. Their output is provisional until independently checked against
  code, tests, equations, and artifacts.
- Model recommendations do not authorize task creation. The user chooses and
  assigns every model.

## DeepSeek Boundary

- Suitable work includes a second derivation, small-diff review, focused unit
  tests from an approved specification, log classification, and concise
  summaries.
- DeepSeek must not independently approve scientific claims, declare a method
  better than CN/PSOR, unblind the holdout, merge or push evidence, or make final
  acceptance decisions.
- Limit every DeepSeek task to explicitly permitted files. Do not send API keys,
  credentials, authentication files, unpublished material outside the task's
  need, or the whole private repository.
- Treat the local OpenCodex-to-DeepSeek route as a third-party provider path.
  Validate tool use and instruction compliance with a small qualification task
  before assigning repository modifications.

## Git And Evidence

- Work from the branch and worktree named by the user. Preserve unrelated user
  changes.
- Review focused commits or diffs, not whole stale branches. Prefer selective
  cherry-picks when a method branch has an old base.
- Never commit virtual environments. Keep compact result summaries and artifact
  hash manifests; keep large reproducible outputs ignored.
- The manager must inspect every returned result. A worker saying "done" is not
  acceptance.
- Push only reviewed commits at meaningful checkpoints. Do not merge to `main`
  without the user's requested PR and acceptance workflow.
