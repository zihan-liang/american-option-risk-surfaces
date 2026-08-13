# Specialist Task Protocol

Use specialist tasks only when the user creates them and assigns their models.
Tasks must be isolated by branch, result folder, and responsibility. The
coordinator owns decision logic and prepares task packets; workers own execution
in their assigned worktrees. This file does not authorize the manager to create
threads, tasks, subagents, or worktrees on the user's behalf.

## Roles

### Coordinator

- Identifies the accepted base branch and required worktree state, then gives
  the user the two-stage creation prompts from
  `docs/campaign/token_budget_policy.md`.
- Assigns one branch/worktree per worker.
- Defines gates, common commands, result schema, and artifact names.
- Compares reviewed evidence and decides whether to promote, combine, or stop.

### Worker

- Edits only the assigned worktree.
- Keeps method-specific outputs under method-specific result folders.
- Runs the required tests and experiment commands for that branch.
- Returns a final report with changed files, commands, result paths, metrics, manifest evidence, and risks.

### Reviewer

- Checks claim safety, metric fairness, provenance, and overfitting risk.
- Verifies no accidental PSOR replacement language.
- Confirms result schema and manifest fields.
- Stays read-only unless explicitly asked to fix a scoped issue.

## Final report format

Every worker and reviewer report should include:

```text
Branch:
Worktree:
Commit:
Changed files:
Commands run:
Result paths:
Metric table:
Manifest/provenance evidence:
Risks and limitations:
Recommended next step:
```

## Token-limit and usage-limit handling

If a subagent hits a token or usage limit:

1. Treat partial output as incomplete.
2. Ask the agent, if possible, for a compact final report using the required report format.
3. If the agent cannot continue, the coordinator inspects the branch/worktree directly.
4. Do not promote a method based only on an interrupted narrative; require files, metrics, and verification evidence.

## Callback And Polling Rules

- Every full worker prompt includes the Project Manager task ID.
- Workers send only blocker and completion callbacks to the manager when task
  messaging is available.
- The coordinator does not use routine `sleep` loops or transcript polling to
  follow active work.
- The coordinator reviews the final commit, diff, artifacts, and focused checks
  directly; it does not reread the full worker history.
- Long numerical runs execute locally with logs and exit-status evidence, not
  inside a continuously monitored agent turn.

## Branch ownership

Workers do not edit each other's branches. If a shared fix is needed, the
coordinator specifies the required branch state and asks the user to create or
assign a separate specialist task, or the user explicitly authorizes a scoped
patch in an existing task.

## Closing specialist tasks

Close or stop tracking specialist tasks only after:

- their final report is captured or their branch has been independently inspected;
- their worktree status is understood;
- result paths and commit SHAs are recorded;
- open risks are moved into `progress.md`, `feature_list.json`, or the summary branch.
