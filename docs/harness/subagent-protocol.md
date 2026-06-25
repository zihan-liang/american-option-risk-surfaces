# Subagent Protocol

Use subagents when tasks can be isolated by branch, result folder, and responsibility. The coordinator owns the decision logic; workers own execution in their assigned worktrees.

## Roles

### Coordinator

- Creates or identifies the base branch and worktree root.
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

## Branch ownership

Workers do not edit each other’s branches. If a shared fix is needed, the coordinator creates a separate fix branch or gives explicit approval for a scoped patch.

## Closing subagents

Close or stop tracking subagents only after:

- their final report is captured or their branch has been independently inspected;
- their worktree status is understood;
- result paths and commit SHAs are recorded;
- open risks are moved into `progress.md`, `feature_list.json`, or the summary branch.
