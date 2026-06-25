# Worktree Policy

This project uses external worktrees so research experiments do not disturb the main checkout or each other.

## Default rule

- Do not implement directly in the main checkout at `/Users/liangzihan/Documents/SURF_0159/SURF/american-option-risk-surfaces`.
- Create feature and experiment worktrees under:

```text
/Users/liangzihan/Documents/SURF_0159/SURF/.worktrees/american-option-risk-surfaces/
```

- Branch from the intended local base branch or commit, not from an accidental dirty state.
- Preserve unrelated dirty and untracked files. Do not use destructive cleanup commands unless the user explicitly asks for them.

## Branch naming

Use scoped branch names that describe the evidence or feature:

- `exp/<experiment-name>` for research experiments.
- `feature/<feature-name>` for reusable tooling or harness work.
- `fix/<issue-name>` for focused bug fixes.

For subagent campaigns, every worker owns exactly one branch/worktree. Reviewers should inspect read-only unless explicitly asked to patch a scoped issue.

## Startup checklist

From the repository root:

```bash
pwd
git status --short --branch
git log --oneline -5
bash scripts/init_harness.sh
```

If the current checkout is not the assigned worktree, stop and move to the correct branch before editing.

## Artifact-copy policy

Large experiment artifacts should stay in method-specific result folders and should not be copied into another branch unless they are curated evidence. When copying evidence across worktrees:

1. Copy only small tables, manifests, and summaries needed for the decision record.
2. Preserve original result paths in the summary.
3. Record source branch, source commit, and verification command.
4. Keep model artifacts and large arrays out of Git unless explicitly required.

## Known project convention

Round-2 and harness work use the shared pilot venv:

```bash
PY=/Users/liangzihan/Documents/SURF_0159/SURF/.worktrees/american-option-risk-surfaces/.venv-pilot/bin/python
export PYTHONPATH=src:experiments
export PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/american_option_pycache"
```
