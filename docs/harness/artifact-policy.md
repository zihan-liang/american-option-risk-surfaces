# Artifact Policy

The repository intentionally keeps large generated artifacts out of Git. Some legacy report-link tests still require those ignored artifacts to exist locally. The harness makes that hidden requirement explicit.

## Generated and ignored

These paths may be regenerated locally and are usually ignored:

- `results/**/*.png`
- `results/**/*.pdf`
- `results/**/*.npz`
- `results/**/*.pt`
- `results/test_stage*/`
- `data/processed/*.npz`

Do not commit generated binaries unless the user explicitly asks for curated evidence artifacts.

## Required for full report-link verification

The full test suite expects LaTeX `\includegraphics{...}` paths to resolve. Verify with:

```bash
bash scripts/verify_report_artifacts.sh
```

If it fails in a fresh worktree, rebuild local artifacts with:

```bash
bash scripts/bootstrap_artifacts.sh
```

This script may run expensive solver/data-generation steps. Use it intentionally, not as a default startup command.

## Commit hygiene

After artifact bootstrap, inspect tracked generated-result churn:

```bash
git status --short
git diff --stat
```

If scripts rewrote tracked result summaries only because of line endings, path separators, or timestamps, keep the harness/code commit scoped and do not include that churn unless it is part of the requested evidence update.
