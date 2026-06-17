# Artifact Manifest

## Source-Controlled Artifacts

The repository should keep small, reviewable research artifacts:

- Markdown documentation in `docs/`.
- LaTeX `.tex` source files in `reports/` and `paper/`.
- YAML configs in `configs/`.
- Python source and tests in `src/`, `experiments/`, and `tests/`.
- Small `.csv` and `.json` audit outputs that support the final claim matrix.

Important source-controlled evidence includes:

- `results/tables/final_claim_matrix.csv`
- `results/experiment_14_claim_audit/claim_audit_summary.json`
- `results/experiment_15_future_extensions/future_extensions_summary.json`
- `results/experiment_16_release_verification/release_verification_summary.json`

## Ignored Local Artifacts

The following are generated locally and should normally remain ignored:

- `.pt` model checkpoints in `results/models/`.
- `.npz` solver surfaces, datasets, and fresh audit cases.
- generated figures such as `.png` files under `results/figures/`.
- generated report logs and auxiliary files.
- build PDF outputs under `build/`.

The build PDF files are deliverables for reading, but they are reproducible from the TeX sources and should not be treated as source artifacts.

## Cleanup Rule

Stale generated root files such as `manuscript_outline.aux`, `manuscript_outline.log`, and `manuscript_outline.pdf` should not be tracked. Use `build/manuscript_outline.pdf` as the local generated manuscript PDF.
