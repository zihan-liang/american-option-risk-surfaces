# Consolidated Experiment Data Build Plan

**Date:** 2026-08-21

## Goal

Build a reproducible, branch-complete experiment-data inventory across:

- `zihan-liang/american-option-risk-surfaces` (all remote branches), and
- `Ruixixu-hub/2026-surf-american-risk-surfaces` (all remote branches),

then normalize machine-readable result artifacts into CSV outputs under
`results/consolidated_csv/` without modifying any raw experiment artifact.

The build must also recompute the final matched high-dimensional PINN price
RMSE values against Ruixi's stored high-accuracy
`DIRK + Policy Iteration + sinh grid (M=480, N=960)` reference.

## Scientific invariants

1. No training, tuning, threshold changes, or raw-result mutation.
2. Preserve existing HD CN/PSOR price RMSE as
   `internal_cnpsor_price_rmse` with scope `internal_diagnostic`.
3. Recompute, for FC-HVI and DeepSets-HVI at d=10,20,50, seed 101:
   - `dirk_price_rmse_hd_domain` on `S/K in [0.2,1.5]`, `tau/T in [0.01,1]`;
   - `dirk_price_rmse_shared_domain` on the verified compact common domain
     `S/K in [0.5,1.5]`, `tau/T in [0.01,1]`.
4. Cross-model comparisons use the shared-domain DIRK metric; HD-only scaling
   uses the HD-domain DIRK metric.
5. Interpolate the stored DIRK reference onto the native HD grid for the
   HD-domain metric. For the shared-domain metric, evaluate all model surfaces
   on one frozen common grid. Never extrapolate.
6. Keep reference type, domain, grid, interpolation method, repo, branch,
   commit, and checksum provenance explicit.
7. Do not silently equate ordinary/scaled Gamma, reduced Gamma, or HD
   covariance-weighted Hessian contractions.
8. Deduplicate identical branch artifacts by Git blob/content hash while
   retaining every branch occurrence in provenance.

## Source references

### Final HD source

- Repo: `zihan-liang/american-option-risk-surfaces`
- Branch: `exp/hd3-matched-american-put-final-boundary-complete`
- Immutable result revision: `c11c3ec3c22a5304b33f394c0ecdb1f6225cb6e1`
- Result directory: `results/hd3_matched_american_put_boundary_complete/`

### Ruixi compact matched source

- Repo: `Ruixixu-hub/2026-surf-american-risk-surfaces`
- Branch: `viz/american-put-pointwise-export`
- Exact regime: `put_T100_s060_r005_q010`
- Compact path: `results/visualization_exports/american_put_matched/put_T100_s060_r005_q010/`

The compact export supplies the verified shared plotting grid, Arm C/D price
surfaces and the stored reference on `S/K in [0.5,1.5]`.

### Full stored DIRK reference

The HD-domain metric requires `S/K < 0.5`, so the compact reference is
insufficient for that metric. The workflow must download Ruixi's official Arm D
result package using the tracked manifest:

- tag: `arm-d-seed101-pilot-v1`
- asset: `arm_d_seed101.zip`
- expected bytes: `677973093`
- expected SHA-256: `237d670ccb889564d94a9f553cf9528082f71c2a6891687585c68e56dedcbc5c`

and extract only the exact stored high-accuracy reference for
`put_T100_s060_r005_q010`.

## Output contract

Write under `results/consolidated_csv/`:

- `README.md`
- `source_artifact_inventory.csv`
- `provenance.csv`
- `source_records.csv`
- `array_artifacts.csv`
- `experiment_runs.csv`
- `price_metrics.csv`
- `greek_metrics.csv`
- `boundary_metrics.csv`
- `free_boundary_points.csv`
- `price_slices.csv`
- `structural_metrics.csv`
- `compute_metrics.csv`
- `classical_solver_metrics.csv`
- `binomial_convergence.csv`
- `ruixi_arm_metrics.csv`
- `hd1_metrics.csv`
- `hd2_candidate_metrics.csv`
- `hd3_metrics.csv`
- `campaign_method_results.csv`
- `experiment_gates.csv`
- `environment_runs.csv`
- `surface_artifacts.csv`
- `dirk_common_reference_metrics.csv`
- `manifest.json`
- deduplicated numeric-array CSV gzip files under `arrays/` when readable.

Empty specialized tables are permitted only when the all-branch inventory
contains no source artifact supporting that schema; headers must still be
present and the README/manifest must say why.

## Implementation steps

1. Add contract tests first and observe an expected RED workflow run.
2. Implement a Python consolidator that:
   - enumerates all remote branches in both local clones;
   - inventories result-like `.csv`, `.json`, `.npz`, `.npy`, `.txt`, `.md`,
     `.pt`, `.pth` files;
   - computes content hashes for unique artifacts;
   - writes occurrence provenance and deduplicated artifact inventory;
   - preserves all parseable CSV/JSON scalar records in a generic long-form CSV;
   - registers/exports readable NPZ/NPY arrays to compressed CSV;
   - parses known structured result schemas into normalized CSVs;
   - recomputes the two DIRK RMSE definitions above.
3. Add a GitHub Actions workflow that checks out this data branch, fetches all
   Zihan refs, clones/fetches all Ruixi refs, downloads/verifies the official
   Arm D package, runs tests, builds the CSVs, verifies outputs, and commits only
   generated `results/consolidated_csv/` files back to this branch.
4. Inspect workflow logs and generated outputs. Fix only through failing tests.
5. Run final verification before claiming completion.

## Non-goals

- No model training or inference requiring missing checkpoints.
- No blind-holdout access.
- No mutation of accepted campaign/result branches.
- No automatic merge into `campaign/independent-methods`.

## Build status

The contract suite is green and this revision intentionally triggers the first
full all-branch build and DIRK rescoring run.
