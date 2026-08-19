# HD3 Visualization Export Design

Date: 2026-08-19

## Purpose

Add a deterministic, post-hoc visualization pipeline for the completed HD3 classical-vs-structured high-dimensional HVI PINN experiment. The pipeline must reuse the preserved trained checkpoints and CN/PSOR oracle grids, add an independent CRR binomial-tree reference under the exact geometric-basket reduction, export auditable source data, and generate publication figures without retraining or accessing the blind holdout.

## Provenance and integrity constraints

The visualization pipeline is read-only with respect to experiment evidence.

- No optimizer steps, retraining, fine-tuning, checkpoint mutation, or parameter updates.
- No blind-holdout access.
- Model inference must load the completed HD3 checkpoint `model_state` exactly as trained.
- Checkpoint verification must be against the training implementation SHA recorded inside each checkpoint, not against the current visualization-code HEAD. This is necessary because the canonical HD3 models were trained at implementation SHA `6c1bd658098698b69d49415c1ac1f0cf280d3c30`, while the compact evidence commit is `951c6a6ad67f1eeedaae5cf4f80ea597909d2a1e` and later visualization commits will move HEAD again.
- The checkpoint `config_hash`, `resolved_objective_hash`, `job_id`, dimension, model ID, and completed-step count must also be verified before inference.
- Canonical dimensions are 10, 20, and 50. Dimension 100 is stress evidence and must remain visibly labeled as stress in derived data and figures.
- Existing `.pt` and `.npz` runtime files remain ignored runtime artifacts; derived CSV/JSON evidence is committed instead.

## Existing inputs

Runtime inputs expected under `results/hd3_classical_vs_structured_high_dimensional_pinn/runtime/`:

- `canonical_d10_FC_HVI/latest_checkpoint.pt`
- `canonical_d10_DEEPSETS_HVI/latest_checkpoint.pt`
- `canonical_d20_FC_HVI/latest_checkpoint.pt`
- `canonical_d20_DEEPSETS_HVI/latest_checkpoint.pt`
- `canonical_d50_FC_HVI/latest_checkpoint.pt`
- `canonical_d50_DEEPSETS_HVI/latest_checkpoint.pt`
- `stress_d100_FC_HVI/latest_checkpoint.pt`
- `stress_d100_DEEPSETS_HVI/latest_checkpoint.pt`
- `oracles/d10.npz`
- `oracles/d20.npz`
- `oracles/d50.npz`
- `oracles/d100.npz`

Committed summary inputs:

- `results/hd3_classical_vs_structured_high_dimensional_pinn/dimension_summary.csv`
- `results/hd3_classical_vs_structured_high_dimensional_pinn/structural_diagnostics.csv`
- `results/hd3_classical_vs_structured_high_dimensional_pinn/resolved_objective.json`
- `results/hd3_classical_vs_structured_high_dimensional_pinn/config_manifest.json`
- `configs/hd3_classical_vs_structured_high_dimensional_pinn.json`

## Reference solutions

### CN/PSOR oracle

The primary numerical reference remains the existing HD3 oracle. For each dimension, the symmetric geometric basket is reduced exactly to a one-dimensional American put with

\[
\sigma_G^2 = \sigma^2\left(\rho + \frac{1-\rho}{d}\right),
\]

and

\[
q_G = q + \frac12\left(\sigma^2 - \sigma_G^2\right).
\]

The existing fine oracle uses a 512 x 512 Crank-Nicolson/PSOR grid and is refinement-checked against a 256 x 256 grid.

### Independent CRR binomial reference

Add a standalone Cox-Ross-Rubinstein American-put solver using the same effective one-dimensional contract. It is an independent numerical cross-check, not the canonical source of truth.

For step counts `N = 500, 1000, 2000, 4000`, compute the American put by backward induction with early exercise at every node. Evaluate CRR/CN-PSOR agreement on the diagonal price grid used for the price-slice figures. Export, for each dimension and N:

- RMSE against CN/PSOR,
- maximum absolute difference against CN/PSOR,
- RMSE against the previous CRR depth where applicable.

The plotting depth is selected deterministically as the smallest tested N satisfying both:

- RMSE versus CN/PSOR <= `1e-4`, and
- RMSE versus the previous tested depth <= `5e-5`.

If no tested depth passes, use N=4000 but mark `reference_agreement_passed=false` in the manifest and figures must label the curve `CRR cross-check` rather than implying numerical equivalence.

## Model loading

Create a visualization-specific completed-model loader that:

1. loads the checkpoint on CPU first,
2. validates training implementation SHA against the frozen expected training SHA,
3. validates config hash and resolved-objective hash against committed experiment evidence,
4. validates job ID, dimension, model ID, and exact completed step budget,
5. reconstructs the FC-HVI or DeepSets-HVI architecture with the existing HD3 model initializers,
6. loads `model_state`,
7. switches the model to `eval()`,
8. moves to CUDA when available, otherwise permits CPU inference,
9. performs no optimizer construction and no training operations.

This loader is intentionally separate from the training-time `_load_completed_model`, whose current-HEAD provenance check is correct for resumable training but unsuitable for later post-hoc visualization commits.

## Derived visualization data

Create `results/hd3_visualization/` with the following auditable artifacts.

### `source_inventory.json`

Record for each checkpoint/oracle:

- relative path,
- file size,
- SHA-256,
- dimension,
- model ID,
- run class,
- checkpoint training implementation SHA,
- checkpoint config hash,
- resolved-objective hash,
- completed step count,
- oracle refinement RMSE.

### `reference_solver_agreement.csv`

Columns include dimension, CRR depth, effective volatility, effective dividend yield, CRR-vs-CNPSOR RMSE, max absolute difference, previous-depth RMSE, and pass/fail flags.

### `diagonal_slices.csv`

For dimensions 10, 20, 50, and 100 and representative maturities `tau = 1.0, 0.5, 0.1`, evaluate the symmetric diagonal

\[
S_1=\cdots=S_d=S, \qquad S\in[0.5,1.5].
\]

Use 401 evenly spaced S points. Export CN/PSOR, CRR, FC-HVI, and DeepSets-HVI values together with absolute model errors. Dimension 100 rows are labeled stress.

### `error_surface_{d}d.csv`

For canonical dimensions 10, 20, and 50, evaluate a dense diagonal surface on 201 S points over [0.5, 1.5] and 101 tau points over [0.0025, 1.0]. Export CN/PSOR, FC-HVI, DeepSets-HVI, and absolute errors.

### `exercise_boundaries.csv`

For canonical dimensions 10, 20, and 50, export 128 tau points and:

- CN/PSOR oracle boundary,
- FC-HVI inferred boundary,
- DeepSets-HVI inferred boundary.

Model boundaries use the same exercise-gap tolerance as the HD3 evaluation logic (`V - payoff <= 1e-5`) on a dense diagonal S grid. The figure is diagnostic because existing HD3 boundary metrics indicate weak boundary recovery.

### `figure_metrics.csv`

Collect the summary metrics actually plotted in the scaling and structural figures so figure generation does not depend on ad-hoc parsing logic.

### `visualization_manifest.json`

Record:

- visualization-code Git SHA,
- training implementation SHA,
- canonical evidence commit SHA,
- config hash,
- resolved-objective hash,
- reference-solver settings and selected CRR depths,
- inference device,
- all grid definitions,
- all generated output filenames,
- `optimizer_steps=0`, `retraining=false`, and `blind_holdout_access=false`.

## Publication figures

Create `figures/hd3/`.

### Figure 1: `hd3_scaling_summary.png`

A 2 x 2 panel over dimension 10, 20, 50, 100 showing:

1. normalized price RMSE,
2. Delta RMSE,
3. trusted PDE P95,
4. training seconds.

FC-HVI and DeepSets-HVI are separate curves. The 100D points use a distinct marker and are labeled `stress`; they must not be presented as canonical evidence. Do not use peak-CUDA scaling in the main panel because the current compact metric is not reliable as a per-model scaling trace.

### Figure 2: `hd3_price_slices.png`

A 1 x 3 panel for canonical dimensions 10, 20, 50 at tau=1.0. Each panel shows:

- CN/PSOR Oracle,
- CRR Binomial Tree (or `CRR cross-check` if convergence agreement failed),
- FC-HVI,
- DeepSets-HVI,
- a vertical line at strike K=1.

The x-axis is `Common asset price S (S1 = ... = Sd)` and the y-axis is normalized American put value.

### Figure 3: `hd3_error_heatmaps.png`

A 2 x 3 panel. Rows are FC-HVI and DeepSets-HVI; columns are 10D, 20D, 50D. Axes are common diagonal price S and tau. Color is absolute error against CN/PSOR. Use a common color normalization across all six panels so visual comparisons are meaningful.

### Figure 4: `hd3_structural_diagnostics.png`

Two panels over dimension 10, 20, 50, 100:

- permutation-consistency RMSE,
- transverse-invariance RMSE.

Dimension 100 is labeled stress. A logarithmic y-axis is permitted where it materially improves readability, especially for permutation consistency.

### Figure 5: `hd3_exercise_boundaries.png`

Three canonical-dimension panels showing CN/PSOR, FC-HVI, and DeepSets-HVI exercise boundaries as functions of tau. This figure is explicitly diagnostic and may expose failure rather than agreement.

### Supplement: `hd3_100d_stress_slice.png`

A standalone 100D stress diagonal price-slice comparison at tau=1.0 using the same four reference/model curves, clearly labeled as stress evidence.

No figure legend is added where direct line labeling or shared labels make the plot readable; when a legend is necessary to distinguish overlapping methods, use one shared figure-level legend rather than repeating legends in every panel.

## Implementation structure

The implementation should remain small and testable rather than placing all responsibilities in one monolithic script.

- `experiments/hd3_visualization_export.py`: CLI orchestration, provenance checks, inference grids, CSV/JSON export, and figure generation entry point.
- `src/solvers/crr_american.py`: pure NumPy CRR American-put reference solver and convergence helpers.
- `tests/test_hd3_visualization_export.py`: CPU-safe unit/integration tests using synthetic or temporary checkpoints/oracles; no canonical CUDA run is required in CI.
- `tests/test_crr_american.py`: CRR payoff/bounds/convergence tests and comparison against an existing one-dimensional solver on a small test contract.

The exporter may reuse existing HD3 private helpers where stable, but must not call training entry points.

## CLI

Primary command:

```powershell
& $PythonExe .\experiments\hd3_visualization_export.py --all
```

Useful narrower modes may include `--reference-only`, `--data-only`, and `--figures-only`, but `--all` is the required user-facing path.

The command should fail fast with actionable messages if a checkpoint/oracle is missing or provenance verification fails. Existing outputs may be overwritten deterministically after provenance checks because they are derived post-processing artifacts, not training evidence.

## Testing and acceptance criteria

The implementation is accepted when:

1. CRR unit tests pass and American early-exercise values are finite, non-negative, and no lower than intrinsic value.
2. Provenance tests reject a wrong implementation SHA, wrong config hash, wrong objective hash, wrong dimension/model, and incomplete checkpoint.
3. A CPU-only synthetic export test demonstrates that loading a checkpoint and generating a small diagonal slice performs zero optimizer steps and does not mutate model parameters.
4. `--reference-only` can regenerate reference-agreement data without model checkpoints.
5. `--all` on the user's preserved HD3 runtime completes without retraining and writes every declared CSV/JSON and figure.
6. The manifest records `optimizer_steps=0`, `retraining=false`, and `blind_holdout_access=false`.
7. 100D is consistently labeled stress in all derived artifacts and figures.
8. CRR/CN-PSOR disagreement is surfaced rather than hidden; no plot calls CRR ground truth unless agreement criteria pass.

## Out of scope

- Retraining or tuning HD3 models.
- Re-running HD2 selection.
- Accessing the campaign blind holdout.
- Claiming the reduced geometric-basket oracle represents arbitrary high-dimensional American baskets.
- Adding generic PCA/t-SNE/UMAP visualizations without a direct scientific question.
- Committing `.pt` checkpoints or `.npz` oracle files to ordinary Git history.
