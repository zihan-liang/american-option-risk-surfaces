# Consolidated experiment CSV data

This directory is generated from immutable Git/release artifacts across every branch of both SURF project repositories. Identical blobs are deduplicated in `source_artifact_inventory.csv`, while every branch/path occurrence remains in `provenance.csv`.

## Reference policy

HD internal CN/PSOR RMSE is retained only as `internal_cnpsor_price_rmse`. Final cross-model price comparison uses `dirk_price_rmse_shared_domain` on S/K=[0.5,1.5], tau/T=[0.01,1]. HD-only scaling uses `dirk_price_rmse_hd_domain` on S/K=[0.2,1.5], tau/T=[0.01,1], both against Ruixi's stored DIRK + Policy Iteration + sinh-grid M=480,N=960 reference.

## Independent 50,000-step CRR cross-check

The `crr50000_*` files add a matched American-put independent classical cross-check for K=1, T=1, r=0.05, q=0.10, sigma=0.60. The CRR run used 50,000 steps and completed 1,203 price jobs plus 163/163 free-boundary points. `crr50000_convergence_summary.csv` records the 2,048-to-50,000 step convergence audit; the final 32,768-to-50,000 confirmation RMSE change is below 1e-6 on the nine-point confirmation set.

`crr50000_three_method_headline.csv` and `crr50000_three_method_error_summary.csv` compare CRR50k, CN+PSOR (M=120,N=120), and the existing DIRK + Policy Iteration + sinh-grid reference (M=480,N=960). CRR50k is an independent benchmark, not analytic ground truth, and does not replace `dirk_price_rmse_shared_domain` as the project headline denominator. Delta comparisons are diagnostic; CRR second-difference scaled-Gamma is lattice-sensitive and must not be interpreted as an exact Greek reference. Boundary comparisons also retain an extraction-method caveat because CRR uses premium-threshold bisection whereas the PDE solvers use spatial-grid threshold crossing.

The compact consolidated add-on stores run status, convergence, representative CRR price anchors, boundary summaries, three-method error summaries, provenance, manifests, and SHA-256 hashes. Hashes for the full user-run 1,203-point price/Greek artifacts and 163-point boundary artifact are preserved in `crr50000_input_checksums.json`.

## Generic lossless fallback

All parseable CSV/JSON scalar fields are preserved in `source_records.csv`. Numeric arrays are indexed in `array_artifacts.csv` and exported under `arrays/` when readable. Checkpoints are inventory-only.

## Refinement and table audit

`table_audit.csv` and `table_audit.json` report row counts, duplicate rows, and required-key gaps for every logical table. Specialized metric tables use canonical aggregate artifacts only; histories, pointwise curves, and other raw numerical evidence remain available through `source_records.csv`, `array_artifacts.csv`, and the array exports. Symbolic Git remote HEAD refs are excluded from branch counts.

## Chunked large tables

Logical CSV tables that exceed the GitHub-safe single-file threshold are not dropped or summarized. They are split losslessly into gzip-compressed CSV chunks under `chunks/<table>/` and indexed by `table_chunks.csv`, including row ranges, row counts, compressed byte sizes, and SHA-256 checksums.
