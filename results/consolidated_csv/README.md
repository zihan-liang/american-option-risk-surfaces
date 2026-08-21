# Consolidated experiment CSV data

This directory is generated from immutable Git/release artifacts across every branch of both SURF project repositories. Identical blobs are deduplicated in `source_artifact_inventory.csv`, while every branch/path occurrence remains in `provenance.csv`.

## Reference policy

HD internal CN/PSOR RMSE is retained only as `internal_cnpsor_price_rmse`. Final cross-model price comparison uses `dirk_price_rmse_shared_domain` on S/K=[0.5,1.5], tau/T=[0.01,1]. HD-only scaling uses `dirk_price_rmse_hd_domain` on S/K=[0.2,1.5], tau/T=[0.01,1], both against Ruixi's stored DIRK + Policy Iteration + sinh-grid M=480,N=960 reference.

## Generic lossless fallback

All parseable CSV/JSON scalar fields are preserved in `source_records.csv`. Numeric arrays are indexed in `array_artifacts.csv` and exported under `arrays/` when readable. Checkpoints are inventory-only.

## Refinement and table audit

`table_audit.csv` and `table_audit.json` report row counts, duplicate rows, and required-key gaps for every logical table. Specialized metric tables use canonical aggregate artifacts only; histories, pointwise curves, and other raw numerical evidence remain available through `source_records.csv`, `array_artifacts.csv`, and the array exports. Symbolic Git remote HEAD refs are excluded from branch counts.

## Chunked large tables

Logical CSV tables that exceed the GitHub-safe single-file threshold are not dropped or summarized. They are split losslessly into gzip-compressed CSV chunks under `chunks/<table>/` and indexed by `table_chunks.csv`, including row ranges, row counts, compressed byte sizes, and SHA-256 checksums.
