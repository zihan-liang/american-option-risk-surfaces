# Result Schema Policy

Comparison tables are the contract between experiment branches, reviewers, and summary branches. Extra metadata belongs in sidecar manifests, not new comparison columns.

## Required comparison columns

Every pilot or Round-2 comparison CSV must use exactly these columns, in this order:

```text
method
branch
seed
split
option_type
price_rmse
price_mae
boundary_rmse
boundary_mae
delta_rmse
delta_mae
max_obstacle_violation
max_delta_bound_violation
train_seconds
inference_seconds
integrated_cases
claim_status
```

Validate with:

```bash
$PY scripts/check_result_schema.py path/to/comparison.csv
```

For new canonical evidence that should also reject string `"nan"` in constraint columns, use:

```bash
$PY scripts/check_result_schema.py --strict-constraints path/to/comparison.csv
```

## Allowed `claim_status` values

- `pilot_candidate`
- `failed_metric_gate`
- `diagnostic_only`
- `future_work_only`

Blank `claim_status` is allowed only for empty template rows used in tests.

## Constraint fields

- Use blank cells for non-applicable constraints.
- Use finite `0.0` for audited zero violations.
- Do not use string `"nan"` for audited zero-violation evidence.

## Provenance sidecar expectations

Method outputs should include a JSON manifest with:

- `config_path`;
- config SHA256;
- git branch and commit;
- `run_mode`;
- `overrides_applied`;
- `override_keys`;
- source config snapshot;
- effective config snapshot;
- actual train, validation, and test sample counts.

Canonical Round-2 outputs must show full-scale intent and actual counts:

- `train_cap: 150000`;
- `validation_cap: 75000`;
- `test_cap: 150000`;
- `hidden: 128`;
- `depth: 3`;
- `epochs: 30`;
- `batch_size: 8192`;
- `device: auto`;
- `run_mode: round2_full`.
