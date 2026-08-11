# M02 Obstacle-Aware PINN Report

Branch: `exp/m02-obstacle-aware-pinn`

Worktree: `/Users/liangzihan/Documents/SURF_0159/SURF/.worktrees/american-option-risk-surfaces/m02-obstacle-aware-pinn`

Commit: recorded in the manager completion notification after this scoped report is committed.

## Status

Implemented the independent four-way M02 screen. It compares matched
positive-premium models using `supervised`, `masked_pde`,
`separate_complementarity`, and `fischer_burmeister` losses. CN/PSOR remains
the benchmark and label generator. The canonical five-seed, 30-epoch screen
has **not** been launched.

## Changed Files

- `src/models/obstacle_pinn.py`
- `experiments/27_m02_obstacle_aware_pinn.py`
- `configs/independent_m02_obstacle_pinn.yaml`
- `tests/test_m02_obstacle_pinn_losses.py`
- `tests/test_m02_obstacle_pinn_outputs.py`
- `.gitignore`

## Verification

- `bash scripts/init_harness.sh`
- Campaign Python: focused M02 tests plus inherited PINN residual/mask/campaign tests: 16 passed.
- Campaign Python: `compileall -q src tests experiments` passed.
- `git diff --check` passed.

## Smoke Result Paths

- `results/test_m02_obstacle_pinn/out/m02_summary.json`
- `results/test_m02_obstacle_pinn/out/m02_metrics_by_regime.csv`
- `results/test_m02_obstacle_pinn/out/m02_promotion.json`

These are ignored test artifacts and explicitly non-evidence smoke outputs.

## Smoke Metrics

- Whole-regime selection: 24 train points across 6 regimes, 12 validation
  points across 3 regimes, and 12 `legacy_test` points across 3 regimes.
- All four methods completed with zero measured obstacle violation on the tiny
  fixture.
- Validation-only paired bootstrap was exercised at the regime level. No
  candidate reached the 2% primary price improvement gate; all smoke decisions
  were `inconclusive_expand_to_ten_seeds`.

## Manifest and Provenance

- Canonical config freezes seeds `[101, 203, 307, 409, 503]`, float64 CPU,
  width 128, depth 3, 30 epochs, batch size 8192, patience 8, and caps
  150000/75000.
- Selection records actual per-seed counts and deterministic whole regime IDs.
- Outputs rename the inherited split to `legacy_test`; promotion consumes only
  validation regime observations and never accesses the blind holdout.
- Loss scaling and Fischer--Burmeister epsilon are emitted in the summary.

## Risks and Limitations

- The smoke fixture is intentionally tiny and has no usable threshold boundary
  crossings, so boundary diagnostics are explicitly unavailable there.
- Canonical execution is manager-serialized and remains pending review.
- The M02 dispatcher/registry connection is intentionally deferred until the
  shared campaign registry integration is available.

## Recommended Next Step

Manager review of the scoped code/config and report, followed by serialized
canonical execution only after approval.
