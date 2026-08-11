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

## Manager Review Round 1 Follow-Up

- Added a fixed `selection_seed`; whole train, validation, and `legacy_test`
  regime selections are now made once before the model-seed loop and recorded
  once in the manifest.
- Added the valid-VI collocation mask. Complementarity and
  Fischer--Burmeister training uses only put, non-maturity, non-kink,
  non-boundary points while retaining both exercise and continuation samples.
  Masked-PDE training continues to use the narrower PSOR continuation mask.
- Residual diagnostics now use fixed per-regime valid-VI indices, capped at
  `2048` in the canonical config and shared across methods and model seeds.
  Output rows report both valid-VI and masked-PDE sample counts.
- Promotion now requires defined scenario-level price, Delta, K*Gamma,
  boundary, continuation-violation, complementarity, and masked-PDE metrics;
  ordinary diagnostics use the 2% no-regression gate, masked PDE uses 5%, and
  obstacle violation must not exceed `1e-10`. Missing diagnostics block
  promotion explicitly.
- M02 now emits the registry-standard `config_manifest.json`,
  `run_metrics.csv`, `training_history.csv`, `decision.json`, and
  `artifact_hashes.json`, in addition to M02 detail files. The final hash
  manifest covers every compact output except itself.
- Added `experiments/26_independent_method_campaign.py --method M02 --phase
  screen --smoke` dispatch. Smoke records are always
  `diagnostic_only` / `smoke_non_evidence`.
- Added an unmeasured forward/backward/residual warm-up before per-method
  timing, preventing PyTorch first-use overhead from being charged solely to
  the supervised baseline.

Follow-up verification: focused M02/PINN tests passed (13 tests); shared
campaign registry/output/statistics/leakage tests passed (42 tests); compileall
and `git diff --check` passed. Canonical execution remains unlaunched.
