# M02 Obstacle-Aware PINN Report

Branch: `exp/m02-obstacle-aware-pinn`

Worktree: `/Users/liangzihan/Documents/SURF_0159/SURF/.worktrees/american-option-risk-surfaces/m02-obstacle-aware-pinn`

## Status

M02 implements four matched positive-premium variants: `supervised`,
`masked_pde`, `separate_complementarity`, and `fischer_burmeister`. CN/PSOR
labels remain benchmark data. The canonical five-seed screen has not been run.

## Review Round 2 Resolution

The manager-reproduced smoke failure was caused by whole-regime caps below the
stored surface size: each regime has 14,641 points, while the prior smoke caps
were 4,096/2,048. Smoke now predeclares exactly one complete 14,641-point
regime for train, validation, and `legacy_test`, with one epoch, width 8,
depth 1, a 64-point collocation batch, and a 256-point timing workload.

The valid-VI mask now excludes maturity, free-boundary and kink bands,
per-regime minimum/maximum log-spot nodes, and `near_zero_spot` points while
retaining interior exercise and continuation samples. The narrower PSOR
continuation mask remains exclusive to masked-PDE loss. Diagnostics are capped
at fixed, per-regime valid-VI indices and report their sample counts.

Promotion uses paired scenario bootstrap for price, Delta, K*Gamma, boundary,
dual-feasibility violation, complementarity, and masked-PDE residual. The
three primary price comparisons receive a Holm correction at alpha 0.05; an
otherwise promotable candidate is downgraded if its adjusted result fails.
The dual-feasibility diagnostic is named truthfully rather than being called a
continuation violation. Missing required diagnostics blocks promotion.

The standard contract includes config, multi-row run metrics, history,
decision, and hashes. The decision contains a 3,600-second canonical screen
budget; a canonical overrun is `resource_blocked` and cannot promote. Costs
record preprocessing, training, deterministic price inference, residual
diagnostics, model/working-memory definitions, zero fallback cost with an
explicit no-fallback reason, and a null reused-label cost with an explicit
reason. Canonical timing is fixed at five warmups and 30
repetitions; smoke records its 0/1 timing counts and is always
`diagnostic_only` / `smoke_non_evidence`.

## Review Round 3 Resolution

The primary-price Holm family now downgrades every raw promotion to
`inconclusive_holm_unavailable` when any of the three predeclared comparisons
is unavailable. Both the standard config manifest and decision explicitly
record `blind_holdout_accessed: false`. The post-fix smoke below was run only
after code commit `4fa45e9a7fa544ae04808a45104b70151b639e25` existed; every
run record and provenance field names that exact implementation commit.

## Actual-Data Smoke Evidence

Command:

```sh
MPLBACKEND=Agg $PY experiments/26_independent_method_campaign.py \
  --method M02 --phase screen --smoke \
  --config configs/independent_m02_obstacle_pinn.yaml \
  --output-root /private/tmp/m02-manager-smoke-round3-final
```

- Completed in `5.8572572090197355` seconds.
- Selected one complete regime per split: 14,641 train, 14,641 validation,
  and 14,641 `legacy_test` points.
- The selected validation regime retained 13,340 valid-VI points; residual
  diagnostics used the fixed 64-point cap.
- Every comparison is `diagnostic_only` with claim status
  `smoke_non_evidence`; no blind holdout was accessed.
- Artifacts: `/private/tmp/m02-manager-smoke-round3-final/M02/`.
- All nine compact artifact hashes matched. Six JSON files passed strict
  `allow_nan=False` parsing and all populated numeric values across four CSV
  files were finite.
- The five standard run records, config manifest, and decision all record
  commit `4fa45e9a7fa544ae04808a45104b70151b639e25` and
  `blind_holdout_accessed: false`.

## Verification

- Focused M02/PINN tests passed.
- Shared registry/output/statistics/leakage tests passed.
- Full `MPLBACKEND=Agg $PY -m pytest tests -q` passed.
- `compileall`, `git diff --check`, and headless harness startup passed.

## Risks and Next Step

Smoke has one validation regime, so scenario-level promotion diagnostics are
explicitly unavailable and cannot support an evidence claim. Manager review is
the next step; canonical execution remains serialized and unlaunched.
