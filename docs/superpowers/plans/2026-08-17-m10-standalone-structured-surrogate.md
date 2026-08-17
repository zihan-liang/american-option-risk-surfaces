# M10 Structured Surrogate Implementation Plan

**Goal:** Implement the frozen M10 representation-ablation screen comparing direct price, positive early-exercise premium, terminal-scaled premium, and European-anchor correction under identical supervised-learning conditions.

**Architecture:** Reuse the repository CN/PSOR oracle and MLP, but keep M10 representation, training, diagnostics, and statistics in M10-specific modules. Every arm is reconstructed to the same normalized American price before validation/test metrics and receives the same hard known constraints.

**Tech Stack:** Python, NumPy, PyTorch CPU float64, existing `src/solvers` CN/PSOR/Black-Scholes utilities, existing `src/models/mlp.py`.

## Frozen protocol

- Method `M10`; branch `campaign/independent-methods`; start from closed M09 integration commit.
- Canonical screen budget: 3600 seconds.
- Seeds: `[101,203,307,409,503]`.
- Arms: `direct_price`, `positive_premium`, `terminal_scaled_premium`, `european_anchor_correction`.
- Same complete regime split, observations, MLP parameter count, optimizer, learning rate, seeded batch order, and epoch count for every arm.
- Feature normalization and deterministic representation scales are fit from training regimes only.
- Reconstruct normalized American price before every metric.
- Primary gate: point price-RMSE improvement over `direct_price` at least 2%, paired scenario-bootstrap 95% CI lower bound strictly above 0%, Holm-adjusted one-sided paired p-value at most 0.05.
- Delta/Gamma/boundary degradation CI upper bound at most 2%; masked continuation residual degradation CI upper bound at most 5%; hard obstacle violation at most `1e-10`.
- Timing: five warm-ups and 30 repetitions; analytical European anchor is recomputed in timed anchored-arm inference.
- Smoke is always `diagnostic_only`; sealed blind holdout is never accessed.
- Expected outcomes are hypotheses only; never tune after seeing the canonical result.

## Task 1 — representation/statistical contracts

**Files:**
- `configs/standalone/m10_structured_surrogate.json`
- `experiments/standalone/m10_structured_surrogate_core.py`
- `tests/test_m10_structured_surrogate.py`

- [ ] Test exact seeds, arms, budget, timing, and gates.
- [ ] Test the four reconstruction equations and NumPy/Torch equivalence.
- [ ] Test train-only scale fitting, paired complete-regime bootstrap, paired randomization test, Holm adjustment, gate behavior, and smoke no-promotion.
- [ ] Implement the pure core helpers and run focused tests.

## Task 2 — data/oracle and training

**Files:**
- `experiments/standalone/m10_structured_surrogate_data.py`
- `experiments/standalone/m10_structured_surrogate_train.py`
- `experiments/standalone/m10_structured_surrogate.py`

- [ ] Generate complete normalized American-put CN/PSOR surfaces for the predeclared 64 parameter regimes.
- [ ] Compute analytical European anchors on the same nodes.
- [ ] Split complete regimes before node expansion and assert every included regime is complete.
- [ ] Fit feature/output scales on train regimes only.
- [ ] Train all four arms for the same fixed epochs and seeded batch order; validation is diagnostic and does not early-stop or tune an arm.
- [ ] Keep MLP topology and trainable parameter count identical across arms.

## Task 3 — reconstructed-price diagnostics and inference timing

**Files:**
- `experiments/standalone/m10_structured_surrogate_metrics.py`
- `experiments/standalone/m10_structured_surrogate_train.py`
- `tests/test_m10_structured_surrogate.py`

- [ ] Reconstruct the American price before all metrics.
- [ ] Apply payoff, terminal, spatial-boundary, and obstacle projection identically to every arm.
- [ ] Record price/Delta/Gamma/boundary RMSE, obstacle violation, masked continuation residual, complementarity diagnostics, and boundary missing/spurious fractions.
- [ ] Recompute the analytical European anchor inside the timed path so anchor overhead is not hidden.
- [ ] Use five warm-ups and 30 repetitions for canonical deterministic inference.

## Task 4 — decision/evidence safety

**Files:**
- `experiments/standalone/m10_structured_surrogate.py`
- `tests/test_m10_structured_surrogate.py`

- [ ] Average paired seed metrics within each complete test regime, then bootstrap complete regimes.
- [ ] Require point price gain >=2%, CI excluding zero, Holm-adjusted p<=0.05, ordinary degradation <=2%, residual degradation <=5%, and hard obstacle <=1e-10.
- [ ] Select the largest price improvement only among candidates that pass every gate.
- [ ] Canonical runtime over 3600 seconds becomes `resource_blocked`; smoke always remains `diagnostic_only`.
- [ ] Write finite `config_manifest.json`, `metrics.csv`, `training_history.csv`, `decision.json`, `artifact_hashes.json`, plus concise surface-generation timing; record `blind_holdout_accessed=false`.

## Task 5 — verification and campaign commits

- [ ] Run focused M10 tests in the campaign environment.
- [ ] Run `python -m compileall` on M10 files and `git diff --check`.
- [ ] Run the full repository test suite.
- [ ] Commit implementation once as `feat: add standalone M10 structured surrogate` and freeze that SHA.
- [ ] Run one bounded smoke from the implementation commit and inspect compact JSON/CSV evidence.
- [ ] Run the canonical screen once from the same implementation SHA only after smoke is structurally valid.
- [ ] Write `docs/campaign/m10_closure.md` and commit evidence once as `results: close M10 structured surrogate`.
- [ ] Never rerun because metrics are disappointing; preserve negative/blocked outcomes.
