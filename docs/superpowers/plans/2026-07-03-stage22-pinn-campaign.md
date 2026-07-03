# Stage 22 PINN Campaign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Stage 22 campaign harness that tests multiple PINN-inspired methods individually and in combination against the matched supervised positive-premium baseline.

**Architecture:** Extend the existing Stage 21 harness rather than replacing it. Keep residual/mask helpers in `src/models/pinn_residual.py`, add focused premium model variants and campaign utilities, and write one reproducible experiment script that emits comparable CSV/JSON summaries.

**Tech Stack:** Python, NumPy, PyTorch, unittest, existing project CSV/JSON helpers, existing Stage 05 surrogate dataset.

---

### Task 1: Add Campaign Configuration And Experiment Matrix Entry

**Files:**
- Create: `configs/pinn_campaign.yaml`
- Modify: `docs/experiment_matrix.md`
- Modify: `docs/pinn_inspired_premium_experiment_plan.md`

- [ ] **Step 1: Create the campaign config**

Create `configs/pinn_campaign.yaml`:

```yaml
dataset: data/processed/surrogate_dataset.npz
seed: 21
seeds: [21, 22, 23]
option_scope: put
train_cap: 12000
validation_cap: 4000
test_cap: 4000
hidden: 64
depth: 2
batch_size: 2048
epochs: 8
warm_start_epochs: 4
learning_rate: 1.0e-3
patience: 3
device: auto
training_dtype: float64
lambda_sup: 1.0
lambda_obs: 1.0
lambda_pde_values: [0.0, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3]
lambda_pde_default: 1.0e-4
lambda_delta: 1.0e-4
pde_batch_cap: 512
pde_huber_delta: 1.0
gap_epsilon: 1.0e-5
tau_min_for_pde: 0.08
residual_eval_cap: 2048
premium_boundary_threshold: 1.0e-3
obstacle_tolerance: 1.0e-10
success_price_rmse_rel_improvement: 0.005
neutral_price_rmse_rel_worsening: 0.001
residual_rel_worsening_limit: 0.05
boundary_rel_worsening_limit: 0.01
diagnostic_rel_improvement: 0.02
methods:
  - supervised_baseline
  - pde_weight_1e-6
  - pde_weight_1e-5
  - pde_weight_1e-4
  - pde_weight_1e-3
  - pde_ramp
  - terminal_premium
  - residual_sampling
  - delta_loss
  - terminal_plus_ramp
  - terminal_plus_ramp_plus_residual_sampling
feature_names:
  - log_moneyness
  - tau_over_T
  - r
  - q
  - sigma
  - T
  - is_call
```

- [ ] **Step 2: Add Stage 22 row**

Add this row to `docs/experiment_matrix.md` after Stage 21:

```markdown
| 22 | `experiments/22_pinn_campaign.py` | Future-extension PINN-inspired method campaign is evaluated against the matched supervised surrogate baseline | Multi-seed matched-initialization runs; PSOR remains the benchmark; success gates require price/residual/obstacle/boundary diagnostics; no PSOR replacement claim |
```

- [ ] **Step 3: Update Stage 21 plan with Stage 22 pointer**

Append to `docs/pinn_inspired_premium_experiment_plan.md`:

```markdown
## Stage 22 Follow-Up

Stage 22 extends this single-method probe into a campaign. It tests warm-start/ramped PDE loss, hard terminal premium, residual-aware sampling, Delta/Sobolev-style loss, and selected combinations. The comparison target remains the matched supervised positive-premium surrogate, while CN/PSOR remains the label benchmark.
```

- [ ] **Step 4: Commit**

Run:

```bash
git add configs/pinn_campaign.yaml docs/experiment_matrix.md docs/pinn_inspired_premium_experiment_plan.md
git commit -m "docs: plan stage22 pinn campaign"
```

Expected: one commit containing only config/docs changes.

### Task 2: Add Premium Variant And Schedule Helpers

**Files:**
- Modify: `src/models/premium_surrogate.py`
- Create: `src/models/pinn_campaign.py`
- Test: `tests/test_pinn_campaign_helpers.py`

- [ ] **Step 1: Write failing helper tests**

Create `tests/test_pinn_campaign_helpers.py`:

```python
import unittest

import numpy as np
import torch

from models.pinn_campaign import (
    campaign_success_flags,
    pde_weight_for_epoch,
    residual_aware_indices,
)
from models.premium_surrogate import TerminalScaledPositivePremiumModel


class PinnCampaignHelperTests(unittest.TestCase):
    def test_terminal_scaled_premium_is_zero_at_maturity(self):
        model = TerminalScaledPositivePremiumModel.build(input_dim=7, hidden=4, depth=1)
        features = torch.tensor(
            [
                [0.0, 0.0, 0.05, 0.0, 0.2, 1.0, 0.0],
                [0.2, 0.0, 0.05, 0.0, 0.2, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        premium = model(features).reshape(-1)
        self.assertTrue(torch.allclose(premium, torch.zeros_like(premium), atol=1e-7))

    def test_pde_weight_ramp_schedule(self):
        self.assertEqual(pde_weight_for_epoch(1, final_weight=1.0e-4, warm_start_epochs=4, total_epochs=8), 0.0)
        self.assertEqual(pde_weight_for_epoch(4, final_weight=1.0e-4, warm_start_epochs=4, total_epochs=8), 0.0)
        self.assertAlmostEqual(pde_weight_for_epoch(8, final_weight=1.0e-4, warm_start_epochs=4, total_epochs=8), 1.0e-4)

    def test_residual_aware_indices_stay_inside_mask(self):
        rng = np.random.default_rng(7)
        pool = np.array([3, 5, 8, 13, 21])
        residual_abs = np.array([0.1, 0.2, 10.0, 0.3, 0.4])
        selected = residual_aware_indices(pool, residual_abs, size=20, rng=rng)
        self.assertTrue(set(selected.tolist()).issubset(set(pool.tolist())))
        self.assertIn(8, selected.tolist())

    def test_campaign_success_flags_rejects_price_gain_with_residual_regression(self):
        baseline = {
            "price_rmse": 1.0,
            "max_obstacle_violation": 0.0,
            "boundary_rmse": 1.0,
            "residual_mean_abs": 1.0,
        }
        candidate = {
            "price_rmse": 0.99,
            "max_obstacle_violation": 0.0,
            "boundary_rmse": 1.0,
            "residual_mean_abs": 1.2,
        }
        flags = campaign_success_flags(candidate, baseline)
        self.assertFalse(flags["candidate_success"])
        self.assertFalse(flags["diagnostic_success"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
PYTHONPATH=src:experiments /Users/liangzihan/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest tests/test_pinn_campaign_helpers.py
```

Expected: fail because helper module and terminal model do not exist.

- [ ] **Step 3: Implement terminal premium model**

Add to `src/models/premium_surrogate.py`:

```python
class TerminalScaledPositivePremiumModel:
    """Positive premium model that enforces zero premium at theta=tau/T=0."""

    @staticmethod
    def build(input_dim: int, hidden: int = 128, depth: int = 3):
        import torch
        from torch import nn

        base = build_mlp(input_dim, 1, hidden=hidden, depth=depth)

        class _TerminalScaled(nn.Module):
            def __init__(self, net):
                super().__init__()
                self.net = net
                self.softplus = nn.Softplus()

            def forward(self, features):
                theta = torch.clamp(features[:, 1:2], min=0.0)
                return theta * self.softplus(self.net(features))

        return _TerminalScaled(base)
```

- [ ] **Step 4: Implement campaign helpers**

Create `src/models/pinn_campaign.py`:

```python
from __future__ import annotations

import numpy as np


def pde_weight_for_epoch(epoch: int, *, final_weight: float, warm_start_epochs: int, total_epochs: int) -> float:
    if epoch <= warm_start_epochs:
        return 0.0
    ramp_epochs = max(1, total_epochs - warm_start_epochs)
    progress = min(1.0, max(0.0, (epoch - warm_start_epochs) / ramp_epochs))
    return float(final_weight) * progress


def residual_aware_indices(pool_indices: np.ndarray, residual_abs: np.ndarray, *, size: int, rng: np.random.Generator) -> np.ndarray:
    pool_indices = np.asarray(pool_indices)
    residual_abs = np.asarray(residual_abs, dtype=float)
    if len(pool_indices) != len(residual_abs):
        raise ValueError("pool_indices and residual_abs must have the same length")
    if len(pool_indices) == 0:
        return pool_indices
    weights = residual_abs + 1.0e-12
    weights = weights / np.sum(weights)
    replace = len(pool_indices) < size
    return rng.choice(pool_indices, size=size, replace=replace, p=weights)


def campaign_success_flags(
    candidate: dict,
    baseline: dict,
    *,
    price_improvement: float = 0.005,
    neutral_price_worsening: float = 0.001,
    residual_worsening_limit: float = 0.05,
    boundary_worsening_limit: float = 0.01,
    diagnostic_improvement: float = 0.02,
) -> dict[str, bool]:
    price_delta = _relative_delta(candidate["price_rmse"], baseline["price_rmse"])
    residual_delta = _relative_delta(candidate["residual_mean_abs"], baseline["residual_mean_abs"])
    boundary_delta = _relative_delta(candidate.get("boundary_rmse"), baseline.get("boundary_rmse"))
    obstacle_ok = candidate.get("max_obstacle_violation", 0.0) <= 1.0e-10
    residual_ok = residual_delta <= residual_worsening_limit
    boundary_ok = boundary_delta <= boundary_worsening_limit
    price_better = price_delta <= -price_improvement
    price_neutral = price_delta <= neutral_price_worsening
    diagnostic_better = residual_delta <= -diagnostic_improvement or boundary_delta <= -diagnostic_improvement
    return {
        "candidate_success": bool(price_better and obstacle_ok and residual_ok and boundary_ok),
        "diagnostic_success": bool((not price_better) and price_neutral and diagnostic_better and obstacle_ok and residual_ok and boundary_ok),
        "price_rel_delta": float(price_delta),
        "residual_rel_delta": float(residual_delta),
        "boundary_rel_delta": float(boundary_delta),
    }


def _relative_delta(value, baseline) -> float:
    if value is None or baseline is None:
        return 0.0
    baseline = float(baseline)
    if baseline == 0.0:
        return 0.0 if float(value) == 0.0 else float("inf")
    return (float(value) - baseline) / abs(baseline)
```

- [ ] **Step 5: Run tests and commit**

Run:

```bash
PYTHONPATH=src:experiments /Users/liangzihan/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest tests/test_pinn_campaign_helpers.py
```

Expected: pass.

Commit:

```bash
git add src/models/premium_surrogate.py src/models/pinn_campaign.py tests/test_pinn_campaign_helpers.py
git commit -m "feat: add pinn campaign helpers"
```

### Task 3: Implement Stage 22 Campaign Experiment

**Files:**
- Create: `experiments/22_pinn_campaign.py`
- Test: `tests/test_pinn_campaign_outputs.py`

- [ ] **Step 1: Write dry-run output test**

Create `tests/test_pinn_campaign_outputs.py` with a tiny synthetic dataset and assert that `run()` writes:

```text
pinn_campaign_config_manifest.json
pinn_campaign_metrics.csv
pinn_campaign_residual_metrics.csv
pinn_campaign_delta_metrics.csv
pinn_campaign_summary.json
pinn_campaign_training_history.csv
```

The test must use `overrides` with `seeds: [21]`, `epochs: 2`, `train_cap: 48`, `validation_cap: 18`, `test_cap: 18`, `methods: ["supervised_baseline", "pde_weight_1e-6", "terminal_premium"]`, and `device: "cpu"`.

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
PYTHONPATH=src:experiments /Users/liangzihan/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest tests/test_pinn_campaign_outputs.py
```

Expected: fail because `experiments/22_pinn_campaign.py` does not exist.

- [ ] **Step 3: Implement `experiments/22_pinn_campaign.py`**

Implement by adapting Stage 21. Required behavior:

- load config and Stage 05 dataset;
- build PSOR PDE mask with near-boundary, near-kink, and near-maturity exclusions;
- for each seed, build matched initial states for all methods;
- train `supervised_baseline` with `lambda_pde = 0`;
- train fixed PDE variants from config;
- train `pde_ramp` using `pde_weight_for_epoch`;
- train `terminal_premium` with `TerminalScaledPositivePremiumModel`;
- train `residual_sampling` by updating PDE sample probabilities from residual magnitudes after each epoch;
- train `delta_loss` using autograd `p_x / S` or a documented normalized Delta proxy on strict Greek-mask samples;
- evaluate every method on validation/test metrics;
- compute success flags versus the matched supervised baseline for the same seed;
- write required outputs using LF line endings.

- [ ] **Step 4: Run dry-run test and focused suite**

Run:

```bash
PYTHONPATH=src:experiments /Users/liangzihan/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest tests/test_pinn_campaign_helpers.py tests/test_pinn_campaign_outputs.py tests/test_pinn_residual.py tests/test_pinn_masks.py tests/test_pinn_experiment_outputs.py
```

Expected: pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add experiments/22_pinn_campaign.py tests/test_pinn_campaign_outputs.py
git commit -m "feat: add stage22 pinn campaign"
```

### Task 4: Run Campaign, Record Results, And Verify

**Files:**
- Create: `results/experiment_22_pinn_campaign/*.csv`
- Create: `results/experiment_22_pinn_campaign/*.json`
- Modify: `docs/pinn_inspired_premium_experiment_plan.md`

- [ ] **Step 1: Run Stage 22**

Run:

```bash
PYTHONPATH=src:experiments /Users/liangzihan/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 experiments/22_pinn_campaign.py --config configs/pinn_campaign.yaml
```

Expected: completes and writes required Stage 22 outputs.

- [ ] **Step 2: Inspect summary**

Run:

```bash
cat results/experiment_22_pinn_campaign/pinn_campaign_summary.json
```

Expected: summary identifies best method, success flags, and cautious interpretation. If no method passes gates, it must say no improvement claim is supported.

- [ ] **Step 3: Update docs with result note**

Append a short `Stage 22 Campaign Result` section to `docs/pinn_inspired_premium_experiment_plan.md`. Include:

- best method;
- whether `candidate_success` or `diagnostic_success` passed;
- one-sentence interpretation;
- reminder that PSOR remains the benchmark.

- [ ] **Step 4: Verification**

Run:

```bash
PYTHONPATH=src:experiments /Users/liangzihan/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest discover -s tests
PYTHONPATH=src:experiments /Users/liangzihan/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m compileall -q src tests experiments
git diff --check stage-13-future-extensions..HEAD
```

Expected: tests pass, compile passes, diff check clean.

- [ ] **Step 5: Commit results**

Run:

```bash
git add docs/pinn_inspired_premium_experiment_plan.md results/experiment_22_pinn_campaign
git commit -m "docs: record stage22 pinn campaign results"
```

### Task 5: Completion Notification

**Files:**
- No file changes required.

- [ ] **Step 1: Notify the source thread**

When implementation is complete, send a message back to the source thread with:

```text
Stage 22 complete.
Branch:
Commits:
Files changed:
Best method:
Candidate success:
Diagnostic success:
Key metrics:
Tests run:
Remaining unrelated dirty files:
```

This notification is required for the coordinating thread to continue the active goal.
