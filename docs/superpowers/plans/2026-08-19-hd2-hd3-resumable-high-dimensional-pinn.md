# HD2-HD3 Resumable High-Dimensional PINN Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a one-command, resumable Windows pipeline that diagnoses and stabilizes the HD1 HVI objective at d=2/5, freezes one objective or the preregistered fallback, then runs FC-HVI versus DeepSets-HVI scaling at 10D/20D/50D with an optional 100D stress test on one RTX 4070 Laptop 8 GB GPU.

**Architecture:** Reuse the exact HD1 geometric-basket oracle, DeepSets model, Black--Scholes VI operator, and Hutchinson HVP implementation from result commit `5b536bc22b800af2a6415fc2a0af6e1f0a0d4922`. Add an isolated durable-runtime layer for atomic checkpoints, append-and-flush JSONL, provenance/status verification, and resume; build HD2 and HD3 as separate experiment modules above that layer; drive both through one PowerShell entry point.

**Tech Stack:** Python 3.12, PyTorch 2.8-compatible APIs, NumPy, SciPy, pytest, Windows PowerShell, Git/GitHub. CUDA training is performed only on the user's RTX 4070 Laptop; development tests must not claim CUDA execution.

**Spec:**
- `docs/superpowers/specs/2026-08-19-hd2-hvi-objective-stabilization-design.md`
- `docs/superpowers/specs/2026-08-19-hd3-classical-vs-structured-high-dimensional-pinn-scaling-design.md`
- `docs/superpowers/specs/2026-08-19-hd2-hd3-one-command-windows-pipeline-design.md`

## Global Constraints

- Canonical seed is exactly `101`; every result is `diagnostic_only`.
- Financial contract is exactly `K=1`, `T=1`, `r=0.05`, `q=0.02`, `sigma=0.25`, `rho=0.30`.
- HD2 dimensions are exactly `[2, 5]`; HD3 canonical dimensions are exactly `[10, 20, 50]`; HD3 stress dimension is exactly `[100]`.
- HD2 uses the exact HD1 DeepSets architecture `2x64` asset encoder, mean pooling, `4x128` value head, `tanh`, and `v=g+tau*softplus(h)`.
- HD2 mandatory candidates are `SUP-HD2`, `HVI-HD1-anchor`, `HVI-low`, `HVI-warm`, and `HVI-warm-low`; all use 10,000 total Adam steps.
- The historical FB smoothing schedule remains `1e-2 -> 1e-4` during the mandatory HD2 screen.
- CAGrad is conditional on the frozen persistent-conflict rule; 8-probe Hutchinson is conditional on the frozen d=5 exact-vs-estimated diagnostic; ReLoBRaLo is optional only when mandatory stabilization fails and budget remains.
- HD3 uses the HD2-selected objective or the fixed `HVI-warm-low` fallback (`warm_fraction=0.5`, `final_physics_weight=0.01`).
- HD3 FC model is six hidden `tanh` layers of width 128; HD3 DeepSets is the unchanged HD1 model.
- HD3 canonical budget is 6,000 steps per model; 100D stress budget is 4,000 steps; warm-start fraction follows the frozen HD2 objective.
- Two fixed Rademacher Hutchinson probes are the default high-dimensional training estimator.
- Training dtype is CUDA float32; oracle/evaluation construction remains CPU float64 where applicable; mixed precision is disabled.
- Global wall-clock cap is 18 hours. Mandatory work has priority over conditional HD2 methods and the 100D stress test.
- Checkpoint interval is every 250 optimizer steps plus phase transitions.
- Runtime checkpoints/full logs/oracle arrays remain ignored; compact CSV/JSON evidence is tracked.
- The independent M01--M24 blind holdout is never accessed.
- No canonical RTX 4070 result may be claimed until the user runs the implementation on the laptop.

---

## File Structure

Create focused modules rather than enlarging the 31k-line-equivalent HD1 experiment script:

```text
configs/hd2_hvi_objective_stabilization.json
configs/hd3_classical_vs_structured_high_dimensional_pinn.json
src/hd_pipeline/runtime.py
src/hd_pipeline/gradient_diagnostics.py
src/hd_pipeline/__init__.py
src/models/hd3_high_dimensional_pinn.py
src/physics/hd3_high_dimensional_sampling.py
experiments/hd2_hvi_objective_stabilization.py
experiments/hd3_classical_vs_structured_high_dimensional_pinn.py
experiments/hd_pipeline_windows.py
scripts/run_hd_pipeline_windows.ps1
scripts/check_hd_pipeline_status.ps1
tests/test_hd2_hvi_objective_stabilization.py
tests/test_hd3_high_dimensional_pinn.py
tests/test_hd_pipeline_resume.py
docs/campaign/hd2_hd3_windows_execution_instructions.md
```

Reuse without semantic modification:

```text
src/models/high_dimensional_pinn.py
src/physics/high_dimensional_black_scholes.py
experiments/hd1_windows_high_dimensional_pinn_48h.py
```

The implementation branch must be anchored on HD1 result commit `5b536bc22b800af2a6415fc2a0af6e1f0a0d4922` before source implementation so those reusable files are physically present. Preserve the approved spec/plan documents when re-anchoring.

---

### Task 1: Re-anchor the implementation branch and freeze machine-readable configs

**Files:**
- Create: `configs/hd2_hvi_objective_stabilization.json`
- Create: `configs/hd3_classical_vs_structured_high_dimensional_pinn.json`
- Create: `tests/test_hd2_hvi_objective_stabilization.py`
- Create: `tests/test_hd3_high_dimensional_pinn.py`

**Interfaces:**
- Consumes: HD1 result commit `5b536bc22b800af2a6415fc2a0af6e1f0a0d4922` and the three approved design specs.
- Produces: frozen JSON config contracts consumed by every later task.

- [ ] **Step 1: Rebuild the working branch on the HD1 result commit while preserving approved docs**

Use the GitHub commit/tree API to create one commit whose parent is `5b536bc22b800af2a6415fc2a0af6e1f0a0d4922` and whose tree contains the three approved spec files plus this plan. Fast-forward or explicitly replace only `exp/hd2-hvi-objective-stabilization`; do not modify `campaign/independent-methods` or the HD1 branch.

- [ ] **Step 2: Write failing config-contract tests**

```python
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_hd2_frozen_config_contract():
    cfg = json.loads((ROOT / "configs/hd2_hvi_objective_stabilization.json").read_text())
    assert cfg["seed"] == 101
    assert cfg["dimensions"] == [2, 5]
    assert cfg["candidates"] == [
        "SUP-HD2", "HVI-HD1-anchor", "HVI-low", "HVI-warm", "HVI-warm-low"
    ]
    assert cfg["total_steps"] == 10000
    assert cfg["checkpoint_interval"] == 250
    assert cfg["gradient_diagnostic_interval"] == 500
    assert cfg["hutchinson_probe_count"] == 2
    assert cfg["max_cuda_allocation_gib"] == 7.2


def test_hd3_frozen_config_contract():
    cfg = json.loads((ROOT / "configs/hd3_classical_vs_structured_high_dimensional_pinn.json").read_text())
    assert cfg["seed"] == 101
    assert cfg["canonical_dimensions"] == [10, 20, 50]
    assert cfg["stress_dimensions"] == [100]
    assert cfg["canonical_steps"] == 6000
    assert cfg["stress_steps"] == 4000
    assert cfg["models"] == ["FC_HVI", "DEEPSETS_HVI"]
    assert cfg["fallback_objective"] == {"candidate": "HVI-warm-low", "warm_fraction": 0.5, "final_physics_weight": 0.01}
```

- [ ] **Step 3: Run tests and verify they fail because configs do not exist**

Run:

```bash
python -m pytest -q -p no:cacheprovider tests/test_hd2_hvi_objective_stabilization.py tests/test_hd3_high_dimensional_pinn.py
```

Expected: FAIL with `FileNotFoundError` for the new config paths.

- [ ] **Step 4: Add exact frozen JSON configs**

The HD2 config must encode the five candidates exactly as:

```json
{
  "SUP-HD2": {"warm_start_steps": 10000, "final_physics_weight": 0.0},
  "HVI-HD1-anchor": {"warm_start_steps": 2000, "final_physics_weight": 0.1},
  "HVI-low": {"warm_start_steps": 2000, "final_physics_weight": 0.01},
  "HVI-warm": {"warm_start_steps": 5000, "final_physics_weight": 0.1},
  "HVI-warm-low": {"warm_start_steps": 5000, "final_physics_weight": 0.01}
}
```

Also encode fresh Sobol seeds distinct from HD1 for `train`, `physics`, `validation`, and `test`; the implementation must compare them with the HD1 config and reject accidental equality.

The HD3 config must encode `transverse_scale=0.35`, the aggregate ranges from the spec, the FC `6x128` architecture, the exact HD1 DeepSets identity, two-probe Hutchinson, and the 18-hour pipeline budget.

- [ ] **Step 5: Run focused config tests**

Run the same pytest command. Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add configs/hd2_hvi_objective_stabilization.json configs/hd3_classical_vs_structured_high_dimensional_pinn.json tests/test_hd2_hvi_objective_stabilization.py tests/test_hd3_high_dimensional_pinn.py docs/superpowers/specs docs/superpowers/plans/2026-08-19-hd2-hd3-resumable-high-dimensional-pinn.md
git commit -m "test(hd2-hd3): freeze experiment contracts"
```

---

### Task 2: Add durable checkpoint, JSONL, status, and provenance primitives

**Files:**
- Create: `src/hd_pipeline/__init__.py`
- Create: `src/hd_pipeline/runtime.py`
- Create: `tests/test_hd_pipeline_resume.py`

**Interfaces:**
- Consumes: `Path`, PyTorch RNG APIs, Git SHA/config hashes.
- Produces:
  - `atomic_json(path: Path, payload: dict) -> None`
  - `append_jsonl(path: Path, payload: dict) -> None`
  - `read_jsonl_recover(path: Path) -> list[dict]`
  - `save_checkpoint_atomic(path: Path, payload: dict) -> None`
  - `load_checkpoint_verified(path: Path, *, implementation_sha: str, config_hash: str, job_id: str) -> dict`
  - `capture_rng_state() -> dict`
  - `restore_rng_state(state: dict) -> None`
  - `sha256_file(path: Path) -> str`
  - `verified_completed_status(path: Path, *, implementation_sha: str, config_hash: str) -> bool`

- [ ] **Step 1: Write failing runtime tests**

```python
from __future__ import annotations
import json
from pathlib import Path
import torch
import pytest

from hd_pipeline.runtime import (
    append_jsonl, load_checkpoint_verified, read_jsonl_recover,
    save_checkpoint_atomic, verified_completed_status,
)


def test_jsonl_recovery_ignores_only_incomplete_final_line(tmp_path: Path):
    p = tmp_path / "history.jsonl"
    append_jsonl(p, {"step": 100, "loss": 1.0})
    append_jsonl(p, {"step": 200, "loss": 0.5})
    with p.open("ab") as fh:
        fh.write(b'{"step":300')
    assert [r["step"] for r in read_jsonl_recover(p)] == [100, 200]


def test_checkpoint_roundtrip_and_stale_provenance_rejection(tmp_path: Path):
    p = tmp_path / "latest_checkpoint.pt"
    payload = {
        "implementation_sha": "abc", "config_hash": "cfg", "job_id": "d5_HVI-low",
        "completed_step": 750, "model_state": {"w": torch.tensor([1.0])},
    }
    save_checkpoint_atomic(p, payload)
    got = load_checkpoint_verified(p, implementation_sha="abc", config_hash="cfg", job_id="d5_HVI-low")
    assert got["completed_step"] == 750
    with pytest.raises(RuntimeError, match="stale_provenance"):
        load_checkpoint_verified(p, implementation_sha="different", config_hash="cfg", job_id="d5_HVI-low")


def test_completed_status_requires_current_hashes(tmp_path: Path):
    artifact = tmp_path / "metrics.json"
    artifact.write_text('{"ok": true}\n')
    status = tmp_path / "status.json"
    from hd_pipeline.runtime import sha256_file
    status.write_text(json.dumps({
        "status": "completed", "implementation_sha": "abc", "config_hash": "cfg",
        "artifacts": {str(artifact): sha256_file(artifact)}
    }))
    assert verified_completed_status(status, implementation_sha="abc", config_hash="cfg")
    artifact.write_text('{"ok": false}\n')
    assert not verified_completed_status(status, implementation_sha="abc", config_hash="cfg")
```

- [ ] **Step 2: Run runtime tests and verify import failure**

```bash
PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_hd_pipeline_resume.py
```

Expected: FAIL because `hd_pipeline.runtime` does not exist.

- [ ] **Step 3: Implement atomic and crash-tolerant primitives**

Use `tempfile.NamedTemporaryFile(delete=False, dir=path.parent)` followed by `os.replace` for JSON/checkpoint writes. `append_jsonl` must `flush()` and `os.fsync()` after each record. `read_jsonl_recover` may ignore a malformed final non-empty line only; malformed earlier lines raise `RuntimeError("corrupt_jsonl")`.

Checkpoint serialization must use `torch.save` to a temporary file followed by `os.replace`, never directly overwrite the last valid checkpoint.

- [ ] **Step 4: Add RNG capture/restore coverage**

Append this test:

```python
def test_rng_roundtrip_restores_torch_stream():
    from hd_pipeline.runtime import capture_rng_state, restore_rng_state
    torch.manual_seed(123)
    state = capture_rng_state()
    expected = torch.rand(4)
    torch.rand(8)
    restore_rng_state(state)
    assert torch.equal(torch.rand(4), expected)
```

- [ ] **Step 5: Run focused tests**

```bash
PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_hd_pipeline_resume.py
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/hd_pipeline tests/test_hd_pipeline_resume.py
git commit -m "feat(hd-pipeline): add durable resume primitives"
```

---

### Task 3: Implement objective-specific gradient diagnostics without retaining graphs

**Files:**
- Create: `src/hd_pipeline/gradient_diagnostics.py`
- Modify: `tests/test_hd2_hvi_objective_stabilization.py`

**Interfaces:**
- Consumes: scalar `loss_sup`, scalar `loss_phys`, model parameter iterables, `lambda_weight`.
- Produces:
  - `flatten_grads(grads: tuple[torch.Tensor | None, ...]) -> torch.Tensor`
  - `objective_gradient_diagnostics(loss_sup: torch.Tensor, loss_phys: torch.Tensor, model: torch.nn.Module, lambda_weight: float) -> dict[str, float]`
  - `layerwise_gradient_diagnostics(loss_sup: torch.Tensor, loss_phys: torch.Tensor, named_parameter_groups: dict[str, list[torch.nn.Parameter]]) -> list[dict]`

- [ ] **Step 1: Write failing analytic gradient test**

```python
import torch
from hd_pipeline.gradient_diagnostics import objective_gradient_diagnostics


def test_gradient_diagnostics_detect_known_alignment_and_conflict():
    w = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float64))
    model = torch.nn.ParameterList([w])
    sup = (w[0] + 2.0 * w[1]) ** 2
    phys_aligned = 3.0 * (w[0] + 2.0 * w[1]) ** 2
    d1 = objective_gradient_diagnostics(sup, phys_aligned, model, lambda_weight=0.1)
    assert d1["cosine"] == pytest.approx(1.0, abs=1e-12)
    phys_conflict = -0.5 * (w[0] + 2.0 * w[1]) ** 2
    d2 = objective_gradient_diagnostics(sup, phys_conflict, model, lambda_weight=0.1)
    assert d2["cosine"] == pytest.approx(-1.0, abs=1e-12)
```

- [ ] **Step 2: Run and verify failure**

```bash
PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_hd2_hvi_objective_stabilization.py
```

- [ ] **Step 3: Implement diagnostics with `torch.autograd.grad`**

Use:

```python
g_sup = torch.autograd.grad(loss_sup, params, retain_graph=True, create_graph=False, allow_unused=True)
g_phys = torch.autograd.grad(loss_phys, params, retain_graph=True, create_graph=False, allow_unused=True)
```

Convert missing gradients to zeros of the matching parameter shape. Flatten, compute norms/cosine, and return Python floats detached to CPU. Do not return graph-connected tensors.

Return at least:

```text
sup_grad_norm
phys_grad_norm
raw_norm_ratio
weighted_norm_ratio
cosine
sup_dot_total
phys_dot_total
```

where `g_total = g_sup + lambda_weight * g_phys`.

- [ ] **Step 4: Add graph-release regression test**

Construct a small MLP, call the diagnostic function repeatedly in a loop, then perform a normal `loss.backward()`/optimizer step after each diagnostic; assert all parameters remain finite and no `backward through graph a second time` error occurs.

- [ ] **Step 5: Run tests**

```bash
PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_hd2_hvi_objective_stabilization.py tests/test_hd_pipeline_resume.py
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/hd_pipeline/gradient_diagnostics.py tests/test_hd2_hvi_objective_stabilization.py
git commit -m "feat(hd2): add objective-gradient diagnostics"
```

---

### Task 4: Implement resumable HD2 training, exact-vs-Hutchinson diagnostics, and deterministic selection

**Files:**
- Create: `experiments/hd2_hvi_objective_stabilization.py`
- Modify: `tests/test_hd2_hvi_objective_stabilization.py`

**Interfaces:**
- Consumes: HD1 `initialize_model`, `build_oracle`/oracle label logic, `black_scholes_vi_residual`, `fischer_burmeister`, runtime primitives, HD2 config.
- Produces:
  - `curriculum(step: int, total_steps: int, warm_start_steps: int, final_weight: float) -> tuple[float, float]`
  - `train_candidate(d: int, candidate: str, *, resume: bool = True, max_steps_override: int | None = None) -> dict`
  - `hutchinson_gradient_diagnostic(d: int, model: torch.nn.Module, step: int) -> dict`
  - `persistent_conflict(records: list[dict]) -> bool`
  - `select_hd2_candidate(metrics_rows: list[dict]) -> dict`
  - CLI phases `preflight`, `phase-a`, `screen`, `conditional`, `select`, `finalize`, `status`.

- [ ] **Step 1: Write failing curriculum and selection tests**

```python
from hd2_hvi_objective_stabilization import curriculum, select_hd2_candidate


def test_candidate_curricula_are_exact():
    assert curriculum(1999, 10000, 2000, 0.1)[0] == 0.0
    assert curriculum(9999, 10000, 2000, 0.1)[0] == pytest.approx(0.1)
    assert curriculum(4999, 10000, 5000, 0.01)[0] == 0.0
    assert curriculum(9999, 10000, 5000, 0.01)[0] == pytest.approx(0.01)


def test_selection_is_lexicographic_and_falls_back_when_price_fails():
    sup2 = {"dimension": 2, "candidate": "SUP-HD2", "price_rmse": 1.0, "gamma_rmse": 1.0, "boundary_rmse": 1.0, "pde_p95": 1.0, "fb_p95": 1.0, "integrity": True}
    sup5 = {"dimension": 5, "candidate": "SUP-HD2", "price_rmse": 1.0, "gamma_rmse": 1.0, "boundary_rmse": 1.0, "pde_p95": 1.0, "fb_p95": 1.0, "integrity": True}
    bad = [{**sup2, "candidate": "HVI-low", "price_rmse": 1.06, "pde_p95": 0.1, "fb_p95": 0.1}, {**sup5, "candidate": "HVI-low", "price_rmse": 1.06, "pde_p95": 0.1, "fb_p95": 0.1}]
    decision = select_hd2_candidate([sup2, sup5, *bad])
    assert decision["selection_status"] == "no_stabilized_candidate"
    assert decision["resolved_objective"]["candidate"] == "HVI-warm-low"
    assert decision["resolved_objective"]["fallback_objective"] is True
```

- [ ] **Step 2: Run and verify failure**

```bash
PYTHONPATH=src:experiments python -m pytest -q -p no:cacheprovider tests/test_hd2_hvi_objective_stabilization.py
```

- [ ] **Step 3: Implement candidate-independent training core by adapting HD1, not copying behavior blindly**

Preserve HD1 oracle/metric definitions. Replace HD1's hard-coded curriculum with the candidate-specific function. The optimizer loop must:

```text
load/verify checkpoint if present
restore RNG and Sobol counters
for step from completed_step to target_steps-1:
  draw exact resumed supervised minibatch
  compute supervised loss
  compute candidate lambda/epsilon
  draw physics Sobol batch only when physics active
  compute FB loss with full Hessian at d=2 and two-probe Hutchinson at d=5
  run sparse diagnostics before optimizer backward when due
  optimizer.zero_grad(set_to_none=True)
  total_loss.backward()
  optimizer.step()
  append/flush JSONL records
  save atomic checkpoint every 250 steps
```

A resume run must not restart the Sobol stream; record consumed draw counts and use `SobolEngine.fast_forward(count)` on reconstruction.

- [ ] **Step 4: Implement forced-interruption equivalence hook for tests**

`max_steps_override` stops cleanly after writing a checkpoint/status but must be labeled `interrupted_test_hook`, never `completed`. Tests then call `train_candidate(..., resume=True)` to finish. This hook is disabled from the Windows canonical CLI.

- [ ] **Step 5: Add deterministic resume test on CPU with a tiny injected test model/operator path**

Use a dependency-injected tiny training backend or a `--test-mode` internal function that runs 20 deterministic optimizer steps on CPU. Compare uninterrupted final state with `7 steps -> resume -> 20 steps`; require bitwise-equal or strict tensor equality for model parameters and identical consumed Sobol counters.

- [ ] **Step 6: Implement d=5 exact-vs-Hutchinson parameter-gradient diagnostic**

For the fixed 16-state diagnostic batch at steps `2000`, `5000`, `10000`, compute exact and two-probe physics losses from the same model/state batch, then obtain parameter gradients with `torch.autograd.grad`. Record contraction relative error, exact/Hutch gradient norm ratio, gradient cosine, time, and peak CUDA memory. Trigger the 8-probe conditional flag if `relative_error > 0.05` OR `cosine < 0.95`.

- [ ] **Step 7: Implement persistent conflict rule exactly**

Group gradient records into consecutive 500-step windows after physics activation. A pair of adjacent windows triggers conflict only when each has a majority of negative cosines, their combined median cosine is `<= -0.1`, and validation price does not improve relative to the pre-physics trend. Return the evidence window IDs in `conditional_methods.json`.

- [ ] **Step 8: Implement lexicographic selection**

A candidate is eligible only if both d=2 and d=5 satisfy:

```text
price <= 1.05 * SUP
gamma <= 1.10 * SUP
boundary <= 1.10 * SUP
pde_p95 <= 0.75 * SUP
fb_p95 <= 0.75 * SUP
integrity == true
```

Choose the eligible fixed-schedule candidate with lowest geometric mean of the d2/d5 normalized PDE and FB ratios. Conditional adaptive/conflict candidates may win only if they are strictly better on that physics criterion without worse price ratio; otherwise prefer the fixed schedule. If none pass, emit the exact fallback object from the config.

- [ ] **Step 9: Run focused HD2 tests**

```bash
PYTHONPATH=src:experiments python -m pytest -q -p no:cacheprovider tests/test_hd1_high_dimensional_pinn.py tests/test_hd2_hvi_objective_stabilization.py tests/test_hd_pipeline_resume.py
python -m compileall -q src/hd_pipeline experiments/hd2_hvi_objective_stabilization.py
```

Expected: PASS; no CUDA canonical run is performed.

- [ ] **Step 10: Commit**

```bash
git add experiments/hd2_hvi_objective_stabilization.py tests/test_hd2_hvi_objective_stabilization.py
git commit -m "feat(hd2): implement resumable HVI stabilization"
```

---

### Task 5: Implement HD3 FC model, transverse-state sampling, and structural diagnostics

**Files:**
- Create: `src/models/hd3_high_dimensional_pinn.py`
- Create: `src/physics/hd3_high_dimensional_sampling.py`
- Modify: `tests/test_hd3_high_dimensional_pinn.py`

**Interfaces:**
- Produces:
  - `FCPINN(d: int, width: int = 128, hidden_layers: int = 6)`
  - `initialize_fc_model(d: int, seed: int = 101, *, device=None, dtype=torch.float32) -> FCPINN`
  - `sample_transverse_states(n: int, d: int, *, seed: int, m_bounds: tuple[float, float], tau_bounds: tuple[float, float], transverse_scale: float, skip: int = 0, device="cpu", dtype=torch.float64) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]`
  - `permutation_consistency_rmse(model, x, tau, permutation) -> float`
  - `transverse_invariance_rmse(model, x_grouped, tau_grouped) -> float`

- [ ] **Step 1: Write failing model/sampling tests**

```python
from models.hd3_high_dimensional_pinn import FCPINN
from physics.hd3_high_dimensional_sampling import sample_transverse_states


def test_fc_model_accepts_all_declared_dimensions_and_hard_terminal():
    for d in (10, 20, 50, 100):
        model = FCPINN(d).double()
        x = torch.zeros(3, d, dtype=torch.float64)
        tau = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
        y = model(x, tau)
        assert y.shape == (3,)
        assert y[0].item() == pytest.approx(0.0, abs=0.0)
        assert torch.all(y >= 0)


def test_transverse_sampler_preserves_aggregate_log_geometric_mean():
    x, tau, m = sample_transverse_states(
        256, 50, seed=901, m_bounds=(math.log(0.5), math.log(1.5)),
        tau_bounds=(0.0025, 1.0), transverse_scale=0.35, dtype=torch.float64,
    )
    assert torch.allclose(x.mean(dim=1), m, atol=1e-12, rtol=0)
    assert torch.allclose(torch.exp(x.mean(dim=1)), torch.exp(m), atol=1e-12, rtol=0)
```

- [ ] **Step 2: Run and verify failure**

```bash
PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_hd3_high_dimensional_pinn.py
```

- [ ] **Step 3: Implement `FCPINN` with the same physical output transform**

The raw MLP is:

```text
Linear(d+1,128), Tanh
5 x [Linear(128,128), Tanh]
Linear(128,1)
```

`forward(x,tau)` returns `geometric_put_payoff(x) + tau * softplus(raw_head)` exactly like HD1.

- [ ] **Step 4: Implement aggregate-preserving Sobol sampling**

Use one scrambled Sobol engine of dimension `d+2`; one coordinate maps to `m`, `d` coordinates create `u_i in [-1,1]`, and one maps to `tau`. Compute:

```python
delta = transverse_scale * (u - u.mean(dim=1, keepdim=True))
x = m[:, None] + delta
```

Support deterministic `skip` via `SobolEngine.fast_forward(skip)` so resume can reconstruct the stream exactly.

- [ ] **Step 5: Add structural-diagnostic tests**

Verify the HD1 DeepSets model gives permutation consistency at float64 tolerance; verify the FC diagnostic returns finite values but is not assumed invariant by construction. Build grouped states with the same `m,tau` and distinct zero-mean deviations and verify the diagnostic is zero for a synthetic oracle `f(x,t)=exp(mean(x))+t`.

- [ ] **Step 6: Run focused tests and compile**

```bash
PYTHONPATH=src python -m pytest -q -p no:cacheprovider tests/test_hd3_high_dimensional_pinn.py tests/test_hd1_high_dimensional_pinn.py
python -m compileall -q src/models/hd3_high_dimensional_pinn.py src/physics/hd3_high_dimensional_sampling.py
```

- [ ] **Step 7: Commit**

```bash
git add src/models/hd3_high_dimensional_pinn.py src/physics/hd3_high_dimensional_sampling.py tests/test_hd3_high_dimensional_pinn.py
git commit -m "feat(hd3): add FC PINN and transverse sampler"
```

---

### Task 6: Implement resumable HD3 scaling/evaluation with immutable HD2 objective handoff

**Files:**
- Create: `experiments/hd3_classical_vs_structured_high_dimensional_pinn.py`
- Modify: `tests/test_hd3_high_dimensional_pinn.py`
- Modify: `tests/test_hd_pipeline_resume.py`

**Interfaces:**
- Consumes: `selection_decision.json`, HD3 config, runtime primitives, FC and HD1 DeepSets models, transverse sampler, HD1 oracle/operator utilities.
- Produces:
  - `resolve_objective(selection_path: Path, hd3_config: dict) -> dict`
  - `freeze_objective(path: Path, objective: dict) -> str`
  - `train_hd3_job(d: int, model_id: str, *, run_class: str, resume: bool = True) -> dict`
  - `evaluate_hd3_job(...) -> dict`
  - CLI phases `preflight`, `job`, `stress-job`, `finalize`, `status`.

- [ ] **Step 1: Write failing objective-freeze tests**

```python
def test_hd3_objective_fallback_is_exact_when_hd2_has_no_candidate(tmp_path):
    from hd3_classical_vs_structured_high_dimensional_pinn import resolve_objective
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"selection_status": "no_stabilized_candidate"}))
    cfg = json.loads((ROOT / "configs/hd3_classical_vs_structured_high_dimensional_pinn.json").read_text())
    got = resolve_objective(selection, cfg)
    assert got["candidate"] == "HVI-warm-low"
    assert got["warm_fraction"] == 0.5
    assert got["final_physics_weight"] == 0.01
    assert got["fallback_objective"] is True


def test_frozen_objective_cannot_change_after_canonical_started(tmp_path):
    from hd3_classical_vs_structured_high_dimensional_pinn import freeze_objective
    p = tmp_path / "resolved_objective.json"
    freeze_objective(p, {"candidate": "HVI-low", "warm_fraction": 0.2, "final_physics_weight": 0.01})
    (tmp_path / "canonical_started.flag").write_text("started")
    with pytest.raises(RuntimeError, match="objective_immutable"):
        freeze_objective(p, {"candidate": "HVI-warm-low", "warm_fraction": 0.5, "final_physics_weight": 0.01}, canonical_started=True)
```

- [ ] **Step 2: Run and verify failure**

```bash
PYTHONPATH=src:experiments python -m pytest -q -p no:cacheprovider tests/test_hd3_high_dimensional_pinn.py tests/test_hd_pipeline_resume.py
```

- [ ] **Step 3: Implement objective resolution and hash freeze**

Write `resolved_objective.json` atomically before the first HD3 canonical job. Store its SHA-256 in every checkpoint/status. A resumed job rejects a changed objective hash.

- [ ] **Step 4: Implement shared HD3 training loop**

Both models use identical supervised and physics streams, optimizer, objective schedule, two-probe Hutchinson estimator, step count, and batch fallbacks. The only architecture switch is `FC_HVI` versus `DEEPSETS_HVI`.

For canonical 6,000 steps, convert `warm_fraction` to `round(6000 * warm_fraction)`; for stress 4,000 steps use `round(4000 * warm_fraction)`. Keep final physics weight unchanged.

Checkpoint/resume behavior is identical to HD2.

- [ ] **Step 5: Implement HD3 evaluation and structural metrics**

For each run write price/Delta/Gamma-contraction/PDE/FB/boundary metrics plus permutation-consistency RMSE, transverse-invariance RMSE, training seconds, seconds/step, derivative seconds, inference seconds, peak CUDA GiB, parameter count, batch size, OOM count, and status.

100D rows must contain `run_class="stress"`; 10/20/50D rows contain `run_class="canonical"`.

- [ ] **Step 6: Add CPU resume smoke for both architecture IDs**

Use the same tiny backend approach as HD2 to verify interruption/resume does not alter final parameter state, current step, or Sobol counters for both model selectors.

- [ ] **Step 7: Run tests and compile**

```bash
PYTHONPATH=src:experiments python -m pytest -q -p no:cacheprovider tests/test_hd1_high_dimensional_pinn.py tests/test_hd2_hvi_objective_stabilization.py tests/test_hd3_high_dimensional_pinn.py tests/test_hd_pipeline_resume.py
python -m compileall -q src experiments/hd2_hvi_objective_stabilization.py experiments/hd3_classical_vs_structured_high_dimensional_pinn.py
```

- [ ] **Step 8: Commit**

```bash
git add experiments/hd3_classical_vs_structured_high_dimensional_pinn.py tests/test_hd3_high_dimensional_pinn.py tests/test_hd_pipeline_resume.py
git commit -m "feat(hd3): implement resumable high-dimensional scaling"
```

---

### Task 7: Implement the one-command Python state machine and Windows PowerShell runner

**Files:**
- Create: `experiments/hd_pipeline_windows.py`
- Create: `scripts/run_hd_pipeline_windows.ps1`
- Create: `scripts/check_hd_pipeline_status.ps1`
- Modify: `tests/test_hd_pipeline_resume.py`

**Interfaces:**
- Python phases: `preflight`, `hd2`, `hd2-select`, `hd3`, `finalize`, `status`, `all`.
- PowerShell user command:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_hd_pipeline_windows.ps1 -Phase all
```

- [ ] **Step 1: Write failing state-machine tests**

Use temporary fake job functions so tests do not run CUDA:

```python
def test_pipeline_resumes_after_middle_phase_without_rerunning_completed(tmp_path):
    from hd_pipeline_windows import run_state_machine
    calls = []
    jobs = {
        "preflight": lambda: calls.append("preflight"),
        "hd2": lambda: calls.append("hd2"),
        "hd2_select": lambda: calls.append("hd2_select"),
        "hd3": lambda: calls.append("hd3"),
        "finalize": lambda: calls.append("finalize"),
    }
    run_state_machine(tmp_path, jobs, stop_after="hd2_select")
    assert calls == ["preflight", "hd2", "hd2_select"]
    calls.clear()
    run_state_machine(tmp_path, jobs)
    assert calls == ["hd3", "finalize"]
```

Also test stale implementation/config provenance blocks resume rather than silently rerunning.

- [ ] **Step 2: Run and verify failure**

```bash
PYTHONPATH=src:experiments python -m pytest -q -p no:cacheprovider tests/test_hd_pipeline_resume.py
```

- [ ] **Step 3: Implement Python orchestration**

The scheduler priority is:

```text
preflight
HD2 mandatory diagnosis/screen
HD2 deterministic selection/fallback
HD3 10D
HD3 20D
HD3 50D
finalization
HD2 optional conditional jobs if still scientifically needed and budget-safe
HD3 100D stress if budget-safe
```

However, CAGrad/8-probe diagnostics that determine HD2 selection must run before selection when their frozen trigger is met. ReLoBRaLo remains optional and is skipped if its projected time threatens mandatory HD3 completion.

Maintain `results/hd_pipeline/run_ledger.csv` by regenerating it from verified status JSONs. Never manually append canonical rows that could duplicate after resume.

- [ ] **Step 4: Add preflight environment/provenance checks**

Preflight must verify:

```text
exact branch/implementation SHA recorded
worktree has no tracked source changes
CUDA available
GPU total memory >= 7 GiB
nvidia-smi works
focused tests pass
oracle smoke/refinement passes
exact-vs-Hutchinson d2 check passes
checkpoint write/read passes
free disk >= 5 GiB
results directories writable
```

It records Windows/Python/PyTorch/CUDA/driver/GPU/CPU/RAM/package versions. Do not reject a 4070 solely because its name string differs; enforce CUDA and memory requirements.

- [ ] **Step 5: Implement PowerShell wrapper**

`run_hd_pipeline_windows.ps1` must:

```text
resolve repo root
resolve Python path (default .\.venv-pinn\Scripts\python.exe)
create ignored logs directory
invoke Python orchestrator with Tee-Object logging
propagate exit code
trap Ctrl+C where possible
support -Phase all|preflight|hd2|hd3|finalize|status
```

Do not duplicate scientific decision logic in PowerShell; Python status/config files are the source of truth.

- [ ] **Step 6: Implement read-only status script**

Print:

```text
phase | job | d | model | status | step/target | elapsed | ETA | batch | VRAM | checkpoint age
```

plus resolved objective, 18-hour remaining budget, next runnable job, and blockers. `status` must not modify files.

- [ ] **Step 7: Add source-level PowerShell contract tests**

Since the development environment may not have Windows PowerShell, tests inspect the script text and assert it contains the accepted phase names, the Python experiment path, `$LASTEXITCODE` propagation, and no package-install commands such as `pip install`, `conda install`, or `winget`.

- [ ] **Step 8: Run focused tests**

```bash
PYTHONPATH=src:experiments python -m pytest -q -p no:cacheprovider tests/test_hd_pipeline_resume.py tests/test_hd2_hvi_objective_stabilization.py tests/test_hd3_high_dimensional_pinn.py
python -m compileall -q experiments/hd_pipeline_windows.py
```

- [ ] **Step 9: Commit**

```bash
git add experiments/hd_pipeline_windows.py scripts/run_hd_pipeline_windows.ps1 scripts/check_hd_pipeline_status.ps1 tests/test_hd_pipeline_resume.py
git commit -m "feat(hd-pipeline): add one-command Windows runner"
```

---

### Task 8: Add detailed finalization, execution instructions, and verification gate

**Files:**
- Modify: `experiments/hd2_hvi_objective_stabilization.py`
- Modify: `experiments/hd3_classical_vs_structured_high_dimensional_pinn.py`
- Modify: `experiments/hd_pipeline_windows.py`
- Create: `docs/campaign/hd2_hd3_windows_execution_instructions.md`
- Modify: `tests/test_hd_pipeline_resume.py`

**Interfaces:**
- Produces compact evidence under:

```text
results/hd2_hvi_objective_stabilization/
results/hd3_classical_vs_structured_high_dimensional_pinn/
results/hd_pipeline/
```

- [ ] **Step 1: Write failing compact-output/hash tests**

Assert the finalizer requires and hashes:

```text
HD2: environment_manifest.json, config_manifest.json, run_ledger.csv,
run_metrics.csv, training_history.csv, validation_history.csv,
gradient_diagnostics.csv, layerwise_gradient_diagnostics.csv,
hutchinson_gradient_validation.csv, conditional_methods.json,
selection_decision.json, dimension_summary.csv, artifact_hashes.json

HD3: resolved_objective.json, run_metrics.csv, training_history.csv,
validation_history.csv, structural_diagnostics.csv,
dimension_summary.csv, artifact_hashes.json

Pipeline: run_ledger.csv, environment_manifest.json, final_status.json,
artifact_hashes.json
```

Corrupt one generated file after hashing and assert verification fails.

- [ ] **Step 2: Implement deterministic compact regeneration from JSONL/status records**

CSV outputs are rewritten atomically from durable runtime records so rerunning finalization cannot duplicate rows. Sort keys deterministically by phase, dimension, model, candidate, seed, and step before writing.

- [ ] **Step 3: Write the Windows execution guide**

The guide must contain exactly these normal-user commands:

```powershell
git fetch origin
git switch exp/hd2-hvi-objective-stabilization
git pull --ff-only
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_hd_pipeline_windows.ps1 -Phase preflight
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_hd_pipeline_windows.ps1 -Phase all
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_hd_pipeline_windows.ps1 -Phase status
```

Explain that after interruption/reboot the user reruns the same `-Phase all` command. Document local runtime paths, checkpoint interval, what is committed versus ignored, how to inspect `run_ledger.csv`, and how to make a separate result commit after completion. Never ask for or print a GitHub token.

- [ ] **Step 4: Run complete focused verification**

```bash
PYTHONPATH=src:experiments python -m pytest -q -p no:cacheprovider tests/test_hd1_high_dimensional_pinn.py tests/test_hd2_hvi_objective_stabilization.py tests/test_hd3_high_dimensional_pinn.py tests/test_hd_pipeline_resume.py
python -m compileall -q src tests experiments
```

Then run the repository harness checks available on the branch without CUDA canonical training:

```bash
bash scripts/verify_harness.sh
git diff --check
```

- [ ] **Step 5: Inspect branch diff against HD1 result commit**

```bash
git diff --stat 5b536bc22b800af2a6415fc2a0af6e1f0a0d4922..HEAD
git diff --check 5b536bc22b800af2a6415fc2a0af6e1f0a0d4922..HEAD
```

Confirm no HD1 compact result file was modified and no checkpoint/log/oracle array is staged.

- [ ] **Step 6: Commit implementation-ready handoff**

```bash
git add docs/campaign/hd2_hd3_windows_execution_instructions.md experiments tests src scripts configs
git commit -m "docs(hd2-hd3): add laptop execution handoff"
```

- [ ] **Step 7: Push branch without claiming laptop results**

Push `exp/hd2-hvi-objective-stabilization`. The completion report must explicitly state that CUDA canonical/stress jobs were not run by the implementation session and that evidence begins only after the user's RTX 4070 Laptop run.

---

## Plan Self-Review

- **Spec coverage:** HD2 stabilization, gradient geometry, exact-vs-Hutchinson gradient validation, deterministic selection/fallback, HD3 FC/DeepSets scaling, aggregate-preserving sampling, 100D stress, 18-hour budget, one-command execution, resume, detailed logs, compact hashes, and claim limits all map to explicit tasks above.
- **Placeholder scan:** no `TBD`, `TODO`, or unspecified implementation step remains.
- **Type/interface consistency:** runtime, diagnostic, HD2, HD3, and orchestration interfaces are named once and reused consistently.
- **Scope:** no unrelated campaign method, report-writing, blind-holdout access, or architecture sweep is included.
