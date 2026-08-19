# HD3 Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic post-hoc HD3 visualization pipeline that reuses the preserved trained checkpoints and CN/PSOR oracles, adds an independent CRR binomial cross-check, exports auditable derived data, and generates publication figures without retraining.

**Architecture:** Keep the numerical reference solver independent in `src/solvers/crr_american.py`; keep HD3-specific provenance, checkpoint loading, dense inference, derived-data export, and figure orchestration in `experiments/hd3_visualization_export.py`. Tests are split between pure CRR numerics and CPU-safe visualization/provenance behavior so CI never needs the canonical CUDA runtime assets.

**Tech Stack:** Python 3.11, NumPy, SciPy, PyTorch, matplotlib, pytest, existing HD3 model/oracle/runtime helpers.

**Spec:** `docs/superpowers/specs/2026-08-19-hd3-visualization-design.md`

## Global Constraints

- No optimizer steps, retraining, fine-tuning, checkpoint mutation, or parameter updates.
- No blind-holdout access.
- Canonical dimensions are 10, 20, and 50; dimension 100 is stress evidence.
- Training implementation SHA is `6c1bd658098698b69d49415c1ac1f0cf280d3c30`.
- Canonical compact-evidence commit is `951c6a6ad67f1eeedaae5cf4f80ea597909d2a1e`.
- Checkpoints are verified against their training implementation SHA, config hash, resolved-objective hash, job ID, dimension, model ID, and exact completed-step budget; visualization HEAD is not used as the training-provenance identity.
- Existing `.pt` and `.npz` runtime artifacts stay ignored and are never committed through this feature.
- CRR is an independent cross-check; CN/PSOR remains the primary numerical oracle.
- No blind-holdout path or API is imported by the exporter.
- `optimizer_steps=0`, `retraining=false`, and `blind_holdout_access=false` must be written to `visualization_manifest.json`.

---

### Task 1: Independent CRR American-put solver

**Files:**
- Create: `src/solvers/crr_american.py`
- Create: `tests/test_crr_american.py`

**Interfaces:**
- Produces: `crr_american_put_price(spot: float, strike: float, maturity: float, rate: float, dividend_yield: float, volatility: float, steps: int) -> float`
- Produces: `crr_american_put_curve(spots: np.ndarray, *, strike: float, maturity: float, rate: float, dividend_yield: float, volatility: float, steps: int) -> np.ndarray`
- Produces: `crr_convergence_rows(spots: np.ndarray, cnpsor_values: np.ndarray, *, strike: float, maturity: float, rate: float, dividend_yield: float, volatility: float, depths: tuple[int, ...]) -> list[dict[str, float | int | bool]]`

- [ ] **Step 1: Write the failing pricing/bounds tests**

```python
import numpy as np
import pytest

from solvers.crr_american import crr_american_put_curve, crr_american_put_price


def test_crr_american_put_respects_intrinsic_and_basic_bounds():
    value = crr_american_put_price(
        spot=0.8,
        strike=1.0,
        maturity=1.0,
        rate=0.05,
        dividend_yield=0.02,
        volatility=0.25,
        steps=500,
    )
    assert np.isfinite(value)
    assert value >= pytest.approx(0.2, abs=1e-12)
    assert value <= 1.0


def test_crr_curve_matches_scalar_prices():
    spots = np.array([0.7, 1.0, 1.3])
    curve = crr_american_put_curve(
        spots,
        strike=1.0,
        maturity=1.0,
        rate=0.05,
        dividend_yield=0.02,
        volatility=0.25,
        steps=300,
    )
    expected = np.array([
        crr_american_put_price(
            spot=float(s), strike=1.0, maturity=1.0, rate=0.05,
            dividend_yield=0.02, volatility=0.25, steps=300,
        )
        for s in spots
    ])
    np.testing.assert_allclose(curve, expected, rtol=0.0, atol=1e-12)
```

- [ ] **Step 2: Run the focused tests and verify they fail because the module does not exist**

Run:

```bash
python -m pytest tests/test_crr_american.py -q
```

Expected: import failure for `solvers.crr_american`.

- [ ] **Step 3: Implement the minimal vector-safe CRR solver**

```python
from __future__ import annotations

import math
import numpy as np


def _validate_inputs(spot, strike, maturity, volatility, steps):
    if spot <= 0 or strike <= 0:
        raise ValueError("spot and strike must be positive")
    if maturity < 0 or volatility < 0:
        raise ValueError("maturity and volatility must be nonnegative")
    if steps < 1:
        raise ValueError("steps must be positive")


def crr_american_put_price(spot, strike, maturity, rate, dividend_yield, volatility, steps):
    _validate_inputs(spot, strike, maturity, volatility, steps)
    if maturity == 0:
        return float(max(strike - spot, 0.0))
    dt = maturity / steps
    if volatility == 0:
        # deterministic path; compare discounted continuation with exercise at each step
        values = []
        for n in range(steps + 1):
            t = n * dt
            s = spot * math.exp((rate - dividend_yield) * t)
            values.append(math.exp(-rate * t) * max(strike - s, 0.0))
        return float(max(values))
    up = math.exp(volatility * math.sqrt(dt))
    down = 1.0 / up
    growth = math.exp((rate - dividend_yield) * dt)
    probability = (growth - down) / (up - down)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("invalid CRR risk-neutral probability")
    j = np.arange(steps + 1, dtype=float)
    terminal_spots = spot * up ** j * down ** (steps - j)
    values = np.maximum(strike - terminal_spots, 0.0)
    discount = math.exp(-rate * dt)
    for n in range(steps - 1, -1, -1):
        continuation = discount * (probability * values[1:n + 2] + (1.0 - probability) * values[:n + 1])
        j = np.arange(n + 1, dtype=float)
        node_spots = spot * up ** j * down ** (n - j)
        exercise = np.maximum(strike - node_spots, 0.0)
        values = np.maximum(exercise, continuation)
    return float(values[0])


def crr_american_put_curve(spots, **kwargs):
    spots = np.asarray(spots, dtype=float)
    return np.asarray([crr_american_put_price(float(s), **kwargs) for s in spots], dtype=float)
```

- [ ] **Step 4: Run the focused tests and make them pass**

Run:

```bash
python -m pytest tests/test_crr_american.py -q
```

Expected: PASS.

- [ ] **Step 5: Add convergence-row tests**

```python
from solvers.crr_american import crr_convergence_rows


def test_crr_convergence_rows_report_reference_and_previous_depth_errors():
    spots = np.array([0.8, 1.0, 1.2])
    reference = crr_american_put_curve(
        spots, strike=1.0, maturity=1.0, rate=0.05,
        dividend_yield=0.02, volatility=0.25, steps=1200,
    )
    rows = crr_convergence_rows(
        spots, reference,
        strike=1.0, maturity=1.0, rate=0.05,
        dividend_yield=0.02, volatility=0.25,
        depths=(100, 200, 400),
    )
    assert [row["depth"] for row in rows] == [100, 200, 400]
    assert rows[0]["previous_depth_rmse"] is None
    assert rows[1]["previous_depth_rmse"] is not None
    assert rows[-1]["cnpsor_rmse"] <= rows[0]["cnpsor_rmse"] * 2.0
```

- [ ] **Step 6: Implement `crr_convergence_rows` with explicit RMSE/max metrics and deterministic pass fields**

```python
def crr_convergence_rows(spots, cnpsor_values, *, strike, maturity, rate, dividend_yield, volatility, depths):
    spots = np.asarray(spots, dtype=float)
    reference = np.asarray(cnpsor_values, dtype=float)
    rows = []
    previous = None
    for depth in depths:
        curve = crr_american_put_curve(
            spots, strike=strike, maturity=maturity, rate=rate,
            dividend_yield=dividend_yield, volatility=volatility, steps=int(depth),
        )
        diff = curve - reference
        previous_rmse = None if previous is None else float(np.sqrt(np.mean((curve - previous) ** 2)))
        rows.append({
            "depth": int(depth),
            "cnpsor_rmse": float(np.sqrt(np.mean(diff ** 2))),
            "cnpsor_max_abs": float(np.max(np.abs(diff))),
            "previous_depth_rmse": previous_rmse,
        })
        previous = curve
    return rows
```

- [ ] **Step 7: Run CRR tests**

Run:

```bash
python -m pytest tests/test_crr_american.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit the CRR solver task**

```bash
git add src/solvers/crr_american.py tests/test_crr_american.py
git commit -m "feat(hd3): add independent CRR American reference"
```

---

### Task 2: Visualization-specific checkpoint provenance and CPU-safe model loading

**Files:**
- Create: `experiments/hd3_visualization_export.py`
- Create: `tests/test_hd3_visualization_export.py`

**Interfaces:**
- Produces: `CheckpointExpectation` dataclass with `training_sha`, `config_hash`, `objective_hash`, `job_id`, `dimension`, `model_id`, `completed_steps`, `run_class`.
- Produces: `validate_checkpoint_metadata(checkpoint: dict, expected: CheckpointExpectation) -> None`.
- Produces: `load_completed_model_for_visualization(path: Path, expected: CheckpointExpectation, cfg: dict, device: torch.device) -> tuple[torch.nn.Module, dict]`.
- Reuses existing HD3 `_initialize_model(model_id, d, cfg, device)` but never calls a training entry point or creates an optimizer.

- [ ] **Step 1: Write failing provenance tests for every required rejection**

```python
import copy
import pytest

from hd3_visualization_export import CheckpointExpectation, validate_checkpoint_metadata


def expectation():
    return CheckpointExpectation(
        training_sha="train-sha",
        config_hash="cfg-hash",
        objective_hash="obj-hash",
        job_id="canonical_d10_FC_HVI",
        dimension=10,
        model_id="FC_HVI",
        completed_steps=6000,
        run_class="canonical",
    )


def valid_checkpoint_metadata():
    return {
        "implementation_sha": "train-sha",
        "config_hash": "cfg-hash",
        "resolved_objective_hash": "obj-hash",
        "job_id": "canonical_d10_FC_HVI",
        "dimension": 10,
        "model_id": "FC_HVI",
        "completed_step": 6000,
    }


@pytest.mark.parametrize("field,bad", [
    ("implementation_sha", "wrong"),
    ("config_hash", "wrong"),
    ("resolved_objective_hash", "wrong"),
    ("job_id", "wrong"),
    ("dimension", 20),
    ("model_id", "DEEPSETS_HVI"),
    ("completed_step", 5999),
])
def test_checkpoint_provenance_rejects_mismatch(field, bad):
    payload = copy.deepcopy(valid_checkpoint_metadata())
    payload[field] = bad
    with pytest.raises(RuntimeError):
        validate_checkpoint_metadata(payload, expectation())
```

- [ ] **Step 2: Run the tests and verify import failure**

Run:

```bash
python -m pytest tests/test_hd3_visualization_export.py -q
```

Expected: import failure for the new exporter.

- [ ] **Step 3: Add exporter import setup, constants, dataclass, and strict metadata validator**

```python
TRAINING_IMPLEMENTATION_SHA = "6c1bd658098698b69d49415c1ac1f0cf280d3c30"
CANONICAL_EVIDENCE_SHA = "951c6a6ad67f1eeedaae5cf4f80ea597909d2a1e"

@dataclass(frozen=True)
class CheckpointExpectation:
    training_sha: str
    config_hash: str
    objective_hash: str
    job_id: str
    dimension: int
    model_id: str
    completed_steps: int
    run_class: str


def validate_checkpoint_metadata(checkpoint, expected):
    required = {
        "implementation_sha": expected.training_sha,
        "config_hash": expected.config_hash,
        "resolved_objective_hash": expected.objective_hash,
        "job_id": expected.job_id,
        "dimension": expected.dimension,
        "model_id": expected.model_id,
        "completed_step": expected.completed_steps,
    }
    mismatches = {k: (checkpoint.get(k), v) for k, v in required.items() if checkpoint.get(k) != v}
    if mismatches:
        raise RuntimeError(f"checkpoint_provenance_mismatch:{mismatches}")
```

- [ ] **Step 4: Add a CPU-safe synthetic load test that verifies parameters do not change**

```python
def test_visualization_loader_loads_state_without_parameter_mutation(tmp_path, monkeypatch):
    cfg = minimal_hd3_cfg()
    device = torch.device("cpu")
    model = export_module._initialize_model("FC_HVI", 10, cfg, device)
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    checkpoint = valid_checkpoint_metadata() | {"model_state": state}
    path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, path)

    loaded, loaded_checkpoint = load_completed_model_for_visualization(path, expectation(), cfg, device)
    assert not loaded.training
    for key, tensor in state.items():
        torch.testing.assert_close(loaded.state_dict()[key], tensor)
    assert loaded_checkpoint["completed_step"] == 6000
```

- [ ] **Step 5: Implement the visualization loader without optimizer construction**

```python
def load_completed_model_for_visualization(path, expected, cfg, device):
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    validate_checkpoint_metadata(checkpoint, expected)
    model = _initialize_model(expected.model_id, expected.dimension, cfg, torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    model.to(device)
    return model, checkpoint
```

- [ ] **Step 6: Add a parameter-hash helper and test that inference leaves state unchanged**

```python
def state_dict_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def infer_prices(model, x, tau, batch_size=4096):
    before = state_dict_digest(model)
    chunks = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            chunks.append(model(x[start:start + batch_size], tau[start:start + batch_size]).detach().cpu())
    if state_dict_digest(model) != before:
        raise RuntimeError("model_state_mutated_during_visualization")
    return torch.cat(chunks).numpy()
```

- [ ] **Step 7: Run the provenance/loader tests**

Run:

```bash
python -m pytest tests/test_hd3_visualization_export.py -q
```

Expected: PASS for metadata and synthetic model-loading tests.

- [ ] **Step 8: Commit the provenance/loading task**

```bash
git add experiments/hd3_visualization_export.py tests/test_hd3_visualization_export.py
git commit -m "feat(hd3): add visualization checkpoint loader"
```

---

### Task 3: Reference audit and auditable derived-data export

**Files:**
- Modify: `experiments/hd3_visualization_export.py`
- Modify: `tests/test_hd3_visualization_export.py`

**Interfaces:**
- Produces: `effective_contract(d: int, cfg: dict) -> dict[str, float]` using existing `effective_geometric_basket_parameters`.
- Produces: `select_crr_depth(rows: list[dict]) -> tuple[int, bool]` with thresholds RMSE vs CN/PSOR `<=1e-4` and previous-depth RMSE `<=5e-5`.
- Produces: `export_reference_agreement(...) -> dict[int, int]` writing `reference_solver_agreement.csv`.
- Produces: `export_diagonal_slices(...)`, `export_error_surfaces(...)`, `export_exercise_boundaries(...)`, and `export_source_inventory(...)`.

- [ ] **Step 1: Add failing CRR-depth-selection tests**

```python
def test_select_crr_depth_chooses_smallest_passing_depth():
    rows = [
        {"depth": 500, "cnpsor_rmse": 2e-4, "previous_depth_rmse": None},
        {"depth": 1000, "cnpsor_rmse": 8e-5, "previous_depth_rmse": 7e-5},
        {"depth": 2000, "cnpsor_rmse": 6e-5, "previous_depth_rmse": 4e-5},
        {"depth": 4000, "cnpsor_rmse": 5e-5, "previous_depth_rmse": 2e-5},
    ]
    assert select_crr_depth(rows) == (2000, True)


def test_select_crr_depth_falls_back_to_largest_depth_and_marks_failure():
    rows = [
        {"depth": 500, "cnpsor_rmse": 3e-4, "previous_depth_rmse": None},
        {"depth": 4000, "cnpsor_rmse": 2e-4, "previous_depth_rmse": 9e-5},
    ]
    assert select_crr_depth(rows) == (4000, False)
```

- [ ] **Step 2: Implement the deterministic selection rule**

```python
def select_crr_depth(rows):
    for row in rows:
        prev = row["previous_depth_rmse"]
        if prev is not None and row["cnpsor_rmse"] <= 1e-4 and prev <= 5e-5:
            return int(row["depth"]), True
    return int(rows[-1]["depth"]), False
```

- [ ] **Step 3: Add a failing diagonal-grid test proving high-dimensional inputs are the symmetric diagonal**

```python
def test_diagonal_states_have_equal_log_prices():
    x, tau, spots = diagonal_states(d=20, tau_value=0.5, count=5, device=torch.device("cpu"))
    assert x.shape == (5, 20)
    torch.testing.assert_close(x, x[:, :1].repeat(1, 20))
    np.testing.assert_allclose(np.exp(x[:, 0].numpy()), spots)
    np.testing.assert_allclose(tau.numpy(), np.full(5, 0.5))
```

- [ ] **Step 4: Implement `diagonal_states` and CN/PSOR interpolation through the existing oracle grid**

```python
def diagonal_states(d, tau_value, count, device):
    spots = np.linspace(0.5, 1.5, count)
    log_spots = torch.tensor(np.log(spots), dtype=torch.float32, device=device)
    x = log_spots[:, None].repeat(1, d)
    tau = torch.full((count,), float(tau_value), dtype=torch.float32, device=device)
    return x, tau, spots
```

Use the existing HD1/HD3 `interp2`/oracle fields rather than reimplementing interpolation.

- [ ] **Step 5: Add a temporary-directory export test for exact filenames and 100D stress labels**

```python
def test_export_metadata_labels_100d_as_stress(tmp_path):
    rows = build_minimal_slice_rows_for_test()
    write_csv(tmp_path / "diagonal_slices.csv", rows)
    loaded = list(csv.DictReader((tmp_path / "diagonal_slices.csv").open()))
    stress = [r for r in loaded if int(r["dimension"]) == 100]
    assert stress
    assert {r["run_class"] for r in stress} == {"stress"}
```

- [ ] **Step 6: Implement source inventory and SHA-256 hashing**

The inventory must include relative path, size, file SHA-256, dimension/model/run class, checkpoint provenance fields, completed step, and oracle refinement RMSE. Missing assets raise `FileNotFoundError` naming the exact path.

- [ ] **Step 7: Implement reference-agreement export for dimensions 10, 20, 50, 100**

Use diagonal S grid of 401 points at `tau=1.0`, CRR depths `(500, 1000, 2000, 4000)`, and existing dimension-specific `sigma_g`/`q_g`. Write one row per `(dimension, depth)` and return selected depth/pass status by dimension.

- [ ] **Step 8: Implement diagonal-slice export**

For each dimension 10/20/50/100 and `tau in (1.0, 0.5, 0.1)`, write 401 rows with fields:

```text
dimension,run_class,tau,S,cnpsor,crr,fc_hvi,deepsets_hvi,fc_abs_error,deepsets_abs_error,crr_depth,reference_agreement_passed
```

Dimension 100 is `stress`; others are `canonical`.

- [ ] **Step 9: Implement canonical error-surface export**

For dimensions 10/20/50 only, write one CSV per dimension on a `201 x 101` grid with:

```text
dimension,tau,S,cnpsor,fc_hvi,deepsets_hvi,fc_abs_error,deepsets_abs_error
```

Inference is batched; CN/PSOR values are interpolated from the existing oracle.

- [ ] **Step 10: Implement exercise-boundary export using the frozen gap tolerance**

Use 128 tau points, a dense diagonal S grid on `[0.5, 1.2]`, and the existing HD3 tolerance `V - payoff <= 1e-5`. Export CN/PSOR, FC-HVI, and DeepSets-HVI boundaries for canonical dimensions only.

- [ ] **Step 11: Run the export-focused tests**

Run:

```bash
python -m pytest tests/test_hd3_visualization_export.py -q
```

Expected: PASS.

- [ ] **Step 12: Commit the derived-data export task**

```bash
git add experiments/hd3_visualization_export.py tests/test_hd3_visualization_export.py
git commit -m "feat(hd3): export auditable visualization data"
```

---

### Task 4: Publication figure generation

**Files:**
- Modify: `experiments/hd3_visualization_export.py`
- Modify: `tests/test_hd3_visualization_export.py`

**Interfaces:**
- Produces: `build_figure_metrics(...) -> Path`.
- Produces: `plot_scaling_summary(...)`, `plot_price_slices(...)`, `plot_error_heatmaps(...)`, `plot_structural_diagnostics(...)`, `plot_exercise_boundaries(...)`, and `plot_100d_stress_slice(...)`.
- All plotting functions consume exported CSVs instead of re-running inference.

- [ ] **Step 1: Add a failing smoke test that figure functions create the declared files from tiny synthetic CSVs**

```python
def test_plotting_smoke_creates_all_declared_pngs(tmp_path):
    data_dir = tmp_path / "results" / "hd3_visualization"
    figure_dir = tmp_path / "figures" / "hd3"
    write_synthetic_visualization_csvs(data_dir)
    generate_all_figures(data_dir=data_dir, figure_dir=figure_dir)
    expected = {
        "hd3_scaling_summary.png",
        "hd3_price_slices.png",
        "hd3_error_heatmaps.png",
        "hd3_structural_diagnostics.png",
        "hd3_exercise_boundaries.png",
        "hd3_100d_stress_slice.png",
    }
    assert expected == {p.name for p in figure_dir.glob("*.png")}
    assert all((figure_dir / name).stat().st_size > 0 for name in expected)
```

- [ ] **Step 2: Implement `figure_metrics.csv` extraction from committed `dimension_summary.csv` and `structural_diagnostics.csv`**

The scaling rows must include `normalized_price_rmse`, `delta_rmse`, `trusted_pde_p95`, and `training_seconds`. Structural rows must include `permutation_consistency_rmse` and `transverse_invariance_rmse`. Preserve `run_class` so 100D is visibly stress.

- [ ] **Step 3: Implement scaling and structural figures**

Use matplotlib with separate axes/panels, no seaborn. Plot FC-HVI and DeepSets-HVI as distinct line/marker series. Mark 100D stress distinctly and annotate `stress`. Use logarithmic y-scale for permutation-consistency RMSE because DeepSets is approximately `1e-8` while FC is approximately `1e-3`.

- [ ] **Step 4: Implement canonical price-slice figure**

Read `diagonal_slices.csv`, filter `tau=1.0` and dimensions 10/20/50, then draw CN/PSOR, selected CRR, FC-HVI, and DeepSets-HVI. Add a vertical strike line at `S=1`. Use one shared figure-level legend only if direct labeling is not readable.

- [ ] **Step 5: Implement error heatmaps with one common normalization**

Read all three canonical surface CSVs, calculate a common `vmax` across FC and DeepSets errors, and draw a 2 x 3 matrix. Add one shared colorbar. Do not independently normalize panels.

- [ ] **Step 6: Implement exercise-boundary and 100D stress figures**

Exercise-boundary panels show CN/PSOR, FC-HVI, and DeepSets-HVI for 10/20/50D. The 100D figure reads only stress rows at `tau=1.0` and labels the title `100D stress`.

- [ ] **Step 7: Run plotting smoke tests**

Run:

```bash
python -m pytest tests/test_hd3_visualization_export.py -q
```

Expected: PASS using a non-interactive matplotlib backend.

- [ ] **Step 8: Commit the figure task**

```bash
git add experiments/hd3_visualization_export.py tests/test_hd3_visualization_export.py
git commit -m "feat(hd3): generate publication visualization figures"
```

---

### Task 5: CLI orchestration, manifest, and end-to-end integrity tests

**Files:**
- Modify: `experiments/hd3_visualization_export.py`
- Modify: `tests/test_hd3_visualization_export.py`

**Interfaces:**
- CLI modes: `--all`, `--reference-only`, `--data-only`, `--figures-only`.
- Produces `results/hd3_visualization/visualization_manifest.json`.
- Main user path: `python experiments/hd3_visualization_export.py --all`.

- [ ] **Step 1: Add failing CLI argument tests**

```python
@pytest.mark.parametrize("mode", ["--all", "--reference-only", "--data-only", "--figures-only"])
def test_cli_accepts_declared_modes(monkeypatch, mode):
    parser = build_parser()
    args = parser.parse_args([mode])
    assert sum(bool(getattr(args, name)) for name in ("all", "reference_only", "data_only", "figures_only")) == 1
```

- [ ] **Step 2: Implement mutually exclusive CLI modes and fail-fast asset checks**

`--all` requires all 8 checkpoints and 4 oracle files. `--reference-only` requires only the 4 oracle files. `--figures-only` requires the derived CSVs. Error messages include the missing relative path.

- [ ] **Step 3: Add a failing manifest-content test**

```python
def test_manifest_records_zero_training_and_no_blind_holdout(tmp_path):
    path = write_visualization_manifest_for_test(tmp_path)
    payload = json.loads(path.read_text())
    assert payload["optimizer_steps"] == 0
    assert payload["retraining"] is False
    assert payload["blind_holdout_access"] is False
    assert payload["training_implementation_sha"] == TRAINING_IMPLEMENTATION_SHA
    assert payload["canonical_evidence_commit"] == CANONICAL_EVIDENCE_SHA
```

- [ ] **Step 4: Implement manifest generation**

Record current visualization Git SHA separately from training/evidence SHA, config hash, objective hash, selected CRR depths and pass flags, inference device, exact grid sizes, declared output files, and integrity booleans.

- [ ] **Step 5: Add a CPU-only synthetic end-to-end export test**

Monkeypatch asset locations to a temporary directory with tiny synthetic checkpoints/oracles and reduced grid sizes. Run the exporter orchestration through data export and figure generation. Snapshot model state digests before and after; assert equality and assert no `torch.optim` object is constructed by the exporter module.

- [ ] **Step 6: Run all new tests**

Run:

```bash
python -m pytest tests/test_crr_american.py tests/test_hd3_visualization_export.py -q
```

Expected: PASS.

- [ ] **Step 7: Run compile checks**

Run:

```bash
python -m compileall src/solvers/crr_american.py experiments/hd3_visualization_export.py tests/test_crr_american.py tests/test_hd3_visualization_export.py
```

Expected: no syntax errors.

- [ ] **Step 8: Run the existing HD3-focused tests to detect regressions**

Run:

```bash
python -m pytest tests/test_hd3_classical_vs_structured_high_dimensional_pinn.py -q
```

If the repository uses a differently named HD3 test module, locate the existing HD3 test file and run that exact module instead; do not skip the regression check.

- [ ] **Step 9: Commit CLI/integrity completion**

```bash
git add experiments/hd3_visualization_export.py tests/test_hd3_visualization_export.py
git commit -m "feat(hd3): finalize visualization export pipeline"
```

---

### Task 6: Canonical laptop execution and evidence commit

**Files:**
- Generated locally: `results/hd3_visualization/*.csv`, `results/hd3_visualization/*.json`
- Generated locally: `figures/hd3/*.png`
- No canonical `.pt` or `.npz` file is added to Git.

**Interfaces:**
- Consumes the preserved runtime assets already present on the user's RTX 4070 worktree.
- Produces the finalized derived evidence package and publication figures.

- [ ] **Step 1: Pull the implementation commits into the existing HD3 worktree without deleting runtime artifacts**

Use a normal fast-forward/pull path only after checking `git status`; never `git clean`, `reset --hard`, or re-create the worktree because the ignored checkpoint/oracle assets are valuable local evidence.

- [ ] **Step 2: Run the unit/regression suite in the known working Python environment**

```powershell
& $PythonExe -m pytest tests/test_crr_american.py tests/test_hd3_visualization_export.py -q
& $PythonExe -m compileall .\src\solvers\crr_american.py .\experiments\hd3_visualization_export.py
```

- [ ] **Step 3: Run the full post-processing exporter**

```powershell
& $PythonExe .\experiments\hd3_visualization_export.py --all
```

Expected behavior: loads existing checkpoints/oracles, performs reference cross-check and inference only, produces all declared CSV/JSON/PNG outputs, and reports no training.

- [ ] **Step 4: Audit the manifest and reference agreement before committing derived evidence**

Check:

```powershell
Get-Content .\results\hd3_visualization\visualization_manifest.json
Import-Csv .\results\hd3_visualization\reference_solver_agreement.csv | Format-Table
```

Required: `optimizer_steps=0`, `retraining=false`, `blind_holdout_access=false`; selected CRR depth/pass status must be explicit for each dimension.

- [ ] **Step 5: Inspect staged scope before commit**

```powershell
git status --short
git add results/hd3_visualization figures/hd3
git diff --cached --check
git diff --cached --stat
```

Do not force-add any `.pt` or `.npz` file.

- [ ] **Step 6: Commit derived evidence**

```powershell
git commit -m "results(hd3): add visualization evidence and figures"
```

- [ ] **Step 7: Push explicitly to the correct repository, never the known-misconfigured local `origin`**

```powershell
git -c http.version=HTTP/1.1 push https://github.com/zihan-liang/american-option-risk-surfaces.git HEAD:exp/hd2-hvi-objective-stabilization
```

- [ ] **Step 8: Verify remote PR head and leave PR draft/unmerged**

Confirm PR #3 points at the new results commit. Update the stale PR description only after the canonical visualization evidence has been inspected; do not merge automatically.
