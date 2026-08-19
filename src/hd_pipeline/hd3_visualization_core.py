"""Read-only HD3 visualization provenance, reference, and data-export core.

The exporter is deliberately read-only with respect to trained model evidence:
it reconstructs the frozen HD3 architectures, loads completed checkpoint model
states, and performs inference only. It never creates an optimizer or imports
campaign blind-holdout helpers.
"""
from __future__ import annotations

import hashlib
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
EXPERIMENTS = ROOT / "experiments"
for _path in (SRC, EXPERIMENTS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

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


def validate_checkpoint_metadata(
    checkpoint: dict[str, Any], expected: CheckpointExpectation
) -> None:
    """Reject any checkpoint that is not the exact completed HD3 job expected."""
    required: dict[str, Any] = {
        "implementation_sha": expected.training_sha,
        "config_hash": expected.config_hash,
        "resolved_objective_hash": expected.objective_hash,
        "job_id": expected.job_id,
        "dimension": expected.dimension,
        "model_id": expected.model_id,
        "run_class": expected.run_class,
        "completed_step": expected.completed_steps,
        "target_steps": expected.completed_steps,
    }
    mismatches = {
        key: {"actual": checkpoint.get(key), "expected": value}
        for key, value in required.items()
        if checkpoint.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"checkpoint_provenance_mismatch:{mismatches}")
    if "model_state" not in checkpoint:
        raise RuntimeError("checkpoint_provenance_mismatch:missing_model_state")


def _hd3_module():
    return importlib.import_module("hd3_classical_vs_structured_high_dimensional_pinn")


def _initialize_model(
    model_id: str, d: int, cfg: dict[str, Any], device: torch.device
) -> torch.nn.Module:
    return _hd3_module()._initialize_model(model_id, d, cfg, device)


def load_completed_model_for_visualization(
    path: Path,
    expected: CheckpointExpectation,
    cfg: dict[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load a completed model state without restoring any training machinery."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError("checkpoint_provenance_mismatch:checkpoint_not_mapping")
    validate_checkpoint_metadata(checkpoint, expected)
    model = _initialize_model(
        expected.model_id, expected.dimension, cfg, torch.device("cpu")
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    model.to(device)
    return model, checkpoint


def state_dict_digest(model: torch.nn.Module) -> str:
    """Stable SHA-256 of all model buffers/parameters for mutation detection."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        try:
            return next(model.buffers()).device
        except StopIteration:
            return torch.device("cpu")


def infer_prices(
    model: torch.nn.Module,
    x: torch.Tensor,
    tau: torch.Tensor,
    batch_size: int = 4096,
) -> np.ndarray:
    """Batched read-only inference, guarded by a before/after state digest."""
    if x.ndim != 2 or tau.ndim != 1 or len(x) != len(tau):
        raise ValueError("x/tau shapes are incompatible")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if len(x) == 0:
        return np.empty((0,), dtype=float)

    before = state_dict_digest(model)
    device = _model_device(model)
    outputs: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(x), int(batch_size)):
            xb = x[start : start + batch_size].to(device=device, dtype=torch.float32)
            tb = tau[start : start + batch_size].to(device=device, dtype=torch.float32)
            prediction = model(xb, tb).reshape(-1)
            outputs.append(prediction.detach().cpu())
    after = state_dict_digest(model)
    if after != before:
        raise RuntimeError("model_state_mutated_during_visualization")
    return torch.cat(outputs).numpy()

# --- Numerical-reference and derived-data helpers ---
import csv
import json

HD3_RESULT_DIR = ROOT / "results" / "hd3_classical_vs_structured_high_dimensional_pinn"
HD3_RUNTIME_DIR = HD3_RESULT_DIR / "runtime"
HD3_ORACLE_DIR = HD3_RUNTIME_DIR / "oracles"
VISUALIZATION_DATA_DIR = ROOT / "results" / "hd3_visualization"
FIGURE_DIR = ROOT / "figures" / "hd3"
CONFIG_PATH = ROOT / "configs" / "hd3_classical_vs_structured_high_dimensional_pinn.json"
CONFIG_MANIFEST_PATH = HD3_RESULT_DIR / "config_manifest.json"
RESOLVED_OBJECTIVE_PATH = HD3_RESULT_DIR / "resolved_objective.json"
DIMENSION_SUMMARY_PATH = HD3_RESULT_DIR / "dimension_summary.csv"
STRUCTURAL_DIAGNOSTICS_PATH = HD3_RESULT_DIR / "structural_diagnostics.csv"
CANONICAL_DIMENSIONS = (10, 20, 50)
STRESS_DIMENSIONS = (100,)
ALL_DIMENSIONS = CANONICAL_DIMENSIONS + STRESS_DIMENSIONS
MODEL_IDS = ("FC_HVI", "DEEPSETS_HVI")
CRR_DEPTHS = (500, 1000, 2000, 4000)


@dataclass(frozen=True)
class OracleAsset:
    d: int
    sigma_g: float
    q_g: float
    spots: np.ndarray
    taus: np.ndarray
    values: np.ndarray
    boundary: np.ndarray
    refinement_rmse: float
    seconds: float


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
    tmp.replace(path)
    return path


def write_json(path: Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def _json_default(value: Any):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def select_crr_depth(rows: list[dict[str, Any]]) -> tuple[int, bool]:
    if not rows:
        raise ValueError("CRR convergence rows are empty")
    for row in rows:
        previous = row.get("previous_depth_rmse")
        if (
            previous is not None
            and float(row["cnpsor_rmse"]) <= 1.0e-4
            and float(previous) <= 5.0e-5
        ):
            return int(row["depth"]), True
    return int(rows[-1]["depth"]), False


def diagonal_states(
    d: int,
    tau_value: float,
    count: int,
    device: torch.device,
    *,
    spot_min: float = 0.5,
    spot_max: float = 1.5,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    if d < 1 or count < 2:
        raise ValueError("d must be positive and count must be at least two")
    if spot_min <= 0.0 or spot_max <= spot_min:
        raise ValueError("invalid spot interval")
    spots = np.linspace(float(spot_min), float(spot_max), int(count), dtype=float)
    log_spots = torch.tensor(np.log(spots), dtype=torch.float32, device=device)
    x = log_spots[:, None].repeat(1, int(d))
    tau = torch.full((int(count),), float(tau_value), dtype=torch.float32, device=device)
    return x, tau, spots


def _effective_geometric_basket_parameters(d: int, sigma: float, q: float, rho: float) -> tuple[float, float]:
    module = importlib.import_module("physics.high_dimensional_black_scholes")
    return module.effective_geometric_basket_parameters(d, sigma, q, rho)


def effective_contract(d: int, cfg: dict[str, Any]) -> dict[str, float]:
    contract = cfg["contract"]
    sigma_g, q_g = _effective_geometric_basket_parameters(
        int(d), float(contract["sigma"]), float(contract["q"]), float(contract["rho"])
    )
    return {
        "K": float(contract["K"]),
        "T": float(contract["T"]),
        "r": float(contract["r"]),
        "sigma_g": float(sigma_g),
        "q_g": float(q_g),
    }


def load_oracle_asset(path: Path, d: int) -> OracleAsset:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as data:
        actual_d = int(data["d"])
        if actual_d != int(d):
            raise RuntimeError(f"oracle_dimension_mismatch:{actual_d}!={d}")
        return OracleAsset(
            d=actual_d,
            sigma_g=float(data["sigma_g"]),
            q_g=float(data["q_g"]),
            spots=np.asarray(data["spots"], dtype=float),
            taus=np.asarray(data["taus"], dtype=float),
            values=np.asarray(data["values"], dtype=float),
            boundary=np.asarray(data["boundary"], dtype=float),
            refinement_rmse=float(data["refinement_rmse"]),
            seconds=float(data["seconds"]),
        )


def _interp2(spots: np.ndarray, taus: np.ndarray, values: np.ndarray, g: np.ndarray, tau: np.ndarray) -> np.ndarray:
    module = importlib.import_module("hd1_windows_high_dimensional_pinn_48h")
    return np.asarray(module.interp2(spots, taus, values, g, tau), dtype=float)


def oracle_price(oracle: OracleAsset, spots: np.ndarray, taus: np.ndarray) -> np.ndarray:
    spots = np.asarray(spots, dtype=float).reshape(-1)
    taus = np.asarray(taus, dtype=float).reshape(-1)
    if spots.shape != taus.shape:
        raise ValueError("spots and taus must have the same shape")
    return _interp2(oracle.spots, oracle.taus, oracle.values, spots, taus)


def run_class_for_dimension(d: int) -> str:
    if int(d) in CANONICAL_DIMENSIONS:
        return "canonical"
    if int(d) in STRESS_DIMENSIONS:
        return "stress"
    raise ValueError(f"unknown HD3 dimension: {d}")


def checkpoint_path(d: int, model_id: str) -> Path:
    run_class = run_class_for_dimension(d)
    return HD3_RUNTIME_DIR / f"{run_class}_d{int(d)}_{model_id}" / "latest_checkpoint.pt"


def oracle_path(d: int) -> Path:
    return HD3_ORACLE_DIR / f"d{int(d)}.npz"


def load_frozen_evidence() -> tuple[dict[str, Any], dict[str, Any], str, str]:
    """Load the executed HD3 contract and its frozen provenance identities.

    The checkpoint ``config_hash`` is the byte hash captured during the
    canonical run. Later commits may serialize the same JSON contract with
    different insignificant whitespace. For post-hoc reconstruction, require
    exact semantic equality with the committed config manifest and keep the
    training-time hash as the checkpoint identity rather than substituting a
    hash of today's serialization.
    """
    for path in (CONFIG_MANIFEST_PATH, RESOLVED_OBJECTIVE_PATH, CONFIG_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = json.loads(CONFIG_MANIFEST_PATH.read_text(encoding="utf-8"))
    current_cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    manifest_cfg = manifest.get("config")
    if not isinstance(manifest_cfg, dict):
        raise RuntimeError("visualization_config_manifest_missing_config")
    if current_cfg != manifest_cfg:
        raise RuntimeError("visualization_config_manifest_content_mismatch")
    objective = json.loads(RESOLVED_OBJECTIVE_PATH.read_text(encoding="utf-8"))
    expected_cfg_hash = str(manifest["config_sha256"])
    objective_hash = sha256_path(RESOLVED_OBJECTIVE_PATH)
    if objective_hash != str(manifest["resolved_objective_hash"]):
        raise RuntimeError("visualization_objective_hash_mismatch")
    if str(manifest["git_commit"]) != TRAINING_IMPLEMENTATION_SHA:
        raise RuntimeError("visualization_training_sha_mismatch")
    return dict(manifest_cfg), objective, expected_cfg_hash, objective_hash


def checkpoint_expectation(
    d: int, model_id: str, *, config_hash: str, objective_hash: str
) -> CheckpointExpectation:
    run_class = run_class_for_dimension(d)
    completed_steps = 6000 if run_class == "canonical" else 4000
    return CheckpointExpectation(
        training_sha=TRAINING_IMPLEMENTATION_SHA,
        config_hash=config_hash,
        objective_hash=objective_hash,
        job_id=f"{run_class}_d{int(d)}_{model_id}",
        dimension=int(d),
        model_id=str(model_id),
        completed_steps=completed_steps,
        run_class=run_class,
    )
