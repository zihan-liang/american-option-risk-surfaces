"""Derived HD3 visualization data exports built from frozen evidence."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from solvers.crr_american import crr_american_put_curve, crr_convergence_rows
from hd_pipeline.hd3_visualization_core import (
    ROOT, ALL_DIMENSIONS, CANONICAL_DIMENSIONS, MODEL_IDS, CRR_DEPTHS,
    OracleAsset, checkpoint_path, oracle_path, load_oracle_asset,
    validate_checkpoint_metadata, checkpoint_expectation, sha256_path, write_json,
    write_rows_csv, effective_contract, oracle_price, run_class_for_dimension,
    select_crr_depth, diagonal_states, load_completed_model_for_visualization,
    _model_device, infer_prices,
)

def export_source_inventory(
    *,
    output_path: Path,
    config_hash: str,
    objective_hash: str,
    include_checkpoints: bool = True,
) -> Path:
    assets: list[dict[str, Any]] = []
    if include_checkpoints:
        for d in ALL_DIMENSIONS:
            for model_id in MODEL_IDS:
                path = checkpoint_path(d, model_id)
                if not path.is_file():
                    raise FileNotFoundError(path)
                expected = checkpoint_expectation(
                    d, model_id, config_hash=config_hash, objective_hash=objective_hash
                )
                checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                validate_checkpoint_metadata(checkpoint, expected)
                assets.append(
                    {
                        "asset_type": "checkpoint",
                        "relative_path": path.relative_to(ROOT).as_posix(),
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256_path(path),
                        "dimension": d,
                        "model_id": model_id,
                        "run_class": expected.run_class,
                        "training_implementation_sha": checkpoint["implementation_sha"],
                        "config_hash": checkpoint["config_hash"],
                        "resolved_objective_hash": checkpoint["resolved_objective_hash"],
                        "completed_step": checkpoint["completed_step"],
                    }
                )
    for d in ALL_DIMENSIONS:
        path = oracle_path(d)
        oracle = load_oracle_asset(path, d)
        assets.append(
            {
                "asset_type": "oracle",
                "relative_path": path.relative_to(ROOT).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_path(path),
                "dimension": d,
                "model_id": None,
                "run_class": run_class_for_dimension(d),
                "oracle_refinement_rmse": oracle.refinement_rmse,
                "sigma_g": oracle.sigma_g,
                "q_g": oracle.q_g,
            }
        )
    return write_json(output_path, {"assets": assets})


def export_reference_agreement(
    *,
    cfg: dict[str, Any],
    oracles: dict[int, OracleAsset],
    output_path: Path,
    depths: tuple[int, ...] = CRR_DEPTHS,
    spot_count: int = 401,
) -> dict[int, dict[str, Any]]:
    rows_out: list[dict[str, Any]] = []
    selected: dict[int, dict[str, Any]] = {}
    spots = np.linspace(0.5, 1.5, int(spot_count), dtype=float)
    taus = np.full_like(spots, 1.0)
    for d in ALL_DIMENSIONS:
        oracle = oracles[int(d)]
        contract = effective_contract(int(d), cfg)
        cnpsor = oracle_price(oracle, spots, taus)
        rows = crr_convergence_rows(
            spots,
            cnpsor,
            strike=contract["K"],
            maturity=1.0,
            rate=contract["r"],
            dividend_yield=contract["q_g"],
            volatility=contract["sigma_g"],
            depths=depths,
        )
        depth, passed = select_crr_depth(rows)
        selected[int(d)] = {"depth": int(depth), "passed": bool(passed)}
        for row in rows:
            rows_out.append(
                {
                    "dimension": int(d),
                    "run_class": run_class_for_dimension(int(d)),
                    "depth": int(row["depth"]),
                    "sigma_g": contract["sigma_g"],
                    "q_g": contract["q_g"],
                    "cnpsor_rmse": row["cnpsor_rmse"],
                    "cnpsor_max_abs": row["cnpsor_max_abs"],
                    "previous_depth_rmse": row["previous_depth_rmse"],
                    "selected_depth": int(row["depth"]) == int(depth),
                    "reference_agreement_passed": bool(passed),
                }
            )
    write_rows_csv(output_path, rows_out)
    return selected


def load_all_oracles() -> dict[int, OracleAsset]:
    return {d: load_oracle_asset(oracle_path(d), d) for d in ALL_DIMENSIONS}


def load_all_models(
    *, cfg: dict[str, Any], config_hash: str, objective_hash: str, device: torch.device
) -> tuple[dict[int, dict[str, torch.nn.Module]], dict[int, dict[str, dict[str, Any]]]]:
    models: dict[int, dict[str, torch.nn.Module]] = {}
    checkpoints: dict[int, dict[str, dict[str, Any]]] = {}
    for d in ALL_DIMENSIONS:
        models[d] = {}
        checkpoints[d] = {}
        for model_id in MODEL_IDS:
            expected = checkpoint_expectation(
                d, model_id, config_hash=config_hash, objective_hash=objective_hash
            )
            model, checkpoint = load_completed_model_for_visualization(
                checkpoint_path(d, model_id), expected, cfg, device
            )
            models[d][model_id] = model
            checkpoints[d][model_id] = checkpoint
    return models, checkpoints


def export_diagonal_slices(
    *,
    cfg: dict[str, Any],
    oracles: dict[int, OracleAsset],
    models: dict[int, dict[str, torch.nn.Module]],
    selected_crr: dict[int, dict[str, Any]],
    output_path: Path,
    maturities: tuple[float, ...] = (1.0, 0.5, 0.1),
    spot_count: int = 401,
    batch_size: int = 4096,
) -> Path:
    rows: list[dict[str, Any]] = []
    for d in ALL_DIMENSIONS:
        oracle = oracles[d]
        contract = effective_contract(d, cfg)
        for tau_value in maturities:
            x, tau, spots = diagonal_states(d, tau_value, spot_count, device=torch.device("cpu"))
            cnpsor = oracle_price(oracle, spots, np.full(len(spots), tau_value))
            crr = crr_american_put_curve(
                spots,
                strike=contract["K"],
                maturity=float(tau_value),
                rate=contract["r"],
                dividend_yield=contract["q_g"],
                volatility=contract["sigma_g"],
                steps=int(selected_crr[d]["depth"]),
            )
            fc = infer_prices(models[d]["FC_HVI"], x, tau, batch_size=batch_size)
            ds = infer_prices(models[d]["DEEPSETS_HVI"], x, tau, batch_size=batch_size)
            for i, spot in enumerate(spots):
                rows.append(
                    {
                        "dimension": d,
                        "run_class": run_class_for_dimension(d),
                        "tau": float(tau_value),
                        "S": float(spot),
                        "cnpsor": float(cnpsor[i]),
                        "crr": float(crr[i]),
                        "fc_hvi": float(fc[i]),
                        "deepsets_hvi": float(ds[i]),
                        "fc_abs_error": float(abs(fc[i] - cnpsor[i])),
                        "deepsets_abs_error": float(abs(ds[i] - cnpsor[i])),
                        "crr_depth": int(selected_crr[d]["depth"]),
                        "reference_agreement_passed": bool(selected_crr[d]["passed"]),
                    }
                )
    return write_rows_csv(output_path, rows)


def export_error_surfaces(
    *,
    cfg: dict[str, Any],
    oracles: dict[int, OracleAsset],
    models: dict[int, dict[str, torch.nn.Module]],
    output_dir: Path,
    spot_count: int = 201,
    tau_count: int = 101,
    batch_size: int = 4096,
) -> list[Path]:
    output_paths: list[Path] = []
    spots = np.linspace(0.5, 1.5, int(spot_count), dtype=float)
    taus = np.linspace(0.0025, 1.0, int(tau_count), dtype=float)
    ss, tt = np.meshgrid(spots, taus)
    flat_spots, flat_taus = ss.ravel(), tt.ravel()
    for d in CANONICAL_DIMENSIONS:
        x = torch.tensor(np.log(flat_spots), dtype=torch.float32)[:, None].repeat(1, d)
        tau = torch.tensor(flat_taus, dtype=torch.float32)
        cnpsor = oracle_price(oracles[d], flat_spots, flat_taus)
        fc = infer_prices(models[d]["FC_HVI"], x, tau, batch_size=batch_size)
        ds = infer_prices(models[d]["DEEPSETS_HVI"], x, tau, batch_size=batch_size)
        rows = [
            {
                "dimension": d,
                "tau": float(flat_taus[i]),
                "S": float(flat_spots[i]),
                "cnpsor": float(cnpsor[i]),
                "fc_hvi": float(fc[i]),
                "deepsets_hvi": float(ds[i]),
                "fc_abs_error": float(abs(fc[i] - cnpsor[i])),
                "deepsets_abs_error": float(abs(ds[i] - cnpsor[i])),
            }
            for i in range(len(flat_spots))
        ]
        path = Path(output_dir) / f"error_surface_{d}d.csv"
        write_rows_csv(path, rows)
        output_paths.append(path)
    return output_paths


def _predicted_boundaries(
    model: torch.nn.Module,
    *,
    d: int,
    taus: np.ndarray,
    spots: np.ndarray,
    strike: float,
    tolerance: float = 1.0e-5,
    batch_size: int = 4096,
) -> np.ndarray:
    """Infer an entire diagonal exercise-boundary curve in one batched pass."""
    taus = np.asarray(taus, dtype=float)
    spots = np.asarray(spots, dtype=float)
    ss, tt = np.meshgrid(spots, taus)
    flat_spots = ss.ravel()
    flat_taus = tt.ravel()
    x = torch.tensor(np.log(flat_spots), dtype=torch.float32)[:, None].repeat(1, int(d))
    tau_tensor = torch.tensor(flat_taus, dtype=torch.float32)
    values = infer_prices(model, x, tau_tensor, batch_size=batch_size).reshape(len(taus), len(spots))
    payoff = np.maximum(float(strike) - spots, 0.0)
    boundaries = np.zeros(len(taus), dtype=float)
    eligible_spot = spots <= float(strike)
    for i, row in enumerate(values):
        exercise = np.flatnonzero(eligible_spot & ((row - payoff) <= float(tolerance)))
        boundaries[i] = float(spots[exercise[-1]]) if exercise.size else 0.0
    return boundaries


def export_exercise_boundaries(
    *,
    cfg: dict[str, Any],
    oracles: dict[int, OracleAsset],
    models: dict[int, dict[str, torch.nn.Module]],
    output_path: Path,
    tau_count: int = 128,
    spot_count: int = 512,
    batch_size: int = 4096,
) -> Path:
    taus = np.linspace(0.0025, 1.0, int(tau_count), dtype=float)
    spots = np.linspace(0.5, 1.2, int(spot_count), dtype=float)
    strike = float(cfg["contract"]["K"])
    rows: list[dict[str, Any]] = []
    for d in CANONICAL_DIMENSIONS:
        oracle = oracles[d]
        true_boundaries = np.interp(taus, oracle.taus, oracle.boundary)
        fc_boundaries = _predicted_boundaries(
            models[d]["FC_HVI"], d=d, taus=taus, spots=spots, strike=strike,
            batch_size=batch_size,
        )
        deepsets_boundaries = _predicted_boundaries(
            models[d]["DEEPSETS_HVI"], d=d, taus=taus, spots=spots, strike=strike,
            batch_size=batch_size,
        )
        for i, tau_value in enumerate(taus):
            rows.append(
                {
                    "dimension": d,
                    "run_class": "canonical",
                    "tau": float(tau_value),
                    "cnpsor_boundary": float(true_boundaries[i]),
                    "fc_hvi_boundary": float(fc_boundaries[i]),
                    "deepsets_hvi_boundary": float(deepsets_boundaries[i]),
                    "exercise_gap_tolerance": 1.0e-5,
                }
            )
    return write_rows_csv(output_path, rows)
