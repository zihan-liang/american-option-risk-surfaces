"""Build supervised surrogate datasets from validated PSOR surfaces."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json

import numpy as np

from solvers.boundaries import boundary_curve
from solvers.cn_psor import SolverConfig, solve_option
from solvers.greeks import delta_gamma, greek_diagnostic_masks


def solver_config_hash(cfg: SolverConfig) -> str:
    payload = json.dumps(asdict(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_surface_samples(
    cfg: SolverConfig,
    *,
    regime_key_value: str | None = None,
    split_name: str = "unsplit",
    diagnostic_status: str = "clean",
    source_stage: str = "stage05",
) -> dict[str, np.ndarray]:
    result = solve_option(cfg)
    boundary = boundary_curve(result.spots, result.values, result.payoff, cfg.option_type)
    delta, gamma = delta_gamma(result.values, result.spots)
    masks = greek_diagnostic_masks(result.spots, boundary, cfg.k)
    premium = np.maximum(result.values - result.payoff[None, :], 0.0)
    spots_mesh, tau_mesh = np.meshgrid(result.spots, result.taus)
    node_count = spots_mesh.size
    regime_key_value = regime_key_value or str(asdict(cfg))
    cfg_hash = solver_config_hash(cfg)
    payoff_surface = np.repeat(result.payoff[None, :], len(result.taus), axis=0)
    exercise_indicator = (result.values - result.payoff[None, :]) <= cfg.tol
    features = np.column_stack(
        [
            np.log(np.maximum(spots_mesh.ravel(), 1e-12) / cfg.k),
            (tau_mesh.ravel() / cfg.t) if cfg.t > 0 else tau_mesh.ravel(),
            np.full(spots_mesh.size, cfg.r),
            np.full(spots_mesh.size, cfg.q),
            np.full(spots_mesh.size, cfg.sigma),
            np.full(spots_mesh.size, cfg.t),
            np.full(spots_mesh.size, 1.0 if cfg.option_type == "call" else 0.0),
        ]
    ).astype(np.float32)
    labels = np.column_stack(
        [
            (result.values / cfg.k).ravel(),
            (premium / cfg.k).ravel(),
            (payoff_surface / cfg.k).ravel(),
            delta.ravel(),
            (cfg.k * gamma).ravel(),
        ]
    ).astype(np.float32)
    boundary_repeated = np.repeat(boundary[:, None], len(result.spots), axis=1).ravel() / cfg.k
    return {
        "features": features,
        "labels": labels,
        "boundary": boundary_repeated.astype(np.float32),
        "exercise_indicator": exercise_indicator.ravel(),
        "strict_greek_mask": masks["strict"].ravel(),
        "near_kink": masks["near_kink"].ravel(),
        "near_boundary": masks["near_boundary"].ravel(),
        "near_zero_spot": masks["near_zero_spot"].ravel(),
        "greek_mask": masks["strict"].ravel(),
        "regime_key": _string_array(regime_key_value, node_count),
        "split": _string_array(split_name, node_count),
        "diagnostic_status": _string_array(diagnostic_status, node_count),
        "source_stage": _string_array(source_stage, node_count),
        "solver_config_hash": _string_array(cfg_hash, node_count),
        "config": np.array(str(asdict(cfg))),
    }


def _string_array(value: str, length: int) -> np.ndarray:
    encoded = value.encode("utf-8")
    return np.full(length, encoded, dtype=f"S{max(1, len(encoded))}")
