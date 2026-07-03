"""Helpers for the Stage 22 PINN-inspired method campaign."""

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
) -> dict[str, bool | float]:
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
