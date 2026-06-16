"""Surrogate metric helpers."""

from __future__ import annotations

import numpy as np


def error_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    diff = np.asarray(prediction, dtype=float) - np.asarray(target, dtype=float)
    if diff.size == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "max_error": float("nan")}
    return {
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "mae": float(np.mean(np.abs(diff))),
        "max_error": float(np.max(np.abs(diff))),
    }


def max_obstacle_violation(price: np.ndarray, payoff: np.ndarray) -> float:
    if len(price) == 0:
        return float("nan")
    return float(np.max(np.maximum(payoff - price, 0.0)))


def delta_bound_violation(delta: np.ndarray, is_call: np.ndarray) -> float:
    lower = np.where(is_call > 0.5, 0.0, -1.0)
    upper = np.where(is_call > 0.5, 1.0, 0.0)
    return float(np.max(np.maximum(lower - delta, 0.0) + np.maximum(delta - upper, 0.0)))


def boundary_band_error(prediction: np.ndarray, target: np.ndarray, near_boundary: np.ndarray) -> dict[str, float | int]:
    mask = np.asarray(near_boundary, dtype=bool)
    samples = int(np.count_nonzero(mask))
    if samples == 0:
        return {"boundary_samples": 0, "boundary_rmse": float("nan"), "boundary_mae": float("nan")}
    metrics = error_metrics(prediction[mask], target[mask])
    return {
        "boundary_samples": samples,
        "boundary_rmse": metrics["rmse"],
        "boundary_mae": metrics["mae"],
    }


def surface_shape_violations(surface: np.ndarray, option_type: str) -> dict[str, float]:
    values = np.asarray(surface, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    diffs = np.diff(values, axis=1)
    if diffs.size == 0:
        monotonicity = 0.0
    elif option_type == "put":
        monotonicity = float(np.max(np.maximum(diffs, 0.0)))
    elif option_type == "call":
        monotonicity = float(np.max(np.maximum(-diffs, 0.0)))
    else:
        raise ValueError(f"unknown option_type: {option_type}")
    second = np.diff(values, n=2, axis=1)
    convexity = 0.0 if second.size == 0 else float(np.max(np.maximum(-second, 0.0)))
    return {"monotonicity_violation": monotonicity, "convexity_violation": convexity}


def grouped_price_metrics(
    *,
    prediction: np.ndarray,
    target: np.ndarray,
    payoff: np.ndarray,
    splits: np.ndarray,
    is_call: np.ndarray,
    model_name: str,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    split_values = _to_str_array(splits)
    option_values = np.where(is_call > 0.5, "call", "put")
    for split in sorted(set(split_values)):
        for option_type in ["put", "call"]:
            mask = (split_values == split) & (option_values == option_type)
            if not np.any(mask):
                continue
            metrics = error_metrics(prediction[mask], target[mask])
            rows.append(
                {
                    "model": model_name,
                    "split": split,
                    "option_type": option_type,
                    "samples": int(np.count_nonzero(mask)),
                    **metrics,
                    "max_obstacle_violation": max_obstacle_violation(prediction[mask], payoff[mask]),
                }
            )
    return rows


def _to_str_array(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind == "S":
        return values.astype(str)
    return values.astype(str)
