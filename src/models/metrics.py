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
    if len(delta) == 0:
        return float("nan")
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


def grouped_premium_metrics(
    *,
    prediction: np.ndarray,
    target: np.ndarray,
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
                    **{f"premium_{key}": value for key, value in metrics.items()},
                }
            )
    return rows


def masked_error_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    *,
    prefix: str,
) -> dict[str, float | int]:
    valid = np.asarray(mask, dtype=bool)
    samples = int(np.count_nonzero(valid))
    if samples == 0:
        return {f"{prefix}_samples": 0, f"{prefix}_rmse": float("nan"), f"{prefix}_mae": float("nan"), f"{prefix}_max_error": float("nan")}
    metrics = error_metrics(prediction[valid], target[valid])
    return {
        f"{prefix}_samples": samples,
        f"{prefix}_rmse": metrics["rmse"],
        f"{prefix}_mae": metrics["mae"],
        f"{prefix}_max_error": metrics["max_error"],
    }


def finite_boundary_mask(boundary: np.ndarray) -> np.ndarray:
    return np.isfinite(np.asarray(boundary, dtype=float))


def grouped_boundary_metrics(
    *,
    prediction: np.ndarray,
    target: np.ndarray,
    splits: np.ndarray,
    is_call: np.ndarray,
    model_name: str,
    lower: float | None = None,
    upper: float | None = None,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    split_values = _to_str_array(splits)
    option_values = np.where(is_call > 0.5, "call", "put")
    for split in sorted(set(split_values)):
        for option_type in ["all", "put", "call"]:
            if option_type == "all":
                mask = split_values == split
            else:
                mask = (split_values == split) & (option_values == option_type)
            if not np.any(mask):
                continue
            metrics = error_metrics(prediction[mask], target[mask])
            row = {
                "model": model_name,
                "split": split,
                "option_type": option_type,
                "samples": int(np.count_nonzero(mask)),
                "boundary_rmse": metrics["rmse"],
                "boundary_mae": metrics["mae"],
                "boundary_max_error": metrics["max_error"],
            }
            if lower is not None and upper is not None:
                row.update(prediction_range_diagnostics(prediction[mask], lower=lower, upper=upper))
            rows.append(row)
    return rows


def grouped_delta_metrics(
    *,
    prediction: np.ndarray,
    target: np.ndarray,
    splits: np.ndarray,
    is_call: np.ndarray,
    model_name: str,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    split_values = _to_str_array(splits)
    option_values = np.where(is_call > 0.5, "call", "put")
    for split in sorted(set(split_values)):
        for option_type in ["all", "put", "call"]:
            if option_type == "all":
                mask = split_values == split
            else:
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
                    "delta_rmse": metrics["rmse"],
                    "delta_mae": metrics["mae"],
                    "delta_max_error": metrics["max_error"],
                    "max_delta_bound_violation": delta_bound_violation(prediction[mask], is_call[mask]),
                }
            )
    return rows


def prediction_range_diagnostics(prediction: np.ndarray, *, lower: float, upper: float) -> dict[str, float | int]:
    values = np.asarray(prediction, dtype=float)
    samples = int(values.size)
    if samples == 0:
        return {
            "samples": 0,
            "prediction_min": float("nan"),
            "prediction_max": float("nan"),
            "out_of_training_range_fraction": float("nan"),
        }
    outside = (values < lower) | (values > upper)
    return {
        "samples": samples,
        "prediction_min": float(np.min(values)),
        "prediction_max": float(np.max(values)),
        "out_of_training_range_fraction": float(np.mean(outside)),
    }


def _to_str_array(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind == "S":
        return values.astype(str)
    return values.astype(str)
