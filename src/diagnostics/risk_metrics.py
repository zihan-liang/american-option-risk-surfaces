"""Application-stage risk summaries for American option surfaces."""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable

import numpy as np

from solvers.boundaries import BoundaryResult


def classify_regime(option_type: str, q: float, *, dividend_tol: float = 1e-12) -> str:
    if option_type == "put":
        return "put"
    if option_type == "call" and abs(q) <= dividend_tol:
        return "no_dividend_call_control"
    if option_type == "call":
        return "dividend_call"
    raise ValueError(f"unknown option_type: {option_type}")


def exercise_fraction(values: np.ndarray, payoff: np.ndarray, *, tol: float = 1e-8) -> float:
    exercise = values - payoff[None, :] <= tol
    return float(np.count_nonzero(exercise) / exercise.size)


def boundary_status_counts(details: Iterable[BoundaryResult]) -> dict[str, int]:
    counter = Counter(detail.status for detail in details)
    return {
        "boundary_found": int(counter["boundary_found"]),
        "no_exercise": int(counter["no_exercise"]),
        "all_exercise": int(counter["all_exercise"]),
    }


def finite_summary(values: Iterable[float]) -> dict[str, float | int]:
    arr = np.asarray(list(values), dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {
            "samples": 0,
            "initial": float("nan"),
            "mean": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "samples": int(finite.size),
        "initial": float(finite[-1]),
        "mean": float(np.mean(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def greek_strict_summary(delta: np.ndarray, gamma: np.ndarray, strict_mask: np.ndarray) -> dict[str, float | int]:
    active = strict_mask & np.isfinite(delta) & np.isfinite(gamma)
    samples = int(np.count_nonzero(active))
    if samples == 0:
        return {
            "strict_samples": 0,
            "delta_strict_mean": float("nan"),
            "delta_strict_min": float("nan"),
            "delta_strict_max": float("nan"),
            "gamma_strict_mean_abs": float("nan"),
            "gamma_strict_max_abs": float("nan"),
        }
    strict_delta = delta[active]
    strict_gamma = gamma[active]
    return {
        "strict_samples": samples,
        "delta_strict_mean": float(np.mean(strict_delta)),
        "delta_strict_min": float(np.min(strict_delta)),
        "delta_strict_max": float(np.max(strict_delta)),
        "gamma_strict_mean_abs": float(np.mean(np.abs(strict_gamma))),
        "gamma_strict_max_abs": float(np.max(np.abs(strict_gamma))),
    }


def regime_flags(
    *,
    option_type: str,
    regime_class: str,
    q: float,
    boundary_found_count: int,
    obstacle_violation: float,
    complementarity_error: float,
    monotonicity_violation: float,
    convexity_violation: float,
    diagnostic_tol: float = 1e-8,
) -> dict[str, bool]:
    diagnostic_failure = any(
        value > diagnostic_tol
        for value in [
            obstacle_violation,
            complementarity_error,
            monotonicity_violation,
            convexity_violation,
        ]
        if math.isfinite(value)
    )
    is_control = regime_class == "no_dividend_call_control"
    call_no_exercise = option_type == "call" and boundary_found_count == 0
    call_dividend_exercise_found = regime_class == "dividend_call" and boundary_found_count > 0
    put_boundary_unusual = option_type == "put" and boundary_found_count == 0
    delicate_regime = (
        diagnostic_failure
        or put_boundary_unusual
        or (regime_class == "dividend_call" and call_no_exercise)
        or (option_type == "call" and q > 0.0 and not call_dividend_exercise_found)
    )
    return {
        "is_control": is_control,
        "put_boundary_unusual": put_boundary_unusual,
        "call_no_exercise": call_no_exercise,
        "call_dividend_exercise_found": call_dividend_exercise_found,
        "diagnostic_failure": diagnostic_failure,
        "delicate_regime": delicate_regime,
    }
