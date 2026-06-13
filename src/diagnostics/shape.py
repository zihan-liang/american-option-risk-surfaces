"""Shape diagnostics for option price surfaces."""

from __future__ import annotations

import numpy as np


def obstacle_violation(values: np.ndarray, payoff: np.ndarray) -> float:
    return float(np.max(np.maximum(payoff[None, :] - values, 0.0)))


def monotonicity_violation(values: np.ndarray, option_type: str) -> float:
    diffs = np.diff(values, axis=1)
    if option_type == "put":
        return float(np.max(np.maximum(diffs, 0.0)))
    if option_type == "call":
        return float(np.max(np.maximum(-diffs, 0.0)))
    raise ValueError(f"unknown option_type: {option_type}")


def convexity_violation(values: np.ndarray) -> float:
    second = np.diff(values, n=2, axis=1)
    if second.size == 0:
        return 0.0
    return float(np.max(np.maximum(-second, 0.0)))


def american_dominates_european_violation(american: np.ndarray, european: np.ndarray) -> float:
    return float(np.max(np.maximum(european - american, 0.0)))
