"""Surrogate metric helpers."""

from __future__ import annotations

import numpy as np


def max_obstacle_violation(price: np.ndarray, payoff: np.ndarray) -> float:
    return float(np.max(np.maximum(payoff - price, 0.0)))


def delta_bound_violation(delta: np.ndarray, is_call: np.ndarray) -> float:
    lower = np.where(is_call > 0.5, 0.0, -1.0)
    upper = np.where(is_call > 0.5, 1.0, 0.0)
    return float(np.max(np.maximum(lower - delta, 0.0) + np.maximum(delta - upper, 0.0)))
