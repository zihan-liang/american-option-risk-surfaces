"""Early-exercise boundary extraction from continuation premium."""

from __future__ import annotations

import numpy as np

from solvers.black_scholes import OptionType


def extract_boundary(
    spots: np.ndarray,
    premium: np.ndarray,
    option_type: OptionType,
    *,
    tol: float = 1e-8,
) -> float:
    exercise = premium <= tol
    if option_type == "put":
        continuation = np.flatnonzero(~exercise)
        if len(continuation) == 0:
            return float(spots[-1])
        first_continuation = int(continuation[0])
        if first_continuation == 0:
            return float("nan")
        i = first_continuation - 1
        return _interpolate_zero(spots[i], spots[i + 1], premium[i], premium[i + 1])
    if option_type == "call":
        continuation = np.flatnonzero(~exercise)
        if len(continuation) == 0:
            return float(spots[0])
        last_continuation = int(continuation[-1])
        if last_continuation == len(spots) - 1:
            return float("nan")
        i = last_continuation + 1
        return _interpolate_zero(spots[i - 1], spots[i], premium[i - 1], premium[i])
    raise ValueError(f"unknown option_type: {option_type}")


def boundary_curve(
    spots: np.ndarray,
    values: np.ndarray,
    payoff_values: np.ndarray,
    option_type: OptionType,
    *,
    tol: float = 1e-8,
) -> np.ndarray:
    premium = values - payoff_values[None, :]
    return np.array([extract_boundary(spots, row, option_type, tol=tol) for row in premium])


def _interpolate_zero(s0: float, s1: float, p0: float, p1: float) -> float:
    if abs(p1 - p0) < 1e-15:
        return float(s0)
    return float(s0 - p0 * (s1 - s0) / (p1 - p0))
