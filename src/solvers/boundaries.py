"""Early-exercise boundary extraction from continuation premium."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from solvers.black_scholes import OptionType


@dataclass(frozen=True)
class BoundaryResult:
    value: float
    status: str
    exercise_nodes: int
    continuation_nodes: int


def extract_boundary(
    spots: np.ndarray,
    premium: np.ndarray,
    option_type: OptionType,
    *,
    tol: float = 1e-8,
) -> float:
    return extract_boundary_result(spots, premium, option_type, tol=tol).value


def extract_boundary_result(
    spots: np.ndarray,
    premium: np.ndarray,
    option_type: OptionType,
    *,
    tol: float = 1e-8,
) -> BoundaryResult:
    exercise = premium <= tol
    exercise_nodes = int(np.count_nonzero(exercise))
    continuation_nodes = int(np.count_nonzero(~exercise))
    if exercise_nodes == 0:
        return BoundaryResult(float("nan"), "no_exercise", exercise_nodes, continuation_nodes)
    if continuation_nodes == 0:
        edge = spots[-1] if option_type == "put" else spots[0]
        return BoundaryResult(float(edge), "all_exercise", exercise_nodes, continuation_nodes)

    if option_type == "put":
        continuation = np.flatnonzero(~exercise)
        first_continuation = int(continuation[0])
        if first_continuation == 0:
            return BoundaryResult(float("nan"), "no_exercise", exercise_nodes, continuation_nodes)
        i = first_continuation - 1
        value = _interpolate_zero(spots[i], spots[i + 1], premium[i], premium[i + 1])
        return BoundaryResult(value, "boundary_found", exercise_nodes, continuation_nodes)
    if option_type == "call":
        continuation = np.flatnonzero(~exercise)
        last_continuation = int(continuation[-1])
        if last_continuation == len(spots) - 1:
            return BoundaryResult(float("nan"), "no_exercise", exercise_nodes, continuation_nodes)
        i = last_continuation + 1
        value = _interpolate_zero(spots[i - 1], spots[i], premium[i - 1], premium[i])
        return BoundaryResult(value, "boundary_found", exercise_nodes, continuation_nodes)
    raise ValueError(f"unknown option_type: {option_type}")


def boundary_curve(
    spots: np.ndarray,
    values: np.ndarray,
    payoff_values: np.ndarray,
    option_type: OptionType,
    *,
    tol: float = 1e-8,
) -> np.ndarray:
    return np.array([result.value for result in boundary_curve_details(spots, values, payoff_values, option_type, tol=tol)])


def boundary_curve_details(
    spots: np.ndarray,
    values: np.ndarray,
    payoff_values: np.ndarray,
    option_type: OptionType,
    *,
    tol: float = 1e-8,
) -> list[BoundaryResult]:
    premium = values - payoff_values[None, :]
    return [extract_boundary_result(spots, row, option_type, tol=tol) for row in premium]


def _interpolate_zero(s0: float, s1: float, p0: float, p1: float) -> float:
    if abs(p1 - p0) < 1e-15:
        return float(s0)
    return float(s0 - p0 * (s1 - s0) / (p1 - p0))
