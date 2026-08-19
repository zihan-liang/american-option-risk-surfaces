"""Independent Cox--Ross--Rubinstein reference for American puts.

This module is intentionally NumPy-only. It is used as a numerical cross-check
for the reduced one-dimensional geometric-basket contract in HD3; CN/PSOR
remains the canonical oracle.
"""
from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np


def _validate_common(
    *,
    strike: float,
    maturity: float,
    volatility: float,
    steps: int,
) -> None:
    if not math.isfinite(strike) or strike <= 0.0:
        raise ValueError("strike must be positive and finite")
    if not math.isfinite(maturity) or maturity < 0.0:
        raise ValueError("maturity must be nonnegative and finite")
    if not math.isfinite(volatility) or volatility < 0.0:
        raise ValueError("volatility must be nonnegative and finite")
    if isinstance(steps, bool) or int(steps) != steps or int(steps) < 1:
        raise ValueError("steps must be a positive integer")


def _validate_spots(spots: np.ndarray) -> np.ndarray:
    spots = np.asarray(spots, dtype=float)
    if spots.ndim != 1:
        raise ValueError("spots must be one-dimensional")
    if spots.size == 0:
        raise ValueError("spots must be non-empty")
    if not np.all(np.isfinite(spots)) or np.any(spots <= 0.0):
        raise ValueError("spots must be positive and finite")
    return spots


def _deterministic_american_put_curve(
    spots: np.ndarray,
    *,
    strike: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    steps: int,
) -> np.ndarray:
    """Value the zero-volatility contract by checking every exercise date."""
    if maturity == 0.0:
        return np.maximum(strike - spots, 0.0)
    times = np.linspace(0.0, maturity, int(steps) + 1, dtype=float)
    growth = np.exp((float(rate) - float(dividend_yield)) * times)
    discount = np.exp(-float(rate) * times)
    exercise = np.maximum(strike - spots[:, None] * growth[None, :], 0.0)
    return np.max(exercise * discount[None, :], axis=1)


def crr_american_put_curve(
    spots: np.ndarray,
    *,
    strike: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    steps: int,
) -> np.ndarray:
    """Price an American put for many initial spots on one CRR time grid.

    The roots are evaluated simultaneously so the Python loop is over time
    levels rather than over both spots and time levels. This keeps the HD3
    401-point reference audit practical even for the 4000-step cross-check.
    """
    spots = _validate_spots(spots)
    _validate_common(
        strike=float(strike),
        maturity=float(maturity),
        volatility=float(volatility),
        steps=int(steps),
    )
    strike = float(strike)
    maturity = float(maturity)
    rate = float(rate)
    dividend_yield = float(dividend_yield)
    volatility = float(volatility)
    steps = int(steps)

    if not all(math.isfinite(v) for v in (rate, dividend_yield)):
        raise ValueError("rate and dividend_yield must be finite")
    if maturity == 0.0:
        return np.maximum(strike - spots, 0.0)
    if volatility == 0.0:
        return _deterministic_american_put_curve(
            spots,
            strike=strike,
            maturity=maturity,
            rate=rate,
            dividend_yield=dividend_yield,
            steps=steps,
        )

    dt = maturity / steps
    up = math.exp(volatility * math.sqrt(dt))
    down = 1.0 / up
    growth = math.exp((rate - dividend_yield) * dt)
    probability = (growth - down) / (up - down)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("invalid CRR risk-neutral probability")
    discount = math.exp(-rate * dt)

    # factors[j] is S_{n,j}/S_0. Backward recursion can update factors by
    # dropping the up-most terminal node and dividing the rest by down.
    j = np.arange(steps + 1, dtype=float)
    factors = (down**steps) * np.power(up / down, j)
    values = np.maximum(strike - spots[:, None] * factors[None, :], 0.0)

    for _n in range(steps - 1, -1, -1):
        continuation = discount * (
            probability * values[:, 1:] + (1.0 - probability) * values[:, :-1]
        )
        factors = factors[:-1] / down
        exercise = np.maximum(strike - spots[:, None] * factors[None, :], 0.0)
        values = np.maximum(exercise, continuation)

    return np.asarray(values[:, 0], dtype=float)


def crr_american_put_price(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    steps: int,
) -> float:
    """Price one American put with a Cox--Ross--Rubinstein tree."""
    if not math.isfinite(float(spot)) or float(spot) <= 0.0:
        raise ValueError("spot must be positive and finite")
    value = crr_american_put_curve(
        np.asarray([float(spot)], dtype=float),
        strike=float(strike),
        maturity=float(maturity),
        rate=float(rate),
        dividend_yield=float(dividend_yield),
        volatility=float(volatility),
        steps=int(steps),
    )
    return float(value[0])


def crr_convergence_rows(
    spots: np.ndarray,
    cnpsor_values: np.ndarray,
    *,
    strike: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    depths: Iterable[int],
) -> list[dict[str, float | int | None]]:
    """Return CRR agreement metrics for a predeclared sequence of depths."""
    spots = _validate_spots(spots)
    reference = np.asarray(cnpsor_values, dtype=float)
    if reference.shape != spots.shape:
        raise ValueError("cnpsor_values must have the same shape as spots")
    if not np.all(np.isfinite(reference)):
        raise ValueError("cnpsor_values must be finite")
    depth_tuple = tuple(int(v) for v in depths)
    if not depth_tuple or any(v < 1 for v in depth_tuple):
        raise ValueError("depths must contain positive integers")
    if tuple(sorted(set(depth_tuple))) != depth_tuple:
        raise ValueError("depths must be strictly increasing")

    rows: list[dict[str, float | int | None]] = []
    previous: np.ndarray | None = None
    for depth in depth_tuple:
        curve = crr_american_put_curve(
            spots,
            strike=strike,
            maturity=maturity,
            rate=rate,
            dividend_yield=dividend_yield,
            volatility=volatility,
            steps=depth,
        )
        diff = curve - reference
        previous_rmse = (
            None
            if previous is None
            else float(np.sqrt(np.mean(np.square(curve - previous))))
        )
        rows.append(
            {
                "depth": depth,
                "cnpsor_rmse": float(np.sqrt(np.mean(np.square(diff)))),
                "cnpsor_max_abs": float(np.max(np.abs(diff))),
                "previous_depth_rmse": previous_rmse,
            }
        )
        previous = curve
    return rows
