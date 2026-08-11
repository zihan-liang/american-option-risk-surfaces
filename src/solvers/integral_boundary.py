"""Independent early-exercise boundary check from the premium integral equation."""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr

from solvers.black_scholes import black_scholes_price


def american_put_boundary_integral(
    *,
    k: float,
    t: float,
    r: float,
    q: float,
    sigma: float,
    steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the American-put boundary Volterra equation on a uniform time grid.

    This is an independent validation calculation, not the production label
    generator. The sequential quadrature follows the early-exercise-premium
    representation and uses only already-computed boundary values.
    """
    if not all(math.isfinite(value) for value in (k, t, r, q, sigma)):
        raise ValueError("integral-boundary inputs must be finite")
    if k <= 0.0 or t <= 0.0 or sigma <= 0.0 or steps < 2:
        raise ValueError("k, t, and sigma must be positive and steps at least two")
    if r <= 0.0 or q < 0.0:
        raise ValueError("the current integral-equation check requires r > 0 and q >= 0")

    taus = np.linspace(0.0, t, steps + 1)
    boundary = np.empty(steps + 1, dtype=float)
    boundary[0] = k if q <= r else min(k, r * k / q)

    for index in range(1, steps + 1):
        tau = float(taus[index])

        def equation(candidate: float) -> float:
            european = float(black_scholes_price(candidate, k, tau, r, q, sigma, "put"))
            integrand = np.empty(index + 1, dtype=float)
            for previous in range(index):
                elapsed = tau - float(taus[previous])
                comparison_boundary = boundary[previous]
                root_elapsed = math.sqrt(elapsed)
                d1 = (
                    math.log(candidate / comparison_boundary)
                    + (r - q + 0.5 * sigma**2) * elapsed
                ) / (sigma * root_elapsed)
                d2 = d1 - sigma * root_elapsed
                integrand[previous] = (
                    r * k * math.exp(-r * elapsed) * ndtr(-d2)
                    - q * candidate * math.exp(-q * elapsed) * ndtr(-d1)
                )
            integrand[index] = 0.5 * (r * k - q * candidate)
            premium = float(np.trapezoid(integrand, taus[: index + 1]))
            return k - candidate - european - premium

        boundary[index] = _largest_bracketed_root(equation, k)

    return taus, boundary


def _largest_bracketed_root(function, upper: float) -> float:
    lower = max(np.finfo(float).tiny, upper * 1e-6)
    candidates = np.linspace(lower, upper * (1.0 - 1e-12), 320)
    values = np.array([function(float(candidate)) for candidate in candidates])
    finite = np.isfinite(values)
    brackets: list[tuple[float, float]] = []
    for index in range(len(candidates) - 1):
        if not finite[index] or not finite[index + 1]:
            continue
        if values[index] == 0.0:
            brackets.append((float(candidates[index]), float(candidates[index])))
        elif values[index] * values[index + 1] < 0.0:
            brackets.append((float(candidates[index]), float(candidates[index + 1])))
    if not brackets:
        best = int(np.nanargmin(np.abs(values)))
        if abs(float(values[best])) > 1e-6:
            raise RuntimeError("integral boundary equation did not contain a resolved root")
        return float(candidates[best])
    left, right = brackets[-1]
    if left == right:
        return left
    return float(brentq(function, left, right, xtol=1e-12, rtol=1e-12, maxiter=200))
