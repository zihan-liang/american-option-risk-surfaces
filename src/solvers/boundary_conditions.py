"""Boundary conditions for finite-difference American option solves."""

from __future__ import annotations

from solvers.black_scholes import OptionType


def boundary_values(
    option_type: OptionType,
    tau: float,
    k: float,
    s_max: float,
    r: float,
    q: float,
) -> tuple[float, float]:
    if option_type == "put":
        return k, 0.0
    if option_type == "call":
        immediate = s_max - k
        european_asymptotic = s_max * _exp(-q * tau) - k * _exp(-r * tau)
        return 0.0, max(immediate, european_asymptotic, 0.0)
    raise ValueError(f"unknown option_type: {option_type}")


def _exp(x: float) -> float:
    import math

    return math.exp(x)
