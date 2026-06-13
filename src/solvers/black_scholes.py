"""Black-Scholes formulas and payoffs used for validation."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

OptionType = Literal["put", "call"]


def payoff(spots: np.ndarray | float, k: float, option_type: OptionType) -> np.ndarray:
    s = np.asarray(spots, dtype=float)
    if option_type == "put":
        return np.maximum(k - s, 0.0)
    if option_type == "call":
        return np.maximum(s - k, 0.0)
    raise ValueError(f"unknown option_type: {option_type}")


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    return np.vectorize(lambda z: 0.5 * (1.0 + math.erf(float(z) / math.sqrt(2.0))))(x)


def black_scholes_price(
    spots: np.ndarray | float,
    k: float,
    tau: np.ndarray | float,
    r: float,
    q: float,
    sigma: float,
    option_type: OptionType,
) -> np.ndarray | float:
    """European Black-Scholes price with continuous dividend yield."""
    s = np.asarray(spots, dtype=float)
    tau_arr = np.asarray(tau, dtype=float)
    s_b, tau_b = np.broadcast_arrays(s, tau_arr)
    shape = s_b.shape
    s_flat = s_b.reshape(-1)
    tau_flat = tau_b.reshape(-1)
    out = payoff(s_flat, k, option_type)

    active = tau_flat > 0
    if np.any(active):
        safe_s = np.maximum(s_flat[active], 1e-300)
        safe_tau = tau_flat[active]
        vol_sqrt = sigma * np.sqrt(safe_tau)
        d1 = (np.log(safe_s / k) + (r - q + 0.5 * sigma**2) * safe_tau) / vol_sqrt
        d2 = d1 - vol_sqrt
        if option_type == "call":
            out[active] = safe_s * np.exp(-q * safe_tau) * _norm_cdf(d1) - k * np.exp(
                -r * safe_tau
            ) * _norm_cdf(d2)
        elif option_type == "put":
            out[active] = k * np.exp(-r * safe_tau) * _norm_cdf(-d2) - safe_s * np.exp(
                -q * safe_tau
            ) * _norm_cdf(-d1)
        else:
            raise ValueError(f"unknown option_type: {option_type}")

    if np.isscalar(spots) and np.isscalar(tau):
        return float(out.item())
    return out.reshape(shape)
