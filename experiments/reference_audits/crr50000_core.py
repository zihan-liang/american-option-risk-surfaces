"""Frozen CRR pricing core from the uploaded 50,000-step benchmark script.

This module preserves the deterministic American-put pricing routine used by
``binomial_benchmark_50000.py``. The original uploaded script SHA-256 is
``19a31a152b18d55c17960a779e1feeee4a4ed850f5572693b11354f62b9af075``.
The benchmark is a convergence-checked numerical cross-check, not an analytic
exact solution.
"""

from __future__ import annotations

import math

import numpy as np


CONTRACT = {
    "option_type": "put",
    "K": 1.0,
    "T": 1.0,
    "r": 0.05,
    "q": 0.10,
    "sigma": 0.60,
}

TREE_STEPS = 50_000
PRICE_S_MIN = 0.5
PRICE_S_MAX = 1.5
PRICE_S_POINTS = 401
PRICE_TAUS = (1.0, 0.5, 0.1)
SOURCE_SCRIPT_SHA256 = "19a31a152b18d55c17960a779e1feeee4a4ed850f5572693b11354f62b9af075"
SOURCE_PRICE_SLICES_SHA256 = "4b1f85845d45177e356eb377a1ce03e610209595997febc8a6d85794fc29dbe6"
SOURCE_ARCHIVE_SHA256 = "87f94df662614d4647c3fbf2e4f4064fc1b5e6ae8f044f24760ed7fdcb912ed6"


def price_spots() -> np.ndarray:
    """Return the frozen common 401-point price grid."""
    return np.linspace(PRICE_S_MIN, PRICE_S_MAX, PRICE_S_POINTS, dtype=np.float64)


def crr_american_put(
    spot: float,
    tau: float,
    *,
    steps: int = TREE_STEPS,
    k: float = CONTRACT["K"],
    r: float = CONTRACT["r"],
    q: float = CONTRACT["q"],
    sigma: float = CONTRACT["sigma"],
) -> float:
    """Memory-efficient float64 CRR American-put backward induction.

    This is the pricing routine from the uploaded benchmark script. Only the
    surrounding export/resume machinery is omitted here.
    """
    spot = float(spot)
    tau = float(tau)
    k = float(k)
    r = float(r)
    q = float(q)
    sigma = float(sigma)
    steps = int(steps)

    if steps < 1:
        raise ValueError("steps must be >= 1")
    if spot < 0.0 or k <= 0.0 or sigma <= 0.0:
        raise ValueError("require spot>=0, K>0 and sigma>0")
    if tau <= 0.0:
        return max(k - spot, 0.0)

    dt = tau / steps
    log_u = sigma * math.sqrt(dt)
    u = math.exp(log_u)
    d = 1.0 / u
    growth = math.exp((r - q) * dt)
    p = (growth - d) / (u - d)
    if not (0.0 <= p <= 1.0):
        raise RuntimeError(f"invalid_crr_probability:{p}")
    disc = math.exp(-r * dt)

    j = np.arange(steps + 1, dtype=np.float64)
    stock = spot * np.exp((2.0 * j - steps) * log_u)
    value = np.maximum(k - stock, 0.0)

    for _ in range(steps - 1, -1, -1):
        value = disc * (p * value[1:] + (1.0 - p) * value[:-1])
        stock = stock[:-1] * u
        value = np.maximum(value, np.maximum(k - stock, 0.0))

    return float(value[0])
