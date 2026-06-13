"""Finite-difference Delta and Gamma from solver surfaces."""

from __future__ import annotations

import numpy as np


def delta_gamma(values: np.ndarray, spots: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if values.ndim != 2:
        raise ValueError("values must be a time-by-spot matrix")
    delta = np.gradient(values, spots, axis=1, edge_order=1)
    gamma = np.gradient(delta, spots, axis=1, edge_order=1)
    return delta, gamma


def greek_mask(spots: np.ndarray, boundary: np.ndarray, k: float, *, width: float = 0.03) -> np.ndarray:
    s = spots[None, :]
    base = (np.abs(s - k) > width * k) & (spots[None, :] > 0.05 * k)
    mask = np.tile(base, (len(boundary), 1))
    finite_boundary = np.isfinite(boundary)
    if np.any(finite_boundary):
        boundary_distance = np.abs(s - boundary[:, None])
        mask &= boundary_distance > width * k
    return mask
