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
    return greek_diagnostic_masks(spots, boundary, k, kink_width=width, boundary_width=width)["strict"]


def greek_diagnostic_masks(
    spots: np.ndarray,
    boundary: np.ndarray,
    k: float,
    *,
    kink_width: float = 0.03,
    boundary_width: float = 0.03,
    zero_spot_width: float = 0.05,
) -> dict[str, np.ndarray]:
    s = spots[None, :]
    near_kink = np.tile(np.abs(s - k) <= kink_width * k, (len(boundary), 1))
    near_zero_spot = np.tile(spots[None, :] <= zero_spot_width * k, (len(boundary), 1))
    near_boundary = np.zeros((len(boundary), len(spots)), dtype=bool)
    finite_boundary = np.isfinite(boundary)
    if np.any(finite_boundary):
        boundary_distance = np.abs(s - boundary[:, None])
        near_boundary = boundary_distance <= boundary_width * k
        near_boundary[~finite_boundary, :] = False
    strict = ~(near_kink | near_zero_spot | near_boundary)
    return {
        "strict": strict,
        "near_kink": near_kink,
        "near_boundary": near_boundary,
        "near_zero_spot": near_zero_spot,
    }
