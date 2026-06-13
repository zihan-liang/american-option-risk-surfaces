"""Build supervised surrogate datasets from validated PSOR surfaces."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from solvers.boundaries import boundary_curve
from solvers.cn_psor import SolverConfig, solve_option
from solvers.greeks import delta_gamma, greek_mask


def build_surface_samples(cfg: SolverConfig) -> dict[str, np.ndarray]:
    result = solve_option(cfg)
    boundary = boundary_curve(result.spots, result.values, result.payoff, cfg.option_type)
    delta, gamma = delta_gamma(result.values, result.spots)
    premium = np.maximum(result.values - result.payoff[None, :], 0.0)
    spots_mesh, tau_mesh = np.meshgrid(result.spots, result.taus)
    features = np.column_stack(
        [
            np.log(np.maximum(spots_mesh.ravel(), 1e-12) / cfg.k),
            (tau_mesh.ravel() / cfg.t) if cfg.t > 0 else tau_mesh.ravel(),
            np.full(spots_mesh.size, cfg.r),
            np.full(spots_mesh.size, cfg.q),
            np.full(spots_mesh.size, cfg.sigma),
            np.full(spots_mesh.size, cfg.t),
            np.full(spots_mesh.size, 1.0 if cfg.option_type == "call" else 0.0),
        ]
    )
    labels = np.column_stack(
        [
            (result.values / cfg.k).ravel(),
            (premium / cfg.k).ravel(),
            (result.payoff[None, :] / cfg.k).repeat(len(result.taus), axis=0).ravel(),
            delta.ravel(),
            (cfg.k * gamma).ravel(),
        ]
    )
    boundary_repeated = np.repeat(boundary[:, None], len(result.spots), axis=1).ravel() / cfg.k
    mask = greek_mask(result.spots, boundary, cfg.k).ravel()
    return {
        "features": features,
        "labels": labels,
        "boundary": boundary_repeated,
        "greek_mask": mask,
        "config": np.array(str(asdict(cfg))),
    }
