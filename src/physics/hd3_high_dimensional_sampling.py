"""Aggregate-preserving high-dimensional state sampling and diagnostics."""
from __future__ import annotations

import math
from collections.abc import Sequence

import torch


def sample_transverse_states(
    n: int,
    d: int,
    *,
    seed: int,
    m_bounds: tuple[float, float],
    tau_bounds: tuple[float, float],
    transverse_scale: float,
    skip: int = 0,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Draw Sobol states with a dimension-invariant geometric-mean marginal.

    The first Sobol coordinate controls aggregate log geometric mean ``m``;
    the next ``d`` coordinates generate deviations whose per-state mean is
    removed exactly; the final coordinate controls ``tau``.
    """
    if n < 1 or d < 1:
        raise ValueError("n and d must be positive")
    if skip < 0:
        raise ValueError("skip must be nonnegative")
    if not m_bounds[0] < m_bounds[1] or not tau_bounds[0] < tau_bounds[1]:
        raise ValueError("sampling bounds must be increasing")
    if transverse_scale < 0:
        raise ValueError("transverse_scale must be nonnegative")
    engine = torch.quasirandom.SobolEngine(d + 2, scramble=True, seed=int(seed))
    if skip:
        engine.fast_forward(int(skip))
    u = engine.draw(int(n)).to(device=device, dtype=dtype)
    m = m_bounds[0] + (m_bounds[1] - m_bounds[0]) * u[:, 0]
    raw = 2.0 * u[:, 1 : d + 1] - 1.0
    delta = float(transverse_scale) * (raw - raw.mean(dim=1, keepdim=True))
    x = m.reshape(-1, 1) + delta
    tau = tau_bounds[0] + (tau_bounds[1] - tau_bounds[0]) * u[:, d + 1]
    return x, tau, m


def _model_device_dtype(model: torch.nn.Module, fallback: torch.Tensor) -> tuple[torch.device, torch.dtype]:
    try:
        parameter = next(model.parameters())
    except StopIteration:
        return fallback.device, fallback.dtype
    return parameter.device, parameter.dtype


def permutation_consistency_rmse(
    model: torch.nn.Module,
    x: torch.Tensor,
    tau: torch.Tensor,
    permutation: Sequence[int] | torch.Tensor,
) -> float:
    """RMSE between predictions before and after a fixed asset permutation."""
    if x.ndim != 2:
        raise ValueError("x must have shape (batch, d)")
    permutation_tensor = torch.as_tensor(permutation, dtype=torch.long)
    if permutation_tensor.numel() != x.shape[1]:
        raise ValueError("permutation length mismatch")
    if sorted(permutation_tensor.tolist()) != list(range(x.shape[1])):
        raise ValueError("invalid permutation")
    device, dtype = _model_device_dtype(model, x)
    x_device = x.to(device=device, dtype=dtype)
    tau_device = tau.to(device=device, dtype=dtype)
    permutation_device = permutation_tensor.to(device=device)
    with torch.no_grad():
        base = model(x_device, tau_device)
        permuted = model(x_device[:, permutation_device], tau_device)
    return float(torch.sqrt((base - permuted).square().mean()).cpu())


def transverse_invariance_rmse(
    model: torch.nn.Module,
    x_grouped: torch.Tensor,
    tau_grouped: torch.Tensor,
) -> float:
    """Measure prediction variation within groups sharing aggregate ``m,tau``.

    ``x_grouped`` has shape ``(groups, repeats, d)`` and ``tau_grouped`` has
    shape ``(groups, repeats)``.  The metric is the RMS deviation from each
    group's prediction mean.
    """
    if x_grouped.ndim != 3:
        raise ValueError("x_grouped must have shape (groups, repeats, d)")
    if tau_grouped.shape != x_grouped.shape[:2]:
        raise ValueError("tau_grouped shape mismatch")
    groups, repeats, d = x_grouped.shape
    flat_x = x_grouped.reshape(groups * repeats, d)
    flat_tau = tau_grouped.reshape(groups * repeats)
    device, dtype = _model_device_dtype(model, flat_x)
    with torch.no_grad():
        prediction = model(
            flat_x.to(device=device, dtype=dtype),
            flat_tau.to(device=device, dtype=dtype),
        ).reshape(groups, repeats)
    centered = prediction - prediction.mean(dim=1, keepdim=True)
    return float(torch.sqrt(centered.square().mean()).cpu())


def grouped_transverse_states(
    *,
    groups: int,
    repeats: int,
    d: int,
    seed: int,
    m_bounds: tuple[float, float],
    tau_bounds: tuple[float, float],
    transverse_scale: float,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build groups with exactly shared aggregate ``m`` and ``tau`` per group."""
    if groups < 1 or repeats < 2:
        raise ValueError("groups must be positive and repeats at least two")
    aggregate_x, aggregate_tau, aggregate_m = sample_transverse_states(
        groups,
        d,
        seed=seed,
        m_bounds=m_bounds,
        tau_bounds=tau_bounds,
        transverse_scale=0.0,
        device=device,
        dtype=dtype,
    )
    # Independent deviation stream.  Its aggregate component is discarded.
    deviation_x, _, _ = sample_transverse_states(
        groups * repeats,
        d,
        seed=seed + 1,
        m_bounds=m_bounds,
        tau_bounds=tau_bounds,
        transverse_scale=transverse_scale,
        device=device,
        dtype=dtype,
    )
    deviations = deviation_x - deviation_x.mean(dim=1, keepdim=True)
    deviations = deviations.reshape(groups, repeats, d)
    x = aggregate_m[:, None, None] + deviations
    tau = aggregate_tau[:, None].repeat(1, repeats)
    return x, tau, aggregate_m
