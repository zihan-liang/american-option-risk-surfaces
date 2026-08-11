"""Loss and sampling helpers for the M02 obstacle-aware PINN screen."""

from __future__ import annotations

import numpy as np


def masked_pde_loss(residual, mask, *, residual_scale: float):
    """Return normalized squared PDE residual on a predeclared mask."""

    import torch

    _positive_finite(residual_scale, "residual_scale")
    values = _finite_tensor(residual, "residual")
    selected = torch.as_tensor(mask, dtype=torch.bool, device=values.device).reshape(-1)
    if selected.numel() != values.numel():
        raise ValueError("mask must match residual length")
    if not bool(torch.any(selected)):
        raise ValueError("masked PDE loss requires at least one selected point")
    return (values[selected] / float(residual_scale)).square().mean()


def separate_complementarity_loss(gap, residual, *, gap_scale: float, residual_scale: float) -> dict:
    """Return normalized dual-feasibility and complementarity-product losses."""

    _positive_finite(gap_scale, "gap_scale")
    _positive_finite(residual_scale, "residual_scale")
    normalized_gap, normalized_residual = _normalized_pair(gap, residual, gap_scale, residual_scale)
    dual_feasibility = normalized_residual.neg().relu().square().mean()
    complementarity = (normalized_gap * normalized_residual).square().mean()
    total = dual_feasibility + complementarity
    _finite_tensor(total, "separate complementarity loss")
    return {
        "dual_feasibility": dual_feasibility,
        "complementarity": complementarity,
        "total": total,
    }


def fischer_burmeister_loss(
    gap,
    residual,
    *,
    epsilon: float,
    gap_scale: float,
    residual_scale: float,
):
    """Return squared smoothed Fischer--Burmeister residual in shared units."""

    import torch

    _positive_finite(epsilon, "epsilon")
    normalized_gap, normalized_residual = _normalized_pair(gap, residual, gap_scale, residual_scale)
    phi = torch.sqrt(normalized_gap.square() + normalized_residual.square() + float(epsilon) ** 2)
    value = (phi - normalized_gap - normalized_residual).square().mean()
    return _finite_tensor(value, "Fischer--Burmeister loss")


def select_whole_regimes(
    candidate_indices: np.ndarray,
    regime_keys: np.ndarray,
    *,
    cap: int,
    seed: int,
) -> tuple[np.ndarray, list[str]]:
    """Select full regimes in deterministic shuffled order without exceeding ``cap``."""

    indices = np.asarray(candidate_indices, dtype=int).reshape(-1)
    keys = np.asarray(regime_keys).astype(str).reshape(-1)
    if indices.size == 0:
        return indices, []
    if np.any(indices < 0) or np.any(indices >= keys.size):
        raise ValueError("candidate_indices must index regime_keys")
    if cap <= 0:
        selected_ids = sorted(set(keys[indices]))
        return np.sort(indices), selected_ids

    by_regime: dict[str, np.ndarray] = {
        regime_id: np.sort(indices[keys[indices] == regime_id])
        for regime_id in sorted(set(keys[indices]))
    }
    rng = np.random.default_rng(int(seed))
    ordered_ids = [str(value) for value in rng.permutation(sorted(by_regime))]
    chosen: list[np.ndarray] = []
    chosen_ids: list[str] = []
    count = 0
    for regime_id in ordered_ids:
        regime_indices = by_regime[regime_id]
        if count + len(regime_indices) > cap:
            continue
        chosen.append(regime_indices)
        chosen_ids.append(regime_id)
        count += len(regime_indices)
    if not chosen:
        return np.array([], dtype=int), []
    return np.sort(np.concatenate(chosen)), chosen_ids


def _normalized_pair(gap, residual, gap_scale: float, residual_scale: float):
    _positive_finite(gap_scale, "gap_scale")
    _positive_finite(residual_scale, "residual_scale")
    gap_values = _finite_tensor(gap, "gap")
    residual_values = _finite_tensor(residual, "residual")
    if gap_values.shape != residual_values.shape:
        raise ValueError("gap and residual must have the same shape")
    return gap_values / float(gap_scale), residual_values / float(residual_scale)


def _finite_tensor(value, name: str):
    import torch

    values = value.reshape(-1)
    if not bool(torch.isfinite(values).all()):
        raise FloatingPointError(f"{name} contains non-finite values")
    return values


def _positive_finite(value: float, name: str) -> None:
    if not np.isfinite(value) or float(value) <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
