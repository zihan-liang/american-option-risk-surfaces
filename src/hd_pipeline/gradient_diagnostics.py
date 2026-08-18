"""Objective-specific gradient geometry diagnostics for HD2.

The helpers intentionally return detached Python scalars only.  They never keep
parameter-gradient tensors or autograd graphs in training histories.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

import torch


def _trainable_parameters(model_or_parameters: Any) -> list[torch.nn.Parameter]:
    if isinstance(model_or_parameters, torch.nn.Module):
        source = model_or_parameters.parameters()
    else:
        source = model_or_parameters
    return [parameter for parameter in source if parameter.requires_grad]


def flatten_grads(
    grads: tuple[torch.Tensor | None, ...],
    parameters: Iterable[torch.nn.Parameter],
) -> torch.Tensor:
    """Flatten a gradient tuple, replacing unused entries with exact zeros."""
    parts: list[torch.Tensor] = []
    for grad, parameter in zip(grads, parameters, strict=True):
        value = torch.zeros_like(parameter) if grad is None else grad
        parts.append(value.reshape(-1))
    if not parts:
        return torch.empty(0, dtype=torch.float64)
    return torch.cat(parts)


def _finite_float(value: torch.Tensor) -> float:
    result = float(value.detach().cpu())
    if not math.isfinite(result):
        raise RuntimeError("nonfinite_gradient_diagnostic")
    return result


def _diagnostics_for_parameters(
    loss_sup: torch.Tensor,
    loss_phys: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    *,
    lambda_weight: float,
) -> dict[str, float]:
    if loss_sup.ndim != 0 or loss_phys.ndim != 0:
        raise ValueError("objective losses must be scalar tensors")
    if not parameters:
        raise ValueError("at least one trainable parameter is required")
    g_sup_tuple = torch.autograd.grad(
        loss_sup,
        parameters,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    g_phys_tuple = torch.autograd.grad(
        loss_phys,
        parameters,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    g_sup = flatten_grads(g_sup_tuple, parameters)
    g_phys = flatten_grads(g_phys_tuple, parameters)
    lambda_tensor = torch.as_tensor(float(lambda_weight), device=g_sup.device, dtype=g_sup.dtype)
    g_total = g_sup + lambda_tensor * g_phys

    eps = torch.finfo(g_sup.dtype).eps
    sup_norm = torch.linalg.vector_norm(g_sup)
    phys_norm = torch.linalg.vector_norm(g_phys)
    denom = sup_norm * phys_norm
    cosine = torch.where(
        denom > eps,
        torch.dot(g_sup, g_phys) / denom,
        torch.zeros((), device=g_sup.device, dtype=g_sup.dtype),
    )
    raw_ratio = phys_norm / torch.clamp(sup_norm, min=eps)
    weighted_ratio = torch.abs(lambda_tensor) * raw_ratio
    return {
        "sup_grad_norm": _finite_float(sup_norm),
        "phys_grad_norm": _finite_float(phys_norm),
        "raw_norm_ratio": _finite_float(raw_ratio),
        "weighted_norm_ratio": _finite_float(weighted_ratio),
        "cosine": _finite_float(cosine),
        "sup_dot_total": _finite_float(torch.dot(g_sup, g_total)),
        "phys_dot_total": _finite_float(torch.dot(g_phys, g_total)),
    }


def objective_gradient_diagnostics(
    loss_sup: torch.Tensor,
    loss_phys: torch.Tensor,
    model: torch.nn.Module | Iterable[torch.nn.Parameter],
    *,
    lambda_weight: float,
) -> dict[str, float]:
    """Return global gradient magnitude/conflict diagnostics for two objectives."""
    return _diagnostics_for_parameters(
        loss_sup,
        loss_phys,
        _trainable_parameters(model),
        lambda_weight=lambda_weight,
    )


def layerwise_gradient_diagnostics(
    loss_sup: torch.Tensor,
    loss_phys: torch.Tensor,
    named_parameter_groups: Mapping[str, Iterable[torch.nn.Parameter]],
    *,
    lambda_weight: float,
) -> list[dict[str, float | str]]:
    """Return the same gradient geometry metrics for named parameter groups."""
    rows: list[dict[str, float | str]] = []
    for name, parameter_group in named_parameter_groups.items():
        parameters = _trainable_parameters(parameter_group)
        if not parameters:
            continue
        row: dict[str, float | str] = {"group": str(name)}
        row.update(
            _diagnostics_for_parameters(
                loss_sup,
                loss_phys,
                parameters,
                lambda_weight=lambda_weight,
            )
        )
        rows.append(row)
    return rows
