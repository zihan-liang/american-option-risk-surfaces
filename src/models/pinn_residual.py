"""PINN-inspired residual and mask helpers for premium surrogates.

The residual helpers are deliberately auxiliary. They do not define a solver
and should only be used with PSOR-supervised labels and PSOR-derived masks.
"""

from __future__ import annotations

import numpy as np


def normalized_payoff_from_features(features):
    """Return payoff/K from dataset features in a differentiable torch form."""

    import torch

    x = features[:, 0]
    is_call = features[:, 6]
    spot_over_k = torch.exp(x)
    call_payoff = torch.relu(spot_over_k - 1.0)
    put_payoff = torch.relu(1.0 - spot_over_k)
    return torch.where(is_call > 0.5, call_payoff, put_payoff)


def premium_model_price(model, features):
    """Reconstruct normalized price from payoff/K plus nonnegative premium."""

    premium = model(features).reshape(-1)
    payoff = normalized_payoff_from_features(features)
    return payoff + premium


def log_space_black_scholes_residual(price, features):
    """Compute the log-space Black-Scholes residual for normalized price.

    Inputs follow the Stage 05 dataset schema:
    ``x=features[:,0]``, ``theta=tau/T=features[:,1]``,
    ``r=features[:,2]``, ``q=features[:,3]``, ``sigma=features[:,4]``,
    and ``T=features[:,5]``.
    """

    import torch

    if not features.requires_grad:
        raise ValueError("features must require gradients for residual evaluation")
    price = price.reshape(-1)
    first = torch.autograd.grad(
        price,
        features,
        grad_outputs=torch.ones_like(price),
        create_graph=True,
        retain_graph=True,
    )[0]
    p_x = first[:, 0]
    p_theta = first[:, 1]
    second = torch.autograd.grad(
        p_x,
        features,
        grad_outputs=torch.ones_like(p_x),
        create_graph=True,
        retain_graph=True,
    )[0]
    p_xx = second[:, 0]
    r = features[:, 2]
    q = features[:, 3]
    sigma = features[:, 4]
    maturity = torch.clamp(features[:, 5], min=torch.finfo(features.dtype).eps)
    return (p_theta / maturity) - 0.5 * sigma.pow(2) * p_xx - (r - q - 0.5 * sigma.pow(2)) * p_x + r * price


def residual_for_premium_model(model, features):
    """Compute residual for a positive-premium model on differentiable features."""

    price = premium_model_price(model, features)
    return log_space_black_scholes_residual(price, features)


def build_psor_continuation_mask(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    near_boundary: np.ndarray | None = None,
    near_kink: np.ndarray | None = None,
    gap_epsilon: float = 1.0e-5,
    tau_min_for_pde: float = 0.05,
    option_scope: str = "put",
) -> np.ndarray:
    """Build the masked continuation region from PSOR labels."""

    features = np.asarray(features)
    labels = np.asarray(labels)
    if features.ndim != 2 or features.shape[1] < 7:
        raise ValueError("features must have shape (n, >=7)")
    if labels.ndim != 2 or labels.shape[1] < 3:
        raise ValueError("labels must have shape (n, >=3)")
    if len(features) != len(labels):
        raise ValueError("features and labels must have the same row count")
    boundary_mask = _optional_bool_mask(near_boundary, len(features), "near_boundary")
    kink_mask = _optional_bool_mask(near_kink, len(features), "near_kink")

    gap = labels[:, 0] - labels[:, 2]
    mask = (gap > float(gap_epsilon)) & (~boundary_mask) & (~kink_mask) & (features[:, 1] > float(tau_min_for_pde))
    if option_scope == "put":
        mask &= features[:, 6] <= 0.5
    elif option_scope == "call":
        mask &= features[:, 6] > 0.5
    elif option_scope == "all":
        pass
    else:
        raise ValueError(f"unknown option_scope: {option_scope}")
    return mask.astype(bool)


def _optional_bool_mask(mask: np.ndarray | None, length: int, name: str) -> np.ndarray:
    values = np.zeros(length, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if len(values) != length:
        raise ValueError(f"{name} must match feature rows")
    return values


def residual_distribution_metrics(residual, *, prefix: str = "residual") -> dict[str, float | int]:
    """Summarize residual values with robust tail diagnostics."""

    if _is_torch_tensor(residual):
        values = residual.detach().cpu().numpy()
    else:
        values = np.asarray(residual)
    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    samples = int(values.size)
    if samples == 0:
        return {
            f"{prefix}_samples": 0,
            f"{prefix}_mean": float("nan"),
            f"{prefix}_mean_abs": float("nan"),
            f"{prefix}_rms": float("nan"),
            f"{prefix}_max_abs": float("nan"),
            f"{prefix}_q50_abs": float("nan"),
            f"{prefix}_q90_abs": float("nan"),
            f"{prefix}_q95_abs": float("nan"),
            f"{prefix}_q99_abs": float("nan"),
        }
    abs_values = np.abs(values)
    return {
        f"{prefix}_samples": samples,
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_mean_abs": float(np.mean(abs_values)),
        f"{prefix}_rms": float(np.sqrt(np.mean(values**2))),
        f"{prefix}_max_abs": float(np.max(abs_values)),
        f"{prefix}_q50_abs": float(np.quantile(abs_values, 0.50)),
        f"{prefix}_q90_abs": float(np.quantile(abs_values, 0.90)),
        f"{prefix}_q95_abs": float(np.quantile(abs_values, 0.95)),
        f"{prefix}_q99_abs": float(np.quantile(abs_values, 0.99)),
    }


def _is_torch_tensor(value) -> bool:
    try:
        import torch

        return isinstance(value, torch.Tensor)
    except Exception:
        return False
