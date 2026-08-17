"""High-dimensional Black--Scholes operator helpers for the HD1 pilot."""
from __future__ import annotations

import math
import torch


def covariance_matrix(d: int, sigma: float, rho: float, *, device=None, dtype=None) -> torch.Tensor:
    if d < 1:
        raise ValueError("d must be positive")
    if not (-1.0 / max(d - 1, 1) < rho < 1.0) and d > 1:
        raise ValueError("rho is not positive-definite for this dimension")
    eye = torch.eye(d, device=device, dtype=dtype)
    ones = torch.ones((d, d), device=device, dtype=dtype)
    return sigma**2 * ((1.0 - rho) * eye + rho * ones)


def effective_geometric_basket_parameters(d: int, sigma: float, q: float, rho: float) -> tuple[float, float]:
    if d < 1:
        raise ValueError("d must be positive")
    sigma_g2 = sigma**2 * (rho + (1.0 - rho) / d)
    sigma_g = math.sqrt(sigma_g2)
    q_g = q + 0.5 * (sigma**2 - sigma_g2)
    return sigma_g, q_g


def geometric_basket_from_log_moneyness(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError("x must have shape (batch, d)")
    return torch.exp(x.mean(dim=1))


def geometric_put_payoff(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(1.0 - geometric_basket_from_log_moneyness(x))


def _first_derivatives(value: torch.Tensor, x: torch.Tensor, tau: torch.Tensor):
    v_x = torch.autograd.grad(value.sum(), x, create_graph=True, retain_graph=True)[0]
    v_tau = torch.autograd.grad(value.sum(), tau, create_graph=True, retain_graph=True)[0].reshape(-1)
    return v_x, v_tau


def full_hessian_contraction(value: torch.Tensor, x: torch.Tensor, sigma_matrix: torch.Tensor) -> torch.Tensor:
    """Exact per-sample tr(Sigma Hessian_x value); canonical HD1 uses this only for d<=2."""
    if x.ndim != 2 or sigma_matrix.shape != (x.shape[1], x.shape[1]):
        raise ValueError("incompatible x/covariance shape")
    grad = torch.autograd.grad(value.sum(), x, create_graph=True, retain_graph=True)[0]
    terms = []
    for i in range(x.shape[1]):
        row = torch.autograd.grad(grad[:, i].sum(), x, create_graph=True, retain_graph=True)[0]
        terms.append((row * sigma_matrix[i].reshape(1, -1)).sum(dim=1))
    return torch.stack(terms, dim=1).sum(dim=1)


def fixed_rademacher_probes(d: int, count: int, seed: int, *, device=None, dtype=None) -> torch.Tensor:
    if count < 1:
        raise ValueError("count must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    probes = torch.randint(0, 2, (count, d), generator=generator, dtype=torch.int64)
    probes = probes.mul(2).sub(1).to(dtype=dtype or torch.get_default_dtype())
    return probes.to(device=device)


def hutchinson_hessian_contraction(
    value: torch.Tensor,
    x: torch.Tensor,
    sigma_matrix: torch.Tensor,
    *,
    probe_count: int = 2,
    probe_seed: int = 101,
) -> torch.Tensor:
    """Fixed-seed Hutchinson estimate of tr(Sigma H) via Hessian-vector products."""
    d = x.shape[1]
    if sigma_matrix.shape != (d, d):
        raise ValueError("incompatible covariance shape")
    grad = torch.autograd.grad(value.sum(), x, create_graph=True, retain_graph=True)[0]
    chol = torch.linalg.cholesky(sigma_matrix)
    xis = fixed_rademacher_probes(d, probe_count, probe_seed, device=x.device, dtype=x.dtype)
    estimates = []
    for xi in xis:
        z = chol @ xi
        directional = (grad * z.reshape(1, -1)).sum(dim=1)
        hz = torch.autograd.grad(directional.sum(), x, create_graph=True, retain_graph=True)[0]
        estimates.append((hz * z.reshape(1, -1)).sum(dim=1))
    return torch.stack(estimates, dim=0).mean(dim=0)


def black_scholes_vi_residual(
    value: torch.Tensor,
    x: torch.Tensor,
    tau: torch.Tensor,
    *,
    r: float,
    q: float,
    sigma: float,
    rho: float,
    hessian_mode: str,
    probe_count: int = 2,
    probe_seed: int = 101,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the frozen HD1 residual and covariance-weighted Hessian contraction."""
    d = x.shape[1]
    covariance = covariance_matrix(d, sigma, rho, device=x.device, dtype=x.dtype)
    v_x, v_tau = _first_derivatives(value, x, tau)
    if hessian_mode == "full":
        contraction = full_hessian_contraction(value, x, covariance)
    elif hessian_mode == "hutchinson":
        contraction = hutchinson_hessian_contraction(
            value, x, covariance, probe_count=probe_count, probe_seed=probe_seed
        )
    else:
        raise ValueError("hessian_mode must be 'full' or 'hutchinson'")
    drift = r - q - 0.5 * sigma**2
    residual = v_tau - 0.5 * contraction - drift * v_x.sum(dim=1) + r * value.reshape(-1)
    return residual, contraction


def fischer_burmeister(gap: torch.Tensor, residual: torch.Tensor, epsilon: float = 0.0) -> torch.Tensor:
    if epsilon < 0:
        raise ValueError("epsilon must be nonnegative")
    return torch.sqrt(gap.square() + residual.square() + float(epsilon) ** 2) - gap - residual
