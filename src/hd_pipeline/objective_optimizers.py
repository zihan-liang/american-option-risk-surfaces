"""Small objective-balancing primitives used by conditional HD2 candidates.

CAGrad follows the official two-task reference implementation from
Cranial-XIX/CAGrad (dual simplex solve, alpha=0.5, rescale=1 by default).
ReLoBRaLo follows the author's reference update equations and repository
defaults T=1, alpha=.999, rho=1 for the two-loss HD2 setting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy.optimize import minimize


def cagrad_combine(
    task1_gradient: torch.Tensor,
    task2_gradient: torch.Tensor,
    *,
    alpha: float = 0.5,
    rescale: int = 1,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Combine two flat gradients using the official CAGrad dual update."""
    g1 = task1_gradient.reshape(-1)
    g2 = task2_gradient.reshape(-1)
    if g1.shape != g2.shape:
        raise ValueError("task gradients must have identical shape")
    if rescale not in (0, 1, 2):
        raise ValueError("rescale must be 0, 1, or 2")
    if alpha < 0:
        raise ValueError("alpha must be nonnegative")
    if not torch.isfinite(g1).all() or not torch.isfinite(g2).all():
        raise RuntimeError("nonfinite_cagrad_input")

    G = torch.stack((g1, g2), dim=1)
    g0 = G.mean(dim=1)
    if alpha == 0.0:
        return g0.detach(), {
            "task1_weight": 0.5,
            "task2_weight": 0.5,
            "lambda": 0.0,
            "solver_success": True,
            "alpha": 0.0,
            "rescale": int(rescale),
        }

    GG = (G.T @ G).detach().double().cpu().numpy()
    b = np.asarray([0.5, 0.5], dtype=np.float64)
    g0_norm = float(np.sqrt(max(float(b @ GG @ b), 0.0) + 1e-8))
    c = float(alpha) * g0_norm

    def objective(x: np.ndarray) -> float:
        quadratic = max(float(x @ GG @ x), 0.0)
        return float(x @ GG @ b + c * np.sqrt(quadratic + 1e-8))

    result = minimize(
        objective,
        b.copy(),
        bounds=((0.0, 1.0), (0.0, 1.0)),
        constraints=({"type": "eq", "fun": lambda x: 1.0 - float(np.sum(x))},),
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError(f"cagrad_dual_failed:{result.message}")
    weights = torch.tensor(result.x, device=G.device, dtype=G.dtype)
    gw = G @ weights
    gw_norm = torch.linalg.vector_norm(gw)
    lam = torch.as_tensor(c, device=G.device, dtype=G.dtype) / (gw_norm + 1e-4)
    combined = g0 + lam * gw
    if rescale == 1:
        combined = combined / (1.0 + float(alpha) ** 2)
    elif rescale == 2:
        combined = combined / (1.0 + float(alpha))
    if not torch.isfinite(combined).all():
        raise RuntimeError("nonfinite_cagrad_output")
    return combined.detach(), {
        "task1_weight": float(result.x[0]),
        "task2_weight": float(result.x[1]),
        "lambda": float(lam.detach().cpu()),
        "solver_success": bool(result.success),
        "alpha": float(alpha),
        "rescale": int(rescale),
    }


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


@dataclass
class ReLoBRaLoState:
    """Resume-safe two-or-more-task ReLoBRaLo reference state."""

    task_count: int
    temperature: float = 1.0
    alpha: float = 0.999
    rho: float = 1.0
    step: int = 0
    weights: list[float] | None = None
    previous_losses: list[float] | None = None
    initial_losses: list[float] | None = None

    def __post_init__(self) -> None:
        if self.task_count < 2:
            raise ValueError("ReLoBRaLo needs at least two objectives")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0.0 <= self.alpha <= 1.0 or not 0.0 <= self.rho <= 1.0:
            raise ValueError("alpha and rho must lie in [0,1]")
        if self.weights is None:
            self.weights = [1.0] * self.task_count
        if self.previous_losses is None:
            self.previous_losses = [1.0] * self.task_count
        if self.initial_losses is None:
            self.initial_losses = [1.0] * self.task_count
        if not (
            len(self.weights) == len(self.previous_losses) == len(self.initial_losses) == self.task_count
        ):
            raise ValueError("ReLoBRaLo state length mismatch")

    def update(self, losses: list[float] | tuple[float, ...]) -> list[float]:
        current = np.asarray(losses, dtype=np.float64)
        if current.shape != (self.task_count,) or not np.all(np.isfinite(current)):
            raise ValueError("invalid ReLoBRaLo losses")
        previous = np.maximum(np.asarray(self.previous_losses, dtype=np.float64), 1e-12)
        initial = np.maximum(np.asarray(self.initial_losses, dtype=np.float64), 1e-12)
        old_weights = np.asarray(self.weights, dtype=np.float64)
        lamb_hat = self.task_count * _softmax(current / (previous * self.temperature + 1e-12))
        lamb0_hat = self.task_count * _softmax(current / (initial * self.temperature + 1e-12))

        # Match the reference training schedule: alpha=1 at first update,
        # alpha=0 at second update, then the configured EMA alpha.
        if self.step == 0:
            alpha_t = 1.0
        elif self.step == 1:
            alpha_t = 0.0
        else:
            alpha_t = float(self.alpha)
        new_weights = (
            float(self.rho) * alpha_t * old_weights
            + (1.0 - float(self.rho)) * alpha_t * lamb0_hat
            + (1.0 - alpha_t) * lamb_hat
        )
        # The official rule preserves a sum equal to the number of tasks up to
        # floating error; normalize defensively without changing proportions.
        new_weights *= self.task_count / np.sum(new_weights)
        self.weights = [float(value) for value in new_weights]
        self.previous_losses = [float(value) for value in current]
        if self.step == 1:
            self.initial_losses = [float(value) for value in current]
        self.step += 1
        return list(self.weights)

    def state_dict(self) -> dict[str, Any]:
        return {
            "task_count": self.task_count,
            "temperature": self.temperature,
            "alpha": self.alpha,
            "rho": self.rho,
            "step": self.step,
            "weights": list(self.weights or []),
            "previous_losses": list(self.previous_losses or []),
            "initial_losses": list(self.initial_losses or []),
        }

    @classmethod
    def from_state_dict(cls, payload: dict[str, Any]) -> "ReLoBRaLoState":
        return cls(
            task_count=int(payload["task_count"]),
            temperature=float(payload["temperature"]),
            alpha=float(payload["alpha"]),
            rho=float(payload["rho"]),
            step=int(payload["step"]),
            weights=[float(value) for value in payload["weights"]],
            previous_losses=[float(value) for value in payload["previous_losses"]],
            initial_losses=[float(value) for value in payload["initial_losses"]],
        )
