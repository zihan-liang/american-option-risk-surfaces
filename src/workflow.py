"""Integrated modular surrogate workflow.

Price, boundary, and Delta remain separate validated components. This module
does not expose or imply a single model that solves all American option tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from models.boundary_head import BoundaryHeadModel
from models.delta_head import BoundedDeltaHead
from models.metrics import delta_bound_violation, error_metrics, masked_error_metrics, max_obstacle_violation
from models.premium_surrogate import PositivePremiumModel
from models.training import choose_device, predict_torch_model


@dataclass
class IntegratedWorkflow:
    price_model: object
    boundary_model: object
    delta_model: object
    device: str = "cpu"

    @classmethod
    def load(
        cls,
        price_model_path: str | Path,
        boundary_model_path: str | Path,
        delta_model_path: str | Path,
        *,
        device: str = "auto",
    ) -> "IntegratedWorkflow":
        resolved_device = choose_device(device)
        price_model = _load_price_model(price_model_path, resolved_device)
        boundary_model = _load_boundary_model(boundary_model_path, resolved_device)
        delta_model = _load_delta_model(delta_model_path, resolved_device)
        return cls(price_model=price_model, boundary_model=boundary_model, delta_model=delta_model, device=resolved_device)

    def predict(self, features: np.ndarray, payoff: np.ndarray, *, batch_size: int = 8192) -> dict[str, np.ndarray]:
        features = np.asarray(features, dtype=np.float32)
        payoff = np.asarray(payoff, dtype=np.float32)
        premium = predict_torch_model(self.price_model, features, batch_size=batch_size, device=self.device)
        price = payoff + premium
        boundary = predict_torch_model(self.boundary_model, features, batch_size=batch_size, device=self.device)
        delta = predict_torch_model(self.delta_model, features, batch_size=batch_size, device=self.device)
        return {"price": price, "premium": premium, "boundary": boundary, "delta": delta}


def integrated_metrics(
    *,
    prediction: dict[str, np.ndarray],
    features: np.ndarray,
    labels: np.ndarray,
    boundary: np.ndarray,
    strict_mask: np.ndarray,
) -> dict[str, float | int]:
    target_price = labels[:, 0]
    payoff = labels[:, 2]
    target_delta = labels[:, 3]
    price_metrics = error_metrics(prediction["price"], target_price)
    finite_boundary = np.isfinite(boundary)
    boundary_samples = int(np.count_nonzero(finite_boundary))
    if boundary_samples:
        boundary_metrics = error_metrics(prediction["boundary"][finite_boundary], boundary[finite_boundary])
    else:
        boundary_metrics = {"rmse": float("nan"), "mae": float("nan"), "max_error": float("nan")}
    strict = np.asarray(strict_mask, dtype=bool)
    strict_delta = masked_error_metrics(prediction["delta"], target_delta, strict, prefix="strict_delta")
    return {
        "price_rmse": price_metrics["rmse"],
        "price_mae": price_metrics["mae"],
        "price_max_error": price_metrics["max_error"],
        "boundary_samples": boundary_samples,
        "boundary_rmse": boundary_metrics["rmse"],
        "boundary_mae": boundary_metrics["mae"],
        "boundary_max_error": boundary_metrics["max_error"],
        **strict_delta,
        "max_price_obstacle_violation": max_obstacle_violation(prediction["price"], payoff),
        "max_delta_bound_violation": delta_bound_violation(prediction["delta"], features[:, 6]),
    }


def safe_speedup(psor_seconds: float, surrogate_seconds: float) -> float:
    if not np.isfinite(psor_seconds) or not np.isfinite(surrogate_seconds):
        return float("nan")
    if surrogate_seconds == 0.0:
        return float("inf")
    return float(psor_seconds / surrogate_seconds)


def _load_price_model(path: str | Path, device: str):
    import torch

    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    metadata = checkpoint.get("metadata", {})
    model = PositivePremiumModel.build(
        int(metadata.get("input_dim", 7)),
        hidden=int(metadata.get("hidden", 128)),
        depth=int(metadata.get("depth", 3)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model


def _load_boundary_model(path: str | Path, device: str):
    import torch

    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    metadata = checkpoint.get("metadata", {})
    model = BoundaryHeadModel.build(
        int(metadata.get("input_dim", 7)),
        hidden=int(metadata.get("hidden", 128)),
        depth=int(metadata.get("depth", 3)),
        lower=metadata.get("lower"),
        upper=metadata.get("upper"),
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model


def _load_delta_model(path: str | Path, device: str):
    import torch

    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    metadata = checkpoint.get("metadata", {})
    model = BoundedDeltaHead.build(
        int(metadata.get("input_dim", 7)),
        hidden=int(metadata.get("hidden", 128)),
        depth=int(metadata.get("depth", 3)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model
