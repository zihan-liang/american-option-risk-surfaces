"""Integrated modular surrogate workflow.

This module intentionally keeps price, boundary, and Delta as separate components.
It does not expose a single "solve everything" neural model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from models.training import predict_linear_baseline


@dataclass
class LinearWorkflow:
    price_weights: np.ndarray
    boundary_weights: np.ndarray
    delta_weights: np.ndarray

    def predict(self, features: np.ndarray, payoff: np.ndarray) -> dict[str, np.ndarray]:
        premium = np.maximum(np.expm1(predict_linear_baseline(features, self.price_weights)), 0.0)
        price = payoff + premium
        boundary = predict_linear_baseline(features, self.boundary_weights)
        raw_delta = predict_linear_baseline(features, self.delta_weights)
        delta = np.where(features[:, 6] > 0.5, np.clip(raw_delta, 0.0, 1.0), np.clip(raw_delta, -1.0, 0.0))
        return {"price": price, "premium": premium, "boundary": boundary, "delta": delta}
