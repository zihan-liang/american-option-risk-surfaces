"""Lightweight training helpers used by surrogate scripts."""

from __future__ import annotations

import numpy as np


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def train_linear_baseline(features: np.ndarray, target: np.ndarray, ridge: float = 1e-8) -> np.ndarray:
    x = np.column_stack([features, np.ones(len(features))])
    lhs = x.T @ x + ridge * np.eye(x.shape[1])
    rhs = x.T @ target
    return np.linalg.solve(lhs, rhs)


def predict_linear_baseline(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    x = np.column_stack([features, np.ones(len(features))])
    return x @ weights
