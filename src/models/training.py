"""Lightweight training helpers used by surrogate scripts."""

from __future__ import annotations

from pathlib import Path
import random

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


def load_dataset_arrays(path: str | Path) -> dict[str, np.ndarray]:
    required = [
        "features",
        "labels",
        "split",
        "regime_key",
        "near_boundary",
    ]
    with np.load(path) as data:
        missing = [key for key in required if key not in data.files]
        if missing:
            raise KeyError(f"dataset missing required arrays: {missing}")
        return {key: data[key] for key in data.files}


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def split_mask(split: np.ndarray, name: str) -> np.ndarray:
    values = split.astype(str) if split.dtype.kind == "S" else split.astype(str)
    return values == name


def capped_indices(mask: np.ndarray, cap: int | None, seed: int) -> np.ndarray:
    indices = np.flatnonzero(mask)
    if cap is None or cap <= 0 or len(indices) <= cap:
        return indices
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=cap, replace=False))


def choose_device(requested: str | None = None) -> str:
    import torch

    if requested and requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def train_torch_price_model(
    model,
    train_x: np.ndarray,
    train_price: np.ndarray,
    train_payoff: np.ndarray,
    val_x: np.ndarray,
    val_price: np.ndarray,
    *,
    penalty_weight: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    device: str,
    model_name: str,
) -> tuple[object, list[dict[str, float | int | str]]]:
    import torch
    import torch.nn.functional as F

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    x_train = torch.as_tensor(train_x, dtype=torch.float32)
    y_train = torch.as_tensor(train_price[:, None], dtype=torch.float32)
    payoff_train = torch.as_tensor(train_payoff[:, None], dtype=torch.float32)
    x_val = torch.as_tensor(val_x, dtype=torch.float32, device=device)
    y_val = torch.as_tensor(val_price[:, None], dtype=torch.float32, device=device)
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val = float("inf")
    stale = 0
    history: list[dict[str, float | int | str]] = []
    n = len(x_train)
    for epoch in range(1, epochs + 1):
        order = torch.randperm(n)
        total_loss = 0.0
        model.train()
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            xb = x_train[idx].to(device)
            yb = y_train[idx].to(device)
            payoffb = payoff_train[idx].to(device)
            pred = model(xb)
            mse = F.mse_loss(pred, yb)
            obstacle_penalty = F.relu(payoffb - pred).pow(2).mean()
            loss = mse + penalty_weight * obstacle_penalty
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(idx)
        model.eval()
        with torch.no_grad():
            val_pred = model(x_val)
            val_rmse = torch.sqrt(F.mse_loss(val_pred, y_val)).item()
        history.append(
            {
                "model": model_name,
                "epoch": epoch,
                "train_loss": total_loss / n,
                "validation_rmse": val_rmse,
                "penalty_weight": penalty_weight,
            }
        )
        if val_rmse < best_val:
            best_val = val_rmse
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return model, history


def train_torch_premium_model(
    model,
    train_x: np.ndarray,
    train_premium: np.ndarray,
    val_x: np.ndarray,
    val_premium: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    device: str,
    model_name: str,
) -> tuple[object, list[dict[str, float | int | str]]]:
    import torch
    import torch.nn.functional as F

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    x_train = torch.as_tensor(train_x, dtype=torch.float32)
    y_train = torch.as_tensor(train_premium[:, None], dtype=torch.float32)
    x_val = torch.as_tensor(val_x, dtype=torch.float32, device=device)
    y_val = torch.as_tensor(val_premium[:, None], dtype=torch.float32, device=device)
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val = float("inf")
    stale = 0
    history: list[dict[str, float | int | str]] = []
    n = len(x_train)
    for epoch in range(1, epochs + 1):
        order = torch.randperm(n)
        total_loss = 0.0
        model.train()
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            xb = x_train[idx].to(device)
            yb = y_train[idx].to(device)
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(idx)
        model.eval()
        with torch.no_grad():
            val_pred = model(x_val)
            val_rmse = torch.sqrt(F.mse_loss(val_pred, y_val)).item()
        history.append(
            {
                "model": model_name,
                "epoch": epoch,
                "train_loss": total_loss / n,
                "validation_premium_rmse": val_rmse,
            }
        )
        if val_rmse < best_val:
            best_val = val_rmse
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return model, history


def train_torch_regression_model(
    model,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    device: str,
    model_name: str,
    metric_name: str = "validation_rmse",
) -> tuple[object, list[dict[str, float | int | str]]]:
    import torch
    import torch.nn.functional as F

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    x_train = torch.as_tensor(train_x, dtype=torch.float32)
    y_train = torch.as_tensor(train_y[:, None], dtype=torch.float32)
    x_val = torch.as_tensor(val_x, dtype=torch.float32, device=device)
    y_val = torch.as_tensor(val_y[:, None], dtype=torch.float32, device=device)
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val = float("inf")
    stale = 0
    history: list[dict[str, float | int | str]] = []
    n = len(x_train)
    for epoch in range(1, epochs + 1):
        order = torch.randperm(n)
        total_loss = 0.0
        model.train()
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            xb = x_train[idx].to(device)
            yb = y_train[idx].to(device)
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(idx)
        model.eval()
        with torch.no_grad():
            val_pred = model(x_val)
            val_rmse = torch.sqrt(F.mse_loss(val_pred, y_val)).item()
        history.append(
            {
                "model": model_name,
                "epoch": epoch,
                "train_loss": total_loss / n,
                metric_name: val_rmse,
            }
        )
        if val_rmse < best_val:
            best_val = val_rmse
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return model, history


def predict_torch_model(model, features: np.ndarray, *, batch_size: int, device: str) -> np.ndarray:
    import torch

    model = model.to(device)
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            xb = torch.as_tensor(features[start : start + batch_size], dtype=torch.float32, device=device)
            preds.append(model(xb).detach().cpu().numpy().reshape(-1))
    return np.concatenate(preds)


def save_torch_model(model, path: str | Path, metadata: dict) -> None:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)
