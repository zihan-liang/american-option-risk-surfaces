"""Training and prediction helpers for M10 structured surrogate."""
from __future__ import annotations

import time
from typing import Any, Mapping

import numpy as np

from models.mlp import build_mlp
import m10_structured_surrogate_core as core
from m10_structured_surrogate_data import FEATURE_NAMES, analytic_european_surface, normalize_features


def build_seeded_model(*, seed: int, hidden: int, depth: int):
    import torch
    torch.manual_seed(int(seed))
    return build_mlp(len(FEATURE_NAMES), 1, hidden=int(hidden), depth=int(depth)).to(device="cpu", dtype=torch.float64)


def model_parameter_count(model: Any) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def predict_reconstructed(model: Any, x: np.ndarray, european: np.ndarray, tau: np.ndarray, *, arm: str, scales: Mapping[str, float], batch_size: int) -> np.ndarray:
    import torch
    model.eval(); rows: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), int(batch_size)):
            stop = start + int(batch_size)
            xb = torch.as_tensor(x[start:stop], dtype=torch.float64)
            eb = torch.as_tensor(european[start:stop, None], dtype=torch.float64)
            tb = torch.as_tensor(tau[start:stop, None], dtype=torch.float64)
            rows.append(core.reconstruct_torch(model(xb), eb, tb, arm, scales).cpu().numpy().reshape(-1))
    return np.concatenate(rows)


def train_model(model: Any, train_x: np.ndarray, train_y: np.ndarray, train_e: np.ndarray, train_tau: np.ndarray,
                val_x: np.ndarray, val_y: np.ndarray, val_e: np.ndarray, val_tau: np.ndarray,
                spec: Mapping[str, Any], *, seed: int, arm: str, scales: Mapping[str, float]):
    import torch
    import torch.nn.functional as F
    torch.manual_seed(int(seed))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(spec["learning_rate"]))
    x = torch.as_tensor(train_x, dtype=torch.float64); y = torch.as_tensor(train_y[:, None], dtype=torch.float64)
    e = torch.as_tensor(train_e[:, None], dtype=torch.float64); tau = torch.as_tensor(train_tau[:, None], dtype=torch.float64)
    batch_size = int(spec["batch_size"]); history: list[dict[str, Any]] = []; started = time.perf_counter()
    for epoch in range(1, int(spec["epochs"]) + 1):
        g = torch.Generator(device="cpu"); g.manual_seed(int(seed) * 1000 + epoch)
        order = torch.randperm(len(x), generator=g); total = 0.0; model.train()
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            pred = core.reconstruct_torch(model(x[idx]), e[idx], tau[idx], arm, scales)
            loss = F.mse_loss(pred, y[idx]); optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            total += float(loss.detach().cpu()) * len(idx)
        val_pred = predict_reconstructed(model, val_x, val_e, val_tau, arm=arm, scales=scales, batch_size=batch_size)
        history.append({"seed": seed, "arm": arm, "epoch": epoch,
                        "train_reconstructed_price_mse": total / max(len(x), 1),
                        "validation_reconstructed_price_rmse": float(np.sqrt(np.mean((val_pred - val_y) ** 2)))})
    return model, history, float(time.perf_counter() - started)


def predict_surface(model: Any, surface: Mapping[str, Any], mean: np.ndarray, std: np.ndarray, model_spec: Mapping[str, Any],
                    *, arm: str, scales: Mapping[str, float], recompute_anchor: bool) -> np.ndarray:
    from m10_structured_surrogate_metrics import enforce_known_constraints
    result, cfg, features = surface["result"], surface["cfg"], surface["features"]
    x = normalize_features(features, mean, std)
    if arm == "direct_price": european = np.zeros(len(features), dtype=np.float64)
    elif recompute_anchor: european = analytic_european_surface(result, cfg).ravel()
    else: european = np.asarray(surface["european"], dtype=np.float64)
    pred = predict_reconstructed(model, x, european, features[:, 1], arm=arm, scales=scales, batch_size=int(model_spec["batch_size"]))
    return enforce_known_constraints((pred * cfg.k).reshape(result.values.shape), result, cfg)


def time_inference(model: Any, surface: Mapping[str, Any], mean: np.ndarray, std: np.ndarray, model_spec: Mapping[str, Any],
                   *, arm: str, scales: Mapping[str, float], timing: Mapping[str, Any]):
    for _ in range(int(timing["warmups"])):
        predict_surface(model, surface, mean, std, model_spec, arm=arm, scales=scales, recompute_anchor=True)
    samples=[]
    for _ in range(int(timing["repetitions"])):
        started=time.perf_counter(); predict_surface(model, surface, mean, std, model_spec, arm=arm, scales=scales, recompute_anchor=True)
        samples.append(time.perf_counter()-started)
    return float(np.mean(samples)), float(np.std(samples, ddof=1)) if len(samples)>1 else 0.0
