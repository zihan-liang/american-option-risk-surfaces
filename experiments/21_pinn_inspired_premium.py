"""Stage 21: PINN-inspired positive-premium future-extension experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from common import ensure_dir, load_config, write_csv, write_json
from models.metrics import boundary_band_error, error_metrics, max_obstacle_violation
from models.pinn_residual import (
    build_psor_continuation_mask,
    premium_model_price,
    residual_distribution_metrics,
    residual_for_premium_model,
)
from models.premium_surrogate import PositivePremiumModel, reconstruct_price
from models.training import capped_indices, choose_device, load_dataset_arrays, set_global_seed, split_mask


DEFAULT_OUTPUT_ROOT = "results/experiment_21_pinn_inspired_premium"


def run(
    config_path: str,
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    model_root: str | Path = "results/models",
    overrides: dict | None = None,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    if overrides:
        cfg = {**cfg, **overrides}
    set_global_seed(int(cfg.get("seed", 21)))
    out_dir = ensure_dir(output_root)
    ensure_dir(model_root)
    write_json(out_dir / "pinn_config_manifest.json", cfg)

    data = load_dataset_arrays(cfg["dataset"])
    features = data["features"].astype(np.float32)
    labels = data["labels"].astype(np.float32)
    split = data["split"]
    price = labels[:, 0]
    premium = labels[:, 1]
    payoff = labels[:, 2]
    near_boundary = data.get("near_boundary", np.zeros(len(features), dtype=bool)).astype(bool)
    option_scope = str(cfg.get("option_scope", "put"))
    scope_mask = _option_scope_mask(features, option_scope)
    pde_mask = build_psor_continuation_mask(
        features,
        labels,
        near_boundary=near_boundary,
        gap_epsilon=float(cfg.get("gap_epsilon", 1.0e-5)),
        tau_min_for_pde=float(cfg.get("tau_min_for_pde", 0.05)),
        option_scope=option_scope,
    )

    seed = int(cfg.get("seed", 21))
    train_idx = capped_indices(split_mask(split, "train") & scope_mask, int(cfg.get("train_cap", 0)), seed)
    val_idx = capped_indices(split_mask(split, "validation") & scope_mask, int(cfg.get("validation_cap", 0)), seed + 1)
    test_idx = capped_indices(split_mask(split, "test") & scope_mask, int(cfg.get("test_cap", 0)), seed + 2)
    pde_train_idx = np.intersect1d(train_idx, np.flatnonzero(pde_mask), assume_unique=False)
    eval_idx = np.concatenate([val_idx, test_idx])

    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0 or len(pde_train_idx) == 0:
        summary = {
            "decision": "insufficient_pde_mask_samples",
            "option_scope": option_scope,
            "train_samples": int(len(train_idx)),
            "validation_samples": int(len(val_idx)),
            "test_samples": int(len(test_idx)),
            "pde_train_samples": int(len(pde_train_idx)),
            "note": "No PINN-inspired model was trained because the PSOR-derived continuation mask had insufficient samples.",
        }
        _write_empty_outputs(out_dir, summary)
        return summary

    device = choose_device(cfg.get("device", "auto"))
    dtype = _torch_dtype(str(cfg.get("training_dtype", "float64")))
    metrics_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    model_summaries: list[dict[str, Any]] = []

    for model_name, use_pde in [
        ("supervised_positive_premium_mlp", False),
        ("pinn_inspired_premium_mlp", True),
    ]:
        model = PositivePremiumModel.build(features.shape[1], hidden=int(cfg.get("hidden", 64)), depth=int(cfg.get("depth", 2)))
        start = time.perf_counter()
        trained, history = _train_model(
            model,
            model_name=model_name,
            features=features,
            premium=premium,
            payoff=payoff,
            train_idx=train_idx,
            val_idx=val_idx,
            pde_train_idx=pde_train_idx,
            cfg=cfg,
            device=device,
            dtype=dtype,
            use_pde=use_pde,
        )
        train_seconds = time.perf_counter() - start
        history_rows.extend(history)
        inference_start = time.perf_counter()
        predicted_premium = _predict_premium(
            trained,
            features[eval_idx],
            batch_size=int(cfg.get("batch_size", 2048)),
            device=device,
            dtype=dtype,
        )
        inference_seconds = time.perf_counter() - inference_start
        predicted_price = reconstruct_price(payoff[eval_idx], predicted_premium)
        _record_metrics(
            model_name,
            predicted_price,
            predicted_premium,
            eval_idx,
            data,
            price,
            premium,
            payoff,
            near_boundary,
            metrics_rows,
            train_seconds=train_seconds,
            inference_seconds=inference_seconds,
            obstacle_tolerance=float(cfg.get("obstacle_tolerance", 1.0e-10)),
        )
        residual_rows.extend(
            _residual_rows(
                model_name,
                trained,
                features,
                eval_idx,
                pde_mask,
                split,
                cfg,
                device=device,
                dtype=dtype,
                seed=seed,
            )
        )
        model_summaries.append(
            {
                "model": model_name,
                "train_seconds": train_seconds,
                "inference_seconds": inference_seconds,
            }
        )

    write_csv(out_dir / "pinn_metrics.csv", metrics_rows)
    write_csv(out_dir / "pinn_residual_metrics.csv", residual_rows)
    write_csv(out_dir / "pinn_training_history.csv", history_rows)
    selected = next(
        row
        for row in metrics_rows
        if row["model"] == "pinn_inspired_premium_mlp" and row["split"] == "test" and row["option_type"] == "all"
    )
    boundary = _boundary_comparison(
        "pinn_inspired_premium_mlp",
        predicted_premium,
        eval_idx,
        data,
        threshold=float(cfg.get("premium_boundary_threshold", 1.0e-3)),
    )
    summary = {
        "decision": "pinn_inspired_future_extension",
        "option_scope": option_scope,
        "models": model_summaries,
        "train_samples": int(len(train_idx)),
        "validation_samples": int(len(val_idx)),
        "test_samples": int(len(test_idx)),
        "pde_train_samples": int(len(pde_train_idx)),
        "pde_total_mask_samples": int(np.count_nonzero(pde_mask)),
        "selected_model": "pinn_inspired_premium_mlp",
        "selected_test_price_rmse": selected["price_rmse"],
        "selected_test_premium_rmse": selected["premium_rmse"],
        "selected_test_max_obstacle_violation": selected["max_obstacle_violation"],
        "boundary_diagnostic": boundary,
        "device": device,
        "training_dtype": str(cfg.get("training_dtype", "float64")),
        "note": "This PINN-inspired future extension uses PSOR labels and a masked continuation-region residual; it does not replace CN/PSOR.",
    }
    write_json(out_dir / "pinn_summary.json", summary)
    return summary


def _train_model(
    model,
    *,
    model_name: str,
    features: np.ndarray,
    premium: np.ndarray,
    payoff: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    pde_train_idx: np.ndarray,
    cfg: dict[str, Any],
    device: str,
    dtype,
    use_pde: bool,
):
    import torch
    import torch.nn.functional as F

    model = model.to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.get("learning_rate", 1.0e-3)))
    x_train = torch.as_tensor(features[train_idx], dtype=dtype)
    y_train = torch.as_tensor(premium[train_idx, None], dtype=dtype)
    payoff_train = torch.as_tensor(payoff[train_idx, None], dtype=dtype)
    x_val = torch.as_tensor(features[val_idx], dtype=dtype, device=device)
    y_val = torch.as_tensor(premium[val_idx, None], dtype=dtype, device=device)
    pde_pool = torch.as_tensor(features[pde_train_idx], dtype=dtype)
    lambda_sup = float(cfg.get("lambda_sup", 1.0))
    lambda_pde = float(cfg.get("lambda_pde", 1.0e-4)) if use_pde else 0.0
    lambda_obs = float(cfg.get("lambda_obs", 1.0))
    pde_batch_size = min(int(cfg.get("pde_batch_cap", 512)), len(pde_pool))
    batch_size = int(cfg.get("batch_size", 2048))
    patience = int(cfg.get("patience", 3))
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_val = float("inf")
    stale = 0
    history: list[dict[str, Any]] = []
    n_train = len(x_train)
    for epoch in range(1, int(cfg.get("epochs", 8)) + 1):
        order = torch.randperm(n_train)
        model.train()
        totals = {"loss": 0.0, "supervised": 0.0, "pde": 0.0, "obstacle": 0.0}
        for start in range(0, n_train, batch_size):
            idx = order[start : start + batch_size]
            xb = x_train[idx].to(device)
            yb = y_train[idx].to(device)
            payoffb = payoff_train[idx].to(device)
            pred_premium = model(xb)
            supervised_loss = F.mse_loss(pred_premium, yb)
            pred_price = payoffb + pred_premium
            obstacle_loss = F.relu(payoffb - pred_price).pow(2).mean()
            pde_loss = torch.zeros((), dtype=dtype, device=device)
            if lambda_pde > 0.0 and pde_batch_size > 0:
                pde_idx = torch.randint(0, len(pde_pool), (pde_batch_size,))
                pde_features = pde_pool[pde_idx].to(device).detach().clone().requires_grad_(True)
                residual = residual_for_premium_model(model, pde_features)
                pde_loss = F.huber_loss(
                    residual,
                    torch.zeros_like(residual),
                    delta=float(cfg.get("pde_huber_delta", 1.0)),
                )
            loss = lambda_sup * supervised_loss + lambda_obs * obstacle_loss + lambda_pde * pde_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_n = len(idx)
            totals["loss"] += float(loss.detach().cpu()) * batch_n
            totals["supervised"] += float(supervised_loss.detach().cpu()) * batch_n
            totals["pde"] += float(pde_loss.detach().cpu()) * batch_n
            totals["obstacle"] += float(obstacle_loss.detach().cpu()) * batch_n
        model.eval()
        with torch.no_grad():
            val_pred = model(x_val)
            val_rmse = torch.sqrt(F.mse_loss(val_pred, y_val)).item()
        history.append(
            {
                "model": model_name,
                "epoch": epoch,
                "train_loss": totals["loss"] / n_train,
                "train_supervised_loss": totals["supervised"] / n_train,
                "train_pde_loss": totals["pde"] / n_train,
                "train_obstacle_loss": totals["obstacle"] / n_train,
                "validation_premium_rmse": val_rmse,
                "lambda_pde": lambda_pde,
            }
        )
        if val_rmse < best_val:
            best_val = val_rmse
            stale = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return model, history


def _predict_premium(model, features: np.ndarray, *, batch_size: int, device: str, dtype) -> np.ndarray:
    import torch

    model = model.to(device=device, dtype=dtype)
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            xb = torch.as_tensor(features[start : start + batch_size], dtype=dtype, device=device)
            preds.append(model(xb).detach().cpu().numpy().reshape(-1))
    return np.concatenate(preds) if preds else np.array([], dtype=float)


def _record_metrics(
    model_name: str,
    predicted_price: np.ndarray,
    predicted_premium: np.ndarray,
    eval_idx: np.ndarray,
    data: dict[str, np.ndarray],
    target_price: np.ndarray,
    target_premium: np.ndarray,
    payoff: np.ndarray,
    near_boundary: np.ndarray,
    metrics_rows: list[dict[str, Any]],
    *,
    train_seconds: float,
    inference_seconds: float,
    obstacle_tolerance: float,
) -> None:
    splits = data["split"].astype(str)[eval_idx]
    is_call = data["features"][eval_idx, 6]
    for split_name in ["validation", "test"]:
        split_selected = splits == split_name
        for option_type in ["all", "put", "call"]:
            if option_type == "all":
                mask = split_selected
            elif option_type == "put":
                mask = split_selected & (is_call <= 0.5)
            else:
                mask = split_selected & (is_call > 0.5)
            if not np.any(mask):
                continue
            price_metrics = error_metrics(predicted_price[mask], target_price[eval_idx][mask])
            premium_metrics = error_metrics(predicted_premium[mask], target_premium[eval_idx][mask])
            boundary = boundary_band_error(predicted_price[mask], target_price[eval_idx][mask], near_boundary[eval_idx][mask])
            violation = np.maximum(payoff[eval_idx][mask] - predicted_price[mask], 0.0)
            metrics_rows.append(
                {
                    "model": model_name,
                    "split": split_name,
                    "option_type": option_type,
                    "samples": int(np.count_nonzero(mask)),
                    "price_rmse": price_metrics["rmse"],
                    "price_mae": price_metrics["mae"],
                    "price_max_error": price_metrics["max_error"],
                    "premium_rmse": premium_metrics["rmse"],
                    "premium_mae": premium_metrics["mae"],
                    "premium_max_error": premium_metrics["max_error"],
                    "max_obstacle_violation": max_obstacle_violation(predicted_price[mask], payoff[eval_idx][mask]),
                    "obstacle_violation_fraction": float(np.mean(violation > obstacle_tolerance)),
                    **boundary,
                    "train_seconds": train_seconds,
                    "inference_seconds": inference_seconds,
                    "inference_samples_per_second": _throughput(len(predicted_price), inference_seconds),
                }
            )


def _residual_rows(
    model_name: str,
    model,
    features: np.ndarray,
    eval_idx: np.ndarray,
    pde_mask: np.ndarray,
    split: np.ndarray,
    cfg: dict[str, Any],
    *,
    device: str,
    dtype,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    split_values = split.astype(str)
    for offset, split_name in enumerate(["validation", "test"]):
        candidate = eval_idx[(split_values[eval_idx] == split_name) & pde_mask[eval_idx]]
        candidate = capped_indices(np.isin(np.arange(len(features)), candidate), int(cfg.get("residual_eval_cap", 2048)), seed + 100 + offset)
        residual = _evaluate_residual(model, features[candidate], batch_size=int(cfg.get("batch_size", 2048)), device=device, dtype=dtype)
        rows.append(
            {
                "model": model_name,
                "split": split_name,
                "mask_source": "psor_continuation_not_boundary_not_maturity",
                **residual_distribution_metrics(residual, prefix="residual"),
            }
        )
    return rows


def _evaluate_residual(model, features: np.ndarray, *, batch_size: int, device: str, dtype) -> np.ndarray:
    import torch

    model = model.to(device=device, dtype=dtype)
    model.eval()
    values = []
    for start in range(0, len(features), batch_size):
        xb = torch.as_tensor(features[start : start + batch_size], dtype=dtype, device=device).detach().clone().requires_grad_(True)
        residual = residual_for_premium_model(model, xb)
        values.append(residual.detach().cpu().numpy())
    return np.concatenate(values) if values else np.array([], dtype=float)


def _boundary_comparison(
    model_name: str,
    predicted_premium: np.ndarray,
    eval_idx: np.ndarray,
    data: dict[str, np.ndarray],
    *,
    threshold: float,
) -> dict[str, Any]:
    if "boundary" not in data:
        return {"model": model_name, "boundary_status": "missing_psor_boundary", "boundary_samples": 0}
    features = data["features"][eval_idx]
    boundary = data["boundary"][eval_idx]
    regimes = data["regime_key"].astype(str)[eval_idx]
    splits = data["split"].astype(str)[eval_idx]
    rows = []
    for split_name in ["validation", "test"]:
        split_mask = (splits == split_name) & (features[:, 6] <= 0.5) & np.isfinite(boundary)
        for regime in sorted(set(regimes[split_mask])):
            regime_mask = split_mask & (regimes == regime)
            theta_values = sorted(set(np.round(features[regime_mask, 1], 8).tolist()))
            for theta in theta_values:
                mask = regime_mask & (np.isclose(features[:, 1], theta))
                if np.count_nonzero(mask) < 3:
                    continue
                spot_over_k = np.exp(features[mask, 0])
                exercise_like = predicted_premium[mask] <= threshold
                if not np.any(exercise_like):
                    continue
                predicted_boundary = float(np.max(spot_over_k[exercise_like]))
                target_boundary = float(np.nanmedian(boundary[mask]))
                rows.append(predicted_boundary - target_boundary)
    if not rows:
        return {
            "model": model_name,
            "boundary_status": "insufficient_threshold_crossings",
            "boundary_samples": 0,
            "boundary_mae": None,
            "boundary_rmse": None,
            "boundary_max_error": None,
            "premium_threshold": threshold,
        }
    diff = np.asarray(rows, dtype=float)
    return {
        "model": model_name,
        "boundary_status": "threshold_proxy",
        "boundary_samples": int(diff.size),
        "boundary_mae": float(np.mean(np.abs(diff))),
        "boundary_rmse": float(np.sqrt(np.mean(diff**2))),
        "boundary_max_error": float(np.max(np.abs(diff))),
        "premium_threshold": threshold,
    }


def _option_scope_mask(features: np.ndarray, option_scope: str) -> np.ndarray:
    if option_scope == "put":
        return features[:, 6] <= 0.5
    if option_scope == "call":
        return features[:, 6] > 0.5
    if option_scope == "all":
        return np.ones(len(features), dtype=bool)
    raise ValueError(f"unknown option_scope: {option_scope}")


def _torch_dtype(name: str):
    import torch

    if name == "float64":
        return torch.float64
    if name == "float32":
        return torch.float32
    raise ValueError(f"unknown training_dtype: {name}")


def _write_empty_outputs(out_dir: Path, summary: dict[str, Any]) -> None:
    _write_csv_with_header(out_dir / "pinn_metrics.csv", [], ["model", "split", "option_type", "samples"])
    _write_csv_with_header(out_dir / "pinn_residual_metrics.csv", [], ["model", "split", "residual_samples"])
    _write_csv_with_header(out_dir / "pinn_training_history.csv", [], ["model", "epoch", "train_loss"])
    write_json(out_dir / "pinn_summary.json", summary)


def _write_csv_with_header(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    ensure_dir(Path(path).parent)
    if rows:
        write_csv(path, rows)
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()


def _throughput(samples: int, seconds: float) -> float:
    return float(samples / seconds) if seconds > 0 else float("inf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pinn_inspired_premium.yaml")
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2, sort_keys=True))
