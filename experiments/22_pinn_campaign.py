"""Stage 22: PINN-inspired premium surrogate campaign.

This campaign keeps CN/PSOR as the benchmark and label generator. Neural
variants are compared only against the matched supervised positive-premium
baseline, not against PSOR as a replacement solver.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from common import ensure_dir, load_config, write_json
from models.metrics import boundary_band_error, error_metrics, max_obstacle_violation
from models.pinn_campaign import campaign_success_flags, pde_weight_for_epoch, residual_aware_indices
from models.pinn_residual import (
    build_psor_continuation_mask,
    premium_model_price,
    residual_distribution_metrics,
    residual_for_premium_model,
)
from models.premium_surrogate import PositivePremiumModel, TerminalScaledPositivePremiumModel, reconstruct_price
from models.training import capped_indices, choose_device, load_dataset_arrays, set_global_seed, split_mask


DEFAULT_OUTPUT_ROOT = "results/experiment_22_pinn_campaign"


def run(
    config_path: str,
    *,
    output_root: str | Path | None = None,
    overrides: dict | None = None,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    if overrides:
        cfg = {**cfg, **overrides}
    if output_root is None:
        output_root = cfg.get("output_root", DEFAULT_OUTPUT_ROOT)
    out_dir = ensure_dir(output_root)
    write_json(out_dir / "pinn_campaign_config_manifest.json", cfg)

    data = load_dataset_arrays(cfg["dataset"])
    features = data["features"].astype(np.float32)
    labels = data["labels"].astype(np.float32)
    split = data["split"]
    price = labels[:, 0]
    premium = labels[:, 1]
    payoff = labels[:, 2]
    target_delta = labels[:, 3]
    near_boundary = data.get("near_boundary", np.zeros(len(features), dtype=bool)).astype(bool)
    near_kink = data.get("near_kink", np.zeros(len(features), dtype=bool)).astype(bool)
    strict_greek_mask = data.get("strict_greek_mask", np.ones(len(features), dtype=bool)).astype(bool)
    option_scope = str(cfg.get("option_scope", "put"))
    scope_mask = _option_scope_mask(features, option_scope)
    pde_mask = build_psor_continuation_mask(
        features,
        labels,
        near_boundary=near_boundary,
        near_kink=near_kink,
        gap_epsilon=float(cfg.get("gap_epsilon", 1.0e-5)),
        tau_min_for_pde=float(cfg.get("tau_min_for_pde", 0.05)),
        option_scope=option_scope,
    )

    device = choose_device(cfg.get("device", "auto"))
    dtype = _torch_dtype(str(cfg.get("training_dtype", "float64")))
    metrics_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    method_specs = [_method_spec(name, cfg) for name in cfg.get("methods", ["supervised_baseline"])]
    seeds = [int(seed) for seed in cfg.get("seeds", [cfg.get("seed", 21)])]

    for seed in seeds:
        set_global_seed(seed)
        train_idx = capped_indices(split_mask(split, "train") & scope_mask, int(cfg.get("train_cap", 0)), seed)
        val_idx = capped_indices(split_mask(split, "validation") & scope_mask, int(cfg.get("validation_cap", 0)), seed + 1)
        test_idx = capped_indices(split_mask(split, "test") & scope_mask, int(cfg.get("test_cap", 0)), seed + 2)
        pde_train_idx = np.intersect1d(train_idx, np.flatnonzero(pde_mask), assume_unique=False)
        delta_train_idx = np.intersect1d(train_idx, np.flatnonzero(strict_greek_mask & scope_mask), assume_unique=False)
        eval_idx = np.concatenate([val_idx, test_idx])
        if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
            continue
        positive_state, positive_net_state = _matched_initial_states(
            features.shape[1],
            hidden=int(cfg.get("hidden", 64)),
            depth=int(cfg.get("depth", 2)),
            seed=seed,
        )
        for spec in method_specs:
            model = _build_model_for_spec(
                spec,
                features.shape[1],
                hidden=int(cfg.get("hidden", 64)),
                depth=int(cfg.get("depth", 2)),
                positive_state=positive_state,
                positive_net_state=positive_net_state,
            )
            start = time.perf_counter()
            trained, history = _train_model(
                model,
                method=spec["method"],
                features=features,
                premium=premium,
                payoff=payoff,
                target_delta=target_delta,
                train_idx=train_idx,
                val_idx=val_idx,
                pde_train_idx=pde_train_idx,
                delta_train_idx=delta_train_idx,
                cfg=cfg,
                device=device,
                dtype=dtype,
                seed=seed,
                spec=spec,
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
            metrics_rows.extend(
                _metrics_rows(
                    spec["method"],
                    seed,
                    predicted_price,
                    predicted_premium,
                    eval_idx,
                    data,
                    price,
                    premium,
                    payoff,
                    near_boundary,
                    train_seconds=train_seconds,
                    inference_seconds=inference_seconds,
                    obstacle_tolerance=float(cfg.get("obstacle_tolerance", 1.0e-10)),
                )
            )
            residual_rows.extend(
                _residual_rows(
                    spec["method"],
                    seed,
                    trained,
                    features,
                    eval_idx,
                    pde_mask,
                    split,
                    cfg,
                    device=device,
                    dtype=dtype,
                )
            )
            delta_rows.extend(
                _delta_rows(
                    spec["method"],
                    seed,
                    trained,
                    features,
                    target_delta,
                    eval_idx,
                    strict_greek_mask & scope_mask,
                    split,
                    cfg,
                    device=device,
                    dtype=dtype,
                )
            )

    _attach_success_flags(metrics_rows, residual_rows, delta_rows, cfg)
    _write_csv(out_dir / "pinn_campaign_metrics.csv", metrics_rows)
    _write_csv(out_dir / "pinn_campaign_residual_metrics.csv", residual_rows)
    _write_csv(out_dir / "pinn_campaign_delta_metrics.csv", delta_rows)
    _write_csv(out_dir / "pinn_campaign_training_history.csv", history_rows)
    summary = _summary(metrics_rows, residual_rows, delta_rows, cfg, seeds, pde_mask)
    write_json(out_dir / "pinn_campaign_summary.json", summary)
    return summary


def _train_model(
    model,
    *,
    method: str,
    features: np.ndarray,
    premium: np.ndarray,
    payoff: np.ndarray,
    target_delta: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    pde_train_idx: np.ndarray,
    delta_train_idx: np.ndarray,
    cfg: dict[str, Any],
    device: str,
    dtype,
    seed: int,
    spec: dict[str, Any],
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
    pde_pool = torch.as_tensor(features[pde_train_idx], dtype=dtype) if len(pde_train_idx) else torch.empty((0, features.shape[1]), dtype=dtype)
    delta_pool = torch.as_tensor(features[delta_train_idx], dtype=dtype) if len(delta_train_idx) else torch.empty((0, features.shape[1]), dtype=dtype)
    delta_target_pool = torch.as_tensor(target_delta[delta_train_idx], dtype=dtype) if len(delta_train_idx) else torch.empty((0,), dtype=dtype)
    order_generator = torch.Generator().manual_seed(seed)
    pde_generator = torch.Generator().manual_seed(seed + 1000)
    delta_generator = torch.Generator().manual_seed(seed + 2000)
    np_rng = np.random.default_rng(seed + 3000)
    residual_abs = np.ones(len(pde_pool), dtype=float)
    batch_size = int(cfg.get("batch_size", 2048))
    pde_batch_size = min(int(cfg.get("pde_batch_cap", 512)), len(pde_pool))
    delta_batch_size = min(int(cfg.get("pde_batch_cap", 512)), len(delta_pool))
    lambda_sup = float(cfg.get("lambda_sup", 1.0))
    lambda_obs = float(cfg.get("lambda_obs", 1.0))
    lambda_delta = float(cfg.get("lambda_delta", 1.0e-4)) if spec.get("delta_loss") else 0.0
    total_epochs = int(cfg.get("epochs", 8))
    patience = int(cfg.get("patience", 3))
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_val = float("inf")
    stale = 0
    history: list[dict[str, Any]] = []
    n_train = len(x_train)

    for epoch in range(1, total_epochs + 1):
        order = torch.randperm(n_train, generator=order_generator)
        model.train()
        epoch_pde_weight = _epoch_pde_weight(spec, cfg, epoch)
        totals = {"loss": 0.0, "supervised": 0.0, "pde": 0.0, "obstacle": 0.0, "delta": 0.0}
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
            if epoch_pde_weight > 0.0 and pde_batch_size > 0:
                pde_positions = _pde_positions(spec, pde_batch_size, len(pde_pool), residual_abs, pde_generator, np_rng)
                pde_features = pde_pool[pde_positions].to(device).detach().clone().requires_grad_(True)
                residual = residual_for_premium_model(model, pde_features)
                pde_loss = F.huber_loss(residual, torch.zeros_like(residual), delta=float(cfg.get("pde_huber_delta", 1.0)))
            delta_loss = torch.zeros((), dtype=dtype, device=device)
            if lambda_delta > 0.0 and delta_batch_size > 0:
                delta_positions = torch.randint(0, len(delta_pool), (delta_batch_size,), generator=delta_generator)
                delta_features = delta_pool[delta_positions].to(device).detach().clone().requires_grad_(True)
                delta_target = delta_target_pool[delta_positions].to(device)
                delta_pred = _delta_from_model(model, delta_features)
                delta_loss = F.huber_loss(delta_pred, delta_target, delta=float(cfg.get("delta_huber_delta", 1.0)))
            loss = lambda_sup * supervised_loss + lambda_obs * obstacle_loss + epoch_pde_weight * pde_loss + lambda_delta * delta_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_n = len(idx)
            totals["loss"] += float(loss.detach().cpu()) * batch_n
            totals["supervised"] += float(supervised_loss.detach().cpu()) * batch_n
            totals["pde"] += float(pde_loss.detach().cpu()) * batch_n
            totals["obstacle"] += float(obstacle_loss.detach().cpu()) * batch_n
            totals["delta"] += float(delta_loss.detach().cpu()) * batch_n
        if spec.get("residual_sampling") and pde_batch_size > 0 and epoch >= int(cfg.get("warm_start_epochs", 4)):
            residual_abs = _residual_abs_on_pool(model, pde_pool, batch_size=batch_size, device=device, dtype=dtype)
        model.eval()
        with torch.no_grad():
            val_pred = model(x_val)
            val_rmse = torch.sqrt(F.mse_loss(val_pred, y_val)).item()
        history.append(
            {
                "method": method,
                "seed": seed,
                "epoch": epoch,
                "train_loss": totals["loss"] / n_train,
                "train_supervised_loss": totals["supervised"] / n_train,
                "train_pde_loss": totals["pde"] / n_train,
                "train_obstacle_loss": totals["obstacle"] / n_train,
                "train_delta_loss": totals["delta"] / n_train,
                "validation_premium_rmse": val_rmse,
                "lambda_pde": epoch_pde_weight,
                "lambda_delta": lambda_delta,
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


def _method_spec(method: str, cfg: dict[str, Any]) -> dict[str, Any]:
    default_pde = float(cfg.get("lambda_pde_default", 1.0e-4))
    if method == "supervised_baseline":
        return {"method": method, "lambda_pde": 0.0}
    if method.startswith("pde_weight_"):
        return {"method": method, "lambda_pde": float(method.removeprefix("pde_weight_").replace("e-", "e-"))}
    if method == "pde_ramp":
        return {"method": method, "lambda_pde": default_pde, "ramp": True}
    if method == "terminal_premium":
        return {"method": method, "lambda_pde": 0.0, "terminal": True}
    if method == "residual_sampling":
        return {"method": method, "lambda_pde": default_pde, "residual_sampling": True}
    if method == "delta_loss":
        return {"method": method, "lambda_pde": 0.0, "delta_loss": True}
    if method == "terminal_plus_ramp":
        return {"method": method, "lambda_pde": default_pde, "ramp": True, "terminal": True}
    if method == "terminal_plus_ramp_plus_residual_sampling":
        return {"method": method, "lambda_pde": default_pde, "ramp": True, "terminal": True, "residual_sampling": True}
    if method.startswith("terminal_pde_weight_"):
        weight_token = method.removeprefix("terminal_pde_weight_")
        residual_sampling = weight_token.endswith("_residual_sampling")
        if residual_sampling:
            weight_token = weight_token.removesuffix("_residual_sampling")
        spec = {"method": method, "lambda_pde": _parse_method_weight(weight_token), "ramp": True, "terminal": True}
        if residual_sampling:
            spec["residual_sampling"] = True
        return spec
    raise ValueError(f"unknown Stage 22 method: {method}")


def _parse_method_weight(token: str) -> float:
    return float(token.replace("p", "."))


def _epoch_pde_weight(spec: dict[str, Any], cfg: dict[str, Any], epoch: int) -> float:
    final_weight = float(spec.get("lambda_pde", 0.0))
    if not spec.get("ramp"):
        return final_weight
    return pde_weight_for_epoch(
        epoch,
        final_weight=final_weight,
        warm_start_epochs=int(cfg.get("warm_start_epochs", 4)),
        total_epochs=int(cfg.get("epochs", 8)),
    )


def _pde_positions(spec, batch_size, pool_size, residual_abs, torch_generator, np_rng):
    import torch

    if spec.get("residual_sampling") and len(residual_abs) == pool_size:
        return residual_aware_indices(np.arange(pool_size), residual_abs, size=batch_size, rng=np_rng)
    return np.asarray(torch.randint(0, pool_size, (batch_size,), generator=torch_generator))


def _matched_initial_states(input_dim: int, *, hidden: int, depth: int, seed: int):
    import torch

    torch.manual_seed(seed)
    positive = PositivePremiumModel.build(input_dim, hidden=hidden, depth=depth)
    return (
        {key: value.detach().clone() for key, value in positive.state_dict().items()},
        {key: value.detach().clone() for key, value in positive[0].state_dict().items()},
    )


def _build_model_for_spec(spec: dict[str, Any], input_dim: int, *, hidden: int, depth: int, positive_state: dict, positive_net_state: dict):
    if spec.get("terminal"):
        model = TerminalScaledPositivePremiumModel.build(input_dim, hidden=hidden, depth=depth)
        model.net.load_state_dict({key: value.clone() for key, value in positive_net_state.items()})
        return model
    model = PositivePremiumModel.build(input_dim, hidden=hidden, depth=depth)
    model.load_state_dict({key: value.clone() for key, value in positive_state.items()})
    return model


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


def _metrics_rows(
    method: str,
    seed: int,
    predicted_price: np.ndarray,
    predicted_premium: np.ndarray,
    eval_idx: np.ndarray,
    data: dict[str, np.ndarray],
    target_price: np.ndarray,
    target_premium: np.ndarray,
    payoff: np.ndarray,
    near_boundary: np.ndarray,
    *,
    train_seconds: float,
    inference_seconds: float,
    obstacle_tolerance: float,
) -> list[dict[str, Any]]:
    rows = []
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
            rows.append(
                {
                    "method": method,
                    "seed": seed,
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
                    "candidate_success": False,
                    "diagnostic_success": False,
                    "price_rel_delta": 0.0,
                    "residual_rel_delta": 0.0,
                    "boundary_rel_delta": 0.0,
                }
            )
    return rows


def _residual_rows(method, seed, model, features, eval_idx, pde_mask, split, cfg, *, device, dtype):
    rows = []
    split_values = split.astype(str)
    for offset, split_name in enumerate(["validation", "test"]):
        candidate = eval_idx[(split_values[eval_idx] == split_name) & pde_mask[eval_idx]]
        candidate = capped_indices(np.isin(np.arange(len(features)), candidate), int(cfg.get("residual_eval_cap", 2048)), seed + 100 + offset)
        residual = _evaluate_residual(model, features[candidate], batch_size=int(cfg.get("batch_size", 2048)), device=device, dtype=dtype)
        rows.append(
            {
                "method": method,
                "seed": seed,
                "split": split_name,
                "mask_source": "psor_continuation_not_boundary_not_kink_not_maturity",
                **residual_distribution_metrics(residual, prefix="residual"),
            }
        )
    return rows


def _delta_rows(method, seed, model, features, target_delta, eval_idx, delta_mask, split, cfg, *, device, dtype):
    rows = []
    split_values = split.astype(str)
    for offset, split_name in enumerate(["validation", "test"]):
        candidate = eval_idx[(split_values[eval_idx] == split_name) & delta_mask[eval_idx]]
        candidate = capped_indices(np.isin(np.arange(len(features)), candidate), int(cfg.get("residual_eval_cap", 2048)), seed + 300 + offset)
        prediction = _evaluate_delta(model, features[candidate], batch_size=int(cfg.get("batch_size", 2048)), device=device, dtype=dtype)
        metrics = error_metrics(prediction, target_delta[candidate]) if len(candidate) else {"rmse": float("nan"), "mae": float("nan"), "max_error": float("nan")}
        rows.append(
            {
                "method": method,
                "seed": seed,
                "split": split_name,
                "strict_delta_samples": int(len(candidate)),
                "strict_delta_rmse": metrics["rmse"],
                "strict_delta_mae": metrics["mae"],
                "strict_delta_max_error": metrics["max_error"],
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


def _residual_abs_on_pool(model, pde_pool, *, batch_size: int, device: str, dtype) -> np.ndarray:
    values = _evaluate_residual(model, pde_pool.detach().cpu().numpy(), batch_size=batch_size, device=device, dtype=dtype)
    return np.abs(values) + 1.0e-12


def _delta_from_model(model, features):
    import torch

    price = premium_model_price(model, features)
    first = torch.autograd.grad(price, features, grad_outputs=torch.ones_like(price), create_graph=True, retain_graph=True)[0]
    return first[:, 0] / torch.exp(features[:, 0])


def _evaluate_delta(model, features: np.ndarray, *, batch_size: int, device: str, dtype) -> np.ndarray:
    import torch

    model = model.to(device=device, dtype=dtype)
    model.eval()
    values = []
    for start in range(0, len(features), batch_size):
        xb = torch.as_tensor(features[start : start + batch_size], dtype=dtype, device=device).detach().clone().requires_grad_(True)
        values.append(_delta_from_model(model, xb).detach().cpu().numpy())
    return np.concatenate(values) if values else np.array([], dtype=float)


def _attach_success_flags(metrics_rows, residual_rows, delta_rows, cfg):
    residual_lookup = {(row["seed"], row["method"], row["split"]): row for row in residual_rows}
    metrics_lookup = {(row["seed"], row["method"], row["split"], row["option_type"]): row for row in metrics_rows}
    for row in metrics_rows:
        if row["split"] != "test" or row["option_type"] != "all" or row["method"] == "supervised_baseline":
            continue
        baseline = metrics_lookup.get((row["seed"], "supervised_baseline", "test", "all"))
        baseline_validation = metrics_lookup.get((row["seed"], "supervised_baseline", "validation", "all"))
        candidate_validation = metrics_lookup.get((row["seed"], row["method"], "validation", "all"))
        baseline_residual = residual_lookup.get((row["seed"], "supervised_baseline", "test"))
        candidate_residual = residual_lookup.get((row["seed"], row["method"], "test"))
        if not baseline or not baseline_residual or not candidate_residual:
            continue
        candidate = {
            "price_rmse": row["price_rmse"],
            "validation_price_rmse": candidate_validation["price_rmse"] if candidate_validation else row["price_rmse"],
            "max_obstacle_violation": row["max_obstacle_violation"],
            "boundary_rmse": row["boundary_rmse"],
            "residual_mean_abs": candidate_residual["residual_mean_abs"],
        }
        baseline_payload = {
            "price_rmse": baseline["price_rmse"],
            "validation_price_rmse": baseline_validation["price_rmse"] if baseline_validation else baseline["price_rmse"],
            "max_obstacle_violation": baseline["max_obstacle_violation"],
            "boundary_rmse": baseline["boundary_rmse"],
            "residual_mean_abs": baseline_residual["residual_mean_abs"],
        }
        flags = campaign_success_flags(
            candidate,
            baseline_payload,
            price_improvement=float(cfg.get("success_price_rmse_rel_improvement", 0.005)),
            neutral_price_worsening=float(cfg.get("neutral_price_rmse_rel_worsening", 0.001)),
            residual_worsening_limit=float(cfg.get("residual_rel_worsening_limit", 0.05)),
            boundary_worsening_limit=float(cfg.get("boundary_rel_worsening_limit", 0.01)),
            diagnostic_improvement=float(cfg.get("diagnostic_rel_improvement", 0.02)),
        )
        row.update(flags)


def _summary(metrics_rows, residual_rows, delta_rows, cfg, seeds, pde_mask):
    test_rows = [row for row in metrics_rows if row["split"] == "test" and row["option_type"] == "all"]
    residual_lookup = {(row["seed"], row["method"], row["split"]): row for row in residual_rows}
    delta_lookup = {(row["seed"], row["method"], row["split"]): row for row in delta_rows}
    method_names = sorted({row["method"] for row in test_rows})
    aggregates = []
    for method in method_names:
        rows = [row for row in test_rows if row["method"] == method]
        if not rows:
            continue
        residual_values = [residual_lookup[(row["seed"], method, "test")]["residual_mean_abs"] for row in rows if (row["seed"], method, "test") in residual_lookup]
        delta_values = [delta_lookup[(row["seed"], method, "test")]["strict_delta_rmse"] for row in rows if (row["seed"], method, "test") in delta_lookup]
        aggregates.append(
            {
                "method": method,
                "mean_test_price_rmse": float(np.mean([row["price_rmse"] for row in rows])),
                "mean_test_boundary_rmse": float(np.mean([row["boundary_rmse"] for row in rows])),
                "mean_test_residual_mean_abs": float(np.mean(residual_values)) if residual_values else float("nan"),
                "mean_test_delta_rmse": float(np.mean(delta_values)) if delta_values else float("nan"),
                "candidate_success_count": int(sum(bool(row["candidate_success"]) for row in rows)),
                "diagnostic_success_count": int(sum(bool(row["diagnostic_success"]) for row in rows)),
            }
        )
    best = _best_aggregate(aggregates)
    any_candidate = any(row["candidate_success_count"] > 0 for row in aggregates)
    any_diagnostic = any(row["diagnostic_success_count"] > 0 for row in aggregates)
    return {
        "decision": "stage22_pinn_campaign",
        "seeds": [int(seed) for seed in seeds],
        "methods": method_names,
        "best_method": best["method"] if best else None,
        "best_combo_method": _best_combo(aggregates),
        "candidate_success": bool(any_candidate),
        "diagnostic_success": bool(any_diagnostic),
        "method_aggregates": aggregates,
        "pde_total_mask_samples": int(np.count_nonzero(pde_mask)),
        "interpretation": _interpretation(any_candidate, any_diagnostic, best),
        "note": "Stage 22 compares PINN-inspired neural variants against the matched supervised surrogate baseline; CN/PSOR remains the benchmark and label generator.",
    }


def _best_aggregate(aggregates):
    if not aggregates:
        return None
    return sorted(
        aggregates,
        key=lambda row: (
            -row["candidate_success_count"],
            -row["diagnostic_success_count"],
            row["mean_test_price_rmse"],
            row["mean_test_residual_mean_abs"],
        ),
    )[0]


def _best_combo(aggregates):
    combo_names = {"terminal_plus_ramp", "terminal_plus_ramp_plus_residual_sampling"}
    combo_rows = [row for row in aggregates if row["method"] in combo_names]
    best = _best_aggregate(combo_rows)
    return best["method"] if best else None


def _interpretation(candidate_success: bool, diagnostic_success: bool, best: dict | None) -> str:
    if not best:
        return "No Stage 22 methods were evaluated."
    if candidate_success:
        return f"{best['method']} passed at least one capped candidate-success gate; this remains a future-extension surrogate diagnostic."
    if diagnostic_success:
        return f"{best['method']} passed at least one capped diagnostic-success gate without becoming a PSOR replacement claim."
    return "No tested Stage 22 method passed the capped candidate or diagnostic success gates; no PINN-inspired improvement claim is supported."


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


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(Path(path).parent)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pinn_campaign.yaml")
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2, sort_keys=True))
