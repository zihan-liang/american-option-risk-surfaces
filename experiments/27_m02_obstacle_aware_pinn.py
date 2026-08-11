"""M02 independent obstacle-aware PINN screen.

CN/PSOR labels remain the benchmark. This script compares regularizers on a
positive-premium surrogate; it is not an LCP solver or a PSOR replacement.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np

from campaign.io import write_method_outputs
from campaign.records import MethodRunRecord, RunPhase, RunStatus
from campaign.registry import get_method_spec
from campaign.statistics import MetricGate, ScenarioObservation, decide_promotion, summarize_paired_scenarios
from common import ensure_dir, load_config, write_json
from models.metrics import error_metrics, max_obstacle_violation
from models.obstacle_pinn import (
    build_vi_collocation_mask,
    fischer_burmeister_loss,
    masked_pde_loss,
    select_whole_regimes,
    separate_complementarity_loss,
)
from models.pinn_residual import (
    build_psor_continuation_mask,
    premium_model_price,
    residual_distribution_metrics,
    residual_for_premium_model,
)
from models.premium_surrogate import PositivePremiumModel
from models.training import choose_device, load_dataset_arrays, set_global_seed, split_mask


DEFAULT_OUTPUT_ROOT = "results/independent_method_campaign/M02"
METHODS = ("supervised", "masked_pde", "separate_complementarity", "fischer_burmeister")
SPLIT_MAP = {"train": "train", "validation": "validation", "legacy_test": "test"}
PROMOTION_METRICS = (
    ("price_rmse", True, False),
    ("delta_rmse", False, False),
    ("k_gamma_rmse", False, False),
    ("boundary_rmse", False, False),
    ("continuation_violation_mean", False, False),
    ("complementarity_product_mean_abs", False, False),
    ("masked_pde_residual_mean_abs", False, True),
)


def run(config_path: str, *, output_root: str | Path | None = None, overrides: dict | None = None) -> dict[str, Any]:
    """Run canonical M02 or its explicitly non-evidence smoke mode."""

    base_cfg = load_config(config_path)
    requested_mode = str((overrides or {}).get("mode", base_cfg.get("mode", "canonical")))
    if requested_mode not in {"canonical", "smoke"}:
        raise ValueError(f"unknown M02 mode: {requested_mode}")
    cfg = {**base_cfg, **base_cfg.get(f"{requested_mode}_settings", {}), **(overrides or {})}
    cfg["mode"] = requested_mode
    cfg["evidence_run"] = requested_mode == "canonical"
    _validate_config(cfg)

    out_dir = ensure_dir(output_root or cfg.get("output_root", DEFAULT_OUTPUT_ROOT))
    data = load_dataset_arrays(cfg["dataset"])
    features = np.asarray(data["features"], dtype=np.float64)
    labels = np.asarray(data["labels"], dtype=np.float64)
    _require_finite(features, "features")
    _require_finite(labels, "labels")
    if labels.ndim != 2 or labels.shape[1] < 5:
        raise ValueError("M02 requires price, premium, payoff, Delta, and Gamma labels")

    price, premium, payoff, target_delta, target_gamma = (labels[:, column] for column in range(5))
    split = data["split"]
    regime_keys = np.asarray(data["regime_key"]).astype(str)
    near_boundary = np.asarray(data.get("near_boundary", np.zeros(len(features), dtype=bool)), dtype=bool)
    near_kink = np.asarray(data.get("near_kink", np.zeros(len(features), dtype=bool)), dtype=bool)
    scope_mask = _option_scope_mask(features, str(cfg["option_scope"]))
    pde_mask = build_psor_continuation_mask(
        features,
        labels,
        near_boundary=near_boundary,
        near_kink=near_kink,
        gap_epsilon=float(cfg["gap_epsilon"]),
        tau_min_for_pde=float(cfg["tau_min_for_pde"]),
        option_scope=str(cfg["option_scope"]),
    )
    vi_mask = build_vi_collocation_mask(
        features,
        near_boundary=near_boundary,
        near_kink=near_kink,
        tau_min_for_pde=float(cfg["tau_min_for_pde"]),
        option_scope=str(cfg["option_scope"]),
    )
    selections = _predeclared_split_selections(split, scope_mask, regime_keys, cfg)
    if any(len(indices) == 0 for indices, _ in selections.values()):
        raise ValueError("M02 requires nonempty train, validation, and legacy_test whole-regime selections")
    vi_train_idx = np.intersect1d(selections["train"][0], np.flatnonzero(vi_mask))
    if len(vi_train_idx) == 0:
        raise ValueError("M02 valid VI collocation mask is empty on the fixed training selection")
    residual_indices = {
        split_name: _predeclared_residual_indices(
            selections[split_name][0],
            regime_keys,
            vi_mask,
            cap=int(cfg["residual_eval_cap"]),
            selection_seed=int(cfg["selection_seed"]),
            split_name=split_name,
        )
        for split_name in ("validation", "legacy_test")
    }
    device = choose_device(str(cfg["device"]))
    dtype = _torch_dtype(str(cfg["training_dtype"]))
    metrics_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []

    for seed in [int(value) for value in cfg["seeds"]]:
        set_global_seed(seed)
        initial_state = _initial_state(features.shape[1], int(cfg["hidden"]), int(cfg["depth"]), seed)
        _warmup_backend(
            initial_state,
            features,
            premium,
            selections["train"][0],
            vi_train_idx,
            cfg,
            device,
            dtype,
        )
        for method in METHODS:
            model = PositivePremiumModel.build(features.shape[1], hidden=int(cfg["hidden"]), depth=int(cfg["depth"]))
            model.load_state_dict({key: value.clone() for key, value in initial_state.items()})
            started_at = time.perf_counter()
            trained, history = _train_method(
                model,
                method=method,
                features=features,
                premium=premium,
                train_idx=selections["train"][0],
                validation_idx=selections["validation"][0],
                masked_pde_idx=np.intersect1d(selections["train"][0], np.flatnonzero(pde_mask)),
                vi_collocation_idx=vi_train_idx,
                cfg=cfg,
                seed=seed,
                device=device,
                dtype=dtype,
            )
            train_seconds = time.perf_counter() - started_at
            history_rows.extend({"method": method, "seed": seed, **row} for row in history)
            for output_split in ("validation", "legacy_test"):
                metrics_rows.extend(
                    _metrics_by_regime(
                        trained,
                        method=method,
                        seed=seed,
                        split_name=output_split,
                        eval_idx=selections[output_split][0],
                        residual_indices=residual_indices[output_split],
                        features=features,
                        price=price,
                        premium=premium,
                        payoff=payoff,
                        target_delta=target_delta,
                        target_gamma=target_gamma,
                        regime_keys=regime_keys,
                        pde_mask=pde_mask,
                        vi_mask=vi_mask,
                        boundary=data.get("boundary"),
                        cfg=cfg,
                        train_seconds=train_seconds,
                        device=device,
                        dtype=dtype,
                    )
                )

    promotion = _promotion_from_validation(metrics_rows, cfg)
    provenance = _git_provenance()
    selection_manifest = _selection_manifest(selections, residual_indices, vi_mask, cfg)
    detail_config = {
        "config": cfg,
        "provenance": provenance,
        "selection": selection_manifest,
        "vi_collocation_total_samples": int(np.count_nonzero(vi_mask)),
        "masked_pde_total_samples": int(np.count_nonzero(pde_mask)),
    }
    write_json(out_dir / "m02_config_manifest.json", detail_config)
    _write_csv(out_dir / "m02_metrics_by_regime.csv", metrics_rows)
    _write_csv(out_dir / "m02_training_history.csv", history_rows)
    write_json(out_dir / "m02_promotion.json", promotion)

    records, decision_record = _method_run_records(metrics_rows, promotion, cfg, provenance)
    write_method_outputs(
        out_dir,
        get_method_spec("M02"),
        records,
        decision_record=decision_record,
        config={**cfg, "config_path": config_path},
        training_history=history_rows,
    )
    summary = {
        "method_id": "M02",
        "mode": cfg["mode"],
        "evidence_run": bool(cfg["evidence_run"]),
        "canonical_run_launched": cfg["mode"] == "canonical",
        "methods": list(METHODS),
        "configuration_count": len(METHODS),
        "seeds": [int(value) for value in cfg["seeds"]],
        "split_names": ["train", "validation", "legacy_test"],
        "selection": selection_manifest,
        "matched_initialization": True,
        "architecture": {"type": "positive_premium_mlp", "width": int(cfg["hidden"]), "depth": int(cfg["depth"])},
        "training": {"epochs": int(cfg["epochs"]), "batch_size": int(cfg["batch_size"]), "patience": int(cfg["patience"]), "device": device, "dtype": str(cfg["training_dtype"])},
        "loss_scaling": {"gap_scale": float(cfg["gap_scale"]), "residual_scale": float(cfg["residual_scale"]), "fischer_burmeister_epsilon": float(cfg["fischer_burmeister_epsilon"])},
        "promotion": promotion,
        "provenance": provenance,
        "note": "CN/PSOR remains the benchmark and label generator. Validation alone is used for selection; legacy_test is descriptive and is not blind.",
    }
    write_json(out_dir / "m02_summary.json", summary)
    _write_artifact_hashes(out_dir)
    return summary


def _validate_config(cfg: dict[str, Any]) -> None:
    seeds = [int(seed) for seed in cfg["seeds"]]
    if len(seeds) != len(set(seeds)):
        raise ValueError("M02 seeds must be unique")
    if cfg["mode"] == "canonical" and seeds != [101, 203, 307, 409, 503]:
        raise ValueError("canonical M02 seeds must be [101, 203, 307, 409, 503]")
    for name in ("gap_scale", "residual_scale", "fischer_burmeister_epsilon"):
        if not np.isfinite(float(cfg[name])) or float(cfg[name]) <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if str(cfg["training_dtype"]) != "float64" or str(cfg["device"]) != "cpu":
        raise ValueError("M02 requires float64 CPU training")
    if len(METHODS) > 8:
        raise ValueError("M02 configuration exceeds the eight-method limit")


def _predeclared_split_selections(split, scope_mask, regime_keys, cfg):
    caps = {"train": "train_cap", "validation": "validation_cap", "legacy_test": "legacy_test_cap"}
    return {
        name: select_whole_regimes(
            np.flatnonzero(split_mask(split, dataset_name) & scope_mask),
            regime_keys,
            cap=int(cfg[caps[name]]),
            seed=int(cfg["selection_seed"]) + offset,
        )
        for offset, (name, dataset_name) in enumerate(SPLIT_MAP.items())
    }


def _predeclared_residual_indices(eval_idx, regime_keys, vi_mask, *, cap, selection_seed, split_name):
    selected: dict[str, np.ndarray] = {}
    for regime_id in sorted(set(regime_keys[eval_idx])):
        candidates = np.sort(eval_idx[(regime_keys[eval_idx] == regime_id) & vi_mask[eval_idx]])
        if cap > 0 and len(candidates) > cap:
            digest = hashlib.sha256(f"{selection_seed}:{split_name}:{regime_id}".encode("utf-8")).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
            candidates = np.sort(rng.choice(candidates, size=cap, replace=False))
        selected[regime_id] = candidates
    return selected


def _selection_manifest(selections, residual_indices, vi_mask, cfg):
    return {
        "selection_seed": int(cfg["selection_seed"]),
        **{
            name: {
                "samples": int(len(indices)),
                "regime_ids": regime_ids,
                "vi_collocation_samples": int(np.count_nonzero(vi_mask[indices])),
                "residual_regime_samples": {regime_id: int(len(value)) for regime_id, value in residual_indices.get(name, {}).items()},
            }
            for name, (indices, regime_ids) in selections.items()
        },
    }


def _initial_state(input_dim, hidden, depth, seed):
    import torch

    torch.manual_seed(seed)
    model = PositivePremiumModel.build(input_dim, hidden=hidden, depth=depth)
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def _warmup_backend(initial_state, features, premium, train_idx, vi_idx, cfg, device, dtype):
    import torch
    import torch.nn.functional as functional

    model = PositivePremiumModel.build(features.shape[1], hidden=int(cfg["hidden"]), depth=int(cfg["depth"])).to(device=device, dtype=dtype)
    model.load_state_dict({key: value.clone() for key, value in initial_state.items()})
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]))
    positions = train_idx[: min(len(train_idx), int(cfg["batch_size"]))]
    xb = torch.as_tensor(features[positions], dtype=dtype, device=device)
    yb = torch.as_tensor(premium[positions, None], dtype=dtype, device=device)
    loss = functional.mse_loss(model(xb), yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    point = torch.as_tensor(features[vi_idx[:1]], dtype=dtype, device=device).detach().clone().requires_grad_(True)
    residual = residual_for_premium_model(model, point)
    optimizer.zero_grad()
    residual.square().mean().backward()


def _train_method(model, *, method, features, premium, train_idx, validation_idx, masked_pde_idx, vi_collocation_idx, cfg, seed, device, dtype):
    import torch
    import torch.nn.functional as functional

    model = model.to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]))
    train_x = torch.as_tensor(features[train_idx], dtype=dtype)
    train_y = torch.as_tensor(premium[train_idx, None], dtype=dtype)
    validation_x = torch.as_tensor(features[validation_idx], dtype=dtype, device=device)
    validation_y = torch.as_tensor(premium[validation_idx, None], dtype=dtype, device=device)
    masked_pool = torch.as_tensor(features[masked_pde_idx], dtype=dtype)
    vi_pool = torch.as_tensor(features[vi_collocation_idx], dtype=dtype)
    if method == "masked_pde" and len(masked_pool) == 0:
        raise ValueError("M02 masked-PDE pool is empty")
    order_generator = torch.Generator().manual_seed(seed)
    collocation_generator = torch.Generator().manual_seed(seed + 1)
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_validation = float("inf")
    stale_epochs = 0
    history = []
    for epoch in range(1, int(cfg["epochs"]) + 1):
        total = {"loss": 0.0, "supervised": 0.0, "masked_pde": 0.0, "dual": 0.0, "complementarity": 0.0, "fb": 0.0}
        model.train()
        order = torch.randperm(len(train_x), generator=order_generator)
        for start in range(0, len(train_x), int(cfg["batch_size"])):
            positions = order[start : start + int(cfg["batch_size"])]
            supervised = functional.mse_loss(model(train_x[positions].to(device)), train_y[positions].to(device))
            physics = _physics_terms(model, method, masked_pool, vi_pool, cfg, device, dtype, collocation_generator)
            loss = supervised + physics["weighted"]
            _require_finite_tensor(loss, "training loss")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_size = len(positions)
            total["loss"] += float(loss.detach().cpu()) * batch_size
            total["supervised"] += float(supervised.detach().cpu()) * batch_size
            for name in ("masked_pde", "dual", "complementarity", "fb"):
                total[name] += float(physics[name].detach().cpu()) * batch_size
        model.eval()
        with torch.no_grad():
            validation_rmse = float(torch.sqrt(functional.mse_loss(model(validation_x), validation_y)).cpu())
        _require_finite([validation_rmse], "validation RMSE")
        history.append({"epoch": epoch, **{name: value / len(train_x) for name, value in total.items()}, "validation_premium_rmse": validation_rmse})
        if validation_rmse < best_validation:
            best_validation = validation_rmse
            stale_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale_epochs += 1
            if stale_epochs >= int(cfg["patience"]):
                break
    model.load_state_dict(best_state)
    return model, history


def _physics_terms(model, method, masked_pool, vi_pool, cfg, device, dtype, generator):
    import torch

    zero = torch.zeros((), dtype=dtype, device=device)
    terms = {"masked_pde": zero, "dual": zero, "complementarity": zero, "fb": zero, "weighted": zero}
    if method == "supervised":
        return terms
    if method == "masked_pde":
        points = _collocation_batch(masked_pool, cfg, generator, device)
        residual = residual_for_premium_model(model, points)
        terms["masked_pde"] = masked_pde_loss(residual, torch.ones(len(points), dtype=torch.bool, device=device), residual_scale=float(cfg["residual_scale"]))
        terms["weighted"] = float(cfg["lambda_masked_pde"]) * terms["masked_pde"]
        return terms
    points = _collocation_batch(vi_pool, cfg, generator, device)
    gap = model(points).reshape(-1)
    residual = residual_for_premium_model(model, points)
    if method == "separate_complementarity":
        separate = separate_complementarity_loss(gap, residual, gap_scale=float(cfg["gap_scale"]), residual_scale=float(cfg["residual_scale"]))
        terms["dual"] = separate["dual_feasibility"]
        terms["complementarity"] = separate["complementarity"]
        terms["weighted"] = float(cfg["lambda_dual"]) * terms["dual"] + float(cfg["lambda_complementarity"]) * terms["complementarity"]
        return terms
    if method == "fischer_burmeister":
        terms["fb"] = fischer_burmeister_loss(gap, residual, epsilon=float(cfg["fischer_burmeister_epsilon"]), gap_scale=float(cfg["gap_scale"]), residual_scale=float(cfg["residual_scale"]))
        terms["weighted"] = float(cfg["lambda_fischer_burmeister"]) * terms["fb"]
        return terms
    raise ValueError(f"unknown M02 method: {method}")


def _collocation_batch(pool, cfg, generator, device):
    import torch

    if len(pool) == 0:
        raise ValueError("M02 collocation pool is empty")
    size = min(len(pool), int(cfg["collocation_cap"]))
    positions = torch.randint(0, len(pool), (size,), generator=generator)
    return pool[positions].to(device).detach().clone().requires_grad_(True)


def _metrics_by_regime(model, *, method, seed, split_name, eval_idx, residual_indices, features, price, premium, payoff, target_delta, target_gamma, regime_keys, pde_mask, vi_mask, boundary, cfg, train_seconds, device, dtype):
    prediction, delta, gamma = _predict_price_delta_gamma(model, features[eval_idx], device=device, dtype=dtype, batch_size=int(cfg["batch_size"]))
    premium_prediction = prediction - payoff[eval_idx]
    rows = []
    for regime_id in sorted(set(regime_keys[eval_idx])):
        mask = regime_keys[eval_idx] == regime_id
        global_idx = eval_idx[mask]
        diagnostics = _residual_diagnostics(model, features[residual_indices[regime_id]], pde_mask[residual_indices[regime_id]], vi_mask[residual_indices[regime_id]], cfg, device, dtype)
        boundary_metrics = _boundary_metrics(premium_prediction[mask], features[global_idx], boundary, global_idx, float(cfg["premium_boundary_threshold"]))
        row = {
            "method": method,
            "seed": seed,
            "split": split_name,
            "regime_id": regime_id,
            "samples": int(np.count_nonzero(mask)),
            **{f"price_{key}": value for key, value in error_metrics(prediction[mask], price[global_idx]).items()},
            **{f"premium_{key}": value for key, value in error_metrics(premium_prediction[mask], premium[global_idx]).items()},
            **{f"delta_{key}": value for key, value in error_metrics(delta[mask], target_delta[global_idx]).items()},
            **{f"k_gamma_{key}": value for key, value in error_metrics(gamma[mask], target_gamma[global_idx]).items()},
            "max_obstacle_violation": max_obstacle_violation(prediction[mask], payoff[global_idx]),
            "obstacle_violation_fraction": float(np.mean(prediction[mask] < payoff[global_idx] - float(cfg["obstacle_tolerance"]))),
            **boundary_metrics,
            **diagnostics,
            "train_seconds": train_seconds,
        }
        _validate_metric_row(row)
        rows.append(row)
    return rows


def _predict_price_delta_gamma(model, features, *, device, dtype, batch_size):
    import torch

    model = model.to(device=device, dtype=dtype)
    model.eval()
    prices, deltas, gammas = [], [], []
    for start in range(0, len(features), batch_size):
        points = torch.as_tensor(features[start : start + batch_size], dtype=dtype, device=device).detach().clone().requires_grad_(True)
        price = premium_model_price(model, points)
        first = torch.autograd.grad(price, points, grad_outputs=torch.ones_like(price), create_graph=True, retain_graph=True)[0][:, 0]
        second = torch.autograd.grad(first, points, grad_outputs=torch.ones_like(first), create_graph=False, retain_graph=False)[0][:, 0]
        spot = torch.exp(points[:, 0])
        prices.append(price.detach().cpu().numpy())
        deltas.append((first / spot).detach().cpu().numpy())
        gammas.append(((second - first) / spot.square()).detach().cpu().numpy())
    values = tuple(np.concatenate(parts) if parts else np.array([], dtype=float) for parts in (prices, deltas, gammas))
    for value, name in zip(values, ("price", "Delta", "K*Gamma")):
        _require_finite(value, name)
    return values


def _residual_diagnostics(model, features, pde_mask, vi_mask, cfg, device, dtype):
    if len(features) == 0:
        return _unavailable_diagnostics("unavailable_no_valid_vi_points")
    if not bool(np.all(vi_mask)):
        raise ValueError("residual diagnostics require the predeclared valid-VI mask")
    residual = _evaluate_residual(model, features, device=device, dtype=dtype, batch_size=int(cfg["batch_size"]))
    gap = _predict_premium(model, features, device=device, dtype=dtype, batch_size=int(cfg["batch_size"]))
    if not np.any(pde_mask):
        masked = {"masked_pde_residual_samples": 0, "masked_pde_residual_mean_abs": None, "masked_pde_residual_rms": None}
    else:
        summary = residual_distribution_metrics(residual[pde_mask], prefix="masked_pde_residual")
        masked = {key: summary[key] for key in ("masked_pde_residual_samples", "masked_pde_residual_mean_abs", "masked_pde_residual_rms")}
    product = gap * residual
    return {
        "vi_collocation_samples": int(len(features)),
        **masked,
        "continuation_violation_mean": float(np.mean(np.maximum(-residual, 0.0))),
        "continuation_violation_max": float(np.max(np.maximum(-residual, 0.0))),
        "complementarity_product_mean_abs": float(np.mean(np.abs(product))),
        "complementarity_product_max_abs": float(np.max(np.abs(product))),
        "diagnostic_status": "defined",
    }


def _unavailable_diagnostics(status):
    return {
        "vi_collocation_samples": 0,
        "masked_pde_residual_samples": 0,
        "masked_pde_residual_mean_abs": None,
        "masked_pde_residual_rms": None,
        "continuation_violation_mean": None,
        "continuation_violation_max": None,
        "complementarity_product_mean_abs": None,
        "complementarity_product_max_abs": None,
        "diagnostic_status": status,
    }


def _evaluate_residual(model, features, *, device, dtype, batch_size):
    import torch

    values = []
    for start in range(0, len(features), batch_size):
        points = torch.as_tensor(features[start : start + batch_size], dtype=dtype, device=device).detach().clone().requires_grad_(True)
        values.append(residual_for_premium_model(model, points).detach().cpu().numpy())
    result = np.concatenate(values) if values else np.array([], dtype=float)
    _require_finite(result, "PDE residual")
    return result


def _predict_premium(model, features, *, device, dtype, batch_size):
    import torch

    values = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            points = torch.as_tensor(features[start : start + batch_size], dtype=dtype, device=device)
            values.append(model(points).detach().cpu().numpy().reshape(-1))
    result = np.concatenate(values) if values else np.array([], dtype=float)
    _require_finite(result, "premium prediction")
    return result


def _boundary_metrics(predicted_premium, features, boundary, global_idx, threshold):
    if boundary is None:
        return {"boundary_status": "unavailable_no_psor_boundary", "boundary_samples": 0, "boundary_rmse": None, "boundary_mae": None}
    target = np.asarray(boundary, dtype=float)[global_idx]
    differences = []
    for theta in sorted(set(np.round(features[:, 1], 10))):
        mask = np.isclose(features[:, 1], theta)
        if not np.any(mask) or not np.any(predicted_premium[mask] <= threshold):
            continue
        differences.append(float(np.max(np.exp(features[mask, 0])[predicted_premium[mask] <= threshold]) - np.nanmedian(target[mask])))
    if not differences:
        return {"boundary_status": "unavailable_no_threshold_crossing", "boundary_samples": 0, "boundary_rmse": None, "boundary_mae": None}
    values = np.asarray(differences)
    return {"boundary_status": "threshold_proxy", "boundary_samples": int(len(values)), "boundary_rmse": float(np.sqrt(np.mean(values ** 2))), "boundary_mae": float(np.mean(np.abs(values)))}


def _promotion_from_validation(metrics_rows, cfg):
    rows = [row for row in metrics_rows if row["split"] == "validation"]
    baseline = [row for row in rows if row["method"] == "supervised"]
    return {
        "selection_split": "validation",
        "legacy_test_used_for_selection": False,
        "comparisons": {method: _paired_promotion(baseline, [row for row in rows if row["method"] == method], cfg) for method in METHODS[1:]},
    }


def _paired_promotion(baseline_rows, candidate_rows, cfg):
    result = {"status": "insufficient_diagnostics", "metrics": {}, "gates": {"ordinary_metrics_percent": 2.0, "masked_pde_residual_percent": 5.0, "max_obstacle_violation": float(cfg["obstacle_tolerance"])}}
    gates = []
    missing = []
    for metric_name, is_primary, is_residual in PROMOTION_METRICS:
        baseline = _scenario_observations(baseline_rows, metric_name)
        candidate = _scenario_observations(candidate_rows, metric_name)
        if len(baseline) < 2 or {item.scenario_id for item in baseline} != {item.scenario_id for item in candidate}:
            result["metrics"][metric_name] = {"status": "unavailable"}
            missing.append(metric_name)
            continue
        try:
            summary = summarize_paired_scenarios(baseline, candidate, lower_is_better=True, bootstrap_iterations=int(cfg["bootstrap_iterations"]), random_seed=int(cfg["bootstrap_seed"]))
        except ValueError as error:
            if "zero baseline mean" not in str(error):
                raise
            result["metrics"][metric_name] = {"status": "unavailable_zero_baseline"}
            missing.append(metric_name)
            continue
        result["metrics"][metric_name] = {"status": "defined", "scenario_count": summary.scenario_count, "relative_percent_change": summary.relative_percent_change, "confidence_interval": list(summary.confidence_interval), "paired_p_value": summary.paired_p_value}
        gates.append(MetricGate(metric_name, summary.relative_percent_change, summary.confidence_interval, lower_is_better=True, is_residual=is_residual, is_primary=is_primary))
    obstacle_values = [row.get("max_obstacle_violation") for row in candidate_rows]
    if not obstacle_values or any(value is None or not np.isfinite(float(value)) for value in obstacle_values):
        result["obstacle"] = {"status": "unavailable"}
        missing.append("max_obstacle_violation")
    else:
        maximum = float(np.max(obstacle_values))
        result["obstacle"] = {"status": "defined", "max_obstacle_violation": maximum, "passes": maximum <= float(cfg["obstacle_tolerance"])}
    if missing:
        result["missing_diagnostics"] = missing
        raw_status = "insufficient_diagnostics"
    elif not result["obstacle"]["passes"]:
        raw_status = "failed_metric_gate"
    else:
        decision = decide_promotion(gates)
        raw_status = decision.status
        result["failed_metrics"] = list(decision.failed_metrics)
        result["improved_primary_metrics"] = list(decision.improved_primary_metrics)
    result["raw_status"] = raw_status
    if cfg["mode"] == "smoke":
        result["status"] = "diagnostic_only"
        result["claim_status"] = "smoke_non_evidence"
    else:
        result["status"] = raw_status
        result["claim_status"] = "diagnostic_only" if raw_status != "promote" else "screen_candidate"
    return result


def _scenario_observations(rows, metric_name):
    grouped: dict[str, list[float]] = {}
    for row in rows:
        value = row.get(metric_name)
        if value is None or not np.isfinite(float(value)):
            continue
        grouped.setdefault(row["regime_id"], []).append(float(value))
    return [ScenarioObservation(regime_id, values) for regime_id, values in sorted(grouped.items())]


def _method_run_records(metrics_rows, promotion, cfg, provenance):
    validation_rows = [row for row in metrics_rows if row["split"] == "validation"]
    records = []
    for method in METHODS:
        for seed in [int(value) for value in cfg["seeds"]]:
            selected = [row for row in validation_rows if row["method"] == method and row["seed"] == seed]
            primary = _mean_metric(selected, "price_rmse")
            baseline = _mean_metric([row for row in validation_rows if row["method"] == "supervised" and row["seed"] == seed], "price_rmse")
            status, claim_status = _record_status(method, promotion, cfg)
            records.append(MethodRunRecord("M02", RunPhase.SCREEN, status, config_id=method, primary_metric_value=primary, baseline_metric_value=baseline, optional_metrics={"delta_rmse": _mean_metric(selected, "delta_rmse"), "k_gamma_rmse": _mean_metric(selected, "k_gamma_rmse")}, scenario_count=len({row["regime_id"] for row in selected}), seed=seed, split="validation", branch=provenance["branch"], commit=provenance["commit"], timings_seconds={"training": _mean_metric(selected, "train_seconds")}, claim_status=claim_status, notes="Validation-only screening record; legacy_test is excluded from selection."))
    decision_status, decision_claim = _decision_status(promotion, cfg)
    baseline_primary = _mean_metric([row for row in validation_rows if row["method"] == "supervised"], "price_rmse")
    decision_record = MethodRunRecord("M02", RunPhase.SCREEN, decision_status, config_id="m02_screen_decision", primary_metric_value=baseline_primary, baseline_metric_value=baseline_primary, scenario_count=len({row["regime_id"] for row in validation_rows}), split="validation", branch=provenance["branch"], commit=provenance["commit"], claim_status=decision_claim, notes="M02 aggregate validation decision; no blind holdout access.")
    records.append(decision_record)
    return records, decision_record


def _record_status(method, promotion, cfg):
    if method == "supervised":
        return (RunStatus.DIAGNOSTIC_ONLY, "smoke_non_evidence" if cfg["mode"] == "smoke" else "baseline")
    status = promotion["comparisons"][method]["status"]
    if status == "promote":
        return RunStatus.COMPLETED, "screen_candidate"
    if status == "failed_metric_gate":
        return RunStatus.FAILED_METRIC_GATE, "failed_metric_gate"
    return RunStatus.DIAGNOSTIC_ONLY, "smoke_non_evidence" if cfg["mode"] == "smoke" else status


def _decision_status(promotion, cfg):
    statuses = [comparison["status"] for comparison in promotion["comparisons"].values()]
    if cfg["mode"] == "smoke":
        return RunStatus.DIAGNOSTIC_ONLY, "smoke_non_evidence"
    if "promote" in statuses:
        return RunStatus.COMPLETED, "screen_candidate"
    if statuses and all(status == "failed_metric_gate" for status in statuses):
        return RunStatus.FAILED_METRIC_GATE, "failed_metric_gate"
    return RunStatus.DIAGNOSTIC_ONLY, "diagnostic_only"


def _mean_metric(rows, name):
    values = [float(row[name]) for row in rows if row.get(name) is not None and np.isfinite(float(row[name]))]
    return float(np.mean(values)) if values else None


def _git_provenance():
    branch = subprocess.run(["git", "branch", "--show-current"], text=True, capture_output=True, check=True).stdout.strip()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=True).stdout.strip()
    return {"branch": branch, "commit": commit}


def _write_artifact_hashes(output_dir):
    paths = sorted(path for path in Path(output_dir).iterdir() if path.is_file() and path.name != "artifact_hashes.json")
    write_json(Path(output_dir) / "artifact_hashes.json", {"algorithm": "sha256", "files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}})


def _option_scope_mask(features, option_scope):
    if option_scope == "put":
        return features[:, 6] <= 0.5
    if option_scope == "call":
        return features[:, 6] > 0.5
    if option_scope == "all":
        return np.ones(len(features), dtype=bool)
    raise ValueError(f"unknown option_scope: {option_scope}")


def _torch_dtype(name):
    import torch

    if name != "float64":
        raise ValueError("M02 requires float64")
    return torch.float64


def _require_finite(values, name):
    if not np.isfinite(np.asarray(values, dtype=float)).all():
        raise FloatingPointError(f"{name} contains non-finite values")


def _require_finite_tensor(value, name):
    import torch

    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{name} contains non-finite values")


def _validate_metric_row(row):
    for name, value in row.items():
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            raise FloatingPointError(f"metric {name} contains a non-finite value")


def _write_csv(path, rows):
    ensure_dir(Path(path).parent)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/independent_m02_obstacle_pinn.yaml")
    parser.add_argument("--mode", choices=("canonical", "smoke"))
    args = parser.parse_args()
    print(json.dumps(run(args.config, overrides={"mode": args.mode} if args.mode else None), indent=2, sort_keys=True))
