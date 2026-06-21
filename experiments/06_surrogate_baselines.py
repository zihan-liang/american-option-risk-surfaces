"""Stage 7: direct-price surrogate baselines and violation audit."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from common import ensure_dir, load_config, write_csv, write_json
from models.metrics import (
    boundary_band_error,
    error_metrics,
    grouped_price_metrics,
    max_obstacle_violation,
    surface_shape_violations,
)
from models.mlp import build_mlp
from models.training import (
    capped_indices,
    choose_device,
    load_dataset_arrays,
    predict_linear_baseline,
    predict_torch_model,
    save_torch_model,
    set_global_seed,
    split_mask,
    train_linear_baseline,
    train_torch_price_model,
)
from plotting.publication import save_bar_figure


def run(
    config_path: str,
    *,
    output_root: str | Path = "results/experiment_06_surrogate_baselines",
    model_root: str | Path = "results/models",
    figure_root: str | Path = "results/figures/stage07_baseline_surrogates",
    overrides: dict | None = None,
) -> dict:
    cfg = load_config(config_path)
    if overrides:
        cfg = {**cfg, **overrides}
    set_global_seed(int(cfg.get("seed", 7)))
    out_dir = ensure_dir(output_root)
    model_dir = ensure_dir(model_root)
    fig_dir = ensure_dir(figure_root)
    write_json(out_dir / "baseline_config_manifest.json", cfg)

    data = load_dataset_arrays(cfg["dataset"])
    features = data["features"].astype(np.float32)
    labels = data["labels"].astype(np.float32)
    split = data["split"]
    price = labels[:, 0]
    payoff = labels[:, 2]
    is_call = features[:, 6]
    near_boundary = data.get("near_boundary", np.zeros(len(features), dtype=bool))
    regime_key = data["regime_key"].astype(str)

    seed = int(cfg.get("seed", 7))
    train_idx = capped_indices(split_mask(split, "train"), int(cfg.get("train_cap", 0)), seed)
    val_idx = capped_indices(split_mask(split, "validation"), int(cfg.get("validation_cap", 0)), seed + 1)
    test_idx = capped_indices(split_mask(split, "test"), int(cfg.get("test_cap", 0)), seed + 2)
    eval_idx = np.concatenate([val_idx, test_idx])

    metrics_rows: list[dict] = []
    by_regime_rows: list[dict] = []
    violation_rows: list[dict] = []
    history_rows: list[dict] = []
    summary_models: list[dict] = []

    linear_start = time.perf_counter()
    weights = train_linear_baseline(features[train_idx], price[train_idx], float(cfg.get("ridge", 1e-6)))
    linear_train_seconds = time.perf_counter() - linear_start
    np.savez_compressed(model_dir / "linear_price_baseline.npz", weights=weights, feature_names=np.array(cfg.get("feature_names", [])))
    linear_inference_start = time.perf_counter()
    linear_pred = predict_linear_baseline(features[eval_idx], weights)
    linear_inference_seconds = time.perf_counter() - linear_inference_start
    _record_model(
        "linear_price_baseline",
        linear_pred,
        eval_idx,
        data,
        price,
        payoff,
        is_call,
        near_boundary,
        regime_key,
        metrics_rows,
        by_regime_rows,
        violation_rows,
        summary_models,
        train_seconds=linear_train_seconds,
        inference_seconds=linear_inference_seconds,
    )
    history_rows.append({"model": "linear_price_baseline", "epoch": 0, "train_loss": float("nan"), "validation_rmse": _split_rmse(linear_pred, eval_idx, split, price, "validation"), "penalty_weight": 0.0})

    device = choose_device(cfg.get("device", "auto"))
    penalty_weights = [float(v) for v in cfg.get("penalty_weights", [0.0])]
    for penalty_weight in penalty_weights:
        model_name = "mlp_direct_price" if penalty_weight == 0.0 else f"mlp_obstacle_penalty_{penalty_weight:g}"
        model = build_mlp(features.shape[1], 1, hidden=int(cfg.get("hidden", 128)), depth=int(cfg.get("depth", 3)))
        start = time.perf_counter()
        trained, history = train_torch_price_model(
            model,
            features[train_idx],
            price[train_idx],
            payoff[train_idx],
            features[val_idx],
            price[val_idx],
            penalty_weight=penalty_weight,
            epochs=int(cfg.get("epochs", 20)),
            batch_size=int(cfg.get("batch_size", 8192)),
            learning_rate=float(cfg.get("learning_rate", 1e-3)),
            patience=int(cfg.get("patience", 4)),
            device=device,
            model_name=model_name,
        )
        train_seconds = time.perf_counter() - start
        checkpoint_path = model_dir / f"{model_name}.pt"
        save_torch_model(trained, checkpoint_path, {"model": model_name, "penalty_weight": penalty_weight, "config": cfg})
        inference_start = time.perf_counter()
        pred = predict_torch_model(trained, features[eval_idx], batch_size=int(cfg.get("batch_size", 8192)), device=device)
        inference_seconds = time.perf_counter() - inference_start
        history_rows.extend(history)
        _record_model(
            model_name,
            pred,
            eval_idx,
            data,
            price,
            payoff,
            is_call,
            near_boundary,
            regime_key,
            metrics_rows,
            by_regime_rows,
            violation_rows,
            summary_models,
            train_seconds=train_seconds,
            inference_seconds=inference_seconds,
        )

    write_csv(out_dir / "baseline_metrics.csv", metrics_rows)
    write_csv(out_dir / "baseline_by_regime.csv", by_regime_rows)
    write_csv(out_dir / "violation_table.csv", violation_rows)
    write_csv(out_dir / "training_history.csv", history_rows)
    _save_figures(metrics_rows, violation_rows, fig_dir)
    best = min((row for row in metrics_rows if row["split"] == "validation" and row["option_type"] == "all"), key=lambda row: row["rmse"])
    summary = {
        "models": len(summary_models),
        "best_validation_model": best["model"],
        "best_validation_rmse": best["rmse"],
        "train_samples": int(len(train_idx)),
        "validation_samples": int(len(val_idx)),
        "test_samples": int(len(test_idx)),
        "device": device,
        "note": "Direct-price baselines are not obstacle-guaranteed and are not Greek engines.",
    }
    write_json(out_dir / "baseline_summary.json", summary)
    return summary


def _record_model(
    model_name: str,
    prediction: np.ndarray,
    eval_idx: np.ndarray,
    data: dict,
    price: np.ndarray,
    payoff: np.ndarray,
    is_call: np.ndarray,
    near_boundary: np.ndarray,
    regime_key: np.ndarray,
    metrics_rows: list[dict],
    by_regime_rows: list[dict],
    violation_rows: list[dict],
    summary_models: list[dict],
    *,
    train_seconds: float,
    inference_seconds: float,
) -> None:
    split = data["split"][eval_idx]
    pred_target = price[eval_idx]
    pred_payoff = payoff[eval_idx]
    pred_is_call = is_call[eval_idx]
    pred_boundary = near_boundary[eval_idx]
    grouped = grouped_price_metrics(
        prediction=prediction,
        target=pred_target,
        payoff=pred_payoff,
        splits=split,
        is_call=pred_is_call,
        model_name=model_name,
    )
    metrics_rows.extend(grouped)
    for split_name in ["validation", "test"]:
        mask = _split_mask_local(split, split_name)
        if not np.any(mask):
            continue
        metrics = error_metrics(prediction[mask], pred_target[mask])
        metrics_rows.append(
            {
                "model": model_name,
                "split": split_name,
                "option_type": "all",
                "samples": int(np.count_nonzero(mask)),
                **metrics,
                "max_obstacle_violation": max_obstacle_violation(prediction[mask], pred_payoff[mask]),
            }
        )
        boundary = boundary_band_error(prediction[mask], pred_target[mask], pred_boundary[mask])
        shape = _approx_shape_by_regime(prediction[mask], data["features"][eval_idx][mask], regime_key[eval_idx][mask])
        violation_rows.append(
            {
                "model": model_name,
                "split": split_name,
                "max_obstacle_violation": max_obstacle_violation(prediction[mask], pred_payoff[mask]),
                **boundary,
                **shape,
                "train_seconds": train_seconds,
                "inference_seconds": inference_seconds,
                "inference_samples_per_second": _throughput(len(prediction), inference_seconds),
            }
        )
    by_regime_rows.extend(_by_regime_metrics(model_name, prediction, eval_idx, data, price, payoff, regime_key))
    summary_models.append({"model": model_name, "train_seconds": train_seconds, "inference_seconds": inference_seconds})


def _by_regime_metrics(
    model_name: str,
    prediction: np.ndarray,
    eval_idx: np.ndarray,
    data: dict,
    price: np.ndarray,
    payoff: np.ndarray,
    regime_key: np.ndarray,
) -> list[dict]:
    rows = []
    split = data["split"][eval_idx].astype(str)
    keys = regime_key[eval_idx]
    for key in sorted(set(keys)):
        mask = keys == key
        metrics = error_metrics(prediction[mask], price[eval_idx][mask])
        rows.append(
            {
                "model": model_name,
                "regime_key": key,
                "split": split[mask][0],
                "samples": int(np.count_nonzero(mask)),
                **metrics,
                "max_obstacle_violation": max_obstacle_violation(prediction[mask], payoff[eval_idx][mask]),
            }
        )
    return rows


def _approx_shape_by_regime(prediction: np.ndarray, features: np.ndarray, keys: np.ndarray) -> dict[str, float]:
    max_mono = 0.0
    max_convex = 0.0
    for key in sorted(set(keys)):
        mask = keys == key
        if np.count_nonzero(mask) < 4:
            continue
        option_type = "call" if np.mean(features[mask, 6]) > 0.5 else "put"
        tau_values = np.unique(features[mask, 1])
        for tau in tau_values[:10]:
            tau_mask = mask & np.isclose(features[:, 1], tau)
            if np.count_nonzero(tau_mask) < 4:
                continue
            order = np.argsort(features[tau_mask, 0])
            surface = prediction[tau_mask][order]
            shape = surface_shape_violations(surface, option_type)
            max_mono = max(max_mono, shape["monotonicity_violation"])
            max_convex = max(max_convex, shape["convexity_violation"])
    return {"monotonicity_violation": max_mono, "convexity_violation": max_convex}


def _split_rmse(prediction: np.ndarray, eval_idx: np.ndarray, split: np.ndarray, price: np.ndarray, split_name: str) -> float:
    values = split[eval_idx].astype(str)
    mask = values == split_name
    if not np.any(mask):
        return float("nan")
    return error_metrics(prediction[mask], price[eval_idx][mask])["rmse"]


def _split_mask_local(split: np.ndarray, name: str) -> np.ndarray:
    return split.astype(str) == name


def _throughput(samples: int, seconds: float) -> float:
    return float(samples / seconds) if seconds > 0 else float("inf")


def _save_figures(metrics_rows: list[dict], violation_rows: list[dict], figure_root: Path) -> None:
    import matplotlib.pyplot as plt

    ensure_dir(figure_root)
    validation = [row for row in metrics_rows if row["split"] == "validation" and row["option_type"] == "all"]
    if validation:
        save_bar_figure(
            figure_root / "validation_rmse.png",
            [row["model"] for row in validation],
            [row["rmse"] for row in validation],
            "Validation price RMSE",
            "Price RMSE",
            orientation="horizontal",
        )
    test_violations = [row for row in violation_rows if row["split"] == "test"]
    if test_violations:
        save_bar_figure(
            figure_root / "test_obstacle_violation.png",
            [row["model"] for row in test_violations],
            [row["max_obstacle_violation"] for row in test_violations],
            "Maximum test obstacle violation",
            "Obstacle violation",
            orientation="horizontal",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline_surrogates.yaml")
    args = parser.parse_args()
    print(run(args.config))
