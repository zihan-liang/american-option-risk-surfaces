"""Stage 8: positive continuation-premium price surrogate audit."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from common import ensure_dir, load_config, write_csv, write_json
from models.metrics import (
    boundary_band_error,
    error_metrics,
    grouped_premium_metrics,
    grouped_price_metrics,
    masked_error_metrics,
    max_obstacle_violation,
)
from models.premium_surrogate import PositivePremiumModel, reconstruct_price
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
    train_torch_premium_model,
)


def run(
    config_path: str,
    *,
    output_root: str | Path = "results/experiment_07_positive_premium_tuning",
    model_root: str | Path = "results/models",
    figure_root: str | Path = "results/figures/stage08_positive_premium_surrogate",
    overrides: dict | None = None,
) -> dict:
    cfg = load_config(config_path)
    if overrides:
        cfg = {**cfg, **overrides}
    set_global_seed(int(cfg.get("seed", 7)))
    out_dir = ensure_dir(output_root)
    model_dir = ensure_dir(model_root)
    fig_dir = ensure_dir(figure_root)
    write_json(out_dir / "positive_premium_config_manifest.json", cfg)

    data = load_dataset_arrays(cfg["dataset"])
    features = data["features"].astype(np.float32)
    labels = data["labels"].astype(np.float32)
    split = data["split"]
    price = labels[:, 0]
    premium = labels[:, 1]
    payoff = labels[:, 2]
    is_call = features[:, 6]
    regime_key = data["regime_key"].astype(str)
    near_boundary = data.get("near_boundary", np.zeros(len(features), dtype=bool))
    strict_mask = data.get("strict_greek_mask", np.ones(len(features), dtype=bool))

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
    linear_weights = train_linear_baseline(features[train_idx], np.log1p(premium[train_idx]), float(cfg.get("ridge", 1e-6)))
    linear_train_seconds = time.perf_counter() - linear_start
    linear_model_path = model_dir / "positive_premium_linear.npz"
    np.savez_compressed(linear_model_path, weights=linear_weights, feature_names=np.array(cfg.get("feature_names", [])))
    linear_inference_start = time.perf_counter()
    linear_premium = np.maximum(np.expm1(predict_linear_baseline(features[eval_idx], linear_weights)), 0.0)
    linear_inference_seconds = time.perf_counter() - linear_inference_start
    linear_price = reconstruct_price(payoff[eval_idx], linear_premium)
    _record_model(
        "positive_premium_linear",
        linear_price,
        linear_premium,
        eval_idx,
        data,
        price,
        premium,
        payoff,
        is_call,
        near_boundary,
        strict_mask,
        regime_key,
        metrics_rows,
        by_regime_rows,
        violation_rows,
        summary_models,
        train_seconds=linear_train_seconds,
        inference_seconds=linear_inference_seconds,
    )
    history_rows.append(
        {
            "model": "positive_premium_linear",
            "epoch": 0,
            "train_loss": float("nan"),
            "validation_premium_rmse": _split_rmse(linear_premium, eval_idx, split, premium, "validation"),
        }
    )

    device = choose_device(cfg.get("device", "auto"))
    model = PositivePremiumModel.build(features.shape[1], hidden=int(cfg.get("hidden", 128)), depth=int(cfg.get("depth", 3)))
    start = time.perf_counter()
    trained, history = train_torch_premium_model(
        model,
        features[train_idx],
        premium[train_idx],
        features[val_idx],
        premium[val_idx],
        epochs=int(cfg.get("epochs", 30)),
        batch_size=int(cfg.get("batch_size", 8192)),
        learning_rate=float(cfg.get("learning_rate", 1e-3)),
        patience=int(cfg.get("patience", 5)),
        device=device,
        model_name="positive_premium_mlp",
    )
    mlp_train_seconds = time.perf_counter() - start
    inference_start = time.perf_counter()
    mlp_premium = predict_torch_model(trained, features[eval_idx], batch_size=int(cfg.get("batch_size", 8192)), device=device)
    mlp_inference_seconds = time.perf_counter() - inference_start
    mlp_price = reconstruct_price(payoff[eval_idx], mlp_premium)
    history_rows.extend(history)
    _record_model(
        "positive_premium_mlp",
        mlp_price,
        mlp_premium,
        eval_idx,
        data,
        price,
        premium,
        payoff,
        is_call,
        near_boundary,
        strict_mask,
        regime_key,
        metrics_rows,
        by_regime_rows,
        violation_rows,
        summary_models,
        train_seconds=mlp_train_seconds,
        inference_seconds=mlp_inference_seconds,
    )

    best = min((row for row in metrics_rows if row["split"] == "validation" and row["option_type"] == "all"), key=lambda row: row["rmse"])
    selected_model = "positive_premium_mlp"
    selected_path = Path(cfg.get("model_out", model_dir / "selected_final_price_premium.pt"))
    save_torch_model(
        trained,
        selected_path,
        {
            "model": selected_model,
            "input_dim": int(features.shape[1]),
            "hidden": int(cfg.get("hidden", 128)),
            "depth": int(cfg.get("depth", 3)),
            "config": cfg,
            "target": "continuation_premium",
            "price_rule": "payoff_plus_softplus_premium",
            "selection_note": "The .pt artifact is the selected neural positive-premium model; linear remains a sanity baseline.",
        },
    )
    mlp_validation = next(
        row
        for row in metrics_rows
        if row["model"] == "positive_premium_mlp" and row["split"] == "validation" and row["option_type"] == "all"
    )

    write_csv(out_dir / "positive_premium_metrics.csv", metrics_rows)
    write_csv(out_dir / "positive_premium_by_regime.csv", by_regime_rows)
    write_csv(out_dir / "positive_premium_violation_table.csv", violation_rows)
    write_csv(out_dir / "positive_premium_training_history.csv", history_rows)
    _save_figures(metrics_rows, violation_rows, fig_dir)
    summary = {
        "models": len(summary_models),
        "best_validation_model": best["model"],
        "best_validation_rmse": best["rmse"],
        "selected_model": selected_model,
        "selected_validation_rmse": mlp_validation["rmse"],
        "selected_model_path": str(selected_path),
        "train_samples": int(len(train_idx)),
        "validation_samples": int(len(val_idx)),
        "test_samples": int(len(test_idx)),
        "device": device,
        "max_obstacle_violation": float(max(row["max_obstacle_violation"] for row in violation_rows)),
        "note": "Positive continuation-premium model is price/premium only; it is not a boundary or Greek engine.",
    }
    write_json(out_dir / "positive_premium_summary.json", summary)
    return summary


def _record_model(
    model_name: str,
    price_prediction: np.ndarray,
    premium_prediction: np.ndarray,
    eval_idx: np.ndarray,
    data: dict,
    price: np.ndarray,
    premium: np.ndarray,
    payoff: np.ndarray,
    is_call: np.ndarray,
    near_boundary: np.ndarray,
    strict_mask: np.ndarray,
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
    target_price = price[eval_idx]
    target_premium = premium[eval_idx]
    target_payoff = payoff[eval_idx]
    pred_is_call = is_call[eval_idx]
    grouped_price = grouped_price_metrics(
        prediction=price_prediction,
        target=target_price,
        payoff=target_payoff,
        splits=split,
        is_call=pred_is_call,
        model_name=model_name,
    )
    metrics_rows.extend(grouped_price)
    metrics_rows.extend(
        grouped_premium_metrics(
            prediction=premium_prediction,
            target=target_premium,
            splits=split,
            is_call=pred_is_call,
            model_name=model_name,
        )
    )
    for split_name in ["validation", "test"]:
        mask = split.astype(str) == split_name
        if not np.any(mask):
            continue
        price_metrics = error_metrics(price_prediction[mask], target_price[mask])
        premium_metrics = error_metrics(premium_prediction[mask], target_premium[mask])
        metrics_rows.append(
            {
                "model": model_name,
                "split": split_name,
                "option_type": "all",
                "samples": int(np.count_nonzero(mask)),
                **price_metrics,
                **{f"premium_{key}": value for key, value in premium_metrics.items()},
                "max_obstacle_violation": max_obstacle_violation(price_prediction[mask], target_payoff[mask]),
            }
        )
        boundary = boundary_band_error(price_prediction[mask], target_price[mask], near_boundary[eval_idx][mask])
        strict = masked_error_metrics(price_prediction[mask], target_price[mask], strict_mask[eval_idx][mask], prefix="strict_mask_price")
        violation_rows.append(
            {
                "model": model_name,
                "split": split_name,
                "max_obstacle_violation": max_obstacle_violation(price_prediction[mask], target_payoff[mask]),
                **boundary,
                **strict,
                "train_seconds": train_seconds,
                "inference_seconds": inference_seconds,
                "inference_samples_per_second": _throughput(len(price_prediction), inference_seconds),
            }
        )
    by_regime_rows.extend(_by_regime_metrics(model_name, price_prediction, premium_prediction, eval_idx, data, price, premium, payoff, regime_key))
    summary_models.append({"model": model_name, "train_seconds": train_seconds, "inference_seconds": inference_seconds})


def _by_regime_metrics(
    model_name: str,
    price_prediction: np.ndarray,
    premium_prediction: np.ndarray,
    eval_idx: np.ndarray,
    data: dict,
    price: np.ndarray,
    premium: np.ndarray,
    payoff: np.ndarray,
    regime_key: np.ndarray,
) -> list[dict]:
    rows = []
    split = data["split"][eval_idx].astype(str)
    keys = regime_key[eval_idx]
    for key in sorted(set(keys)):
        mask = keys == key
        price_metrics = error_metrics(price_prediction[mask], price[eval_idx][mask])
        premium_metrics = error_metrics(premium_prediction[mask], premium[eval_idx][mask])
        rows.append(
            {
                "model": model_name,
                "regime_key": key,
                "split": split[mask][0],
                "samples": int(np.count_nonzero(mask)),
                **price_metrics,
                **{f"premium_{metric}": value for metric, value in premium_metrics.items()},
                "max_obstacle_violation": max_obstacle_violation(price_prediction[mask], payoff[eval_idx][mask]),
            }
        )
    return rows


def _split_rmse(prediction: np.ndarray, eval_idx: np.ndarray, split: np.ndarray, target: np.ndarray, split_name: str) -> float:
    mask = split[eval_idx].astype(str) == split_name
    if not np.any(mask):
        return float("nan")
    return error_metrics(prediction[mask], target[eval_idx][mask])["rmse"]


def _throughput(samples: int, seconds: float) -> float:
    return float(samples / seconds) if seconds > 0 else float("inf")


def _save_figures(metrics_rows: list[dict], violation_rows: list[dict], figure_root: Path) -> None:
    import matplotlib.pyplot as plt

    ensure_dir(figure_root)
    validation = [row for row in metrics_rows if row["split"] == "validation" and row["option_type"] == "all" and "rmse" in row]
    if validation:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar([row["model"] for row in validation], [row["rmse"] for row in validation])
        ax.set_ylabel("validation price RMSE")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(figure_root / "validation_price_rmse.png", dpi=160)
        plt.close(fig)
    test_rows = [row for row in violation_rows if row["split"] == "test"]
    if test_rows:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar([row["model"] for row in test_rows], [row["max_obstacle_violation"] for row in test_rows])
        ax.set_ylabel("test obstacle violation")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(figure_root / "test_obstacle_violation.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/positive_premium.yaml")
    args = parser.parse_args()
    print(run(args.config))
