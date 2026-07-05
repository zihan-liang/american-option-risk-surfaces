"""Stage 9: direct supervised early-exercise boundary diagnostic head."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from common import ensure_dir, load_config, write_csv, write_json
from models.boundary_head import BoundaryHeadModel
from models.metrics import (
    error_metrics,
    finite_boundary_mask,
    grouped_boundary_metrics,
    prediction_range_diagnostics,
)
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
    train_torch_regression_model,
)
from plotting.publication import save_bar_figure
from plotting.report_visuals import format_display_label


def run(
    config_path: str,
    *,
    output_root: str | Path = "results/experiment_09_boundary_head",
    model_root: str | Path = "results/models",
    figure_root: str | Path = "results/figures/stage09_boundary_head",
    overrides: dict | None = None,
) -> dict:
    cfg = load_config(config_path)
    if overrides:
        cfg = {**cfg, **overrides}
    set_global_seed(int(cfg.get("seed", 7)))
    out_dir = ensure_dir(output_root)
    model_dir = ensure_dir(model_root)
    fig_dir = ensure_dir(figure_root)
    write_json(out_dir / "boundary_config_manifest.json", cfg)

    data = load_dataset_arrays(cfg["dataset"])
    if "boundary" not in data:
        raise KeyError("dataset missing required boundary labels")
    features = data["features"].astype(np.float32)
    boundary = data["boundary"].astype(np.float32)
    split = data["split"]
    is_call = features[:, 6]
    regime_key = data["regime_key"].astype(str)

    finite = finite_boundary_mask(boundary)
    seed = int(cfg.get("seed", 7))
    train_idx = capped_indices(split_mask(split, "train") & finite, int(cfg.get("train_cap", 0)), seed)
    val_idx = capped_indices(split_mask(split, "validation") & finite, int(cfg.get("validation_cap", 0)), seed + 1)
    test_idx = capped_indices(split_mask(split, "test") & finite, int(cfg.get("test_cap", 0)), seed + 2)
    eval_idx = np.concatenate([val_idx, test_idx])
    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        summary = {
            "decision": "insufficient_boundary_data",
            "finite_boundary_samples": int(np.count_nonzero(finite)),
            "train_samples": int(len(train_idx)),
            "validation_samples": int(len(val_idx)),
            "test_samples": int(len(test_idx)),
        }
        _write_empty_outputs(out_dir, summary)
        return summary

    lower = float(np.min(boundary[train_idx]))
    upper = float(np.max(boundary[train_idx]))
    if np.isclose(lower, upper):
        upper = lower + 1e-6

    metrics_rows: list[dict] = []
    by_regime_rows: list[dict] = []
    history_rows: list[dict] = []
    range_rows: list[dict] = []

    start = time.perf_counter()
    linear_weights = train_linear_baseline(features[train_idx], boundary[train_idx], float(cfg.get("ridge", 1e-6)))
    linear_train_seconds = time.perf_counter() - start
    linear_path = model_dir / "boundary_head_linear.npz"
    np.savez_compressed(linear_path, weights=linear_weights, lower=lower, upper=upper)
    linear_start = time.perf_counter()
    linear_pred = predict_linear_baseline(features[eval_idx], linear_weights)
    linear_inference_seconds = time.perf_counter() - linear_start
    _record_boundary_model(
        model_name="boundary_head_linear",
        prediction=linear_pred,
        eval_idx=eval_idx,
        boundary=boundary,
        split=split,
        is_call=is_call,
        regime_key=regime_key,
        lower=lower,
        upper=upper,
        train_seconds=linear_train_seconds,
        inference_seconds=linear_inference_seconds,
        metrics_rows=metrics_rows,
        by_regime_rows=by_regime_rows,
        range_rows=range_rows,
    )
    history_rows.append(
        {
            "model": "boundary_head_linear",
            "epoch": 0,
            "train_loss": float("nan"),
            "validation_boundary_rmse": _split_rmse(linear_pred, eval_idx, split, boundary, "validation"),
        }
    )

    device = choose_device(cfg.get("device", "auto"))
    model = BoundaryHeadModel.build(
        features.shape[1],
        hidden=int(cfg.get("hidden", 128)),
        depth=int(cfg.get("depth", 3)),
        lower=lower,
        upper=upper,
    )
    start = time.perf_counter()
    trained, history = train_torch_regression_model(
        model,
        features[train_idx],
        boundary[train_idx],
        features[val_idx],
        boundary[val_idx],
        epochs=int(cfg.get("epochs", 30)),
        batch_size=int(cfg.get("batch_size", 8192)),
        learning_rate=float(cfg.get("learning_rate", 1e-3)),
        patience=int(cfg.get("patience", 5)),
        device=device,
        model_name="boundary_head_mlp",
        metric_name="validation_boundary_rmse",
    )
    mlp_train_seconds = time.perf_counter() - start
    inference_start = time.perf_counter()
    mlp_pred = predict_torch_model(trained, features[eval_idx], batch_size=int(cfg.get("batch_size", 8192)), device=device)
    mlp_inference_seconds = time.perf_counter() - inference_start
    history_rows.extend(history)
    _record_boundary_model(
        model_name="boundary_head_mlp",
        prediction=mlp_pred,
        eval_idx=eval_idx,
        boundary=boundary,
        split=split,
        is_call=is_call,
        regime_key=regime_key,
        lower=lower,
        upper=upper,
        train_seconds=mlp_train_seconds,
        inference_seconds=mlp_inference_seconds,
        metrics_rows=metrics_rows,
        by_regime_rows=by_regime_rows,
        range_rows=range_rows,
    )
    mlp_path = model_dir / "boundary_head_mlp.pt"
    save_torch_model(
        trained,
        mlp_path,
        {
            "model": "boundary_head_mlp",
            "input_dim": int(features.shape[1]),
            "hidden": int(cfg.get("hidden", 128)),
            "depth": int(cfg.get("depth", 3)),
            "lower": lower,
            "upper": upper,
            "target": "boundary_over_K",
            "claim_scope": "diagnostic_boundary_estimate_only",
        },
    )

    write_csv(out_dir / "boundary_metrics.csv", metrics_rows)
    write_csv(out_dir / "boundary_by_regime.csv", by_regime_rows)
    write_csv(out_dir / "boundary_training_history.csv", history_rows)
    _save_figures(metrics_rows, range_rows, fig_dir)
    test_mlp = next(
        row
        for row in metrics_rows
        if row["model"] == "boundary_head_mlp" and row["split"] == "test" and row["option_type"] == "all"
    )
    summary = {
        "decision": "scoped_boundary_diagnostic",
        "finite_boundary_samples": int(np.count_nonzero(finite)),
        "train_samples": int(len(train_idx)),
        "validation_samples": int(len(val_idx)),
        "test_samples": int(len(test_idx)),
        "boundary_label_min_train": lower,
        "boundary_label_max_train": upper,
        "selected_model": "boundary_head_mlp",
        "selected_test_boundary_rmse": test_mlp["boundary_rmse"],
        "selected_test_boundary_mae": test_mlp["boundary_mae"],
        "device": device,
        "note": "Boundary labels come from validated PSOR dataset artifacts; no price-threshold proxy is used.",
    }
    write_json(out_dir / "boundary_summary.json", summary)
    return summary


def _record_boundary_model(
    *,
    model_name: str,
    prediction: np.ndarray,
    eval_idx: np.ndarray,
    boundary: np.ndarray,
    split: np.ndarray,
    is_call: np.ndarray,
    regime_key: np.ndarray,
    lower: float,
    upper: float,
    train_seconds: float,
    inference_seconds: float,
    metrics_rows: list[dict],
    by_regime_rows: list[dict],
    range_rows: list[dict],
) -> None:
    eval_split = split[eval_idx]
    target = boundary[eval_idx]
    rows = grouped_boundary_metrics(
        prediction=prediction,
        target=target,
        splits=eval_split,
        is_call=is_call[eval_idx],
        model_name=model_name,
        lower=lower,
        upper=upper,
    )
    for row in rows:
        row["train_seconds"] = train_seconds
        row["inference_seconds"] = inference_seconds
        row["inference_samples_per_second"] = _throughput(len(prediction), inference_seconds)
        metrics_rows.append(row)
        if row["option_type"] == "all":
            range_rows.append(row)
    by_regime_rows.extend(_by_regime_metrics(model_name, prediction, eval_idx, boundary, split, is_call, regime_key, lower, upper))


def _by_regime_metrics(
    model_name: str,
    prediction: np.ndarray,
    eval_idx: np.ndarray,
    boundary: np.ndarray,
    split: np.ndarray,
    is_call: np.ndarray,
    regime_key: np.ndarray,
    lower: float,
    upper: float,
) -> list[dict]:
    rows = []
    keys = regime_key[eval_idx]
    split_values = split[eval_idx].astype(str)
    option_values = np.where(is_call[eval_idx] > 0.5, "call", "put")
    for key in sorted(set(keys)):
        mask = keys == key
        metrics = error_metrics(prediction[mask], boundary[eval_idx][mask])
        range_diag = prediction_range_diagnostics(prediction[mask], lower=lower, upper=upper)
        rows.append(
            {
                "model": model_name,
                "regime_key": key,
                "split": split_values[mask][0],
                "option_type": _option_summary(option_values[mask]),
                "samples": int(np.count_nonzero(mask)),
                "boundary_rmse": metrics["rmse"],
                "boundary_mae": metrics["mae"],
                "boundary_max_error": metrics["max_error"],
                "prediction_min": range_diag["prediction_min"],
                "prediction_max": range_diag["prediction_max"],
                "out_of_training_range_fraction": range_diag["out_of_training_range_fraction"],
            }
        )
    return rows


def _split_rmse(prediction: np.ndarray, eval_idx: np.ndarray, split: np.ndarray, boundary: np.ndarray, split_name: str) -> float:
    mask = split[eval_idx].astype(str) == split_name
    if not np.any(mask):
        return float("nan")
    return error_metrics(prediction[mask], boundary[eval_idx][mask])["rmse"]


def _option_summary(values: np.ndarray) -> str:
    unique = sorted(set(values.tolist()))
    return unique[0] if len(unique) == 1 else "mixed"


def _throughput(samples: int, seconds: float) -> float:
    return float(samples / seconds) if seconds > 0 else float("inf")


def _write_empty_outputs(out_dir: Path, summary: dict) -> None:
    write_csv(out_dir / "boundary_metrics.csv", [summary])
    write_csv(out_dir / "boundary_by_regime.csv", [summary])
    write_csv(out_dir / "boundary_training_history.csv", [summary])
    write_json(out_dir / "boundary_summary.json", summary)


def _save_figures(metrics_rows: list[dict], range_rows: list[dict], figure_root: Path) -> None:
    import matplotlib.pyplot as plt

    ensure_dir(figure_root)
    test_rows = [row for row in metrics_rows if row["split"] == "test" and row["option_type"] in {"put", "call"}]
    if test_rows:
        labels = [f"{row['model']}|{row['option_type']}" for row in test_rows]
        aliases = {
            label: f"{format_display_label(row['model'])}: {row['option_type'].title()}"
            for label, row in zip(labels, test_rows)
        }
        save_bar_figure(
            figure_root / "test_boundary_mae_by_option.png",
            labels,
            [row["boundary_mae"] for row in test_rows],
            "Test boundary MAE by option type",
            "Boundary MAE",
            orientation="horizontal",
            display_labels=aliases,
        )
    if range_rows:
        fig, ax = plt.subplots(figsize=(6, 4))
        labels = [f"{row['model']}\n{row['split']}" for row in range_rows]
        ax.bar(labels, [row["out_of_training_range_fraction"] for row in range_rows])
        ax.set_ylabel("out-of-training-range fraction")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(figure_root / "boundary_prediction_range.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/boundary_head.yaml")
    args = parser.parse_args()
    print(run(args.config))
