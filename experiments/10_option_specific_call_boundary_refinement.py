"""Stage 9 refinement: option-specific dividend-call boundary diagnostic head."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from common import ensure_dir, load_config, write_csv, write_json
from models.boundary_head import BoundaryHeadModel
from models.metrics import error_metrics, finite_boundary_mask, grouped_boundary_metrics, prediction_range_diagnostics
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


def run(
    config_path: str,
    *,
    output_root: str | Path = "results/experiment_10_option_specific_call_boundary",
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
    write_json(out_dir / "call_boundary_config_manifest.json", cfg)

    data = load_dataset_arrays(cfg["dataset"])
    if "boundary" not in data:
        raise KeyError("dataset missing required boundary labels")
    features = data["features"].astype(np.float32)
    boundary = data["boundary"].astype(np.float32)
    split = data["split"]
    is_call = features[:, 6] > 0.5
    finite = finite_boundary_mask(boundary)
    call_finite = is_call & finite
    seed = int(cfg.get("seed", 7))
    pooled_train_idx = capped_indices(split_mask(split, "train") & finite, int(cfg.get("train_cap", 0)), seed)
    call_train_idx = capped_indices(split_mask(split, "train") & call_finite, int(cfg.get("train_cap", 0)), seed)
    call_val_idx = capped_indices(split_mask(split, "validation") & call_finite, int(cfg.get("validation_cap", 0)), seed + 1)
    call_test_idx = capped_indices(split_mask(split, "test") & call_finite, int(cfg.get("test_cap", 0)), seed + 2)
    call_eval_idx = np.concatenate([call_val_idx, call_test_idx])

    if len(call_train_idx) == 0 or len(call_val_idx) == 0 or len(call_test_idx) == 0:
        summary = {
            "decision": "insufficient_call_data",
            "finite_call_boundary_samples": int(np.count_nonzero(call_finite)),
            "call_train_samples": int(len(call_train_idx)),
            "call_validation_samples": int(len(call_val_idx)),
            "call_test_samples": int(len(call_test_idx)),
            "note": "No dividend-call boundary labels were fabricated.",
        }
        write_csv(out_dir / "call_boundary_metrics.csv", [summary])
        write_csv(out_dir / "call_boundary_by_regime.csv", [summary])
        write_json(out_dir / "call_boundary_summary.json", summary)
        return summary

    lower = float(np.min(boundary[call_train_idx]))
    upper = float(np.max(boundary[call_train_idx]))
    if np.isclose(lower, upper):
        upper = lower + 1e-6

    metrics_rows: list[dict] = []
    by_regime_rows: list[dict] = []
    history_rows: list[dict] = []

    pooled_weights = train_linear_baseline(features[pooled_train_idx], boundary[pooled_train_idx], float(cfg.get("ridge", 1e-6)))
    pooled_pred = predict_linear_baseline(features[call_eval_idx], pooled_weights)
    _record_call_model("pooled_boundary_head_linear_on_calls", pooled_pred, call_eval_idx, boundary, split, features[:, 6], data["regime_key"].astype(str), lower, upper, metrics_rows, by_regime_rows)

    call_weights = train_linear_baseline(features[call_train_idx], boundary[call_train_idx], float(cfg.get("ridge", 1e-6)))
    np.savez_compressed(model_dir / "call_boundary_head_linear.npz", weights=call_weights, lower=lower, upper=upper)
    call_linear_pred = predict_linear_baseline(features[call_eval_idx], call_weights)
    _record_call_model("call_boundary_head_linear", call_linear_pred, call_eval_idx, boundary, split, features[:, 6], data["regime_key"].astype(str), lower, upper, metrics_rows, by_regime_rows)
    history_rows.append(
        {
            "model": "call_boundary_head_linear",
            "epoch": 0,
            "train_loss": float("nan"),
            "validation_boundary_rmse": _split_rmse(call_linear_pred, call_eval_idx, split, boundary, "validation"),
        }
    )

    device = choose_device(cfg.get("device", "auto"))
    start = time.perf_counter()
    call_model = BoundaryHeadModel.build(
        features.shape[1],
        hidden=int(cfg.get("hidden", 128)),
        depth=int(cfg.get("depth", 3)),
        lower=lower,
        upper=upper,
    )
    trained, history = train_torch_regression_model(
        call_model,
        features[call_train_idx],
        boundary[call_train_idx],
        features[call_val_idx],
        boundary[call_val_idx],
        epochs=int(cfg.get("epochs", 30)),
        batch_size=int(cfg.get("batch_size", 8192)),
        learning_rate=float(cfg.get("learning_rate", 1e-3)),
        patience=int(cfg.get("patience", 5)),
        device=device,
        model_name="call_boundary_head_mlp",
        metric_name="validation_boundary_rmse",
    )
    train_seconds = time.perf_counter() - start
    pred_start = time.perf_counter()
    call_mlp_pred = predict_torch_model(trained, features[call_eval_idx], batch_size=int(cfg.get("batch_size", 8192)), device=device)
    inference_seconds = time.perf_counter() - pred_start
    history_rows.extend(history)
    _record_call_model(
        "call_boundary_head_mlp",
        call_mlp_pred,
        call_eval_idx,
        boundary,
        split,
        features[:, 6],
        data["regime_key"].astype(str),
        lower,
        upper,
        metrics_rows,
        by_regime_rows,
        train_seconds=train_seconds,
        inference_seconds=inference_seconds,
    )
    save_torch_model(
        trained,
        model_dir / "call_boundary_head_mlp.pt",
        {
            "model": "call_boundary_head_mlp",
            "input_dim": int(features.shape[1]),
            "hidden": int(cfg.get("hidden", 128)),
            "depth": int(cfg.get("depth", 3)),
            "lower": lower,
            "upper": upper,
            "target": "dividend_call_boundary_over_K",
            "claim_scope": "scoped_call_boundary_diagnostic_only",
        },
    )

    write_csv(out_dir / "call_boundary_metrics.csv", metrics_rows)
    write_csv(out_dir / "call_boundary_by_regime.csv", by_regime_rows)
    write_csv(out_dir / "call_boundary_training_history.csv", history_rows)
    _save_figures(metrics_rows, fig_dir)
    selected = next(row for row in metrics_rows if row["model"] == "call_boundary_head_mlp" and row["split"] == "test" and row["option_type"] == "call")
    summary = {
        "decision": "scoped_call_boundary_diagnostic",
        "finite_call_boundary_samples": int(np.count_nonzero(call_finite)),
        "call_train_samples": int(len(call_train_idx)),
        "call_validation_samples": int(len(call_val_idx)),
        "call_test_samples": int(len(call_test_idx)),
        "selected_model": "call_boundary_head_mlp",
        "selected_test_boundary_rmse": selected["boundary_rmse"],
        "selected_test_boundary_mae": selected["boundary_mae"],
        "comparison_models": sorted({row["model"] for row in metrics_rows}),
        "device": device,
        "note": "Call refinement uses only finite dividend-call boundary labels; no-boundary cases are counted, not relabeled.",
    }
    write_json(out_dir / "call_boundary_summary.json", summary)
    return summary


def _record_call_model(
    model_name: str,
    prediction: np.ndarray,
    eval_idx: np.ndarray,
    boundary: np.ndarray,
    split: np.ndarray,
    is_call: np.ndarray,
    regime_key: np.ndarray,
    lower: float,
    upper: float,
    metrics_rows: list[dict],
    by_regime_rows: list[dict],
    *,
    train_seconds: float = float("nan"),
    inference_seconds: float = float("nan"),
) -> None:
    rows = grouped_boundary_metrics(
        prediction=prediction,
        target=boundary[eval_idx],
        splits=split[eval_idx],
        is_call=is_call[eval_idx],
        model_name=model_name,
        lower=lower,
        upper=upper,
    )
    for row in rows:
        if row["option_type"] == "call":
            row["train_seconds"] = train_seconds
            row["inference_seconds"] = inference_seconds
            metrics_rows.append(row)
    keys = regime_key[eval_idx]
    split_values = split[eval_idx].astype(str)
    for key in sorted(set(keys)):
        mask = keys == key
        metrics = error_metrics(prediction[mask], boundary[eval_idx][mask])
        range_diag = prediction_range_diagnostics(prediction[mask], lower=lower, upper=upper)
        by_regime_rows.append(
            {
                "model": model_name,
                "regime_key": key,
                "split": split_values[mask][0],
                "option_type": "call",
                "samples": int(np.count_nonzero(mask)),
                "boundary_rmse": metrics["rmse"],
                "boundary_mae": metrics["mae"],
                "boundary_max_error": metrics["max_error"],
                "prediction_min": range_diag["prediction_min"],
                "prediction_max": range_diag["prediction_max"],
                "out_of_training_range_fraction": range_diag["out_of_training_range_fraction"],
            }
        )


def _split_rmse(prediction: np.ndarray, eval_idx: np.ndarray, split: np.ndarray, boundary: np.ndarray, split_name: str) -> float:
    mask = split[eval_idx].astype(str) == split_name
    if not np.any(mask):
        return float("nan")
    return error_metrics(prediction[mask], boundary[eval_idx][mask])["rmse"]


def _save_figures(metrics_rows: list[dict], figure_root: Path) -> None:
    import matplotlib.pyplot as plt

    ensure_dir(figure_root)
    test_rows = [row for row in metrics_rows if row["split"] == "test" and row["option_type"] == "call"]
    if test_rows:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar([row["model"] for row in test_rows], [row["boundary_mae"] for row in test_rows])
        ax.set_ylabel("call test boundary MAE")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(figure_root / "call_boundary_refinement_mae.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/call_boundary_refinement.yaml")
    args = parser.parse_args()
    print(run(args.config))
