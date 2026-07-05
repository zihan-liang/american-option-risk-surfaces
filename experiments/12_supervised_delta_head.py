"""Stage 10: supervised bounded Delta diagnostic head."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from common import ensure_dir, load_config, write_csv, write_json
from models.delta_head import BoundedDeltaHead
from models.metrics import delta_bound_violation, error_metrics, grouped_delta_metrics, masked_error_metrics
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
    output_root: str | Path = "results/experiment_12_supervised_delta_head",
    model_root: str | Path = "results/models",
    figure_root: str | Path = "results/figures/stage10_delta_head",
    overrides: dict | None = None,
) -> dict:
    cfg = load_config(config_path)
    if overrides:
        cfg = {**cfg, **overrides}
    set_global_seed(int(cfg.get("seed", 7)))
    out_dir = ensure_dir(output_root)
    model_dir = ensure_dir(model_root)
    fig_dir = ensure_dir(figure_root)
    write_json(out_dir / "delta_head_config_manifest.json", cfg)
    data = load_dataset_arrays(cfg["dataset"])
    if "strict_greek_mask" not in data:
        raise KeyError("dataset missing strict_greek_mask")
    features = data["features"].astype(np.float32)
    labels = data["labels"].astype(np.float32)
    target_delta = labels[:, 3]
    strict = data["strict_greek_mask"].astype(bool)
    split = data["split"]
    is_call = features[:, 6]
    seed = int(cfg.get("seed", 7))
    train_idx = capped_indices(split_mask(split, "train") & strict, int(cfg.get("train_cap", 150000)), seed)
    val_idx = capped_indices(split_mask(split, "validation") & strict, int(cfg.get("validation_cap", 75000)), seed + 1)
    test_idx = capped_indices(split_mask(split, "test") & strict, int(cfg.get("test_cap", 150000)), seed + 2)
    eval_idx = np.concatenate([val_idx, test_idx])
    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        summary = {
            "decision": "insufficient_strict_delta_data",
            "uses_strict_greek_mask": True,
            "train_samples": int(len(train_idx)),
            "validation_samples": int(len(val_idx)),
            "test_samples": int(len(test_idx)),
        }
        _write_empty_outputs(out_dir, summary)
        return summary

    metrics_rows: list[dict] = []
    by_regime_rows: list[dict] = []
    violation_rows: list[dict] = []
    history_rows: list[dict] = []
    start = time.perf_counter()
    weights = train_linear_baseline(features[train_idx], target_delta[train_idx], float(cfg.get("ridge", 1e-6)))
    linear_train_seconds = time.perf_counter() - start
    np.savez_compressed(model_dir / "delta_head_linear.npz", weights=weights)
    linear_start = time.perf_counter()
    linear_pred = _clip_delta(predict_linear_baseline(features[eval_idx], weights), is_call[eval_idx])
    linear_inference_seconds = time.perf_counter() - linear_start
    _record_delta_model(
        "bounded_delta_linear",
        linear_pred,
        eval_idx,
        data,
        target_delta,
        metrics_rows,
        by_regime_rows,
        violation_rows,
        train_seconds=linear_train_seconds,
        inference_seconds=linear_inference_seconds,
    )
    history_rows.append(
        {
            "model": "bounded_delta_linear",
            "epoch": 0,
            "train_loss": float("nan"),
            "validation_delta_rmse": _split_rmse(linear_pred, eval_idx, split, target_delta, "validation"),
        }
    )

    device = choose_device(cfg.get("device", "auto"))
    model = BoundedDeltaHead.build(features.shape[1], hidden=int(cfg.get("hidden", 128)), depth=int(cfg.get("depth", 3)))
    start = time.perf_counter()
    trained, history = train_torch_regression_model(
        model,
        features[train_idx],
        target_delta[train_idx],
        features[val_idx],
        target_delta[val_idx],
        epochs=int(cfg.get("epochs", 30)),
        batch_size=int(cfg.get("batch_size", 8192)),
        learning_rate=float(cfg.get("learning_rate", 1e-3)),
        patience=int(cfg.get("patience", 5)),
        device=device,
        model_name="bounded_delta_head_mlp",
        metric_name="validation_delta_rmse",
    )
    mlp_train_seconds = time.perf_counter() - start
    inference_start = time.perf_counter()
    mlp_pred = predict_torch_model(trained, features[eval_idx], batch_size=int(cfg.get("batch_size", 8192)), device=device)
    mlp_inference_seconds = time.perf_counter() - inference_start
    history_rows.extend(history)
    _record_delta_model(
        "bounded_delta_head_mlp",
        mlp_pred,
        eval_idx,
        data,
        target_delta,
        metrics_rows,
        by_regime_rows,
        violation_rows,
        train_seconds=mlp_train_seconds,
        inference_seconds=mlp_inference_seconds,
    )
    save_torch_model(
        trained,
        model_dir / "bounded_delta_head_mlp.pt",
        {
            "model": "bounded_delta_head_mlp",
            "input_dim": int(features.shape[1]),
            "hidden": int(cfg.get("hidden", 128)),
            "depth": int(cfg.get("depth", 3)),
            "target": "strict_mask_delta",
            "claim_scope": "bounded_supervised_delta_diagnostic_only",
        },
    )

    write_csv(out_dir / "delta_head_metrics.csv", metrics_rows)
    write_csv(out_dir / "delta_head_by_regime.csv", by_regime_rows)
    write_csv(out_dir / "delta_head_training_history.csv", history_rows)
    write_csv(out_dir / "delta_head_violation_table.csv", violation_rows)
    _save_figures(metrics_rows, violation_rows, fig_dir)
    selected = next(row for row in metrics_rows if row["model"] == "bounded_delta_head_mlp" and row["split"] == "test" and row["option_type"] == "all")
    summary = {
        "decision": "bounded_supervised_delta_diagnostic",
        "uses_strict_greek_mask": True,
        "strict_samples": int(np.count_nonzero(strict)),
        "train_samples": int(len(train_idx)),
        "validation_samples": int(len(val_idx)),
        "test_samples": int(len(test_idx)),
        "selected_model": "bounded_delta_head_mlp",
        "selected_test_delta_rmse": selected["delta_rmse"],
        "selected_test_delta_mae": selected["delta_mae"],
        "selected_max_delta_bound_violation": selected["max_delta_bound_violation"],
        "device": device,
        "note": "Delta head is supervised and bounded; it is separate from price, boundary, and Gamma models.",
    }
    write_json(out_dir / "delta_head_summary.json", summary)
    return summary


def _record_delta_model(
    model_name: str,
    prediction: np.ndarray,
    eval_idx: np.ndarray,
    data: dict,
    target_delta: np.ndarray,
    metrics_rows: list[dict],
    by_regime_rows: list[dict],
    violation_rows: list[dict],
    *,
    train_seconds: float,
    inference_seconds: float,
) -> None:
    split = data["split"][eval_idx]
    features = data["features"][eval_idx]
    target = target_delta[eval_idx]
    rows = grouped_delta_metrics(
        prediction=prediction,
        target=target,
        splits=split,
        is_call=features[:, 6],
        model_name=model_name,
    )
    for row in rows:
        row["train_seconds"] = train_seconds
        row["inference_seconds"] = inference_seconds
        row["inference_samples_per_second"] = _throughput(len(prediction), inference_seconds)
        metrics_rows.append(row)
    for split_name in ["validation", "test"]:
        mask = split.astype(str) == split_name
        if not np.any(mask):
            continue
        violation = {
            "model": model_name,
            "split": split_name,
            "samples": int(np.count_nonzero(mask)),
            "max_delta_bound_violation": delta_bound_violation(prediction[mask], features[mask, 6]),
            "train_seconds": train_seconds,
            "inference_seconds": inference_seconds,
        }
        for name, source in [
            ("near_boundary_delta", data.get("near_boundary", np.zeros(len(data["features"]), dtype=bool))[eval_idx]),
            ("near_kink_delta", data.get("near_kink", np.zeros(len(data["features"]), dtype=bool))[eval_idx]),
            ("near_zero_spot_delta", data.get("near_zero_spot", np.zeros(len(data["features"]), dtype=bool))[eval_idx]),
        ]:
            violation.update(masked_error_metrics(prediction[mask], target[mask], source[mask], prefix=name))
        violation_rows.append(violation)
    by_regime_rows.extend(_by_regime_rows(model_name, prediction, eval_idx, data, target_delta))


def _by_regime_rows(model_name: str, prediction: np.ndarray, eval_idx: np.ndarray, data: dict, target_delta: np.ndarray) -> list[dict]:
    regime = data["regime_key"].astype(str)[eval_idx]
    split = data["split"].astype(str)[eval_idx]
    features = data["features"][eval_idx]
    target = target_delta[eval_idx]
    rows = []
    for key in sorted(set(regime)):
        mask = regime == key
        metrics = error_metrics(prediction[mask], target[mask])
        option_values = np.where(features[mask, 6] > 0.5, "call", "put")
        rows.append(
            {
                "model": model_name,
                "regime_key": key,
                "split": split[mask][0],
                "option_type": sorted(set(option_values.tolist()))[0] if len(set(option_values.tolist())) == 1 else "mixed",
                "samples": int(np.count_nonzero(mask)),
                "delta_rmse": metrics["rmse"],
                "delta_mae": metrics["mae"],
                "delta_max_error": metrics["max_error"],
                "max_delta_bound_violation": delta_bound_violation(prediction[mask], features[mask, 6]),
            }
        )
    return rows


def _clip_delta(delta: np.ndarray, is_call: np.ndarray) -> np.ndarray:
    return np.where(is_call > 0.5, np.clip(delta, 0.0, 1.0), np.clip(delta, -1.0, 0.0))


def _split_rmse(prediction: np.ndarray, eval_idx: np.ndarray, split: np.ndarray, target_delta: np.ndarray, split_name: str) -> float:
    mask = split[eval_idx].astype(str) == split_name
    if not np.any(mask):
        return float("nan")
    return error_metrics(prediction[mask], target_delta[eval_idx][mask])["rmse"]


def _throughput(samples: int, seconds: float) -> float:
    return float(samples / seconds) if seconds > 0 else float("inf")


def _write_empty_outputs(out_dir: Path, summary: dict) -> None:
    write_csv(out_dir / "delta_head_metrics.csv", [summary])
    write_csv(out_dir / "delta_head_by_regime.csv", [summary])
    write_csv(out_dir / "delta_head_training_history.csv", [summary])
    write_csv(out_dir / "delta_head_violation_table.csv", [summary])
    write_json(out_dir / "delta_head_summary.json", summary)


def _save_figures(metrics_rows: list[dict], violation_rows: list[dict], figure_root: Path) -> None:
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
            figure_root / "test_delta_mae_by_option.png",
            labels,
            [row["delta_mae"] for row in test_rows],
            r"Strict-mask test $\Delta$ MAE",
            r"$\Delta$ MAE",
            orientation="horizontal",
            display_labels=aliases,
        )
    test_violations = [row for row in violation_rows if row["split"] == "test"]
    if test_violations:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar([row["model"] for row in test_violations], [row["max_delta_bound_violation"] for row in test_violations])
        ax.set_ylabel("test Delta bound violation")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(figure_root / "test_delta_bound_violation.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/delta_head.yaml")
    args = parser.parse_args()
    print(run(args.config))
