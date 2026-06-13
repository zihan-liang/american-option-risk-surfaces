"""Stage 11: supervised bounded Delta diagnostic head."""

from __future__ import annotations

import argparse

import numpy as np

from common import ensure_dir, load_config, write_csv, write_json
from models.metrics import delta_bound_violation
from models.training import predict_linear_baseline, rmse, train_linear_baseline


def _clip_delta(delta: np.ndarray, is_call: np.ndarray) -> np.ndarray:
    return np.where(is_call > 0.5, np.clip(delta, 0.0, 1.0), np.clip(delta, -1.0, 0.0))


def run(config_path: str) -> dict:
    cfg = load_config(config_path)
    data = np.load(cfg["dataset"])
    x, y, split, mask = data["features"], data["labels"], data["split"], data["greek_mask"]
    train = (split == "train") & mask
    test = (split == "test") & mask
    target = y[:, 3]
    weights = train_linear_baseline(x[train], target[train], cfg["ridge"])
    pred = _clip_delta(predict_linear_baseline(x[test], weights), x[test, 6])
    row = {"model": "bounded_delta_linear", "strict_delta_rmse": rmse(pred, target[test]), "strict_delta_mae": float(np.mean(np.abs(pred - target[test]))), "max_delta_bound_violation": delta_bound_violation(pred, x[test, 6])}
    np.savez_compressed(cfg["model_out"], weights=weights)
    out_dir = ensure_dir("results/experiment_12_supervised_delta_head")
    write_csv(out_dir / "delta_head_metrics.csv", [row])
    write_json(out_dir / "delta_head_summary.json", row)
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/delta_head.yaml")
    args = parser.parse_args()
    print(run(args.config))
