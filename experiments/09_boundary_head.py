"""Stage 10: direct boundary diagnostic head."""

from __future__ import annotations

import argparse

import numpy as np

from common import ensure_dir, load_config, write_csv, write_json
from models.training import predict_linear_baseline, rmse, train_linear_baseline


def run(config_path: str) -> dict:
    cfg = load_config(config_path)
    data = np.load(cfg["dataset"])
    x, boundary, split = data["features"], data["boundary"], data["split"]
    train = split == "train"
    test = split == "test"
    valid_train = np.isfinite(boundary[train])
    valid_test = np.isfinite(boundary[test])
    weights = train_linear_baseline(x[train][valid_train], boundary[train][valid_train], cfg["ridge"])
    pred = predict_linear_baseline(x[test][valid_test], weights)
    out_dir = ensure_dir("results/experiment_09_boundary_head")
    row = {"model": "boundary_head_linear", "boundary_mae": float(np.mean(np.abs(pred - boundary[test][valid_test]))), "boundary_rmse": rmse(pred, boundary[test][valid_test])}
    np.savez_compressed(cfg["model_out"], weights=weights)
    write_csv(out_dir / "boundary_metrics.csv", [row])
    write_json(out_dir / "boundary_summary.json", row)
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/boundary_head.yaml")
    args = parser.parse_args()
    print(run(args.config))
