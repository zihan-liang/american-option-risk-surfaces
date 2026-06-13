"""Stage 10 repair: option-specific dividend-call boundary head."""

from __future__ import annotations

import argparse

import numpy as np

from common import ensure_dir, load_config, write_csv, write_json
from models.training import predict_linear_baseline, rmse, train_linear_baseline


def run(config_path: str) -> dict:
    cfg = load_config(config_path)
    data = np.load(cfg["dataset"])
    x, boundary, split = data["features"], data["boundary"], data["split"]
    is_call = x[:, 6] > 0.5
    train = (split == "train") & is_call & np.isfinite(boundary)
    test = (split == "test") & is_call & np.isfinite(boundary)
    out_dir = ensure_dir("results/experiment_10_option_specific_call_boundary")
    if not np.any(train) or not np.any(test):
        row = {"model": "call_boundary_head_linear", "boundary_mae": float("nan"), "boundary_rmse": float("nan"), "decision": "insufficient_call_data"}
    else:
        weights = train_linear_baseline(x[train], boundary[train], cfg["ridge"])
        pred = predict_linear_baseline(x[test], weights)
        np.savez_compressed(cfg["model_out"], weights=weights)
        row = {"model": "call_boundary_head_linear", "boundary_mae": float(np.mean(np.abs(pred - boundary[test]))), "boundary_rmse": rmse(pred, boundary[test]), "decision": "scoped_call_diagnostic"}
    write_csv(out_dir / "call_boundary_metrics.csv", [row])
    write_json(out_dir / "call_boundary_summary.json", row)
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/call_boundary_refinement.yaml")
    args = parser.parse_args()
    print(run(args.config))
