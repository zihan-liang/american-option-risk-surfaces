"""Stage 12: integrated modular workflow evaluation on held-out samples."""

from __future__ import annotations

import argparse
import time

import numpy as np

from common import ensure_dir, load_config, write_csv, write_json
from models.metrics import delta_bound_violation, max_obstacle_violation
from models.training import predict_linear_baseline, rmse


def run(config_path: str) -> dict:
    cfg = load_config(config_path)
    data = np.load(cfg["dataset"])
    price_model = np.load(cfg["price_model"])
    boundary_model = np.load(cfg["boundary_model"])
    delta_model = np.load(cfg["delta_model"])
    x, y, split, boundary, mask = data["features"], data["labels"], data["split"], data["boundary"], data["greek_mask"]
    test = split == "test"
    start = time.perf_counter()
    premium = np.maximum(np.expm1(predict_linear_baseline(x[test], price_model["weights"])), 0.0)
    price = y[test, 2] + premium
    boundary_pred = predict_linear_baseline(x[test], boundary_model["weights"])
    delta = predict_linear_baseline(x[test], delta_model["weights"])
    delta = np.where(x[test, 6] > 0.5, np.clip(delta, 0.0, 1.0), np.clip(delta, -1.0, 0.0))
    elapsed = time.perf_counter() - start
    valid_boundary = np.isfinite(boundary[test])
    strict = mask[test]
    row = {
        "price_rmse": rmse(price, y[test, 0]),
        "boundary_mae": float(np.mean(np.abs(boundary_pred[valid_boundary] - boundary[test][valid_boundary]))),
        "strict_delta_rmse": rmse(delta[strict], y[test, 3][strict]),
        "max_price_obstacle_violation": max_obstacle_violation(price, y[test, 2]),
        "max_delta_bound_violation": delta_bound_violation(delta, x[test, 6]),
        "surrogate_eval_seconds": elapsed,
        "decision": "scenario_screening_with_psor_audit",
    }
    out_dir = ensure_dir("results/experiment_13_integrated_application_workflow")
    write_csv(out_dir / "integrated_case_metrics.csv", [row])
    write_json(out_dir / "integrated_summary.json", row)
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/integrated_workflow.yaml")
    args = parser.parse_args()
    print(run(args.config))
