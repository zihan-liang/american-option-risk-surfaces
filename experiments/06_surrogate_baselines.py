"""Stage 8: ordinary surrogate baselines and obstacle audit."""

from __future__ import annotations

import argparse

import numpy as np

from common import ensure_dir, load_config, write_csv, write_json
from models.metrics import max_obstacle_violation
from models.training import predict_linear_baseline, rmse, train_linear_baseline


def run(config_path: str) -> dict:
    cfg = load_config(config_path)
    data = np.load(cfg["dataset"])
    x, y, split = data["features"], data["labels"], data["split"]
    train = split == "train"
    test = split == "test"
    price = y[:, 0]
    payoff = y[:, 2]
    weights = train_linear_baseline(x[train], price[train], cfg["ridge"])
    pred = predict_linear_baseline(x[test], weights)
    out_dir = ensure_dir("results/experiment_06_surrogate_baselines")
    rows = [
        {
            "model": "linear_price_baseline",
            "regime_price_rmse": rmse(pred, price[test]),
            "max_obstacle_violation": max_obstacle_violation(pred, payoff[test]),
            "decision": "audit_required",
        }
    ]
    np.savez_compressed("results/models/linear_price_baseline.npz", weights=weights)
    write_csv(out_dir / "baseline_metrics.csv", rows)
    summary = rows[0]
    write_json(out_dir / "baseline_summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline_surrogates.yaml")
    args = parser.parse_args()
    print(run(args.config))
