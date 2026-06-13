"""Stage 9: positive continuation-premium surrogate."""

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
    premium = y[:, 1]
    payoff = y[:, 2]
    price = y[:, 0]
    weights = train_linear_baseline(x[train], np.log1p(premium[train]), cfg["ridge"])
    pred_premium = np.expm1(predict_linear_baseline(x[test], weights))
    pred_premium = np.maximum(pred_premium, 0.0)
    pred_price = payoff[test] + pred_premium
    out_dir = ensure_dir("results/experiment_07_positive_premium_tuning")
    rows = [
        {
            "model": "positive_premium_linear",
            "regime_price_rmse": rmse(pred_price, price[test]),
            "premium_rmse": rmse(pred_premium, premium[test]),
            "max_obstacle_violation": max_obstacle_violation(pred_price, payoff[test]),
            "decision": "retain_if_better_than_plain",
        }
    ]
    np.savez_compressed(cfg["model_out"], weights=weights)
    write_csv(out_dir / "positive_premium_metrics.csv", rows)
    write_json(out_dir / "positive_premium_summary.json", rows[0])
    return rows[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/positive_premium.yaml")
    args = parser.parse_args()
    print(run(args.config))
