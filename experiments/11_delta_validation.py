"""Stage 11: audit whether the price model behaves as a Greek engine."""

from __future__ import annotations

import argparse

import numpy as np

from common import ensure_dir, load_config, write_csv, write_json
from models.training import predict_linear_baseline, rmse


def run(config_path: str) -> dict:
    cfg = load_config(config_path)
    data = np.load(cfg["dataset"])
    model = np.load(cfg["price_model"])
    x, y, split, mask = data["features"], data["labels"], data["split"], data["greek_mask"]
    test = (split == "test") & mask & (np.exp(x[:, 0]) > 0.05)
    eps = 1e-4
    x_up = x[test].copy()
    x_dn = x[test].copy()
    x_up[:, 0] += eps
    x_dn[:, 0] -= eps
    p_up = np.maximum(np.expm1(predict_linear_baseline(x_up, model["weights"])), 0.0)
    p_dn = np.maximum(np.expm1(predict_linear_baseline(x_dn, model["weights"])), 0.0)
    spot = np.exp(x[test, 0])
    autograd_like_delta = (p_up - p_dn) / (2 * eps * spot)
    target_delta = y[test, 3]
    row = {"method": "price_model_finite_difference_delta", "strict_delta_rmse": rmse(autograd_like_delta, target_delta), "strict_delta_mae": float(np.mean(np.abs(autograd_like_delta - target_delta))), "decision": "do_not_claim_price_model_as_greek_engine"}
    out_dir = ensure_dir("results/experiment_11_delta_validation")
    write_csv(out_dir / "delta_validation_metrics.csv", [row])
    write_json(out_dir / "delta_validation_summary.json", row)
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/delta_validation.yaml")
    args = parser.parse_args()
    print(run(args.config))
