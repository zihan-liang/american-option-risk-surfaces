"""Stage 9 diagnostic: price error by distance to extracted boundary."""

from __future__ import annotations

import argparse

import numpy as np

from common import ensure_dir, load_config, write_csv, write_json
from models.training import predict_linear_baseline, rmse


def run(config_path: str) -> dict:
    cfg = load_config(config_path)
    data = np.load(cfg["dataset"])
    model = np.load(cfg["model"])
    x, y, split, boundary = data["features"], data["labels"], data["split"], data["boundary"]
    test = split == "test"
    spot = np.exp(x[:, 0])
    distance = np.abs(spot - boundary)
    pred_premium = np.maximum(np.expm1(predict_linear_baseline(x[test], model["weights"])), 0.0)
    pred_price = y[test, 2] + pred_premium
    rows = []
    for label, mask in [
        ("near_boundary", distance[test] <= 0.05),
        ("away_boundary", distance[test] > 0.05),
    ]:
        if np.any(mask):
            rows.append({"band": label, "price_rmse": rmse(pred_price[mask], y[test, 0][mask]), "samples": int(mask.sum())})
    out_dir = ensure_dir("results/experiment_08_boundary_distance_profiles")
    write_csv(out_dir / "boundary_distance_metrics.csv", rows)
    summary = {"bands": len(rows)}
    write_json(out_dir / "boundary_distance_summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/boundary_distance_profiles.yaml")
    args = parser.parse_args()
    print(run(args.config))
