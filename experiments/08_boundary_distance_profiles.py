"""Stage 8 diagnostic: positive-premium price error by delicate-region masks."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import ensure_dir, load_config, write_csv, write_json
from models.metrics import error_metrics
from models.premium_surrogate import PositivePremiumModel, reconstruct_price
from models.training import choose_device, load_dataset_arrays, predict_torch_model, split_mask


def run(
    config_path: str,
    *,
    output_root: str | Path = "results/experiment_08_boundary_distance_profiles",
    overrides: dict | None = None,
) -> dict:
    cfg = load_config(config_path)
    if overrides:
        cfg = {**cfg, **overrides}
    out_dir = ensure_dir(output_root)
    data = load_dataset_arrays(cfg["dataset"])
    features = data["features"].astype(np.float32)
    labels = data["labels"].astype(np.float32)
    price = labels[:, 0]
    payoff = labels[:, 2]
    split = data["split"]
    test = split_mask(split, "test")
    device = choose_device(cfg.get("device", "auto"))
    model = _load_positive_model(cfg["model"], device)
    pred_premium = predict_torch_model(model, features[test], batch_size=int(cfg.get("batch_size", 8192)), device=device)
    pred_price = reconstruct_price(payoff[test], pred_premium)

    near_boundary = data.get("near_boundary", np.zeros(len(features), dtype=bool))[test]
    near_kink = data.get("near_kink", np.zeros(len(features), dtype=bool))[test]
    strict = data.get("strict_greek_mask", np.ones(len(features), dtype=bool))[test]
    away_boundary = ~near_boundary
    rows = []
    for band, mask in [
        ("near_boundary", near_boundary),
        ("away_boundary", away_boundary),
        ("near_kink", near_kink),
        ("strict_mask", strict),
    ]:
        if not np.any(mask):
            rows.append({"band": band, "samples": 0, "price_rmse": float("nan"), "price_mae": float("nan"), "price_max_error": float("nan")})
            continue
        metrics = error_metrics(pred_price[mask], price[test][mask])
        rows.append(
            {
                "band": band,
                "samples": int(np.count_nonzero(mask)),
                "price_rmse": metrics["rmse"],
                "price_mae": metrics["mae"],
                "price_max_error": metrics["max_error"],
            }
        )
    write_csv(out_dir / "boundary_distance_metrics.csv", rows)
    summary = {
        "bands": len(rows),
        "infers_boundary": False,
        "note": "This diagnostic uses Stage 6 masks/boundaries only; it does not infer a boundary from the price surrogate.",
    }
    write_json(out_dir / "boundary_distance_summary.json", summary)
    return summary


def _load_positive_model(path: str | Path, device: str):
    import torch

    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    metadata = checkpoint.get("metadata", {})
    model = PositivePremiumModel.build(
        int(metadata.get("input_dim", 7)),
        hidden=int(metadata.get("hidden", 128)),
        depth=int(metadata.get("depth", 3)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/boundary_distance_profiles.yaml")
    args = parser.parse_args()
    print(run(args.config))
