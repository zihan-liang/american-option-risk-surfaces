"""Stage 10: validate whether the price surrogate behaves as a Delta engine."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import ensure_dir, load_config, write_csv, write_json
from models.metrics import error_metrics, grouped_delta_metrics, masked_error_metrics
from models.premium_surrogate import PositivePremiumModel
from models.training import capped_indices, choose_device, load_dataset_arrays, split_mask


def run(
    config_path: str,
    *,
    output_root: str | Path = "results/experiment_11_delta_validation",
    overrides: dict | None = None,
) -> dict:
    cfg = load_config(config_path)
    if overrides:
        cfg = {**cfg, **overrides}
    out_dir = ensure_dir(output_root)
    data = load_dataset_arrays(cfg["dataset"])
    if "strict_greek_mask" not in data:
        raise KeyError("dataset missing strict_greek_mask")
    features = data["features"].astype(np.float32)
    labels = data["labels"].astype(np.float32)
    target_delta = labels[:, 3]
    strict_mask = data["strict_greek_mask"].astype(bool)
    split = data["split"]
    is_call = features[:, 6]
    seed = int(cfg.get("seed", 7))
    test_idx = capped_indices(split_mask(split, "test") & strict_mask, int(cfg.get("test_cap", 150000)), seed + 2)
    if len(test_idx) == 0:
        summary = {
            "decision": "insufficient_strict_delta_data",
            "uses_strict_greek_mask": True,
            "test_samples": 0,
        }
        write_csv(out_dir / "delta_validation_metrics.csv", [summary])
        write_csv(out_dir / "delta_validation_by_regime.csv", [summary])
        write_json(out_dir / "delta_validation_summary.json", summary)
        return summary

    device = choose_device(cfg.get("device", "auto"))
    model = _load_positive_model(cfg["price_model"], device)
    batch_size = int(cfg.get("batch_size", 8192))
    eps = float(cfg.get("finite_difference_eps", 1e-4))
    eval_features = features[test_idx]
    autograd_delta = _autograd_delta(model, eval_features, batch_size=batch_size, device=device)
    fd_delta = _finite_difference_delta(model, eval_features, eps=eps, batch_size=batch_size, device=device)
    target = target_delta[test_idx]
    eval_split = split[test_idx]
    eval_is_call = is_call[test_idx]

    metrics_rows = []
    metrics_rows.extend(
        grouped_delta_metrics(
            prediction=autograd_delta,
            target=target,
            splits=eval_split,
            is_call=eval_is_call,
            model_name="price_model_autograd_delta",
        )
    )
    metrics_rows.extend(
        grouped_delta_metrics(
            prediction=fd_delta,
            target=target,
            splits=eval_split,
            is_call=eval_is_call,
            model_name="price_model_finite_difference_delta",
        )
    )
    near_boundary = data.get("near_boundary", np.zeros(len(features), dtype=bool))[test_idx]
    near_kink = data.get("near_kink", np.zeros(len(features), dtype=bool))[test_idx]
    near_zero = data.get("near_zero_spot", np.zeros(len(features), dtype=bool))[test_idx]
    for method, prediction in [("price_model_autograd_delta", autograd_delta), ("price_model_finite_difference_delta", fd_delta)]:
        row = {"model": method, "split": "test", "option_type": "mask_profiles"}
        row.update(masked_error_metrics(prediction, target, near_boundary, prefix="near_boundary_delta"))
        row.update(masked_error_metrics(prediction, target, near_kink, prefix="near_kink_delta"))
        row.update(masked_error_metrics(prediction, target, near_zero, prefix="near_zero_spot_delta"))
        metrics_rows.append(row)
    by_regime_rows = _by_regime_rows("price_model_autograd_delta", autograd_delta, test_idx, data, target_delta)
    by_regime_rows.extend(_by_regime_rows("price_model_finite_difference_delta", fd_delta, test_idx, data, target_delta))
    write_csv(out_dir / "delta_validation_metrics.csv", metrics_rows)
    write_csv(out_dir / "delta_validation_by_regime.csv", by_regime_rows)
    autograd_metrics = error_metrics(autograd_delta, target)
    fd_metrics = error_metrics(fd_delta, target)
    summary = {
        "decision": "do_not_claim_price_model_as_greek_engine",
        "uses_strict_greek_mask": True,
        "test_samples": int(len(test_idx)),
        "autograd_delta_rmse": autograd_metrics["rmse"],
        "autograd_delta_mae": autograd_metrics["mae"],
        "finite_difference_delta_rmse": fd_metrics["rmse"],
        "finite_difference_delta_mae": fd_metrics["mae"],
        "price_model": str(cfg["price_model"]),
        "device": device,
        "note": "Price-model derivative diagnostics are separate evidence and are not used as the final Delta claim.",
    }
    write_json(out_dir / "delta_validation_summary.json", summary)
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
    return model.to(device)


def _autograd_delta(model, features: np.ndarray, *, batch_size: int, device: str) -> np.ndarray:
    import torch

    model.eval()
    out = []
    for start in range(0, len(features), batch_size):
        xb = torch.as_tensor(features[start : start + batch_size], dtype=torch.float32, device=device)
        xb.requires_grad_(True)
        price = _normalized_price(model, xb)
        grad = torch.autograd.grad(price.sum(), xb, create_graph=False)[0][:, 0]
        spot = torch.exp(xb[:, 0])
        out.append((grad / spot).detach().cpu().numpy())
    return np.concatenate(out)


def _finite_difference_delta(model, features: np.ndarray, *, eps: float, batch_size: int, device: str) -> np.ndarray:
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            base = torch.as_tensor(features[start : start + batch_size], dtype=torch.float32, device=device)
            up = base.clone()
            down = base.clone()
            up[:, 0] += eps
            down[:, 0] -= eps
            price_up = _normalized_price(model, up)
            price_down = _normalized_price(model, down)
            spot = torch.exp(base[:, 0])
            out.append(((price_up - price_down) / (2.0 * eps) / spot).detach().cpu().numpy())
    return np.concatenate(out)


def _normalized_price(model, features):
    import torch

    spot = torch.exp(features[:, 0])
    is_call = features[:, 6] > 0.5
    payoff = torch.where(is_call, torch.relu(spot - 1.0), torch.relu(1.0 - spot))
    return payoff + model(features).reshape(-1)


def _by_regime_rows(model_name: str, prediction: np.ndarray, eval_idx: np.ndarray, data: dict, target_delta: np.ndarray) -> list[dict]:
    regime = data["regime_key"].astype(str)[eval_idx]
    split = data["split"].astype(str)[eval_idx]
    is_call = data["features"][eval_idx, 6]
    target = target_delta[eval_idx]
    rows = []
    for key in sorted(set(regime)):
        mask = regime == key
        metrics = error_metrics(prediction[mask], target[mask])
        option_values = np.where(is_call[mask] > 0.5, "call", "put")
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
            }
        )
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/delta_validation.yaml")
    args = parser.parse_args()
    print(run(args.config))
