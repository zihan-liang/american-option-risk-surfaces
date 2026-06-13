"""Stage 7: construct a regime-split surrogate dataset from PSOR labels."""

from __future__ import annotations

import argparse
import itertools

import numpy as np

from common import ensure_dir, load_config, write_csv, write_json
from data.dataset_builder import build_surface_samples
from data.splits import regime_key, regime_split
from experiment_utils import solver_config_from_dict


def run(config_path: str) -> dict:
    raw = load_config(config_path)
    out_dir = ensure_dir("results/experiment_05_surrogate_dataset")
    ensure_dir("data/processed")
    regimes = [
        {"option_type": option_type, "t": t, "sigma": sigma, "r": r, "q": q}
        for option_type, t, sigma, r, q in itertools.product(
            raw["option_types"], raw["t_values"], raw["sigma_values"], raw["r_values"], raw["q_values"]
        )
    ]
    train_regimes, test_regimes = regime_split(regimes, raw["test_fraction"], raw["seed"])
    arrays = []
    split_rows = []
    for split_name, split_regimes in [("train", train_regimes), ("test", test_regimes)]:
        for regime in split_regimes:
            cfg = solver_config_from_dict(
                {
                    **regime,
                    "k": raw["k"],
                    "s_max": raw["s_max"],
                    "m": raw["m"],
                    "n": raw["n"],
                    "omega": raw["omega"],
                    "tol": raw["tol"],
                    "max_iter": raw["max_iter"],
                    "american": True,
                }
            )
            sample = build_surface_samples(cfg)
            sample["split"] = np.full(len(sample["features"]), split_name)
            sample["regime"] = np.full(len(sample["features"]), regime_key(regime))
            arrays.append(sample)
            split_rows.append({"split": split_name, **regime, "key": regime_key(regime)})

    dataset = {
        "features": np.vstack([a["features"] for a in arrays]),
        "labels": np.vstack([a["labels"] for a in arrays]),
        "boundary": np.concatenate([a["boundary"] for a in arrays]),
        "greek_mask": np.concatenate([a["greek_mask"] for a in arrays]),
        "split": np.concatenate([a["split"] for a in arrays]),
        "regime": np.concatenate([a["regime"] for a in arrays]),
    }
    np.savez_compressed("data/processed/surrogate_dataset.npz", **dataset)
    write_csv(out_dir / "split_table.csv", split_rows)
    summary = {
        "samples": int(len(dataset["features"])),
        "features": int(dataset["features"].shape[1]),
        "train_regimes": len(train_regimes),
        "test_regimes": len(test_regimes),
    }
    write_json(out_dir / "dataset_manifest.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/surrogate_dataset.yaml")
    args = parser.parse_args()
    print(run(args.config))
