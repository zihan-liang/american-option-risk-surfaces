import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np

_POSITIVE_PATH = Path("experiments/07_positive_premium_tuning.py")
_POSITIVE_SPEC = importlib.util.spec_from_file_location("stage08_positive_for_boundary", _POSITIVE_PATH)
positive_experiment = importlib.util.module_from_spec(_POSITIVE_SPEC)
sys.modules[_POSITIVE_SPEC.name] = positive_experiment
assert _POSITIVE_SPEC.loader is not None
_POSITIVE_SPEC.loader.exec_module(positive_experiment)

_BOUNDARY_PATH = Path("experiments/08_boundary_distance_profiles.py")
_BOUNDARY_SPEC = importlib.util.spec_from_file_location("stage08_boundary_distance", _BOUNDARY_PATH)
boundary_experiment = importlib.util.module_from_spec(_BOUNDARY_SPEC)
sys.modules[_BOUNDARY_SPEC.name] = boundary_experiment
assert _BOUNDARY_SPEC.loader is not None
_BOUNDARY_SPEC.loader.exec_module(boundary_experiment)


class BoundaryDistanceProfileOutputTests(unittest.TestCase):
    def test_dry_run_writes_boundary_distance_bands_without_boundary_claim(self):
        root = Path("results/test_stage08_boundary_profiles")
        root.mkdir(parents=True, exist_ok=True)
        dataset = root / "tiny_dataset.npz"
        rng = np.random.default_rng(4)
        features = rng.normal(size=(96, 7)).astype(np.float32)
        features[:, 6] = np.array([0.0, 1.0] * 48, dtype=np.float32)
        spot = np.exp(features[:, 0])
        payoff = np.maximum(1.0 - spot, 0.0).astype(np.float32)
        premium = (0.03 + 0.02 * np.maximum(features[:, 1], 0.0)).astype(np.float32)
        price = payoff + premium
        labels = np.column_stack([price, premium, payoff, np.zeros(96), np.zeros(96)]).astype(np.float32)
        split = np.array([b"train"] * 56 + [b"validation"] * 20 + [b"test"] * 20)
        regime_key = np.array([f"regime_{idx // 8}".encode("utf-8") for idx in range(96)])
        boundary = spot + np.where(np.arange(96) % 2 == 0, 0.02, 0.2)
        near_boundary = np.abs(spot - boundary) <= 0.05
        near_kink = np.zeros(96, dtype=bool)
        near_kink[::6] = True
        strict_greek_mask = ~(near_boundary | near_kink)
        np.savez_compressed(
            dataset,
            features=features,
            labels=labels,
            split=split,
            regime_key=regime_key,
            boundary=boundary,
            near_boundary=near_boundary,
            near_kink=near_kink,
            strict_greek_mask=strict_greek_mask,
        )
        model_root = root / "models"
        positive_experiment.run(
            "configs/positive_premium.yaml",
            output_root=root / "positive_out",
            model_root=model_root,
            figure_root=root / "figures",
            overrides={
                "dataset": str(dataset),
                "train_cap": 40,
                "validation_cap": 20,
                "test_cap": 20,
                "epochs": 1,
                "batch_size": 16,
                "hidden": 12,
                "depth": 1,
                "device": "cpu",
                "model_out": str(model_root / "selected_final_price_premium.pt"),
            },
        )

        out_dir = root / "boundary_out"
        summary = boundary_experiment.run(
            "configs/boundary_distance_profiles.yaml",
            output_root=out_dir,
            overrides={
                "dataset": str(dataset),
                "model": str(model_root / "selected_final_price_premium.pt"),
                "batch_size": 16,
                "device": "cpu",
            },
        )

        self.assertGreaterEqual(summary["bands"], 3)
        self.assertFalse(summary["infers_boundary"])
        self.assertTrue((out_dir / "boundary_distance_metrics.csv").exists())
        self.assertTrue((out_dir / "boundary_distance_summary.json").exists())
        with open(out_dir / "boundary_distance_summary.json", encoding="utf-8") as fh:
            loaded = json.load(fh)
        self.assertFalse(loaded["infers_boundary"])


if __name__ == "__main__":
    unittest.main()
