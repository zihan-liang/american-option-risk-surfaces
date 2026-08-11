import csv
import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np


_EXPERIMENT_PATH = Path("experiments/27_m02_obstacle_aware_pinn.py")
_SPEC = importlib.util.spec_from_file_location("m02_obstacle_aware_pinn", _EXPERIMENT_PATH)
experiment = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = experiment
assert _SPEC.loader is not None
_SPEC.loader.exec_module(experiment)


class M02ObstaclePinnOutputTests(unittest.TestCase):
    def test_smoke_run_writes_labeled_outputs_with_all_four_methods(self):
        root = Path("results/test_m02_obstacle_pinn")
        root.mkdir(parents=True, exist_ok=True)
        dataset = root / "tiny_dataset.npz"
        features, labels, split, regime_key = _tiny_dataset()
        np.savez_compressed(
            dataset,
            features=features,
            labels=labels,
            split=split,
            regime_key=regime_key,
            near_boundary=np.zeros(len(features), dtype=bool),
            near_kink=np.zeros(len(features), dtype=bool),
            strict_greek_mask=np.ones(len(features), dtype=bool),
            boundary=np.full(len(features), 0.9, dtype=np.float64),
        )

        output_root = root / "out"
        summary = experiment.run(
            "configs/independent_m02_obstacle_pinn.yaml",
            output_root=output_root,
            overrides={
                "dataset": str(dataset),
                "mode": "smoke",
                "seeds": [101],
                "epochs": 2,
                "patience": 2,
                "batch_size": 8,
                "train_cap": 24,
                "validation_cap": 12,
                "legacy_test_cap": 12,
                "collocation_cap": 8,
                "residual_eval_cap": 8,
                "hidden": 8,
                "depth": 1,
                "device": "cpu",
            },
        )

        self.assertEqual(summary["mode"], "smoke")
        self.assertFalse(summary["evidence_run"])
        self.assertEqual(summary["canonical_run_launched"], False)
        self.assertEqual(
            set(summary["methods"]),
            {"supervised", "masked_pde", "separate_complementarity", "fischer_burmeister"},
        )
        self.assertEqual(summary["split_names"], ["train", "validation", "legacy_test"])
        for name in [
            "m02_config_manifest.json",
            "m02_metrics_by_regime.csv",
            "m02_training_history.csv",
            "m02_promotion.json",
            "m02_summary.json",
        ]:
            path = output_root / name
            self.assertTrue(path.exists(), name)
            if path.suffix == ".csv":
                self.assertNotIn(b"\r\n", path.read_bytes(), name)
        with open(output_root / "m02_metrics_by_regime.csv", newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual({row["method"] for row in rows}, set(summary["methods"]))
        self.assertIn("legacy_test", {row["split"] for row in rows})
        self.assertIn("masked_pde_residual_mean_abs", rows[0])
        self.assertIn("complementarity_product_mean_abs", rows[0])
        with open(output_root / "m02_summary.json", encoding="utf-8") as fh:
            loaded = json.load(fh)
        self.assertEqual(loaded["mode"], "smoke")


def _tiny_dataset():
    rows_per_regime = 4
    regimes = [
        ("train_a", "train"),
        ("train_b", "train"),
        ("train_c", "train"),
        ("train_d", "train"),
        ("train_e", "train"),
        ("train_f", "train"),
        ("validation_a", "validation"),
        ("validation_b", "validation"),
        ("validation_c", "validation"),
        ("legacy_a", "test"),
        ("legacy_b", "test"),
        ("legacy_c", "test"),
    ]
    n = len(regimes) * rows_per_regime
    features = np.zeros((n, 7), dtype=np.float64)
    split = []
    regime_key = []
    for regime_index, (name, split_name) in enumerate(regimes):
        start = regime_index * rows_per_regime
        end = start + rows_per_regime
        features[start:end, 0] = np.linspace(-0.2, 0.2, rows_per_regime)
        features[start:end, 1] = np.linspace(0.2, 0.8, rows_per_regime)
        features[start:end, 2] = 0.05
        features[start:end, 3] = 0.01
        features[start:end, 4] = 0.25
        features[start:end, 5] = 1.0
        split.extend([split_name.encode("utf-8")] * rows_per_regime)
        regime_key.extend([name.encode("utf-8")] * rows_per_regime)
    payoff = np.maximum(1.0 - np.exp(features[:, 0]), 0.0)
    premium = 0.03 + 0.02 * features[:, 1]
    price = payoff + premium
    labels = np.column_stack([price, premium, payoff, np.zeros(n), np.zeros(n)])
    return features, labels, np.asarray(split), np.asarray(regime_key)


if __name__ == "__main__":
    unittest.main()
