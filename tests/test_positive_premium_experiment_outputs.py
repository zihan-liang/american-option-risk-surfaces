import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np

_EXPERIMENT_PATH = Path("experiments/07_positive_premium_tuning.py")
_SPEC = importlib.util.spec_from_file_location("stage08_positive_premium", _EXPERIMENT_PATH)
experiment = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = experiment
assert _SPEC.loader is not None
_SPEC.loader.exec_module(experiment)


class PositivePremiumExperimentOutputTests(unittest.TestCase):
    def test_dry_run_writes_required_audit_outputs(self):
        root = Path("results/test_stage08_positive_premium")
        root.mkdir(parents=True, exist_ok=True)
        dataset = root / "tiny_dataset.npz"
        rng = np.random.default_rng(3)
        features = rng.normal(size=(96, 7)).astype(np.float32)
        features[:, 6] = np.array([0.0, 1.0] * 48, dtype=np.float32)
        payoff = np.maximum(1.0 - np.exp(features[:, 0]), 0.0).astype(np.float32)
        premium = (0.02 + 0.03 * np.maximum(features[:, 1], 0.0)).astype(np.float32)
        price = payoff + premium
        labels = np.column_stack([price, premium, payoff, np.zeros(96), np.zeros(96)]).astype(np.float32)
        split = np.array([b"train"] * 56 + [b"validation"] * 20 + [b"test"] * 20)
        regime_key = np.array([f"regime_{idx // 8}".encode("utf-8") for idx in range(96)])
        near_boundary = np.zeros(96, dtype=bool)
        near_boundary[::4] = True
        strict_greek_mask = np.ones(96, dtype=bool)
        near_kink = np.zeros(96, dtype=bool)
        near_kink[1::7] = True
        np.savez_compressed(
            dataset,
            features=features,
            labels=labels,
            split=split,
            regime_key=regime_key,
            near_boundary=near_boundary,
            strict_greek_mask=strict_greek_mask,
            near_kink=near_kink,
        )

        output_root = root / "out"
        model_root = root / "models"
        figure_root = root / "figures"
        summary = experiment.run(
            "configs/positive_premium.yaml",
            output_root=output_root,
            model_root=model_root,
            figure_root=figure_root,
            overrides={
                "dataset": str(dataset),
                "train_cap": 40,
                "validation_cap": 20,
                "test_cap": 20,
                "epochs": 2,
                "batch_size": 16,
                "hidden": 16,
                "depth": 1,
                "device": "cpu",
                "model_out": str(model_root / "selected_final_price_premium.pt"),
            },
        )

        self.assertIn("best_validation_model", summary)
        for name in [
            "positive_premium_metrics.csv",
            "positive_premium_by_regime.csv",
            "positive_premium_violation_table.csv",
            "positive_premium_training_history.csv",
            "positive_premium_summary.json",
            "positive_premium_config_manifest.json",
        ]:
            self.assertTrue((output_root / name).exists(), name)
        self.assertTrue((model_root / "selected_final_price_premium.pt").exists())
        header = (output_root / "positive_premium_violation_table.csv").read_text(encoding="utf-8").splitlines()[0]
        self.assertIn("max_obstacle_violation", header)
        with open(output_root / "positive_premium_summary.json", encoding="utf-8") as fh:
            loaded = json.load(fh)
        self.assertEqual(loaded["best_validation_model"], summary["best_validation_model"])


if __name__ == "__main__":
    unittest.main()
