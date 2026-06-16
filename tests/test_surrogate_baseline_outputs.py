import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np

_EXPERIMENT_PATH = Path("experiments/06_surrogate_baselines.py")
_SPEC = importlib.util.spec_from_file_location("stage07_baselines", _EXPERIMENT_PATH)
experiment = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = experiment
assert _SPEC.loader is not None
_SPEC.loader.exec_module(experiment)


class SurrogateBaselineOutputTests(unittest.TestCase):
    def test_dry_run_writes_required_audit_outputs(self):
        root = Path("results/test_stage07_baselines")
        root.mkdir(parents=True, exist_ok=True)
        dataset = root / "tiny_dataset.npz"
        features = np.random.default_rng(1).normal(size=(90, 7)).astype(np.float32)
        payoff = np.maximum(1.0 - np.exp(features[:, 0]), 0.0).astype(np.float32)
        price = payoff + 0.05 * np.maximum(features[:, 1], 0.0).astype(np.float32)
        labels = np.column_stack(
            [price, price - payoff, payoff, np.zeros(len(price)), np.zeros(len(price))]
        ).astype(np.float32)
        split = np.array([b"train"] * 50 + [b"validation"] * 20 + [b"test"] * 20)
        regime_key = np.array([f"regime_{idx // 10}".encode("utf-8") for idx in range(90)])
        near_boundary = np.zeros(90, dtype=bool)
        near_boundary[::5] = True
        np.savez_compressed(
            dataset,
            features=features,
            labels=labels,
            split=split,
            regime_key=regime_key,
            near_boundary=near_boundary,
            strict_greek_mask=np.ones(90, dtype=bool),
        )
        output_root = root / "out"
        model_root = root / "models"
        figure_root = root / "figures"

        summary = experiment.run(
            "configs/baseline_surrogates.yaml",
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
                "penalty_weights": [0.0],
                "device": "cpu",
            },
        )

        self.assertGreaterEqual(summary["models"], 2)
        for name in [
            "baseline_metrics.csv",
            "baseline_by_regime.csv",
            "violation_table.csv",
            "training_history.csv",
            "baseline_summary.json",
            "baseline_config_manifest.json",
        ]:
            self.assertTrue((output_root / name).exists(), name)
        self.assertTrue((model_root / "linear_price_baseline.npz").exists())
        self.assertTrue(any(model_root.glob("*.pt")))

        header = (output_root / "violation_table.csv").read_text(encoding="utf-8").splitlines()[0]
        self.assertIn("max_obstacle_violation", header)
        self.assertIn("boundary_rmse", header)
        with open(output_root / "baseline_summary.json", encoding="utf-8") as fh:
            loaded = json.load(fh)
        self.assertIn("best_validation_model", loaded)


if __name__ == "__main__":
    unittest.main()
