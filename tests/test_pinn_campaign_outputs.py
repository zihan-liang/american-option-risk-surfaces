import csv
import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np

_EXPERIMENT_PATH = Path("experiments/22_pinn_campaign.py")
_SPEC = importlib.util.spec_from_file_location("stage22_pinn_campaign", _EXPERIMENT_PATH)
experiment = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = experiment
assert _SPEC.loader is not None
_SPEC.loader.exec_module(experiment)


class PinnCampaignOutputTests(unittest.TestCase):
    def test_method_spec_supports_strong_terminal_pde_sweep(self):
        spec = experiment._method_spec("terminal_pde_weight_10_residual_sampling", {})

        self.assertEqual(spec["method"], "terminal_pde_weight_10_residual_sampling")
        self.assertTrue(spec["terminal"])
        self.assertTrue(spec["ramp"])
        self.assertTrue(spec["residual_sampling"])
        self.assertEqual(spec["lambda_pde"], 10.0)

    def test_dry_run_writes_required_campaign_outputs(self):
        root = Path("results/test_stage22_pinn_campaign")
        root.mkdir(parents=True, exist_ok=True)
        dataset = root / "tiny_dataset.npz"
        n = 96
        features = np.zeros((n, 7), dtype=np.float32)
        features[:, 0] = np.linspace(-0.35, 0.35, n, dtype=np.float32)
        features[:, 1] = np.tile(np.linspace(0.05, 0.95, 12, dtype=np.float32), 8)
        features[:, 2] = 0.05
        features[:, 3] = 0.01
        features[:, 4] = 0.25
        features[:, 5] = 1.0
        features[:, 6] = 0.0
        payoff = np.maximum(1.0 - np.exp(features[:, 0]), 0.0).astype(np.float32)
        premium = (0.03 + 0.02 * features[:, 1]).astype(np.float32)
        price = payoff + premium
        labels = np.column_stack([price, premium, payoff, np.zeros(n), np.zeros(n)]).astype(np.float32)
        split = np.array([b"train"] * 60 + [b"validation"] * 18 + [b"test"] * 18)
        regime_key = np.array([f"regime_{idx // 12}".encode("utf-8") for idx in range(n)])
        near_boundary = np.zeros(n, dtype=bool)
        near_boundary[::11] = True
        near_kink = np.zeros(n, dtype=bool)
        near_kink[::13] = True
        strict_greek_mask = np.ones(n, dtype=bool)
        boundary = np.full(n, 0.9, dtype=np.float32)
        np.savez_compressed(
            dataset,
            features=features,
            labels=labels,
            split=split,
            regime_key=regime_key,
            near_boundary=near_boundary,
            near_kink=near_kink,
            strict_greek_mask=strict_greek_mask,
            boundary=boundary,
        )

        output_root = root / "out"
        summary = experiment.run(
            "configs/pinn_campaign.yaml",
            output_root=output_root,
            overrides={
                "dataset": str(dataset),
                "seeds": [21],
                "epochs": 2,
                "train_cap": 48,
                "validation_cap": 18,
                "test_cap": 18,
                "methods": ["supervised_baseline", "pde_weight_1e-6", "terminal_premium"],
                "batch_size": 16,
                "pde_batch_cap": 8,
                "residual_eval_cap": 12,
                "hidden": 8,
                "depth": 1,
                "device": "cpu",
            },
        )

        self.assertEqual(summary["decision"], "stage22_pinn_campaign")
        self.assertIn("best_method", summary)
        for name in [
            "pinn_campaign_config_manifest.json",
            "pinn_campaign_metrics.csv",
            "pinn_campaign_residual_metrics.csv",
            "pinn_campaign_delta_metrics.csv",
            "pinn_campaign_summary.json",
            "pinn_campaign_training_history.csv",
        ]:
            path = output_root / name
            self.assertTrue(path.exists(), name)
            if path.suffix == ".csv":
                self.assertNotIn(b"\r\n", path.read_bytes(), name)
        with open(output_root / "pinn_campaign_metrics.csv", newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        self.assertIn("supervised_baseline", {row["method"] for row in rows})
        with open(output_root / "pinn_campaign_summary.json", encoding="utf-8") as fh:
            loaded = json.load(fh)
        self.assertEqual(loaded["decision"], summary["decision"])


if __name__ == "__main__":
    unittest.main()
