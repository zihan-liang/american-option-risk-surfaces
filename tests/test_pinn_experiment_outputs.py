import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

_EXPERIMENT_PATH = Path("experiments/21_pinn_inspired_premium.py")
_SPEC = importlib.util.spec_from_file_location("stage21_pinn_inspired_premium", _EXPERIMENT_PATH)
experiment = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = experiment
assert _SPEC.loader is not None
_SPEC.loader.exec_module(experiment)


class PinnInspiredExperimentOutputTests(unittest.TestCase):
    def test_matched_model_builder_uses_identical_independent_initial_weights(self):
        baseline, pinn = experiment._build_matched_premium_models(input_dim=7, hidden=8, depth=1, seed=21)

        baseline_state = baseline.state_dict()
        pinn_state = pinn.state_dict()
        self.assertEqual(set(baseline_state), set(pinn_state))
        for key in baseline_state:
            self.assertTrue(torch.equal(baseline_state[key], pinn_state[key]), key)
            self.assertNotEqual(baseline_state[key].data_ptr(), pinn_state[key].data_ptr(), key)

    def test_dry_run_writes_required_pinn_outputs(self):
        root = Path("results/test_stage21_pinn_inspired")
        root.mkdir(parents=True, exist_ok=True)
        dataset = root / "tiny_dataset.npz"
        rng = np.random.default_rng(21)
        n = 96
        features = rng.normal(size=(n, 7)).astype(np.float32)
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
        model_root = root / "models"
        summary = experiment.run(
            "configs/pinn_inspired_premium.yaml",
            output_root=output_root,
            model_root=model_root,
            overrides={
                "dataset": str(dataset),
                "train_cap": 48,
                "validation_cap": 18,
                "test_cap": 18,
                "epochs": 2,
                "batch_size": 16,
                "hidden": 8,
                "depth": 1,
                "device": "cpu",
                "lambda_pde": 1.0e-4,
                "pde_batch_cap": 8,
                "residual_eval_cap": 12,
                "tau_min_for_pde": 0.10,
            },
        )

        self.assertEqual(summary["decision"], "pinn_inspired_future_extension")
        for name in [
            "pinn_config_manifest.json",
            "pinn_metrics.csv",
            "pinn_residual_metrics.csv",
            "pinn_summary.json",
            "pinn_training_history.csv",
        ]:
            self.assertTrue((output_root / name).exists(), name)
        with open(output_root / "pinn_summary.json", encoding="utf-8") as fh:
            loaded = json.load(fh)
        self.assertEqual(loaded["decision"], summary["decision"])
        self.assertTrue(loaded["matched_initialization"])
        self.assertIn("diagnostic_interpretation", loaded)
        self.assertIn("masked continuation-region", loaded["note"])
        for name in ["pinn_metrics.csv", "pinn_residual_metrics.csv", "pinn_training_history.csv"]:
            self.assertNotIn(b"\r\n", (output_root / name).read_bytes(), name)


if __name__ == "__main__":
    unittest.main()
