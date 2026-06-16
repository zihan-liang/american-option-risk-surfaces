import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np

_EXPERIMENT_PATH = Path("experiments/10_option_specific_call_boundary_refinement.py")
_SPEC = importlib.util.spec_from_file_location("stage09_call_boundary", _EXPERIMENT_PATH)
experiment = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = experiment
assert _SPEC.loader is not None
_SPEC.loader.exec_module(experiment)


class CallBoundaryRefinementOutputTests(unittest.TestCase):
    def test_dry_run_writes_call_specific_metrics(self):
        root = Path("results/test_stage09_call_boundary")
        root.mkdir(parents=True, exist_ok=True)
        dataset = root / "tiny_call_dataset.npz"
        features, labels, split, regime_key, boundary = _tiny_dataset(call_boundaries=True)
        np.savez_compressed(
            dataset,
            features=features,
            labels=labels,
            split=split,
            regime_key=regime_key,
            boundary=boundary,
            near_boundary=np.zeros(len(features), dtype=bool),
        )

        output_root = root / "out"
        model_root = root / "models"
        summary = experiment.run(
            "configs/call_boundary_refinement.yaml",
            output_root=output_root,
            model_root=model_root,
            figure_root=root / "figures",
            overrides={
                "dataset": str(dataset),
                "train_cap": 50,
                "validation_cap": 24,
                "test_cap": 24,
                "epochs": 2,
                "batch_size": 16,
                "hidden": 12,
                "depth": 1,
                "device": "cpu",
            },
        )

        self.assertEqual(summary["decision"], "scoped_call_boundary_diagnostic")
        self.assertTrue((output_root / "call_boundary_metrics.csv").exists())
        self.assertTrue((output_root / "call_boundary_by_regime.csv").exists())
        self.assertTrue((output_root / "call_boundary_summary.json").exists())
        self.assertTrue((model_root / "call_boundary_head_mlp.pt").exists())

    def test_no_call_boundary_labels_return_insufficient_data_decision(self):
        root = Path("results/test_stage09_call_boundary_empty")
        root.mkdir(parents=True, exist_ok=True)
        dataset = root / "tiny_no_call_boundaries.npz"
        features, labels, split, regime_key, boundary = _tiny_dataset(call_boundaries=False)
        np.savez_compressed(
            dataset,
            features=features,
            labels=labels,
            split=split,
            regime_key=regime_key,
            boundary=boundary,
            near_boundary=np.zeros(len(features), dtype=bool),
        )

        output_root = root / "out"
        summary = experiment.run(
            "configs/call_boundary_refinement.yaml",
            output_root=output_root,
            model_root=root / "models",
            figure_root=root / "figures",
            overrides={"dataset": str(dataset), "device": "cpu"},
        )

        self.assertEqual(summary["decision"], "insufficient_call_data")
        with open(output_root / "call_boundary_summary.json", encoding="utf-8") as fh:
            loaded = json.load(fh)
        self.assertEqual(loaded["decision"], "insufficient_call_data")


def _tiny_dataset(call_boundaries: bool):
    n = 120
    rng = np.random.default_rng(11)
    features = rng.normal(size=(n, 7)).astype(np.float32)
    features[:, 6] = np.array([0.0, 1.0] * (n // 2), dtype=np.float32)
    payoff = np.maximum(1.0 - np.exp(features[:, 0]), 0.0)
    labels = np.column_stack([payoff + 0.04, np.full(n, 0.04), payoff, np.zeros(n), np.zeros(n)]).astype(np.float32)
    split = np.array([b"train"] * 70 + [b"validation"] * 25 + [b"test"] * 25)
    regime_key = np.array([f"regime_{idx // 10}".encode("utf-8") for idx in range(n)])
    boundary = (0.85 + 0.3 * features[:, 6] + 0.03 * features[:, 4]).astype(np.float32)
    if not call_boundaries:
        boundary[features[:, 6] > 0.5] = np.nan
    return features, labels, split, regime_key, boundary


if __name__ == "__main__":
    unittest.main()
