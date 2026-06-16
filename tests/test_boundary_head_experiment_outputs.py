import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np

_EXPERIMENT_PATH = Path("experiments/09_boundary_head.py")
_SPEC = importlib.util.spec_from_file_location("stage09_boundary_head", _EXPERIMENT_PATH)
experiment = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = experiment
assert _SPEC.loader is not None
_SPEC.loader.exec_module(experiment)


class BoundaryHeadExperimentOutputTests(unittest.TestCase):
    def test_dry_run_writes_boundary_audit_outputs(self):
        root = Path("results/test_stage09_boundary_head")
        root.mkdir(parents=True, exist_ok=True)
        dataset = root / "tiny_dataset.npz"
        features, labels, split, regime_key, boundary = _tiny_boundary_dataset(120)
        np.savez_compressed(
            dataset,
            features=features,
            labels=labels,
            split=split,
            regime_key=regime_key,
            boundary=boundary,
            near_boundary=np.zeros(120, dtype=bool),
        )

        output_root = root / "out"
        model_root = root / "models"
        summary = experiment.run(
            "configs/boundary_head.yaml",
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

        self.assertEqual(summary["decision"], "scoped_boundary_diagnostic")
        for name in [
            "boundary_metrics.csv",
            "boundary_by_regime.csv",
            "boundary_training_history.csv",
            "boundary_summary.json",
            "boundary_config_manifest.json",
        ]:
            self.assertTrue((output_root / name).exists(), name)
        self.assertTrue((model_root / "boundary_head_linear.npz").exists())
        self.assertTrue((model_root / "boundary_head_mlp.pt").exists())
        header = (output_root / "boundary_metrics.csv").read_text(encoding="utf-8").splitlines()[0]
        self.assertIn("out_of_training_range_fraction", header)
        with open(output_root / "boundary_summary.json", encoding="utf-8") as fh:
            loaded = json.load(fh)
        self.assertEqual(loaded["decision"], summary["decision"])


def _tiny_boundary_dataset(n: int):
    rng = np.random.default_rng(9)
    features = rng.normal(size=(n, 7)).astype(np.float32)
    features[:, 6] = np.array([0.0, 1.0] * (n // 2), dtype=np.float32)
    spot = np.exp(features[:, 0])
    payoff = np.maximum(1.0 - spot, 0.0)
    price = payoff + 0.03
    labels = np.column_stack([price, price - payoff, payoff, np.zeros(n), np.zeros(n)]).astype(np.float32)
    split = np.array([b"train"] * 70 + [b"validation"] * 25 + [b"test"] * 25)
    regime_key = np.array([f"regime_{idx // 10}".encode("utf-8") for idx in range(n)])
    boundary = (0.7 + 0.2 * features[:, 6] + 0.04 * features[:, 2]).astype(np.float32)
    boundary[::13] = np.nan
    return features, labels, split, regime_key, boundary


if __name__ == "__main__":
    unittest.main()
