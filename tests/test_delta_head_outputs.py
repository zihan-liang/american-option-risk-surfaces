import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np

_EXPERIMENT_PATH = Path("experiments/12_supervised_delta_head.py")
_SPEC = importlib.util.spec_from_file_location("stage10_delta_head", _EXPERIMENT_PATH)
experiment = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = experiment
assert _SPEC.loader is not None
_SPEC.loader.exec_module(experiment)


class DeltaHeadOutputTests(unittest.TestCase):
    def test_dry_run_writes_bounded_delta_head_outputs(self):
        root = Path("results/test_stage10_delta_head")
        root.mkdir(parents=True, exist_ok=True)
        dataset = root / "tiny_dataset.npz"
        _write_tiny_dataset(dataset, n=120)

        output_root = root / "out"
        model_root = root / "models"
        summary = experiment.run(
            "configs/delta_head.yaml",
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

        self.assertEqual(summary["decision"], "bounded_supervised_delta_diagnostic")
        self.assertTrue(summary["uses_strict_greek_mask"])
        for name in [
            "delta_head_metrics.csv",
            "delta_head_by_regime.csv",
            "delta_head_training_history.csv",
            "delta_head_violation_table.csv",
            "delta_head_summary.json",
            "delta_head_config_manifest.json",
        ]:
            self.assertTrue((output_root / name).exists(), name)
        self.assertTrue((model_root / "delta_head_linear.npz").exists())
        self.assertTrue((model_root / "bounded_delta_head_mlp.pt").exists())
        header = (output_root / "delta_head_violation_table.csv").read_text(encoding="utf-8").splitlines()[0]
        self.assertIn("max_delta_bound_violation", header)
        with open(output_root / "delta_head_summary.json", encoding="utf-8") as fh:
            loaded = json.load(fh)
        self.assertEqual(loaded["decision"], summary["decision"])


def _write_tiny_dataset(path: Path, n: int) -> None:
    rng = np.random.default_rng(19)
    features = rng.normal(size=(n, 7)).astype(np.float32)
    features[:, 6] = np.array([0.0, 1.0] * (n // 2), dtype=np.float32)
    spot = np.exp(features[:, 0])
    payoff = np.where(features[:, 6] > 0.5, np.maximum(spot - 1.0, 0.0), np.maximum(1.0 - spot, 0.0))
    premium = np.full(n, 0.04, dtype=np.float32)
    price = payoff + premium
    delta = np.where(features[:, 6] > 0.5, np.clip(0.4 + 0.15 * features[:, 0], 0.0, 1.0), np.clip(-0.4 + 0.15 * features[:, 0], -1.0, 0.0))
    labels = np.column_stack([price, premium, payoff, delta, np.zeros(n)]).astype(np.float32)
    split = np.array([b"train"] * 70 + [b"validation"] * 25 + [b"test"] * 25)
    regime_key = np.array([f"regime_{idx // 10}".encode("utf-8") for idx in range(n)])
    strict = np.ones(n, dtype=bool)
    strict[::13] = False
    near_boundary = np.zeros(n, dtype=bool)
    near_boundary[::7] = True
    near_kink = np.zeros(n, dtype=bool)
    near_kink[1::9] = True
    near_zero_spot = np.zeros(n, dtype=bool)
    near_zero_spot[2::17] = True
    np.savez_compressed(
        path,
        features=features,
        labels=labels,
        split=split,
        regime_key=regime_key,
        near_boundary=near_boundary,
        near_kink=near_kink,
        near_zero_spot=near_zero_spot,
        strict_greek_mask=strict,
    )


if __name__ == "__main__":
    unittest.main()
