import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np

_PREMIUM_PATH = Path("experiments/07_positive_premium_tuning.py")
_PREMIUM_SPEC = importlib.util.spec_from_file_location("stage10_positive_source", _PREMIUM_PATH)
premium_experiment = importlib.util.module_from_spec(_PREMIUM_SPEC)
sys.modules[_PREMIUM_SPEC.name] = premium_experiment
assert _PREMIUM_SPEC.loader is not None
_PREMIUM_SPEC.loader.exec_module(premium_experiment)

_VALIDATION_PATH = Path("experiments/11_delta_validation.py")
_VALIDATION_SPEC = importlib.util.spec_from_file_location("stage10_delta_validation", _VALIDATION_PATH)
validation_experiment = importlib.util.module_from_spec(_VALIDATION_SPEC)
sys.modules[_VALIDATION_SPEC.name] = validation_experiment
assert _VALIDATION_SPEC.loader is not None
_VALIDATION_SPEC.loader.exec_module(validation_experiment)


class DeltaValidationOutputTests(unittest.TestCase):
    def test_dry_run_writes_price_model_delta_validation_outputs(self):
        root = Path("results/test_stage10_delta_validation")
        root.mkdir(parents=True, exist_ok=True)
        dataset = root / "tiny_dataset.npz"
        _write_tiny_dataset(dataset, n=96)
        model_root = root / "models"
        premium_experiment.run(
            "configs/positive_premium.yaml",
            output_root=root / "premium_out",
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

        output_root = root / "delta_validation_out"
        summary = validation_experiment.run(
            "configs/delta_validation.yaml",
            output_root=output_root,
            overrides={
                "dataset": str(dataset),
                "price_model": str(model_root / "selected_final_price_premium.pt"),
                "test_cap": 20,
                "batch_size": 8,
                "device": "cpu",
            },
        )

        self.assertEqual(summary["decision"], "do_not_claim_price_model_as_greek_engine")
        self.assertTrue(summary["uses_strict_greek_mask"])
        self.assertTrue((output_root / "delta_validation_metrics.csv").exists())
        self.assertTrue((output_root / "delta_validation_by_regime.csv").exists())
        self.assertTrue((output_root / "delta_validation_summary.json").exists())
        with open(output_root / "delta_validation_summary.json", encoding="utf-8") as fh:
            loaded = json.load(fh)
        self.assertEqual(loaded["decision"], "do_not_claim_price_model_as_greek_engine")


def _write_tiny_dataset(path: Path, n: int) -> None:
    rng = np.random.default_rng(17)
    features = rng.normal(size=(n, 7)).astype(np.float32)
    features[:, 6] = np.array([0.0, 1.0] * (n // 2), dtype=np.float32)
    spot = np.exp(features[:, 0])
    payoff = np.where(features[:, 6] > 0.5, np.maximum(spot - 1.0, 0.0), np.maximum(1.0 - spot, 0.0))
    premium = (0.03 + 0.02 * np.maximum(features[:, 1], 0.0)).astype(np.float32)
    price = payoff + premium
    delta = np.where(features[:, 6] > 0.5, np.clip(0.5 + 0.1 * features[:, 0], 0.0, 1.0), np.clip(-0.5 + 0.1 * features[:, 0], -1.0, 0.0))
    labels = np.column_stack([price, premium, payoff, delta, np.zeros(n)]).astype(np.float32)
    split = np.array([b"train"] * 56 + [b"validation"] * 20 + [b"test"] * 20)
    regime_key = np.array([f"regime_{idx // 8}".encode("utf-8") for idx in range(n)])
    strict = np.ones(n, dtype=bool)
    strict[::11] = False
    np.savez_compressed(
        path,
        features=features,
        labels=labels,
        split=split,
        regime_key=regime_key,
        near_boundary=np.zeros(n, dtype=bool),
        near_kink=np.zeros(n, dtype=bool),
        near_zero_spot=np.zeros(n, dtype=bool),
        strict_greek_mask=strict,
    )


if __name__ == "__main__":
    unittest.main()
