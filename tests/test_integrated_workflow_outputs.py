import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np

from models.boundary_head import BoundaryHeadModel
from models.delta_head import BoundedDeltaHead
from models.premium_surrogate import PositivePremiumModel
from models.training import save_torch_model

_EXPERIMENT_PATH = Path("experiments/13_integrated_application_workflow.py")
_SPEC = importlib.util.spec_from_file_location("stage11_integrated_workflow", _EXPERIMENT_PATH)
experiment = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = experiment
assert _SPEC.loader is not None
_SPEC.loader.exec_module(experiment)


class IntegratedWorkflowOutputTests(unittest.TestCase):
    def test_dry_run_writes_required_outputs_without_solver_replacement_claim(self):
        root = Path("results/test_stage11_integrated_outputs")
        root.mkdir(parents=True, exist_ok=True)
        dataset = root / "tiny_fresh_case.npz"
        _write_tiny_fresh_case(dataset)
        model_root = root / "models"
        model_root.mkdir(exist_ok=True)
        price_path = model_root / "price.pt"
        boundary_path = model_root / "boundary.pt"
        delta_path = model_root / "delta.pt"
        save_torch_model(PositivePremiumModel.build(7, hidden=8, depth=1), price_path, {"input_dim": 7, "hidden": 8, "depth": 1})
        save_torch_model(BoundaryHeadModel.build(7, hidden=8, depth=1, lower=0.0, upper=4.0), boundary_path, {"input_dim": 7, "hidden": 8, "depth": 1, "lower": 0.0, "upper": 4.0})
        save_torch_model(BoundedDeltaHead.build(7, hidden=8, depth=1), delta_path, {"input_dim": 7, "hidden": 8, "depth": 1})

        output_root = root / "out"
        summary = experiment.run(
            "configs/integrated_workflow.yaml",
            output_root=output_root,
            figure_root=root / "figures",
            overrides={
                "price_model": str(price_path),
                "boundary_model": str(boundary_path),
                "delta_model": str(delta_path),
                "dry_run_cases": [str(dataset)],
                "batch_size": 8,
                "device": "cpu",
            },
        )

        self.assertEqual(summary["decision"], "scenario_screening_with_psor_audit")
        self.assertFalse(summary["claims_psor_replacement"])
        for name in [
            "integrated_case_metrics.csv",
            "integrated_by_regime.csv",
            "integrated_violation_table.csv",
            "integrated_timing.csv",
            "integrated_summary.json",
            "integrated_config_manifest.json",
        ]:
            self.assertTrue((output_root / name).exists(), name)
        with open(output_root / "integrated_summary.json", encoding="utf-8") as fh:
            loaded = json.load(fh)
        self.assertEqual(loaded["decision"], "scenario_screening_with_psor_audit")
        self.assertNotIn("gamma", json.dumps(loaded).lower())
        self.assertNotIn("replacement", loaded["allowed_claim"])


def _write_tiny_fresh_case(path: Path) -> None:
    n = 32
    rng = np.random.default_rng(23)
    features = rng.normal(size=(n, 7)).astype(np.float32)
    features[:, 6] = np.array([0.0, 1.0] * (n // 2), dtype=np.float32)
    spot = np.exp(features[:, 0])
    payoff = np.where(features[:, 6] > 0.5, np.maximum(spot - 1.0, 0.0), np.maximum(1.0 - spot, 0.0)).astype(np.float32)
    premium = np.full(n, 0.03, dtype=np.float32)
    price = payoff + premium
    delta = np.where(features[:, 6] > 0.5, np.clip(0.5 + 0.1 * features[:, 0], 0.0, 1.0), np.clip(-0.5 + 0.1 * features[:, 0], -1.0, 0.0))
    labels = np.column_stack([price, premium, payoff, delta, np.zeros(n)]).astype(np.float32)
    boundary = np.where(features[:, 6] > 0.5, 1.4, 0.8).astype(np.float32)
    strict = np.ones(n, dtype=bool)
    strict[::5] = False
    np.savez_compressed(
        path,
        features=features,
        labels=labels,
        boundary=boundary,
        strict_greek_mask=strict,
        regime_key=np.array([b"tiny_integrated_case"] * n),
        split=np.array([b"test"] * n),
        psor_seconds=np.array([0.25]),
        option_type=np.array([b"mixed"]),
    )


if __name__ == "__main__":
    unittest.main()
