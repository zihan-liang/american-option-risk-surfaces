import unittest
from pathlib import Path

import numpy as np

from models.boundary_head import BoundaryHeadModel
from models.delta_head import BoundedDeltaHead
from models.premium_surrogate import PositivePremiumModel
from models.training import save_torch_model
from workflow import IntegratedWorkflow


class IntegratedWorkflowModelLoadingTests(unittest.TestCase):
    def test_loads_three_separate_heads_and_predicts_expected_keys(self):
        root = Path("results/test_stage11_workflow_loading")
        root.mkdir(parents=True, exist_ok=True)
        price_path = root / "price.pt"
        boundary_path = root / "boundary.pt"
        delta_path = root / "delta.pt"
        save_torch_model(PositivePremiumModel.build(7, hidden=8, depth=1), price_path, {"input_dim": 7, "hidden": 8, "depth": 1})
        save_torch_model(BoundaryHeadModel.build(7, hidden=8, depth=1, lower=0.0, upper=4.0), boundary_path, {"input_dim": 7, "hidden": 8, "depth": 1, "lower": 0.0, "upper": 4.0})
        save_torch_model(BoundedDeltaHead.build(7, hidden=8, depth=1), delta_path, {"input_dim": 7, "hidden": 8, "depth": 1})

        workflow = IntegratedWorkflow.load(price_path, boundary_path, delta_path, device="cpu")
        features = np.zeros((6, 7), dtype=np.float32)
        features[:, 0] = np.linspace(-0.5, 0.5, 6)
        features[:, 6] = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)
        payoff = np.maximum(1.0 - np.exp(features[:, 0]), 0.0).astype(np.float32)

        prediction = workflow.predict(features, payoff, batch_size=3)

        self.assertEqual(set(prediction), {"price", "premium", "boundary", "delta"})
        self.assertTrue(np.all(prediction["price"] >= payoff))
        self.assertTrue(np.all(prediction["delta"][features[:, 6] <= 0.5] <= 0.0))
        self.assertTrue(np.all(prediction["delta"][features[:, 6] > 0.5] >= 0.0))


if __name__ == "__main__":
    unittest.main()
