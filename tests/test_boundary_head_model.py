import unittest
from pathlib import Path

import torch

from models.boundary_head import BoundaryHeadModel
from models.training import save_torch_model


class BoundaryHeadModelTests(unittest.TestCase):
    def test_boundary_mlp_returns_column_vector(self):
        model = BoundaryHeadModel.build(input_dim=7, hidden=8, depth=1)
        x = torch.zeros((5, 7), dtype=torch.float32)

        prediction = model(x)

        self.assertEqual(tuple(prediction.shape), (5, 1))

    def test_bounded_output_stays_inside_label_range(self):
        model = BoundaryHeadModel.build(input_dim=7, hidden=8, depth=1, lower=0.45, upper=1.75)
        x = torch.randn((16, 7), dtype=torch.float32)

        prediction = model(x)

        self.assertTrue(bool(torch.all(prediction >= 0.45)))
        self.assertTrue(bool(torch.all(prediction <= 1.75)))

    def test_model_save_load_preserves_predictions(self):
        root = Path("results/test_stage09_boundary_model")
        root.mkdir(parents=True, exist_ok=True)
        path = root / "boundary.pt"
        model = BoundaryHeadModel.build(input_dim=7, hidden=8, depth=1, lower=0.4, upper=1.8)
        x = torch.randn((4, 7), dtype=torch.float32)
        before = model(x).detach().clone()

        save_torch_model(model, path, {"model": "boundary_head_test", "input_dim": 7, "hidden": 8, "depth": 1, "lower": 0.4, "upper": 1.8})
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        loaded = BoundaryHeadModel.build(**checkpoint["metadata"])
        loaded.load_state_dict(checkpoint["state_dict"])
        after = loaded(x).detach()

        self.assertTrue(torch.allclose(before, after))


if __name__ == "__main__":
    unittest.main()
