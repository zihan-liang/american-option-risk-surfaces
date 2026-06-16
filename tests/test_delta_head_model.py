import unittest
from pathlib import Path

import torch

from models.delta_head import BoundedDeltaHead, map_delta_unit
from models.training import save_torch_model


class DeltaHeadModelTests(unittest.TestCase):
    def test_bounded_model_returns_column_vector(self):
        model = BoundedDeltaHead.build(input_dim=7, hidden=8, depth=1)
        x = torch.zeros((6, 7), dtype=torch.float32)
        x[:, 6] = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])

        prediction = model(x)

        self.assertEqual(tuple(prediction.shape), (6, 1))

    def test_put_and_call_predictions_stay_inside_delta_bounds(self):
        model = BoundedDeltaHead.build(input_dim=7, hidden=8, depth=1)
        x = torch.randn((20, 7), dtype=torch.float32)
        x[:10, 6] = 0.0
        x[10:, 6] = 1.0

        prediction = model(x).reshape(-1)

        self.assertTrue(bool(torch.all(prediction[:10] >= -1.0)))
        self.assertTrue(bool(torch.all(prediction[:10] <= 0.0)))
        self.assertTrue(bool(torch.all(prediction[10:] >= 0.0)))
        self.assertTrue(bool(torch.all(prediction[10:] <= 1.0)))

    def test_map_delta_unit_respects_option_type(self):
        raw = torch.tensor([-1.0, 0.0, 1.0])
        put = map_delta_unit(raw, torch.zeros(3))
        call = map_delta_unit(raw, torch.ones(3))

        self.assertTrue(bool(torch.all(put >= -1.0) and torch.all(put <= 0.0)))
        self.assertTrue(bool(torch.all(call >= 0.0) and torch.all(call <= 1.0)))

    def test_model_save_load_preserves_predictions(self):
        root = Path("results/test_stage10_delta_model")
        root.mkdir(parents=True, exist_ok=True)
        path = root / "delta.pt"
        model = BoundedDeltaHead.build(input_dim=7, hidden=8, depth=1)
        x = torch.randn((5, 7), dtype=torch.float32)
        x[:, 6] = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0])
        before = model(x).detach().clone()

        save_torch_model(model, path, {"model": "bounded_delta_head_test", "input_dim": 7, "hidden": 8, "depth": 1})
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        loaded = BoundedDeltaHead.build(**checkpoint["metadata"])
        loaded.load_state_dict(checkpoint["state_dict"])
        after = loaded(x).detach()

        self.assertTrue(torch.allclose(before, after))


if __name__ == "__main__":
    unittest.main()
