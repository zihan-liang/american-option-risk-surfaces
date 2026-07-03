import unittest

import numpy as np
import torch

from models.pinn_residual import build_psor_continuation_mask, normalized_payoff_from_features
from models.premium_surrogate import PositivePremiumModel


class PinnMaskTests(unittest.TestCase):
    def test_psor_continuation_mask_excludes_exercise_boundary_maturity_and_calls(self):
        features = np.array(
            [
                [0.0, 0.50, 0.05, 0.0, 0.2, 1.0, 0.0],
                [0.0, 0.50, 0.05, 0.0, 0.2, 1.0, 0.0],
                [0.0, 0.50, 0.05, 0.0, 0.2, 1.0, 0.0],
                [0.0, 0.05, 0.05, 0.0, 0.2, 1.0, 0.0],
                [0.0, 0.50, 0.05, 0.0, 0.2, 1.0, 1.0],
            ],
            dtype=np.float32,
        )
        labels = np.array(
            [
                [0.70, 0.20, 0.50, 0.0, 0.0],
                [0.500001, 0.0, 0.50, 0.0, 0.0],
                [0.70, 0.20, 0.50, 0.0, 0.0],
                [0.70, 0.20, 0.50, 0.0, 0.0],
                [0.70, 0.20, 0.50, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        near_boundary = np.array([False, False, True, False, False])

        mask = build_psor_continuation_mask(
            features,
            labels,
            near_boundary=near_boundary,
            gap_epsilon=1.0e-4,
            tau_min_for_pde=0.10,
            option_scope="put",
        )

        self.assertEqual(mask.tolist(), [True, False, False, False, False])

    def test_positive_premium_reconstruction_cannot_fall_below_payoff(self):
        features = torch.tensor(
            [
                [0.0, 0.5, 0.05, 0.0, 0.2, 1.0, 0.0],
                [0.2, 0.5, 0.05, 0.0, 0.2, 1.0, 1.0],
            ],
            dtype=torch.float32,
        )
        model = PositivePremiumModel.build(input_dim=7, hidden=4, depth=1)

        payoff = normalized_payoff_from_features(features)
        price = payoff + model(features).reshape(-1)

        self.assertTrue(bool(torch.all(price >= payoff)))


if __name__ == "__main__":
    unittest.main()
