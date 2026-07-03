import unittest

import numpy as np
import torch

from models.pinn_campaign import (
    campaign_success_flags,
    pde_weight_for_epoch,
    residual_aware_indices,
)
from models.premium_surrogate import TerminalScaledPositivePremiumModel


class PinnCampaignHelperTests(unittest.TestCase):
    def test_terminal_scaled_premium_is_zero_at_maturity(self):
        model = TerminalScaledPositivePremiumModel.build(input_dim=7, hidden=4, depth=1)
        features = torch.tensor(
            [
                [0.0, 0.0, 0.05, 0.0, 0.2, 1.0, 0.0],
                [0.2, 0.0, 0.05, 0.0, 0.2, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        premium = model(features).reshape(-1)
        self.assertTrue(torch.allclose(premium, torch.zeros_like(premium), atol=1e-7))

    def test_pde_weight_ramp_schedule(self):
        self.assertEqual(pde_weight_for_epoch(1, final_weight=1.0e-4, warm_start_epochs=4, total_epochs=8), 0.0)
        self.assertEqual(pde_weight_for_epoch(4, final_weight=1.0e-4, warm_start_epochs=4, total_epochs=8), 0.0)
        self.assertAlmostEqual(pde_weight_for_epoch(8, final_weight=1.0e-4, warm_start_epochs=4, total_epochs=8), 1.0e-4)

    def test_residual_aware_indices_stay_inside_mask(self):
        rng = np.random.default_rng(7)
        pool = np.array([3, 5, 8, 13, 21])
        residual_abs = np.array([0.1, 0.2, 10.0, 0.3, 0.4])
        selected = residual_aware_indices(pool, residual_abs, size=20, rng=rng)
        self.assertTrue(set(selected.tolist()).issubset(set(pool.tolist())))
        self.assertIn(8, selected.tolist())

    def test_campaign_success_flags_rejects_price_gain_with_residual_regression(self):
        baseline = {
            "price_rmse": 1.0,
            "max_obstacle_violation": 0.0,
            "boundary_rmse": 1.0,
            "residual_mean_abs": 1.0,
        }
        candidate = {
            "price_rmse": 0.99,
            "max_obstacle_violation": 0.0,
            "boundary_rmse": 1.0,
            "residual_mean_abs": 1.2,
        }
        flags = campaign_success_flags(candidate, baseline)
        self.assertFalse(flags["candidate_success"])
        self.assertFalse(flags["diagnostic_success"])


if __name__ == "__main__":
    unittest.main()
