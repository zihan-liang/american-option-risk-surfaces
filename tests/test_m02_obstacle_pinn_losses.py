import unittest

import numpy as np
import torch

from models.obstacle_pinn import (
    fischer_burmeister_loss,
    masked_pde_loss,
    separate_complementarity_loss,
    select_whole_regimes,
)


class ObstaclePinnLossTests(unittest.TestCase):
    def test_masked_pde_loss_uses_only_predeclared_continuation_points(self):
        residual = torch.tensor([1.0, 10.0], dtype=torch.float64)
        mask = torch.tensor([True, False])

        loss = masked_pde_loss(residual, mask, residual_scale=1.0)

        self.assertAlmostEqual(float(loss), 1.0)

    def test_separate_complementarity_reports_dual_and_product_terms(self):
        gap = torch.tensor([2.0, 0.0], dtype=torch.float64)
        residual = torch.tensor([3.0, -1.0], dtype=torch.float64)

        terms = separate_complementarity_loss(
            gap,
            residual,
            gap_scale=1.0,
            residual_scale=1.0,
        )

        self.assertAlmostEqual(float(terms["dual_feasibility"]), 0.5)
        self.assertAlmostEqual(float(terms["complementarity"]), 18.0)
        self.assertAlmostEqual(float(terms["total"]), 18.5)

    def test_fischer_burmeister_loss_applies_predeclared_smoothing(self):
        gap = torch.tensor([0.0], dtype=torch.float64)
        residual = torch.tensor([0.0], dtype=torch.float64)

        loss = fischer_burmeister_loss(
            gap,
            residual,
            epsilon=0.25,
            gap_scale=1.0,
            residual_scale=1.0,
        )

        self.assertAlmostEqual(float(loss), 0.0625)

    def test_whole_regime_sampler_never_splits_a_regime_and_respects_cap(self):
        regime_keys = np.array(["a"] * 5 + ["b"] * 5 + ["c"] * 5)
        selected, regime_ids = select_whole_regimes(
            np.arange(len(regime_keys)),
            regime_keys,
            cap=12,
            seed=203,
        )

        self.assertEqual(len(selected), 10)
        self.assertEqual(len(regime_ids), 2)
        self.assertLessEqual(len(selected), 12)
        for regime_id in regime_ids:
            self.assertEqual(np.count_nonzero(regime_keys[selected] == regime_id), 5)


if __name__ == "__main__":
    unittest.main()
