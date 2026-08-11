import unittest

import numpy as np
import torch

from models.obstacle_pinn import (
    build_vi_collocation_mask,
    fischer_burmeister_loss,
    masked_pde_loss,
    separate_complementarity_loss,
    select_whole_regimes,
)


class ObstaclePinnLossTests(unittest.TestCase):
    def test_vi_collocation_mask_excludes_boundary_kink_and_maturity_but_keeps_exercise_points(self):
        features = np.array(
            [
                [0.0, 0.50, 0.05, 0.0, 0.2, 1.0, 0.0],
                [0.0, 0.04, 0.05, 0.0, 0.2, 1.0, 0.0],
                [0.0, 0.50, 0.05, 0.0, 0.2, 1.0, 0.0],
                [0.0, 0.50, 0.05, 0.0, 0.2, 1.0, 0.0],
                [0.0, 0.50, 0.05, 0.0, 0.2, 1.0, 1.0],
            ],
            dtype=np.float64,
        )

        mask = build_vi_collocation_mask(
            features,
            near_boundary=np.array([False, False, True, False, False]),
            near_kink=np.array([False, False, False, True, False]),
            tau_min_for_pde=0.05,
            option_scope="put",
        )

        self.assertEqual(mask.tolist(), [True, False, False, False, False])
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
