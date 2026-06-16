import unittest

import numpy as np

from solvers.greeks import delta_gamma, greek_diagnostic_masks


class GreekMaskTests(unittest.TestCase):
    def test_delta_gamma_shapes_match_surface(self):
        spots = np.linspace(0.0, 2.0, 9)
        values = np.vstack([spots**2, 2.0 * spots**2])

        delta, gamma = delta_gamma(values, spots)

        self.assertEqual(delta.shape, values.shape)
        self.assertEqual(gamma.shape, values.shape)

    def test_named_masks_exclude_kink_boundary_and_near_zero_spot(self):
        spots = np.array([0.0, 0.02, 0.5, 0.98, 1.0, 1.02, 1.5])
        boundary = np.array([0.5, np.nan])

        masks = greek_diagnostic_masks(spots, boundary, k=1.0, kink_width=0.03, boundary_width=0.04)

        self.assertEqual(set(masks), {"strict", "near_kink", "near_boundary", "near_zero_spot"})
        self.assertTrue(masks["near_zero_spot"][0, 0])
        self.assertTrue(masks["near_zero_spot"][0, 1])
        self.assertTrue(masks["near_kink"][0, 4])
        self.assertTrue(masks["near_boundary"][0, 2])
        self.assertFalse(masks["strict"][0, 0])
        self.assertFalse(masks["strict"][0, 2])
        self.assertFalse(masks["strict"][0, 4])
        self.assertTrue(masks["strict"][0, 6])
        self.assertTrue(masks["strict"][1, 2])
