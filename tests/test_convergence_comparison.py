import unittest

import numpy as np

from diagnostics.convergence import (
    boundary_error,
    masked_error,
    surface_to_reference_grid,
)
from solvers.boundaries import BoundaryResult


class ConvergenceComparisonTests(unittest.TestCase):
    def test_surface_interpolation_preserves_values_on_same_grid(self):
        spots = np.array([0.0, 1.0, 2.0])
        taus = np.array([0.0, 0.5, 1.0])
        surface = np.array(
            [
                [0.0, 1.0, 4.0],
                [0.5, 1.5, 4.5],
                [1.0, 2.0, 5.0],
            ]
        )

        interpolated = surface_to_reference_grid(spots, taus, surface, spots, taus)

        np.testing.assert_allclose(interpolated, surface)

    def test_boundary_error_uses_only_boundary_found_entries(self):
        candidate = [
            BoundaryResult(0.8, "boundary_found", 2, 2),
            BoundaryResult(float("nan"), "no_exercise", 0, 4),
            BoundaryResult(0.7, "boundary_found", 2, 2),
        ]
        reference = [
            BoundaryResult(0.75, "boundary_found", 2, 2),
            BoundaryResult(0.9, "boundary_found", 2, 2),
            BoundaryResult(float("nan"), "no_exercise", 0, 4),
        ]

        metrics = boundary_error(candidate, reference)

        self.assertEqual(metrics["samples"], 1)
        self.assertAlmostEqual(metrics["mae"], 0.05)
        self.assertAlmostEqual(metrics["max"], 0.05)

    def test_masked_error_uses_only_true_mask_entries(self):
        candidate = np.array([[1.0, 2.0], [3.0, 4.0]])
        reference = np.array([[1.0, 4.0], [5.0, 4.0]])
        mask = np.array([[True, False], [True, False]])

        metrics = masked_error(candidate, reference, mask)

        self.assertEqual(metrics["samples"], 2)
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(2.0))
        self.assertAlmostEqual(metrics["mae"], 1.0)
