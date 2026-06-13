import unittest

import numpy as np

from diagnostics.shape import convexity_violation, monotonicity_violation
from solvers.boundaries import extract_boundary
from solvers.greeks import delta_gamma


class DiagnosticsTests(unittest.TestCase):
    def test_boundary_extraction_for_put_uses_last_exercise_node(self):
        spots = np.array([0.0, 0.5, 1.0, 1.5])
        premium = np.array([0.0, 0.0, 0.2, 0.4])

        boundary = extract_boundary(spots, premium, "put")

        self.assertEqual(boundary, 0.5)

    def test_delta_gamma_shapes_match_price_surface(self):
        spots = np.linspace(0.5, 1.5, 5)
        values = np.vstack([spots**2, 2 * spots**2])

        delta, gamma = delta_gamma(values, spots)

        self.assertEqual(delta.shape, values.shape)
        self.assertEqual(gamma.shape, values.shape)

    def test_shape_violation_helpers_report_zero_for_valid_call_curve(self):
        values = np.array([[0.0, 0.1, 0.3, 0.6]])

        self.assertEqual(monotonicity_violation(values, "call"), 0.0)
        self.assertEqual(convexity_violation(values), 0.0)
