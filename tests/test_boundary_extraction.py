import math
import unittest

import numpy as np

from solvers.boundaries import boundary_curve_details, extract_boundary_result


class BoundaryExtractionTests(unittest.TestCase):
    def test_put_boundary_uses_low_spot_exercise_to_continuation_transition(self):
        spots = np.array([0.0, 0.5, 1.0, 1.5])
        premium = np.array([0.0, 0.0, 0.2, 0.4])

        result = extract_boundary_result(spots, premium, "put")

        self.assertEqual(result.status, "boundary_found")
        self.assertEqual(result.exercise_nodes, 2)
        self.assertEqual(result.continuation_nodes, 2)
        self.assertEqual(result.value, 0.5)

    def test_dividend_call_boundary_uses_continuation_to_high_spot_exercise_transition(self):
        spots = np.array([0.0, 1.0, 1.5, 2.0])
        premium = np.array([0.0, 0.3, 0.2, 0.0])

        result = extract_boundary_result(spots, premium, "call")

        self.assertEqual(result.status, "boundary_found")
        self.assertEqual(result.value, 2.0)

    def test_no_exercise_case_does_not_fabricate_boundary(self):
        spots = np.array([0.0, 0.5, 1.0, 1.5])
        premium = np.array([0.2, 0.2, 0.2, 0.2])

        result = extract_boundary_result(spots, premium, "put")

        self.assertEqual(result.status, "no_exercise")
        self.assertTrue(math.isnan(result.value))

    def test_boundary_curve_details_returns_status_for_each_time_level(self):
        spots = np.array([0.0, 0.5, 1.0, 1.5])
        payoff = np.maximum(1.0 - spots, 0.0)
        values = np.vstack([payoff, payoff + np.array([0.0, 0.0, 0.1, 0.2])])

        results = boundary_curve_details(spots, values, payoff, "put")

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].status, "all_exercise")
        self.assertEqual(results[1].status, "boundary_found")
