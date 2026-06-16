import math
import unittest

import numpy as np

from diagnostics.risk_metrics import (
    boundary_status_counts,
    classify_regime,
    exercise_fraction,
    finite_summary,
    greek_strict_summary,
)
from solvers.boundaries import BoundaryResult


class ApplicationRiskMetricsTests(unittest.TestCase):
    def test_exercise_fraction_counts_exercise_nodes_across_surface(self):
        payoff = np.array([1.0, 0.5, 0.0])
        values = np.array(
            [
                [1.0, 0.7, 0.2],
                [1.0, 0.5, 0.1],
            ]
        )

        self.assertAlmostEqual(exercise_fraction(values, payoff, tol=1e-10), 0.5)

    def test_boundary_status_counts_and_finite_summary_ignore_nan_boundaries(self):
        details = [
            BoundaryResult(0.8, "boundary_found", 3, 2),
            BoundaryResult(float("nan"), "no_exercise", 0, 5),
            BoundaryResult(0.7, "boundary_found", 2, 3),
        ]

        counts = boundary_status_counts(details)
        summary = finite_summary([row.value for row in details])

        self.assertEqual(counts["boundary_found"], 2)
        self.assertEqual(counts["no_exercise"], 1)
        self.assertAlmostEqual(summary["mean"], 0.75)
        self.assertAlmostEqual(summary["initial"], 0.7)

    def test_greek_strict_summary_uses_only_strict_mask_entries(self):
        delta = np.array([[0.0, 0.2], [0.4, 0.6]])
        gamma = np.array([[1.0, 2.0], [3.0, 99.0]])
        strict = np.array([[True, False], [True, False]])

        summary = greek_strict_summary(delta, gamma, strict)

        self.assertEqual(summary["strict_samples"], 2)
        self.assertAlmostEqual(summary["delta_strict_mean"], 0.2)
        self.assertAlmostEqual(summary["gamma_strict_mean_abs"], 2.0)
        self.assertAlmostEqual(summary["gamma_strict_max_abs"], 3.0)

    def test_greek_strict_summary_returns_nan_when_mask_empty(self):
        summary = greek_strict_summary(
            np.ones((2, 2)),
            np.ones((2, 2)),
            np.zeros((2, 2), dtype=bool),
        )

        self.assertEqual(summary["strict_samples"], 0)
        self.assertTrue(math.isnan(summary["delta_strict_mean"]))

    def test_classify_regime_separates_put_dividend_call_and_control(self):
        self.assertEqual(classify_regime("put", 0.0), "put")
        self.assertEqual(classify_regime("call", 0.0), "no_dividend_call_control")
        self.assertEqual(classify_regime("call", 0.03), "dividend_call")


if __name__ == "__main__":
    unittest.main()
