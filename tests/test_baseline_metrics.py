import unittest

import numpy as np

from models.metrics import (
    boundary_band_error,
    error_metrics,
    grouped_price_metrics,
    max_obstacle_violation,
    surface_shape_violations,
)


class BaselineMetricTests(unittest.TestCase):
    def test_error_metrics_report_rmse_mae_and_max(self):
        metrics = error_metrics(np.array([1.0, 2.0, 4.0]), np.array([1.0, 1.0, 1.0]))

        self.assertAlmostEqual(metrics["rmse"], np.sqrt((0.0 + 1.0 + 9.0) / 3.0))
        self.assertAlmostEqual(metrics["mae"], 4.0 / 3.0)
        self.assertAlmostEqual(metrics["max_error"], 3.0)

    def test_grouped_price_metrics_keeps_option_type_and_split(self):
        rows = grouped_price_metrics(
            prediction=np.array([1.0, 2.0, 3.0, 5.0]),
            target=np.array([1.0, 1.0, 2.0, 4.0]),
            payoff=np.array([0.5, 1.5, 2.5, 4.5]),
            splits=np.array([b"test", b"test", b"validation", b"validation"]),
            is_call=np.array([0.0, 1.0, 0.0, 1.0]),
            model_name="demo",
        )

        keys = {(row["split"], row["option_type"]) for row in rows}
        self.assertIn(("test", "put"), keys)
        self.assertIn(("test", "call"), keys)
        self.assertIn(("validation", "put"), keys)
        self.assertIn(("validation", "call"), keys)
        self.assertTrue(all("max_obstacle_violation" in row for row in rows))

    def test_surface_shape_violations_detect_monotonicity_and_convexity(self):
        put_surface = np.array([[2.0, 1.0, 1.2, 0.1]])
        call_surface = np.array([[0.0, 1.0, 0.9, 3.0]])
        convex_bad = np.array([[0.0, 2.0, 1.0, 3.0]])

        self.assertGreater(surface_shape_violations(put_surface, "put")["monotonicity_violation"], 0.0)
        self.assertGreater(surface_shape_violations(call_surface, "call")["monotonicity_violation"], 0.0)
        self.assertGreater(surface_shape_violations(convex_bad, "call")["convexity_violation"], 0.0)

    def test_boundary_band_error_uses_boundary_mask(self):
        pred = np.array([1.0, 2.0, 3.0, 10.0])
        target = np.array([1.0, 1.0, 1.0, 10.0])
        near_boundary = np.array([False, True, True, False])

        metrics = boundary_band_error(pred, target, near_boundary)

        self.assertEqual(metrics["boundary_samples"], 2)
        self.assertAlmostEqual(metrics["boundary_rmse"], np.sqrt((1.0 + 4.0) / 2.0))

    def test_max_obstacle_violation_is_zero_when_price_above_payoff(self):
        self.assertEqual(max_obstacle_violation(np.array([1.1, 2.0]), np.array([1.0, 1.5])), 0.0)


if __name__ == "__main__":
    unittest.main()
