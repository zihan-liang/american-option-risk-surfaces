import unittest

import numpy as np

from workflow import integrated_metrics, safe_speedup


class IntegratedWorkflowMetricTests(unittest.TestCase):
    def test_integrated_metrics_handle_empty_masks_safely(self):
        features = np.zeros((4, 7), dtype=np.float32)
        features[:, 6] = np.array([0.0, 1.0, 0.0, 1.0])
        labels = np.column_stack(
            [
                np.array([0.5, 0.6, 0.7, 0.8]),
                np.zeros(4),
                np.array([0.4, 0.5, 0.6, 0.7]),
                np.array([-0.2, 0.3, -0.4, 0.5]),
                np.zeros(4),
            ]
        )
        boundary = np.array([np.nan, np.nan, np.nan, np.nan])
        prediction = {
            "price": np.array([0.51, 0.58, 0.71, 0.79]),
            "boundary": np.array([0.9, 1.0, 1.1, 1.2]),
            "delta": np.array([-0.2, 0.4, -0.5, 0.6]),
        }

        row = integrated_metrics(
            prediction=prediction,
            features=features,
            labels=labels,
            boundary=boundary,
            strict_mask=np.zeros(4, dtype=bool),
        )

        self.assertEqual(row["boundary_samples"], 0)
        self.assertEqual(row["strict_delta_samples"], 0)
        self.assertEqual(row["max_price_obstacle_violation"], 0.0)
        self.assertEqual(row["max_delta_bound_violation"], 0.0)
        self.assertTrue(np.isnan(row["boundary_mae"]))

    def test_safe_speedup_handles_zero_or_missing_timing(self):
        self.assertTrue(np.isinf(safe_speedup(1.0, 0.0)))
        self.assertTrue(np.isnan(safe_speedup(float("nan"), 1.0)))
        self.assertAlmostEqual(safe_speedup(2.0, 0.5), 4.0)


if __name__ == "__main__":
    unittest.main()
