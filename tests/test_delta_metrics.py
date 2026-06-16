import unittest

import numpy as np

from models.metrics import (
    delta_bound_violation,
    grouped_delta_metrics,
    masked_error_metrics,
)


class DeltaMetricTests(unittest.TestCase):
    def test_delta_bound_violation_is_zero_for_bounded_predictions(self):
        delta = np.array([-1.0, -0.4, 0.0, 0.2, 1.0])
        is_call = np.array([0.0, 0.0, 0.0, 1.0, 1.0])

        self.assertEqual(delta_bound_violation(delta, is_call), 0.0)

    def test_grouped_delta_metrics_report_all_put_and_call_rows(self):
        prediction = np.array([-0.8, 0.3, -0.2, 0.7])
        target = np.array([-0.7, 0.4, -0.1, 0.6])
        splits = np.array(["validation", "validation", "test", "test"])
        is_call = np.array([0.0, 1.0, 0.0, 1.0])

        rows = grouped_delta_metrics(
            prediction=prediction,
            target=target,
            splits=splits,
            is_call=is_call,
            model_name="delta_test",
        )

        self.assertIn("all", {row["option_type"] for row in rows})
        self.assertIn("put", {row["option_type"] for row in rows})
        self.assertIn("call", {row["option_type"] for row in rows})
        self.assertEqual({row["split"] for row in rows}, {"validation", "test"})

    def test_masked_error_metrics_handles_empty_strict_mask(self):
        result = masked_error_metrics(
            prediction=np.array([0.1, 0.2]),
            target=np.array([0.0, 0.3]),
            mask=np.array([False, False]),
            prefix="strict_delta",
        )

        self.assertEqual(result["strict_delta_samples"], 0)
        self.assertTrue(np.isnan(result["strict_delta_rmse"]))


if __name__ == "__main__":
    unittest.main()
