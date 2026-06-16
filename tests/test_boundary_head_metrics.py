import unittest

import numpy as np

from models.metrics import (
    finite_boundary_mask,
    grouped_boundary_metrics,
    prediction_range_diagnostics,
)


class BoundaryHeadMetricTests(unittest.TestCase):
    def test_finite_boundary_mask_excludes_nan_and_inf(self):
        boundary = np.array([0.8, np.nan, 1.2, np.inf, -np.inf])

        mask = finite_boundary_mask(boundary)

        self.assertEqual(mask.tolist(), [True, False, True, False, False])

    def test_grouped_boundary_metrics_report_all_put_and_call_rows(self):
        prediction = np.array([0.5, 0.8, 1.2, 1.5])
        target = np.array([0.6, 0.7, 1.0, 1.4])
        splits = np.array(["validation", "validation", "test", "test"])
        is_call = np.array([0.0, 1.0, 0.0, 1.0])

        rows = grouped_boundary_metrics(
            prediction=prediction,
            target=target,
            splits=splits,
            is_call=is_call,
            model_name="boundary_test",
        )

        self.assertIn("all", {row["option_type"] for row in rows})
        self.assertIn("put", {row["option_type"] for row in rows})
        self.assertIn("call", {row["option_type"] for row in rows})
        self.assertEqual({row["split"] for row in rows}, {"validation", "test"})

    def test_prediction_range_diagnostics_handles_empty_and_nonempty_inputs(self):
        empty = prediction_range_diagnostics(np.array([]), lower=0.4, upper=1.8)
        nonempty = prediction_range_diagnostics(np.array([0.3, 0.8, 2.0]), lower=0.4, upper=1.8)

        self.assertEqual(empty["samples"], 0)
        self.assertTrue(np.isnan(empty["prediction_min"]))
        self.assertEqual(nonempty["samples"], 3)
        self.assertAlmostEqual(nonempty["out_of_training_range_fraction"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
