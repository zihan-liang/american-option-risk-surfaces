import unittest

import numpy as np

from models.metrics import (
    boundary_band_error,
    error_metrics,
    grouped_price_metrics,
    max_obstacle_violation,
)
from models.premium_surrogate import reconstruct_price


def grouped_premium_metrics(*, prediction, target, splits, is_call, model_name):
    rows = []
    split_values = splits.astype(str)
    option_values = np.where(is_call > 0.5, "call", "put")
    for split in sorted(set(split_values)):
        for option_type in ["put", "call"]:
            mask = (split_values == split) & (option_values == option_type)
            if np.any(mask):
                rows.append(
                    {
                        "model": model_name,
                        "split": split,
                        "option_type": option_type,
                        "samples": int(mask.sum()),
                        **error_metrics(prediction[mask], target[mask]),
                    }
                )
    return rows


class PositivePremiumMetricTests(unittest.TestCase):
    def test_payoff_plus_nonnegative_premium_has_zero_obstacle_violation(self):
        payoff = np.array([0.4, 0.0, 1.0])
        premium = np.array([0.0, 0.2, 0.05])
        price = reconstruct_price(payoff, premium)

        self.assertEqual(max_obstacle_violation(price, payoff), 0.0)

    def test_grouped_price_and_premium_metrics_include_splits_and_option_types(self):
        prediction = np.array([0.5, 0.6, 0.2, 0.3])
        target = np.array([0.4, 0.7, 0.1, 0.35])
        payoff = np.array([0.2, 0.3, 0.0, 0.1])
        splits = np.array(["validation", "validation", "test", "test"])
        is_call = np.array([0.0, 1.0, 0.0, 1.0])

        price_rows = grouped_price_metrics(
            prediction=prediction,
            target=target,
            payoff=payoff,
            splits=splits,
            is_call=is_call,
            model_name="positive_premium_test",
        )
        premium_rows = grouped_premium_metrics(
            prediction=prediction - payoff,
            target=target - payoff,
            splits=splits,
            is_call=is_call,
            model_name="positive_premium_test",
        )

        self.assertEqual({row["split"] for row in price_rows}, {"validation", "test"})
        self.assertEqual({row["option_type"] for row in price_rows}, {"put", "call"})
        self.assertEqual(len(premium_rows), 4)

    def test_boundary_band_error_handles_empty_masks(self):
        result = boundary_band_error(
            prediction=np.array([1.0, 2.0]),
            target=np.array([1.1, 1.9]),
            near_boundary=np.array([False, False]),
        )

        self.assertEqual(result["boundary_samples"], 0)
        self.assertTrue(np.isnan(result["boundary_rmse"]))


if __name__ == "__main__":
    unittest.main()
