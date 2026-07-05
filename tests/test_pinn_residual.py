import unittest

import torch

from models.pinn_residual import log_space_black_scholes_residual, normalized_payoff_from_features


class PinnResidualTests(unittest.TestCase):
    def test_log_space_residual_uses_theta_over_t_scaling(self):
        features = torch.tensor(
            [
                [0.20, 0.40, 0.05, 0.01, 0.30, 2.00, 0.00],
                [-0.10, 0.60, 0.03, 0.00, 0.25, 0.50, 0.00],
            ],
            dtype=torch.float64,
            requires_grad=True,
        )
        x = features[:, 0]
        theta = features[:, 1]
        r = features[:, 2]
        q = features[:, 3]
        sigma = features[:, 4]
        maturity = features[:, 5]
        price = theta.pow(2) + x.pow(2)

        residual = log_space_black_scholes_residual(price, features)

        expected = (
            (2.0 * theta / maturity)
            - 0.5 * sigma.pow(2) * 2.0
            - (r - q - 0.5 * sigma.pow(2)) * (2.0 * x)
            + r * price
        )
        self.assertTrue(torch.allclose(residual, expected, atol=1e-10))

    def test_normalized_payoff_from_features_supports_puts_and_calls(self):
        features = torch.tensor(
            [
                [torch.log(torch.tensor(0.8)).item(), 0.5, 0.05, 0.0, 0.2, 1.0, 0.0],
                [torch.log(torch.tensor(1.3)).item(), 0.5, 0.05, 0.0, 0.2, 1.0, 1.0],
            ],
            dtype=torch.float64,
        )

        payoff = normalized_payoff_from_features(features)

        expected = torch.tensor([0.2, 0.3], dtype=torch.float64)
        self.assertTrue(torch.allclose(payoff, expected, atol=1e-7))


if __name__ == "__main__":
    unittest.main()
