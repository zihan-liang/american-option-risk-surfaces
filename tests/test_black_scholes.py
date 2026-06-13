import math
import unittest

import numpy as np

from solvers.black_scholes import black_scholes_price, payoff


class BlackScholesTests(unittest.TestCase):
    def test_put_call_payoff_vectors(self):
        spots = np.array([0.5, 1.0, 1.5])

        np.testing.assert_allclose(payoff(spots, 1.0, "put"), [0.5, 0.0, 0.0])
        np.testing.assert_allclose(payoff(spots, 1.0, "call"), [0.0, 0.0, 0.5])

    def test_black_scholes_put_call_parity_with_dividends(self):
        s = 1.2
        k = 1.0
        tau = 0.75
        r = 0.04
        q = 0.02
        sigma = 0.3

        call = black_scholes_price(s, k, tau, r, q, sigma, "call")
        put = black_scholes_price(s, k, tau, r, q, sigma, "put")

        expected = s * math.exp(-q * tau) - k * math.exp(-r * tau)
        self.assertLess(abs((call - put) - expected), 1e-12)

    def test_black_scholes_intrinsic_at_expiry(self):
        spots = np.array([0.8, 1.0, 1.4])

        np.testing.assert_allclose(
            black_scholes_price(spots, 1.0, 0.0, 0.03, 0.0, 0.2, "call"),
            [0.0, 0.0, 0.4],
        )
