import unittest

import numpy as np

from models.premium_surrogate import PositivePremiumModel, reconstruct_price


class PositivePremiumModelTests(unittest.TestCase):
    def test_softplus_premium_output_is_nonnegative(self):
        import torch

        model = PositivePremiumModel.build(input_dim=3, hidden=4, depth=1)
        x = torch.tensor([[0.0, 1.0, -1.0], [2.0, -3.0, 0.5]], dtype=torch.float32)

        premium = model(x)

        self.assertEqual(tuple(premium.shape), (2, 1))
        self.assertTrue(bool(torch.all(premium >= 0.0)))

    def test_reconstruct_price_numpy_preserves_shape_and_obstacle(self):
        payoff = np.array([0.2, 0.0, 1.1])
        premium = np.array([0.0, 0.3, 0.4])

        price = reconstruct_price(payoff, premium)

        self.assertEqual(price.shape, payoff.shape)
        self.assertTrue(np.all(price >= payoff))

    def test_reconstruct_price_torch_preserves_shape_and_obstacle(self):
        import torch

        payoff = torch.tensor([[0.2], [0.0], [1.1]], dtype=torch.float32)
        premium = torch.tensor([[0.0], [0.3], [0.4]], dtype=torch.float32)

        price = reconstruct_price(payoff, premium)

        self.assertEqual(tuple(price.shape), tuple(payoff.shape))
        self.assertTrue(bool(torch.all(price >= payoff)))


if __name__ == "__main__":
    unittest.main()
