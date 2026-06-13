import unittest

import numpy as np

from solvers.cn_psor import SolverConfig, solve_option


class CnPsorShapeTests(unittest.TestCase):
    def test_solver_returns_expected_grid_shapes_and_payoff_initial_condition(self):
        cfg = SolverConfig(option_type="put", m=24, n=12, s_max=4.0, k=1.0)

        result = solve_option(cfg)

        self.assertEqual(result.values.shape, (cfg.n + 1, cfg.m + 1))
        self.assertEqual(result.spots.shape, (cfg.m + 1,))
        self.assertEqual(result.taus.shape, (cfg.n + 1,))
        np.testing.assert_allclose(result.values[0], result.payoff)

    def test_american_put_solution_respects_obstacle(self):
        cfg = SolverConfig(option_type="put", m=32, n=16, s_max=4.0, k=1.0)

        result = solve_option(cfg)

        self.assertGreaterEqual(np.min(result.values - result.payoff[None, :]), -1e-10)
        self.assertGreater(max(result.psor_iterations), 0)
