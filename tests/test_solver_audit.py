import unittest

import numpy as np

from solvers.boundary_conditions import boundary_values
from solvers.cn_psor import SolverConfig, cn_matrices, fd_operator_matrix


class SolverAuditTests(unittest.TestCase):
    def test_operator_matrix_matches_black_scholes_operator_for_linear_function(self):
        cfg = SolverConfig(m=8, n=4, s_max=2.0, r=0.05, q=0.02, sigma=0.3)
        spots = np.linspace(0.0, cfg.s_max, cfg.m + 1)

        l_mat = fd_operator_matrix(cfg, spots)
        linear_values = spots[1:-1]

        numerical = l_mat @ linear_values
        analytical = -cfg.q * spots[1:-1]
        np.testing.assert_allclose(numerical[:-1], analytical[:-1], atol=1e-12)

    def test_crank_nicolson_matrices_use_expected_sign_convention(self):
        cfg = SolverConfig(m=8, n=4, s_max=2.0)
        spots, _, _, a, b = cn_matrices(cfg)
        dt = cfg.t / cfg.n
        l_mat = fd_operator_matrix(cfg, spots)

        np.testing.assert_allclose(a, np.eye(cfg.m - 1) - 0.5 * dt * l_mat)
        np.testing.assert_allclose(b, np.eye(cfg.m - 1) + 0.5 * dt * l_mat)

    def test_american_put_boundary_conditions_are_obstacle_consistent(self):
        left, right = boundary_values("put", tau=0.75, k=1.0, s_max=4.0, r=0.05, q=0.0)

        self.assertEqual(left, 1.0)
        self.assertEqual(right, 0.0)

    def test_dividend_call_boundary_conditions_blend_immediate_and_asymptotic_values(self):
        tau = 0.75
        left, right = boundary_values("call", tau=tau, k=1.0, s_max=4.0, r=0.05, q=0.06)
        immediate = 3.0
        european_asymptotic = 4.0 * np.exp(-0.06 * tau) - np.exp(-0.05 * tau)

        self.assertEqual(left, 0.0)
        self.assertAlmostEqual(right, max(immediate, european_asymptotic, 0.0))
