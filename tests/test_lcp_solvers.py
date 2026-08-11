import unittest

import numpy as np

from solvers.cn_psor import (
    brennan_schwartz_lcp,
    lcp_complementarity_error,
    lcp_natural_residual,
    penalty_newton_lcp,
    policy_iteration_lcp,
    psor_lcp,
)


class LCPSolverTests(unittest.TestCase):
    def setUp(self):
        self.a = np.array([[2.0, -1.0], [-1.0, 2.0]])
        self.b = np.array([1.0, -2.0])
        self.obstacle = np.zeros(2)
        self.expected = np.array([0.5, 0.0])

    def test_psor_and_policy_solve_same_known_lcp(self):
        psor, psor_info = psor_lcp(
            self.a,
            self.b,
            self.obstacle,
            omega=1.0,
            tol=1e-12,
            max_iter=10_000,
        )
        policy, policy_info = policy_iteration_lcp(
            self.a,
            self.b,
            self.obstacle,
            tol=1e-12,
            max_iter=100,
        )

        np.testing.assert_allclose(psor, self.expected, atol=1e-10)
        np.testing.assert_allclose(policy, self.expected, atol=1e-10)
        self.assertTrue(psor_info["converged"])
        self.assertTrue(policy_info["converged"])
        self.assertLessEqual(lcp_natural_residual(self.a, policy, self.b, self.obstacle), 1e-12)
        self.assertLessEqual(lcp_complementarity_error(self.a, policy, self.b, self.obstacle), 1e-12)

    def test_penalty_newton_reports_controlled_penalty_error(self):
        solution, info = penalty_newton_lcp(
            self.a,
            self.b,
            self.obstacle,
            penalty=1e9,
            tol=1e-12,
            max_iter=100,
        )

        self.assertTrue(info["converged"])
        self.assertLess(np.max(np.maximum(self.obstacle - solution, 0.0)), 2e-9)
        self.assertLess(lcp_natural_residual(self.a, solution, self.b, self.obstacle), 2e-9)

    def test_brennan_schwartz_matches_known_put_ordered_lcp(self):
        solution, info = brennan_schwartz_lcp(self.a, self.b, self.obstacle, tol=1e-12)

        np.testing.assert_allclose(solution, self.expected, atol=1e-10)
        self.assertTrue(info["converged"])

    def test_lcp_solvers_reject_non_finite_inputs(self):
        bad = self.b.copy()
        bad[0] = np.nan
        with self.assertRaises(ValueError):
            policy_iteration_lcp(self.a, bad, self.obstacle, tol=1e-10, max_iter=10)


if __name__ == "__main__":
    unittest.main()
