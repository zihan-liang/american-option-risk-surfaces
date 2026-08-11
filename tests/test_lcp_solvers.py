import unittest

import numpy as np

from solvers.black_scholes import payoff
from solvers.cn_psor import (
    SolverConfig,
    brennan_schwartz_lcp,
    cn_matrices,
    lcp_complementarity_error,
    lcp_natural_residual,
    penalty_newton_lcp,
    policy_iteration_lcp,
    psor_lcp,
    time_boundary_values,
)


def _american_put_first_step_lcp(cfg):
    spots, lower, upper, a, b_mat = cn_matrices(cfg)
    obstacle = payoff(spots, cfg.k, "put")[1:-1]
    dt = cfg.t / cfg.n
    left_old, right_old = time_boundary_values(cfg, 0.0)
    left_new, right_new = time_boundary_values(cfg, dt)
    rhs = np.einsum("ij,j->i", b_mat, obstacle, optimize=False)
    rhs[0] += 0.5 * dt * lower[0] * (left_old + left_new)
    rhs[-1] += 0.5 * dt * upper[-1] * (right_old + right_new)
    return a, rhs, obstacle


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
            tol=2e-9,
            max_iter=100,
        )

        self.assertTrue(info["converged"])
        self.assertLess(np.max(np.maximum(self.obstacle - solution, 0.0)), 2e-9)
        self.assertLess(lcp_natural_residual(self.a, solution, self.b, self.obstacle), 2e-9)

    def test_psor_converged_flag_is_defined_by_final_natural_residual(self):
        solution, info = psor_lcp(
            np.array([[1.0]]),
            np.array([0.0]),
            np.array([0.0]),
            omega=1.0,
            tol=1e-12,
            max_iter=1,
            x0=np.array([1.0]),
        )

        np.testing.assert_array_equal(solution, np.array([0.0]))
        self.assertEqual(info["natural_residual"], 0.0)
        self.assertGreater(info["last_update"], 1e-12)
        self.assertTrue(info["converged"])

    def test_policy_converged_flag_is_defined_by_final_natural_residual(self):
        solution, info = policy_iteration_lcp(
            np.array([[1.0]]),
            np.array([0.0]),
            np.array([0.0]),
            tol=1e-12,
            max_iter=1,
            x0=np.array([1.0]),
        )

        np.testing.assert_array_equal(solution, np.array([0.0]))
        self.assertEqual(info["natural_residual"], 0.0)
        self.assertGreater(info["last_update"], 1e-12)
        self.assertTrue(info["converged"])

    def test_penalty_converged_flag_rejects_only_penalized_equation_convergence(self):
        _, info = penalty_newton_lcp(
            np.array([[1.0]]),
            np.array([-1.0]),
            np.array([0.0]),
            penalty=1.0,
            tol=1e-12,
            max_iter=3,
        )

        self.assertEqual(info["equation_residual"], 0.0)
        self.assertEqual(info["natural_residual"], 0.5)
        self.assertFalse(info["converged"])

    def test_brennan_schwartz_matches_known_put_ordered_lcp(self):
        solution, info = brennan_schwartz_lcp(self.a, self.b, self.obstacle, tol=1e-12)

        np.testing.assert_allclose(solution, self.expected, atol=1e-10)
        self.assertTrue(info["converged"])

    def test_brennan_schwartz_solves_positive_rate_american_put_lcp(self):
        cfg = SolverConfig(m=40, n=40, t=0.25, r=0.05, sigma=0.2, tol=1e-12)
        a, rhs, obstacle = _american_put_first_step_lcp(cfg)
        expected, _ = policy_iteration_lcp(
            a, rhs, obstacle, tol=cfg.tol, max_iter=100, x0=obstacle
        )

        solution, info = brennan_schwartz_lcp(
            a,
            rhs,
            obstacle,
            tol=cfg.tol,
            exercise_direction="put",
            interest_rate=cfg.r,
        )

        np.testing.assert_allclose(solution, expected, rtol=0.0, atol=1e-12)
        self.assertLessEqual(info["natural_residual"], cfg.tol)
        self.assertTrue(info["converged"])
        self.assertTrue(info["applicable"])
        self.assertEqual(info["exercise_direction"], "put")

    def test_brennan_schwartz_classifies_negative_rate_put_as_not_applicable(self):
        cfg = SolverConfig(m=40, n=40, t=0.25, r=-0.012, q=-0.016, sigma=0.1)
        a, rhs, obstacle = _american_put_first_step_lcp(cfg)

        solution, info = brennan_schwartz_lcp(
            a,
            rhs,
            obstacle,
            exercise_direction="put",
            interest_rate=cfg.r,
        )

        np.testing.assert_array_equal(solution, obstacle)
        self.assertFalse(info["applicable"])
        self.assertFalse(info["converged"])
        self.assertEqual(info["applicability_reason"], "negative_interest_rate")

    def test_lcp_solvers_reject_non_finite_inputs(self):
        bad = self.b.copy()
        bad[0] = np.nan
        with self.assertRaises(ValueError):
            policy_iteration_lcp(self.a, bad, self.obstacle, tol=1e-10, max_iter=10)


if __name__ == "__main__":
    unittest.main()
