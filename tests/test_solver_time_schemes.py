import unittest

import numpy as np

from diagnostics.solver_checks import closed_form_error_in_region, solver_summary
from solvers.black_scholes import black_scholes_price
from solvers.boundary_conditions import boundary_values
from solvers.cn_psor import SolverConfig, _sdirk2_step, solve_option


class SolverTimeSchemeTests(unittest.TestCase):
    def test_negative_rate_put_lower_boundary_uses_discounted_strike(self):
        left, right = boundary_values(
            "put", tau=2.0, k=3.0, s_max=12.0, r=-0.05, q=0.0
        )

        self.assertAlmostEqual(left, 3.0 * np.exp(0.1), places=14)
        self.assertEqual(right, 0.0)

    def test_negative_rate_put_boundary_propagates_through_solver_time_levels(self):
        cfg = SolverConfig(r=-0.04, t=0.75, m=16, n=5, tol=1e-10)

        result = solve_option(cfg)

        expected = cfg.k * np.exp(-cfg.r * result.taus)
        np.testing.assert_allclose(result.values[:, 0], expected, rtol=0.0, atol=1e-14)

    def test_sdirk2_carries_stage_one_obstacle_reaction_into_second_stage(self):
        cfg = SolverConfig(
            r=-1.0,
            m=3,
            n=1,
            time_scheme="sdirk2",
            lcp_solver="policy_iteration",
            tol=1e-12,
        )
        gamma = 1.0 - 1.0 / np.sqrt(2.0)

        solution, info = _sdirk2_step(
            cfg,
            u_old=np.array([1.0, 0.0]),
            obstacle=np.array([1.0, 0.0]),
            l_mat=np.diag([-1.0, -1.0]),
            lower=np.array([0.5, 0.0]),
            upper=np.array([0.0, 0.0]),
            tau_old=0.0,
            tau_new=1.0,
        )

        expected = (1.0 + gamma * 0.5 * np.exp(1.0)) / (1.0 + gamma)
        self.assertAlmostEqual(solution[0], expected, places=12)
        self.assertGreater(solution[0], 1.0)
        self.assertTrue(info["converged"])

    def test_generic_lcp_fields_keep_psor_compatibility_properties(self):
        result = solve_option(SolverConfig(m=24, n=12, tol=1e-9))

        self.assertEqual(result.lcp_iterations, result.psor_iterations)
        self.assertEqual(result.lcp_converged, result.psor_converged)
        self.assertEqual(len(result.lcp_iterations), result.config.n)

    def test_all_time_schemes_produce_finite_european_prices(self):
        for scheme, rannacher_steps in [
            ("cn", 0),
            ("rannacher_cn", 2),
            ("sdirk2", 0),
        ]:
            with self.subTest(scheme=scheme):
                cfg = SolverConfig(
                    option_type="put",
                    american=False,
                    time_scheme=scheme,
                    rannacher_steps=rannacher_steps,
                    m=48,
                    n=48,
                    s_max=4.0,
                )
                result = solve_option(cfg)
                error = closed_form_error_in_region(result, 0.5, 1.5)
                self.assertTrue(np.isfinite(result.values).all())
                self.assertLess(error["region_rmse"], 0.01)

    def test_rannacher_uses_two_backward_euler_substeps_per_smoothed_interval(self):
        cfg = SolverConfig(
            option_type="put",
            time_scheme="rannacher_cn",
            rannacher_steps=2,
            m=32,
            n=8,
        )
        result = solve_option(cfg)

        self.assertEqual(result.time_substeps[:2], [2, 2])
        self.assertEqual(result.time_substeps[2:], [1] * 6)
        self.assertEqual(len(result.lcp_iterations), cfg.n)

    def test_no_dividend_american_call_matches_european_control_for_each_scheme(self):
        spots = None
        for scheme, rannacher_steps in [
            ("cn", 0),
            ("rannacher_cn", 2),
            ("sdirk2", 0),
        ]:
            with self.subTest(scheme=scheme):
                cfg = SolverConfig(
                    option_type="call",
                    q=0.0,
                    american=True,
                    time_scheme=scheme,
                    rannacher_steps=rannacher_steps,
                    m=64,
                    n=64,
                    s_max=4.0,
                    tol=1e-10,
                )
                result = solve_option(cfg)
                spots = result.spots
                exact = black_scholes_price(spots, cfg.k, cfg.t, cfg.r, cfg.q, cfg.sigma, "call")
                mask = (spots >= 0.5) & (spots <= 1.5)
                rmse = float(np.sqrt(np.mean((result.terminal_values[mask] - exact[mask]) ** 2)))
                self.assertLess(rmse, 0.01)

    def test_policy_and_penalty_are_selectable_lcp_solvers(self):
        for solver in ["policy_iteration", "penalty_newton"]:
            with self.subTest(solver=solver):
                result = solve_option(
                    SolverConfig(
                        option_type="put",
                        lcp_solver=solver,
                        penalty=1e10,
                        m=32,
                        n=16,
                        tol=1e-9,
                    )
                )
                self.assertTrue(np.isfinite(result.values).all())
                self.assertTrue(all(result.lcp_converged))
                if solver == "policy_iteration":
                    self.assertGreaterEqual(float(np.min(result.values - result.payoff)), -1e-12)
                summary = solver_summary(result)
                self.assertTrue(np.isfinite(summary["complementarity_error"]))
                self.assertTrue(np.isfinite(summary["max_lcp_natural_residual"]))
                self.assertEqual(summary["all_lcp_converged"], all(result.lcp_converged))

    def test_invalid_scheme_and_solver_are_rejected(self):
        with self.assertRaises(ValueError):
            solve_option(SolverConfig(time_scheme="bogus"))
        with self.assertRaises(ValueError):
            solve_option(SolverConfig(lcp_solver="bogus"))


if __name__ == "__main__":
    unittest.main()
