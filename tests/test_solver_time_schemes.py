import unittest

import numpy as np

from diagnostics.solver_checks import closed_form_error_in_region, solver_summary
from solvers.black_scholes import black_scholes_price
from solvers.cn_psor import SolverConfig, solve_option


class SolverTimeSchemeTests(unittest.TestCase):
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
