import unittest

import numpy as np

from solvers.boundaries import boundary_curve
from solvers.cn_psor import SolverConfig, solve_option
from solvers.integral_boundary import american_put_boundary_integral


class IntegralBoundaryTests(unittest.TestCase):
    def test_integral_boundary_is_finite_bounded_and_decreases_with_time(self):
        taus, boundary = american_put_boundary_integral(
            k=1.0,
            t=1.0,
            r=0.05,
            q=0.0,
            sigma=0.2,
            steps=64,
        )

        self.assertEqual(len(taus), 65)
        self.assertTrue(np.isfinite(boundary).all())
        self.assertAlmostEqual(boundary[0], 1.0)
        self.assertTrue(np.all((boundary > 0.0) & (boundary <= 1.0)))
        self.assertLessEqual(float(np.max(np.diff(boundary))), 2e-3)

    def test_integral_boundary_agrees_with_independent_fd_boundary_at_maturity(self):
        cfg = SolverConfig(
            option_type="put",
            american=True,
            time_scheme="rannacher_cn",
            rannacher_steps=2,
            lcp_solver="policy_iteration",
            m=160,
            n=160,
            s_max=4.0,
            tol=1e-10,
        )
        result = solve_option(cfg)
        fd_boundary = boundary_curve(result.spots, result.values, result.payoff, "put")
        _, integral_boundary = american_put_boundary_integral(
            k=cfg.k,
            t=cfg.t,
            r=cfg.r,
            q=cfg.q,
            sigma=cfg.sigma,
            steps=160,
        )

        self.assertLess(abs(float(fd_boundary[-1]) - float(integral_boundary[-1])), 0.04)


if __name__ == "__main__":
    unittest.main()
