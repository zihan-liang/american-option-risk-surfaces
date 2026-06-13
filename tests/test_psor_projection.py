import unittest

import numpy as np

from solvers.cn_psor import psor_lcp


class PsorProjectionTests(unittest.TestCase):
    def test_psor_lcp_projects_solution_above_obstacle(self):
        a = np.array(
            [
                [2.0, -0.5, 0.0],
                [-0.5, 2.0, -0.5],
                [0.0, -0.5, 2.0],
            ]
        )
        b = np.array([0.1, 0.1, 0.1])
        obstacle = np.array([0.2, 0.4, 0.2])

        x, info = psor_lcp(a, b, obstacle, omega=1.2, tol=1e-12, max_iter=10_000)

        self.assertTrue(info["converged"])
        self.assertTrue(np.all(x >= obstacle - 1e-12))
        residual = a @ x - b
        self.assertTrue(np.all(residual >= -1e-9))
        self.assertLess(abs(np.dot(x - obstacle, residual)), 1e-8)
