import importlib.util
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiments" / "binomial_benchmark_convergence.py"


def load_module():
    if not MODULE_PATH.is_file():
        raise AssertionError(f"missing production module: {MODULE_PATH}")
    spec = importlib.util.spec_from_file_location("binomial_benchmark_convergence", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def brute_crr_put(spot, tau, *, k, r, q, sigma, steps):
    dt = tau / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    growth = math.exp((r - q) * dt)
    p = (growth - d) / (u - d)
    disc = math.exp(-r * dt)
    stock = [spot * (u**j) * (d ** (steps - j)) for j in range(steps + 1)]
    values = [max(k - s, 0.0) for s in stock]
    for n in range(steps - 1, -1, -1):
        next_values = []
        for j in range(n + 1):
            continuation = disc * (p * values[j + 1] + (1.0 - p) * values[j])
            s = spot * (u**j) * (d ** (n - j))
            next_values.append(max(continuation, k - s, 0.0))
        values = next_values
    return values[0]


class TestBinomialBenchmark(unittest.TestCase):
    def test_crr_matches_independent_small_tree(self):
        m = load_module()
        expected = brute_crr_put(1.0, 1.0, k=1.0, r=0.05, q=0.10, sigma=0.60, steps=8)
        actual = m.crr_american_put(1.0, 1.0, k=1.0, r=0.05, q=0.10, sigma=0.60, steps=8)
        self.assertAlmostEqual(actual, expected, places=13)


class TestBenchmarkPlan(unittest.TestCase):
    def test_default_plan_has_expected_probe_and_confirmation_jobs(self):
        m = load_module()
        plan = m.build_default_job_plan()
        probe = [row for row in plan if row["phase"] == "probe"]
        confirm = [row for row in plan if row["phase"] == "confirm"]
        self.assertEqual(len(probe), 27 * 4)
        self.assertEqual(len(confirm), 9 * 2)
        self.assertEqual(sorted({row["steps"] for row in probe}), [2048, 4096, 8192, 16384])
        self.assertEqual(sorted({row["steps"] for row in confirm}), [32768, 50000])
        self.assertEqual(sorted({row["tau_over_T"] for row in plan}), [0.1, 0.5, 1.0])

    def test_summarize_convergence_uses_matching_point_pairs_only(self):
        m = load_module()
        rows = [
            {"phase": "probe", "S_over_K": 1.0, "tau_over_T": 1.0, "steps": 2048, "V_over_K": 0.20, "seconds": 1.0},
            {"phase": "probe", "S_over_K": 1.0, "tau_over_T": 1.0, "steps": 4096, "V_over_K": 0.21, "seconds": 1.0},
            {"phase": "probe", "S_over_K": 0.8, "tau_over_T": 1.0, "steps": 2048, "V_over_K": 0.30, "seconds": 1.0},
            {"phase": "probe", "S_over_K": 0.8, "tau_over_T": 1.0, "steps": 4096, "V_over_K": 0.32, "seconds": 1.0},
        ]
        summary = m.summarize_convergence(rows)
        row = next(x for x in summary if x["from_steps"] == 2048 and x["to_steps"] == 4096)
        self.assertEqual(row["point_count"], 2)
        self.assertAlmostEqual(row["rmse_change"], (0.01**2 + 0.02**2) ** 0.5 / (2**0.5), places=14)
        self.assertAlmostEqual(row["max_abs_change"], 0.02, places=14)


class TestWindowsRunnerContract(unittest.TestCase):
    def test_powershell_runner_exposes_preflight_and_all_phases(self):
        runner = ROOT / "scripts" / "run_binomial_benchmark_windows.ps1"
        self.assertTrue(runner.is_file(), f"missing runner: {runner}")
        text = runner.read_text(encoding="utf-8")
        self.assertIn("binomial_benchmark_convergence.py", text)
        self.assertIn("-Phase", text)
        self.assertIn("preflight", text)
        self.assertIn("all", text)
        self.assertIn("-Workers", text)


if __name__ == "__main__":
    unittest.main()
