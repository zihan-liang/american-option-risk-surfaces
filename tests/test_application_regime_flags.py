import unittest

from diagnostics.risk_metrics import regime_flags


class ApplicationRegimeFlagTests(unittest.TestCase):
    def test_no_dividend_call_control_is_flagged_without_becoming_failure(self):
        flags = regime_flags(
            option_type="call",
            regime_class="no_dividend_call_control",
            q=0.0,
            boundary_found_count=0,
            obstacle_violation=0.0,
            complementarity_error=1e-12,
            monotonicity_violation=0.0,
            convexity_violation=0.0,
        )

        self.assertTrue(flags["is_control"])
        self.assertTrue(flags["call_no_exercise"])
        self.assertFalse(flags["diagnostic_failure"])

    def test_dividend_call_without_exercise_is_delicate_and_not_hidden(self):
        flags = regime_flags(
            option_type="call",
            regime_class="dividend_call",
            q=0.03,
            boundary_found_count=0,
            obstacle_violation=0.0,
            complementarity_error=1e-12,
            monotonicity_violation=0.0,
            convexity_violation=0.0,
        )

        self.assertFalse(flags["is_control"])
        self.assertTrue(flags["call_no_exercise"])
        self.assertFalse(flags["call_dividend_exercise_found"])
        self.assertTrue(flags["delicate_regime"])

    def test_diagnostic_tolerance_breach_sets_failure_flag(self):
        flags = regime_flags(
            option_type="put",
            regime_class="put",
            q=0.0,
            boundary_found_count=10,
            obstacle_violation=1e-4,
            complementarity_error=1e-12,
            monotonicity_violation=0.0,
            convexity_violation=0.0,
        )

        self.assertTrue(flags["diagnostic_failure"])


if __name__ == "__main__":
    unittest.main()
