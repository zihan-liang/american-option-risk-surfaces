import importlib.util
import sys
import unittest
from pathlib import Path

_EXPERIMENT_PATH = Path("experiments/05_surrogate_dataset.py")
_SPEC = importlib.util.spec_from_file_location("stage06_dataset", _EXPERIMENT_PATH)
experiment = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = experiment
assert _SPEC.loader is not None
_SPEC.loader.exec_module(experiment)


class DatasetRegimeFilteringTests(unittest.TestCase):
    def test_filter_excludes_controls_and_diagnostic_failures(self):
        risk_rows = [
            {"case_id": "put_clean", "regime_class": "put", "option_type": "put", "T": "0.25", "sigma": "0.2", "r": "0.05", "q": "0.0"},
            {"case_id": "call_clean", "regime_class": "dividend_call", "option_type": "call", "T": "0.25", "sigma": "0.2", "r": "0.05", "q": "0.06"},
            {"case_id": "control", "regime_class": "no_dividend_call_control", "option_type": "call", "T": "0.25", "sigma": "0.2", "r": "0.05", "q": "0.0"},
            {"case_id": "put_bad", "regime_class": "put", "option_type": "put", "T": "2.0", "sigma": "0.6", "r": "0.05", "q": "0.1"},
        ]
        flag_rows = [
            {"case_id": "put_clean", "diagnostic_failure": "False", "delicate_regime": "False"},
            {"case_id": "call_clean", "diagnostic_failure": "False", "delicate_regime": "False"},
            {"case_id": "control", "diagnostic_failure": "False", "delicate_regime": "False"},
            {"case_id": "put_bad", "diagnostic_failure": "True", "delicate_regime": "True"},
        ]

        clean, excluded = experiment.select_dataset_regimes(risk_rows, flag_rows)

        self.assertEqual([row["case_id"] for row in clean], ["put_clean", "call_clean"])
        excluded_reasons = {row["case_id"]: row["exclusion_reason"] for row in excluded}
        self.assertEqual(excluded_reasons["control"], "control_regime")
        self.assertIn("diagnostic_failure", excluded_reasons["put_bad"])
        self.assertIn("delicate_regime", excluded_reasons["put_bad"])


if __name__ == "__main__":
    unittest.main()
