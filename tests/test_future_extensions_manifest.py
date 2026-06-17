import importlib.util
import sys
import unittest
from pathlib import Path

_EXPERIMENT_PATH = Path("experiments/15_future_extensions_manifest.py")
_SPEC = importlib.util.spec_from_file_location("stage13_future_extensions", _EXPERIMENT_PATH)
experiment = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = experiment
assert _SPEC.loader is not None
_SPEC.loader.exec_module(experiment)


class FutureExtensionsManifestTests(unittest.TestCase):
    def test_validate_extensions_requires_future_work_only_status_and_validation(self):
        rows = [
            {
                "extension_id": "active_learning",
                "extension_name": "Active learning",
                "motivation": "Reduce PSOR label cost.",
                "current_limitation_addressed": "Fixed grid sweeps are expensive.",
                "required_new_data_or_solver_work": "Additional PSOR labels selected by an acquisition rule.",
                "required_validation": "Compare error reduction per added PSOR regime.",
                "expected_outputs": "Acquisition log and label-efficiency curve.",
                "risk_level": "medium",
                "claim_status": "future_work_only",
            }
        ]

        summary = experiment.validate_extensions(rows)

        self.assertEqual(summary["extensions"], 1)
        self.assertEqual(summary["claim_status"], "future_work_only")
        self.assertEqual(summary["invalid_extensions"], 0)

    def test_validate_extensions_rejects_completed_or_empty_validation_rows(self):
        rows = [
            {
                "extension_id": "uq",
                "extension_name": "Uncertainty quantification",
                "motivation": "Add calibrated intervals.",
                "current_limitation_addressed": "Point estimates lack uncertainty.",
                "required_new_data_or_solver_work": "Calibration set.",
                "required_validation": "",
                "expected_outputs": "Coverage table.",
                "risk_level": "medium",
                "claim_status": "completed",
            }
        ]

        summary = experiment.validate_extensions(rows)

        self.assertEqual(summary["invalid_extensions"], 1)
        self.assertEqual(summary["claim_status"], "future_work_only_with_invalid_rows")


if __name__ == "__main__":
    unittest.main()
