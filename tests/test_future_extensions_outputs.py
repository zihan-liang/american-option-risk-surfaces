import csv
import importlib.util
import json
import sys
import unittest
from pathlib import Path

import yaml

_EXPERIMENT_PATH = Path("experiments/15_future_extensions_manifest.py")
_SPEC = importlib.util.spec_from_file_location("stage13_future_extensions_outputs", _EXPERIMENT_PATH)
experiment = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = experiment
assert _SPEC.loader is not None
_SPEC.loader.exec_module(experiment)


class FutureExtensionsOutputTests(unittest.TestCase):
    def test_dry_run_writes_manifest_validation_matrix_and_summary(self):
        root = Path("results/test_stage13_future_extensions")
        output_dir = root / "out"
        config_path = root / "future_extensions.yaml"
        root.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            yaml.safe_dump(
                {
                    "output_dir": str(output_dir),
                    "extensions": [
                        _extension("active_learning", "Active learning"),
                        _extension("uncertainty_quantification", "Uncertainty quantification"),
                        _extension("shape_constrained_models", "Shape-constrained models"),
                        _extension("alternative_dynamics", "Alternative dynamics"),
                    ],
                }
            ),
            encoding="utf-8",
        )

        summary = experiment.run(str(config_path))

        self.assertEqual(summary["extensions"], 4)
        self.assertEqual(summary["claim_status"], "future_work_only")
        for name in [
            "future_extensions.csv",
            "future_extension_validation_matrix.csv",
            "future_extensions_summary.json",
        ]:
            self.assertTrue((output_dir / name).exists(), name)
        with open(output_dir / "future_extensions.csv", newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        ids = {row["extension_id"] for row in rows}
        self.assertIn("active_learning", ids)
        self.assertIn("uncertainty_quantification", ids)
        self.assertIn("shape_constrained_models", ids)
        self.assertIn("alternative_dynamics", ids)
        with open(output_dir / "future_extensions_summary.json", encoding="utf-8") as fh:
            loaded = json.load(fh)
        self.assertTrue(loaded["no_completed_extension_claims"])


def _extension(extension_id: str, name: str) -> dict:
    return {
        "extension_id": extension_id,
        "extension_name": name,
        "motivation": f"Motivation for {name}.",
        "current_limitation_addressed": "Current project limitation.",
        "required_new_data_or_solver_work": "New audited data or solver work.",
        "required_validation": "A validation table and held-out PSOR audit.",
        "expected_outputs": "A report section and reproducible manifest.",
        "risk_level": "medium",
        "claim_status": "future_work_only",
    }


if __name__ == "__main__":
    unittest.main()
