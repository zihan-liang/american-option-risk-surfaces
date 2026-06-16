import importlib.util
import sys
import unittest
from pathlib import Path

_EXPERIMENT_PATH = Path("experiments/14_claim_audit.py")
_SPEC = importlib.util.spec_from_file_location("stage12_claim_audit_assets", _EXPERIMENT_PATH)
experiment = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = experiment
assert _SPEC.loader is not None
_SPEC.loader.exec_module(experiment)


class FinalReportAssetTests(unittest.TestCase):
    def test_required_artifact_check_reports_missing_paths(self):
        root = Path("results/test_stage12_assets")
        root.mkdir(parents=True, exist_ok=True)
        present = root / "present.csv"
        missing = root / "missing.csv"
        present.write_text("x\n", encoding="utf-8")

        rows = experiment.check_required_artifacts([present, missing])

        self.assertEqual(len(rows), 2)
        by_name = {Path(row["artifact"]).name: row for row in rows}
        self.assertEqual(by_name["present.csv"]["status"], "present")
        self.assertEqual(by_name["missing.csv"]["status"], "missing")


if __name__ == "__main__":
    unittest.main()
