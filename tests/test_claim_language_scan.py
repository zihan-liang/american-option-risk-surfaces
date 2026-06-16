import importlib.util
import sys
import unittest
from pathlib import Path

_EXPERIMENT_PATH = Path("experiments/14_claim_audit.py")
_SPEC = importlib.util.spec_from_file_location("stage12_claim_audit_language", _EXPERIMENT_PATH)
experiment = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = experiment
assert _SPEC.loader is not None
_SPEC.loader.exec_module(experiment)


class ClaimLanguageScanTests(unittest.TestCase):
    def test_scan_flags_positive_forbidden_claims_but_allows_restrictions(self):
        root = Path("results/test_stage12_language_scan")
        root.mkdir(parents=True, exist_ok=True)
        safe = root / "safe.tex"
        unsafe = root / "unsafe.tex"
        safe.write_text(
            "The workflow does not replace PSOR.\n"
            "Do not claim one network solves price boundary and Greeks.\n",
            encoding="utf-8",
        )
        unsafe.write_text("The integrated workflow replaces PSOR in production.\n", encoding="utf-8")

        rows = experiment.scan_forbidden_language(
            [safe, unsafe],
            ["replaces PSOR", "one network solves price boundary and Greeks"],
        )

        violations = [row for row in rows if row["status"] == "violation"]
        self.assertEqual(len(violations), 1)
        self.assertEqual(Path(violations[0]["file"]).name, "unsafe.tex")
        self.assertIn("replaces PSOR", violations[0]["phrase"])


if __name__ == "__main__":
    unittest.main()
