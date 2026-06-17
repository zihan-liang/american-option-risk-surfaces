import importlib.util
import sys
import unittest
from pathlib import Path

_EXPERIMENT_PATH = Path("experiments/15_future_extensions_manifest.py")
_SPEC = importlib.util.spec_from_file_location("stage13_future_extensions_language", _EXPERIMENT_PATH)
experiment = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = experiment
assert _SPEC.loader is not None
_SPEC.loader.exec_module(experiment)


class FutureExtensionsLanguageTests(unittest.TestCase):
    def test_scan_flags_future_work_presented_as_completed(self):
        root = Path("results/test_stage13_future_language")
        root.mkdir(parents=True, exist_ok=True)
        safe = root / "safe.md"
        unsafe = root / "unsafe.md"
        safe.write_text(
            "Active learning remains future work only and requires PSOR validation before any claim.\n",
            encoding="utf-8",
        )
        unsafe.write_text(
            "Active learning improves the surrogate and has validated production performance.\n",
            encoding="utf-8",
        )

        rows = experiment.scan_future_claim_language([safe, unsafe])

        violations = [row for row in rows if row["status"] == "violation"]
        self.assertEqual(len(violations), 1)
        self.assertEqual(Path(violations[0]["file"]).name, "unsafe.md")


if __name__ == "__main__":
    unittest.main()
