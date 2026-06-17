import importlib.util
import json
import sys
import unittest
from pathlib import Path

_EXPERIMENT_PATH = Path("experiments/16_release_verification.py")
_SPEC = importlib.util.spec_from_file_location("stage14_release_verification", _EXPERIMENT_PATH)
experiment = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = experiment
assert _SPEC.loader is not None
_SPEC.loader.exec_module(experiment)


class ReleaseVerificationTests(unittest.TestCase):
    def test_run_reports_required_files_tracked_artifacts_and_stage_summaries(self):
        root = Path("results/test_stage14_release_verification")
        output_root = root / "out"
        required = root / "README.md"
        stage12 = root / "stage12.json"
        stage13 = root / "stage13.json"
        root.mkdir(parents=True, exist_ok=True)
        required.write_text("release handoff\n", encoding="utf-8")
        stage12.write_text(
            json.dumps({"forbidden_language_violations": 0, "missing_artifacts": 0}),
            encoding="utf-8",
        )
        stage13.write_text(
            json.dumps({"invalid_extensions": 0, "language_violations": 0, "claim_status": "future_work_only"}),
            encoding="utf-8",
        )

        summary = experiment.run(
            output_root=output_root,
            required_files=[required],
            tracked_files=["README.md", "manuscript_outline.aux"],
            forbidden_tracked_files=["manuscript_outline.aux"],
            stage12_summary=stage12,
            stage13_summary=stage13,
        )

        self.assertEqual(summary["missing_required_files"], 0)
        self.assertEqual(summary["forbidden_tracked_files"], 1)
        self.assertEqual(summary["stage12_release_gate"], "pass")
        self.assertEqual(summary["stage13_release_gate"], "pass")
        for name in [
            "release_required_files.csv",
            "release_tracked_generated_files.csv",
            "release_stage_summary_checks.csv",
            "release_verification_summary.json",
        ]:
            self.assertTrue((output_root / name).exists(), name)


if __name__ == "__main__":
    unittest.main()
