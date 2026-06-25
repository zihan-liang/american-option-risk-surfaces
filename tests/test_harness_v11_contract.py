import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class HarnessV11ContractTests(unittest.TestCase):
    def test_v11_operational_harness_files_exist(self):
        required = [
            "docs/harness/startup-readiness-checklist.md",
            "docs/harness/artifact-policy.md",
            "docs/harness/clean-state-checklist.md",
            "docs/harness/evaluator-rubric.md",
            "docs/harness/quality-document.md",
            "docs/harness/decision-log.md",
            "docs/harness/feature-list.schema.json",
            "scripts/validate_feature_list.py",
            "scripts/verify_report_artifacts.sh",
            "scripts/bootstrap_artifacts.sh",
            "scripts/check_clean_session.sh",
            "scripts/score_harness.py",
        ]

        missing = [path for path in required if not (REPO_ROOT / path).exists()]

        self.assertEqual(missing, [])

    def test_feature_list_schema_requires_operational_fields_and_wip_limit(self):
        result = subprocess.run(
            [sys.executable, "scripts/validate_feature_list.py", "feature_list.json"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("feature list ok", result.stdout.lower())

        payload = json.loads((REPO_ROOT / "feature_list.json").read_text(encoding="utf-8"))
        active = [feature for feature in payload["features"] if feature["status"] == "in-progress"]
        self.assertLessEqual(len(active), 1)
        for feature in payload["features"]:
            for field in ["priority", "area", "user_visible_behavior", "verification", "notes"]:
                self.assertIn(field, feature)

    def test_feature_list_validator_rejects_two_active_features(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            feature_list = Path(tmpdir) / "feature_list.json"
            feature_list.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "features": [
                            {
                                "id": "feature-a",
                                "name": "Feature A",
                                "description": "First active feature.",
                                "dependencies": [],
                                "status": "in-progress",
                                "priority": "high",
                                "area": "harness",
                                "user_visible_behavior": "Visible behavior A.",
                                "verification": "pytest tests/test_a.py",
                                "evidence": "",
                                "notes": "",
                            },
                            {
                                "id": "feature-b",
                                "name": "Feature B",
                                "description": "Second active feature.",
                                "dependencies": [],
                                "status": "in-progress",
                                "priority": "medium",
                                "area": "harness",
                                "user_visible_behavior": "Visible behavior B.",
                                "verification": "pytest tests/test_b.py",
                                "evidence": "",
                                "notes": "",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [sys.executable, "scripts/validate_feature_list.py", str(feature_list)],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("at most one", result.stderr.lower())

    def test_report_artifact_verifier_detects_missing_and_present_figures(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tex = root / "report.tex"
            tex.write_text(
                "\\documentclass{article}\n"
                "\\usepackage{graphicx}\n"
                "\\begin{document}\n"
                "\\includegraphics{results/figures/example.png}\n"
                "\\end{document}\n",
                encoding="utf-8",
            )

            missing = subprocess.run(
                ["bash", str(REPO_ROOT / "scripts/verify_report_artifacts.sh"), "--root", str(root), "--tex", str(tex)],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            (root / "results/figures").mkdir(parents=True)
            (root / "results/figures/example.png").write_bytes(b"fake image bytes")
            present = subprocess.run(
                ["bash", str(REPO_ROOT / "scripts/verify_report_artifacts.sh"), "--root", str(root), "--tex", str(tex)],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertNotEqual(missing.returncode, 0)
        self.assertIn("missing report artifacts", missing.stderr.lower())
        self.assertEqual(present.returncode, 0, present.stderr)
        self.assertIn("report artifacts ok", present.stdout.lower())

    def test_harness_score_writes_observability_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [sys.executable, "scripts/score_harness.py", "--output-dir", tmpdir],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            score_path = Path(tmpdir) / "harness_score.json"

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(score_path.exists())
            payload = json.loads(score_path.read_text(encoding="utf-8"))

        self.assertIn("overall_score", payload)
        self.assertGreaterEqual(payload["overall_score"], 0)
        self.assertLessEqual(payload["overall_score"], 100)
        self.assertEqual(
            {item["dimension"] for item in payload["dimensions"]},
            {"instructions", "state", "verification", "scope", "lifecycle", "observability"},
        )

    def test_clean_session_check_supports_allow_dirty_for_active_work(self):
        result = subprocess.run(
            ["bash", "scripts/check_clean_session.sh", "--allow-dirty"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("clean-session check ok", result.stdout.lower())


if __name__ == "__main__":
    unittest.main()
