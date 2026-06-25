import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _feature_list_payload() -> dict:
    return {
        "schema_version": 2,
        "features": [
            {
                "id": "feature-a",
                "name": "Feature A",
                "description": "First feature.",
                "dependencies": [],
                "status": "not-started",
                "priority": "high",
                "area": "harness",
                "user_visible_behavior": "Behavior A.",
                "verification": "python -m pytest tests/test_a.py",
                "evidence": "",
                "notes": "",
            },
            {
                "id": "feature-b",
                "name": "Feature B",
                "description": "Second feature.",
                "dependencies": ["feature-a"],
                "status": "not-started",
                "priority": "medium",
                "area": "harness",
                "user_visible_behavior": "Behavior B.",
                "verification": "python -m pytest tests/test_b.py",
                "evidence": "",
                "notes": "",
            },
        ],
    }


class FeatureLifecycleToolTests(unittest.TestCase):
    def test_update_feature_status_starts_one_feature_and_appends_note(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            feature_list = Path(tmpdir) / "feature_list.json"
            feature_list.write_text(json.dumps(_feature_list_payload()), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/update_feature_status.py",
                    "--file",
                    str(feature_list),
                    "--id",
                    "feature-a",
                    "--status",
                    "in-progress",
                    "--note",
                    "Started with TDD.",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            payload = json.loads(feature_list.read_text(encoding="utf-8"))

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(payload["features"][0]["status"], "in-progress")
        self.assertIn("Started with TDD.", payload["features"][0]["notes"])
        self.assertIn("feature updated: feature-a -> in-progress", result.stdout)

    def test_update_feature_status_rejects_second_active_feature(self):
        payload = _feature_list_payload()
        payload["features"][0]["status"] = "in-progress"
        with tempfile.TemporaryDirectory() as tmpdir:
            feature_list = Path(tmpdir) / "feature_list.json"
            feature_list.write_text(json.dumps(payload), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/update_feature_status.py",
                    "--file",
                    str(feature_list),
                    "--id",
                    "feature-b",
                    "--status",
                    "in-progress",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("at most one", result.stderr.lower())

    def test_update_feature_status_requires_evidence_when_done(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            feature_list = Path(tmpdir) / "feature_list.json"
            feature_list.write_text(json.dumps(_feature_list_payload()), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/update_feature_status.py",
                    "--file",
                    str(feature_list),
                    "--id",
                    "feature-a",
                    "--status",
                    "done",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires --evidence", result.stderr.lower())

    def test_update_feature_status_can_mark_done_with_evidence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            feature_list = Path(tmpdir) / "feature_list.json"
            feature_list.write_text(json.dumps(_feature_list_payload()), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/update_feature_status.py",
                    "--file",
                    str(feature_list),
                    "--id",
                    "feature-a",
                    "--status",
                    "done",
                    "--evidence",
                    "pytest passed",
                    "--note",
                    "Committed in abc1234.",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            payload = json.loads(feature_list.read_text(encoding="utf-8"))

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(payload["features"][0]["status"], "done")
        self.assertEqual(payload["features"][0]["evidence"], "pytest passed")
        self.assertIn("Committed in abc1234.", payload["features"][0]["notes"])


if __name__ == "__main__":
    unittest.main()
