import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class HarnessContractTests(unittest.TestCase):
    def test_required_harness_files_exist(self):
        required = [
            "AGENTS.md",
            "feature_list.json",
            "progress.md",
            "init.sh",
            "session-handoff.md",
            "docs/harness/worktree-policy.md",
            "docs/harness/verification-matrix.md",
            "docs/harness/claim-safety-rules.md",
            "docs/harness/result-schema-policy.md",
            "docs/harness/subagent-protocol.md",
            "docs/harness/harness_changelog.md",
            "scripts/init_harness.sh",
            "scripts/verify_harness.sh",
            "scripts/verify_round2_evidence.sh",
            "scripts/check_result_schema.py",
        ]

        missing = [path for path in required if not (REPO_ROOT / path).exists()]

        self.assertEqual(missing, [])

    def test_feature_list_tracks_valid_harness_state(self):
        with (REPO_ROOT / "feature_list.json").open(encoding="utf-8") as fh:
            payload = json.load(fh)

        allowed_statuses = {"not-started", "in-progress", "blocked", "done", "failed_metric_gate", "diagnostic_only"}
        features = payload["features"]
        self.assertGreaterEqual(len(features), 5)
        self.assertEqual(len({feature["id"] for feature in features}), len(features))
        for feature in features:
            self.assertIn(feature["status"], allowed_statuses)
            self.assertIn("evidence", feature)

    def test_agents_file_contains_project_specific_safety_rules(self):
        text = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
        lower = text.lower()

        self.assertIn("external worktree", text)
        self.assertIn("PSOR", text)
        self.assertIn("Definition of Done", text)
        self.assertIn("Verification Commands", text)
        self.assertIn("do not claim", lower)
        for line in lower.splitlines():
            if "psor" in line and "replace" in line:
                self.assertTrue(
                    any(marker in line for marker in ("do not", "not ", "avoid", "forbidden", "unsafe")),
                    msg=f"unsafe PSOR replacement wording in AGENTS.md line: {line}",
                )

    def test_result_schema_validator_accepts_valid_and_rejects_extra_columns(self):
        required = [
            "method",
            "branch",
            "seed",
            "split",
            "option_type",
            "price_rmse",
            "price_mae",
            "boundary_rmse",
            "boundary_mae",
            "delta_rmse",
            "delta_mae",
            "max_obstacle_violation",
            "max_delta_bound_violation",
            "train_seconds",
            "inference_seconds",
            "integrated_cases",
            "claim_status",
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            valid = Path(tmpdir) / "valid.csv"
            invalid = Path(tmpdir) / "invalid.csv"
            missing = Path(tmpdir) / "missing.csv"
            for path, fields in [(valid, required), (invalid, required + ["sample_policy"]), (missing, required[:-1])]:
                with path.open("w", newline="", encoding="utf-8") as fh:
                    writer = csv.DictWriter(fh, fieldnames=fields)
                    writer.writeheader()
                    writer.writerow({field: "" for field in fields})

            ok = subprocess.run(
                [sys.executable, "scripts/check_result_schema.py", str(valid)],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            bad = subprocess.run(
                [sys.executable, "scripts/check_result_schema.py", str(invalid)],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            missing_result = subprocess.run(
                [sys.executable, "scripts/check_result_schema.py", str(missing)],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(ok.returncode, 0, ok.stderr)
        self.assertNotEqual(bad.returncode, 0)
        self.assertIn("schema mismatch", bad.stderr.lower())
        self.assertNotEqual(missing_result.returncode, 0)
        self.assertIn("schema mismatch", missing_result.stderr.lower())

    def test_round2_evidence_script_detects_zero_candidate_decision(self):
        result = subprocess.run(
            ["bash", "scripts/verify_round2_evidence.sh"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Round-2 evidence verification ok", result.stdout)


if __name__ == "__main__":
    unittest.main()
