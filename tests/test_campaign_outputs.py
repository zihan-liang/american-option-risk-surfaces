import csv
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from campaign.io import write_method_outputs
from campaign.records import MethodRunRecord, RunPhase, RunStatus
from campaign.registry import get_method_spec


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_SCRIPT = REPOSITORY_ROOT / "experiments" / "26_independent_method_campaign.py"


class CampaignOutputTests(unittest.TestCase):
    def test_writer_emits_stable_standard_evidence_and_hashes_every_generated_file(self):
        record = MethodRunRecord(
            method_id="M01",
            phase=RunPhase.SCREEN,
            status=RunStatus.COMPLETED,
            primary_metric_value=0.2,
            baseline_metric_value=0.25,
            optional_metrics={"boundary_rmse": 0.1},
            scenario_count=75_000,
            seed=101,
            split="validation",
            branch="feature/campaign-registry",
            commit="deadbeef",
            timings_seconds={"inference": 1.0, "training": 2.0},
            memory_mb=128.0,
            claim_status="pilot_candidate",
            notes="contract test",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "M01"
            paths = write_method_outputs(
                output_dir,
                get_method_spec("M01"),
                record,
                config={"z": 1, "a": {"nested": True}},
                training_history=[{"epoch": 2, "loss": 0.2}, {"epoch": 1, "loss": 0.4}],
            )

            self.assertEqual(
                {path.name for path in paths.values()},
                {
                    "artifact_hashes.json",
                    "config_manifest.json",
                    "decision.json",
                    "run_metrics.csv",
                    "training_history.csv",
                },
            )
            manifest_text = (output_dir / "config_manifest.json").read_text(encoding="utf-8")
            self.assertLess(manifest_text.index('"a"'), manifest_text.index('"z"'))
            manifest = json.loads(manifest_text)
            expected_config_hash = hashlib.sha256(
                json.dumps({"z": 1, "a": {"nested": True}}, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
            ).hexdigest()
            self.assertEqual(manifest["config_sha256"], expected_config_hash)
            self.assertEqual(manifest["provenance"], {"branch": "feature/campaign-registry", "commit": "deadbeef"})
            with (output_dir / "run_metrics.csv").open(newline="", encoding="utf-8") as handle:
                metric_rows = list(csv.DictReader(handle))
            self.assertEqual(len(metric_rows), 1)
            self.assertEqual(metric_rows[0]["method_id"], "M01")
            self.assertEqual(metric_rows[0]["primary_metric_value"], "0.2")

            hashes = json.loads((output_dir / "artifact_hashes.json").read_text(encoding="utf-8"))
            self.assertEqual(hashes["algorithm"], "sha256")
            self.assertEqual(set(hashes["files"]), {"config_manifest.json", "decision.json", "run_metrics.csv", "training_history.csv"})
            for filename, expected_hash in hashes["files"].items():
                self.assertEqual(hashlib.sha256((output_dir / filename).read_bytes()).hexdigest(), expected_hash)

    def test_writer_omits_optional_training_history_when_no_history_is_supplied(self):
        record = MethodRunRecord(
            method_id="M24",
            phase=RunPhase.SCREEN,
            status=RunStatus.NOT_IMPLEMENTED,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "M24"
            write_method_outputs(output_dir, get_method_spec("M24"), record, config={})

            self.assertFalse((output_dir / "training_history.csv").exists())
            hashes = json.loads((output_dir / "artifact_hashes.json").read_text(encoding="utf-8"))
            self.assertNotIn("training_history.csv", hashes["files"])

    def test_writer_removes_stale_history_when_a_later_scaffold_omits_it(self):
        completed_record = MethodRunRecord(
            method_id="M02",
            phase=RunPhase.SCREEN,
            status=RunStatus.COMPLETED,
            primary_metric_value=0.2,
            baseline_metric_value=0.3,
        )
        scaffold_record = MethodRunRecord(
            method_id="M02",
            phase=RunPhase.SCREEN,
            status=RunStatus.NOT_IMPLEMENTED,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "M02"
            write_method_outputs(
                output_dir,
                get_method_spec("M02"),
                completed_record,
                config={},
                training_history=[{"epoch": 1, "loss": 0.2}],
            )
            self.assertTrue((output_dir / "training_history.csv").exists())

            write_method_outputs(output_dir, get_method_spec("M02"), scaffold_record, config={})

            self.assertFalse((output_dir / "training_history.csv").exists())
            hashes = json.loads((output_dir / "artifact_hashes.json").read_text(encoding="utf-8"))
            self.assertNotIn("training_history.csv", hashes["files"])

    def test_completed_method_requiring_history_rejects_missing_or_empty_history(self):
        record = MethodRunRecord(
            method_id="M02",
            phase=RunPhase.SCREEN,
            status=RunStatus.COMPLETED,
            primary_metric_value=0.2,
            baseline_metric_value=0.3,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "training_history"):
                write_method_outputs(Path(temp_dir), get_method_spec("M02"), record, config={})
            with self.assertRaisesRegex(ValueError, "training_history"):
                write_method_outputs(Path(temp_dir), get_method_spec("M02"), record, config={}, training_history=[])


class CampaignCliTests(unittest.TestCase):
    def test_list_methods_and_verify_all_expose_the_campaign_contract(self):
        list_result = self.run_cli("--list-methods")
        listed = json.loads(list_result.stdout)
        self.assertEqual([row["method_id"] for row in listed], [f"M{index:02d}" for index in range(1, 25)])

        verify_result = self.run_cli("--verify-all")
        self.assertEqual(json.loads(verify_result.stdout)["methods"], 24)

    def test_normal_unimplemented_run_fails_without_writing_evidence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self.run_cli("--method", "M01", "--phase", "screen", "--output-root", temp_dir, check=False)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("not implemented", result.stderr.lower())
            self.assertFalse((Path(temp_dir) / "M01").exists())

    def test_dry_run_config_writes_only_a_not_implemented_scaffold(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "dry_run.json"
            config_path.write_text(json.dumps({"dry_run": True, "label": "contract"}), encoding="utf-8")
            output_root = Path(temp_dir) / "outputs"
            result = self.run_cli(
                "--method",
                "M01",
                "--phase",
                "screen",
                "--output-root",
                str(output_root),
                "--config",
                str(config_path),
            )

            payload = json.loads(result.stdout)
            self.assertEqual(payload["status"], "not_implemented")
            self.assertTrue((output_root / "M01" / "decision.json").exists())

    @staticmethod
    def run_cli(*arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["/Users/liangzihan/Documents/SURF_0159/SURF/.worktrees/american-option-risk-surfaces/.venv-pilot/bin/python", str(CAMPAIGN_SCRIPT), *arguments],
            cwd=REPOSITORY_ROOT,
            text=True,
            capture_output=True,
            check=check,
        )
