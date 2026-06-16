import csv
import importlib.util
import json
import sys
import unittest
from pathlib import Path

import yaml

_EXPERIMENT_PATH = Path("experiments/14_claim_audit.py")
_SPEC = importlib.util.spec_from_file_location("stage12_claim_audit", _EXPERIMENT_PATH)
experiment = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = experiment
assert _SPEC.loader is not None
_SPEC.loader.exec_module(experiment)


class ClaimAuditOutputTests(unittest.TestCase):
    def test_run_writes_claim_matrix_summary_scan_and_missing_artifacts(self):
        root = Path("results/test_stage12_claim_audit")
        source = root / "integrated_summary.json"
        scan_file = root / "safe_report.tex"
        existing_artifact = root / "existing.csv"
        missing_artifact = root / "missing.csv"
        output_dir = root / "out"
        final_matrix = root / "tables" / "final_claim_matrix.csv"
        config_path = root / "claim_audit.yaml"
        root.mkdir(parents=True, exist_ok=True)
        source.write_text(
            json.dumps(
                {
                    "decision": "scenario_screening_with_psor_audit",
                    "mean_price_rmse": 0.01,
                    "mean_boundary_mae": 0.15,
                    "claims_psor_replacement": False,
                }
            ),
            encoding="utf-8",
        )
        scan_file.write_text("The workflow does not replace PSOR.\n", encoding="utf-8")
        existing_artifact.write_text("ok\n", encoding="utf-8")
        config_path.write_text(
            yaml.safe_dump(
                {
                    "output_dir": str(output_dir),
                    "final_claim_matrix": str(final_matrix),
                    "claims": [
                        {
                            "claim_id": "integrated_workflow",
                            "claim": "Integrated workflow supports scenario screening with PSOR audit.",
                            "evidence": str(source),
                            "status": "scoped_screening",
                            "allowed_wording": "scenario screening with PSOR audit",
                            "restricted_wording": "workflow replaces PSOR",
                            "limitations": "Boundary estimates remain the weakest component.",
                            "metrics": {
                                "mean_price_rmse": "mean_price_rmse",
                                "mean_boundary_mae": "mean_boundary_mae",
                            },
                        }
                    ],
                    "forbidden_phrases": ["replaces PSOR", "one network solves price boundary and Greeks"],
                    "scan_files": [str(scan_file)],
                    "required_artifacts": [str(existing_artifact), str(missing_artifact)],
                }
            ),
            encoding="utf-8",
        )

        summary = experiment.run(str(config_path))

        self.assertEqual(summary["claims"], 1)
        self.assertEqual(summary["forbidden_language_violations"], 0)
        self.assertEqual(summary["missing_artifacts"], 1)
        for name in [
            "claim_matrix.csv",
            "claim_audit_summary.json",
            "forbidden_language_scan.csv",
            "missing_artifacts.csv",
        ]:
            self.assertTrue((output_dir / name).exists(), name)
        self.assertTrue(final_matrix.exists())
        with open(final_matrix, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(rows[0]["claim_id"], "integrated_workflow")
        self.assertEqual(rows[0]["metric_mean_price_rmse"], "0.01")


if __name__ == "__main__":
    unittest.main()
