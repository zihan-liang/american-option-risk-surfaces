import csv
import importlib.util
import json
import shutil
import uuid
from pathlib import Path
import unittest

import yaml


_SPEC = importlib.util.spec_from_file_location(
    "stage15_publication_audit",
    Path("experiments/17_publication_evidence_audit.py"),
)
publication_audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(publication_audit)


class PublicationAuditOutputsTest(unittest.TestCase):
    def test_run_writes_publication_gap_and_evidence_tables(self):
        root = Path("results") / f"test_stage15_publication_audit_{uuid.uuid4().hex}"
        root.mkdir(parents=True)
        config_path = root / "publication_evidence.yaml"
        with config_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(
                {
                    "output_dir": str(root / "out"),
                    "target_integrated_test_regimes": 34,
                    "claim_matrix": "results/tables/final_claim_matrix.csv",
                    "integrated_summary": "results/experiment_13_integrated_application_workflow/integrated_summary.json",
                    "integrated_by_regime": "results/experiment_13_integrated_application_workflow/integrated_by_regime.csv",
                    "split_table": "results/experiment_05_surrogate_dataset/split_table.csv",
                    "baseline_metrics": "results/experiment_06_surrogate_baselines/baseline_metrics.csv",
                    "positive_premium_metrics": "results/experiment_07_positive_premium_tuning/positive_premium_metrics.csv",
                    "delta_validation_summary": "results/experiment_11_delta_validation/delta_validation_summary.json",
                    "delta_head_summary": "results/experiment_12_supervised_delta_head/delta_head_summary.json",
                    "boundary_by_regime": "results/experiment_09_boundary_head/boundary_by_regime.csv",
                    "call_boundary_by_regime": "results/experiment_10_option_specific_call_boundary/call_boundary_by_regime.csv",
                    "convergence_summary": "results/experiment_03_convergence_domain_runtime/convergence_summary.json",
                    "application_summary": "results/experiment_04_application_risk_surfaces/application_summary.json",
                },
                fh,
            )

        summary = publication_audit.run(str(config_path))

        self.assertEqual(summary["decision"], "journal_style_with_scoped_claims")
        self.assertEqual(summary["target_integrated_test_regimes"], 34)
        self.assertIn("integrated_workflow", summary["demoted_or_limited_claims"])
        for name in [
            "publication_gap_matrix.csv",
            "publication_audit_summary.json",
            "integrated_scope_summary.csv",
            "baseline_comparison.csv",
            "boundary_failure_analysis.csv",
            "numerical_benchmark_summary.csv",
        ]:
            self.assertTrue((root / "out" / name).exists(), name)

        with (root / "out" / "publication_gap_matrix.csv").open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        by_claim = {row["claim_id"]: row for row in rows}
        self.assertEqual(by_claim["boundary_head"]["readiness"], "supplement_only")
        self.assertEqual(by_claim["integrated_workflow"]["readiness"], "supplement_only")


if __name__ == "__main__":
    unittest.main()
