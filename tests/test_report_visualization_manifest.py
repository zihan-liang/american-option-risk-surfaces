import csv
from pathlib import Path
import unittest

import yaml


class ReportVisualizationManifestTest(unittest.TestCase):
    def test_config_plans_figures_for_every_report(self):
        config_path = Path("configs/report_visualization_enrichment.yaml")
        self.assertTrue(config_path.exists())
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        reports = cfg["reports"]
        self.assertGreaterEqual(len(reports), 14)
        for report in reports:
            self.assertTrue(Path(report["path"]).exists(), report["path"])
            self.assertGreaterEqual(len(report["figures"]), 1, report["path"])
            for figure in report["figures"]:
                self.assertTrue(figure["figure_id"])
                self.assertTrue(figure["caption"])
                self.assertTrue(figure["source_artifact"])
                if figure.get("kind") == "bar_from_csv":
                    source = Path(figure["source_artifact"])
                    if source.exists():
                        with source.open(encoding="utf-8", newline="") as fh:
                            reader = csv.DictReader(fh)
                            self.assertIn(figure["x"], reader.fieldnames or [], figure["figure_id"])
                            self.assertIn(figure["y"], reader.fieldnames or [], figure["figure_id"])
        by_path = {report["path"]: report for report in reports}
        self.assertGreaterEqual(len(by_path["reports/stage01_math_notes.tex"]["figures"]), 10)
        self.assertGreaterEqual(len(by_path["reports/stage02_solver_validation.tex"]["figures"]), 10)

    def test_manifest_has_required_fields_and_each_report(self):
        manifest_path = Path("results/figures/report_enrichment/figure_manifest.csv")
        self.assertTrue(manifest_path.exists())
        with manifest_path.open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        self.assertGreaterEqual(len(rows), 70)
        reports = {row["report_path"] for row in rows}
        expected = {
            "reports/stage01_math_notes.tex",
            "reports/stage02_solver_validation.tex",
            "reports/stage03_boundary_greeks_diagnostics.tex",
            "reports/stage04_grid_domain_runtime.tex",
            "reports/stage05_application_risk_surfaces.tex",
            "reports/stage06_surrogate_dataset.tex",
            "reports/stage07_baseline_surrogates.tex",
            "reports/stage08_positive_premium_surrogate.tex",
            "reports/stage09_boundary_head.tex",
            "reports/stage10_delta_validation_head.tex",
            "reports/stage11_integrated_workflow.tex",
            "reports/stage13_future_extensions.tex",
            "reports/final_report.tex",
            "paper/supplementary_material.tex",
        }
        self.assertTrue(expected.issubset(reports))
        for row in rows:
            self.assertTrue(row["caption"])
            self.assertTrue(row["source_artifact"])
            self.assertTrue(Path(row["png_path"]).exists(), row["png_path"])
            self.assertTrue(Path(row["pdf_path"]).exists(), row["pdf_path"])

    def test_stage02_image_audit_covers_all_report_figures(self):
        audit_path = Path("results/figures/report_enrichment/stage02_solver_validation_image_audit.csv")
        self.assertTrue(audit_path.exists())
        with audit_path.open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), 12)
        self.assertEqual({row["report_path"] for row in rows}, {"reports/stage02_solver_validation.tex"})
        self.assertTrue(all(row["plotting_function"].startswith("save_") for row in rows))
        self.assertTrue(all(row["source_artifact"] for row in rows))
        self.assertTrue(all(row["visual_status"].startswith("pass") for row in rows))
        self.assertTrue(all(row["dpi_status"] == "pass_after_300dpi_regeneration" for row in rows))


if __name__ == "__main__":
    unittest.main()
