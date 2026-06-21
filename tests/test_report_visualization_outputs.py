import csv
import importlib.util
from pathlib import Path
import uuid
import unittest

import yaml


_SPEC = importlib.util.spec_from_file_location(
    "stage16_visualization",
    Path("experiments/19_report_visualization_enrichment.py"),
)
visualization = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(visualization)


class ReportVisualizationOutputsTest(unittest.TestCase):
    def test_dry_run_writes_figures_and_manifest(self):
        root = Path("results/test_stage16_visualization_outputs") / uuid.uuid4().hex
        root.mkdir(parents=True)
        config = root / "config.yaml"
        with config.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(
                {
                    "output_dir": str(root / "figures"),
                    "reports": [
                        {
                            "path": "reports/final_report.tex",
                            "figures": [
                                {
                                    "figure_id": "test_workflow",
                                    "kind": "workflow",
                                    "title": "Workflow",
                                    "caption": "Claim-safe workflow figure.",
                                    "source_artifact": "results/tables/final_claim_matrix.csv",
                                },
                                {
                                    "figure_id": "test_bar",
                                    "kind": "bar_from_csv",
                                    "title": "Price RMSE",
                                    "caption": "Evidence-linked metric figure.",
                                    "source_artifact": "results/experiment_13_integrated_application_workflow/integrated_by_regime.csv",
                                    "x": "case_id",
                                    "y": "price_rmse",
                                },
                            ],
                        }
                    ],
                },
                fh,
            )

        summary = visualization.run(str(config))

        self.assertEqual(summary["reports"], 1)
        self.assertEqual(summary["figures"], 2)
        manifest = root / "figures" / "figure_manifest.csv"
        with manifest.open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(Path(row["png_path"]).exists() for row in rows))
        self.assertTrue(all(Path(row["pdf_path"]).exists() for row in rows))


if __name__ == "__main__":
    unittest.main()
