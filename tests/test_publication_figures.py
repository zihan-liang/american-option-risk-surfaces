import csv
import importlib.util
import shutil
import uuid
from pathlib import Path
import unittest

import yaml


_SPEC = importlib.util.spec_from_file_location(
    "stage15_publication_figures",
    Path("experiments/18_publication_figures.py"),
)
publication_figures = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(publication_figures)


class PublicationFiguresTest(unittest.TestCase):
    def test_run_writes_required_png_pdf_figures_and_manifest(self):
        root = Path("results") / f"test_stage15_publication_figures_{uuid.uuid4().hex}"
        root.mkdir(parents=True)
        config_path = root / "publication_figures.yaml"
        with config_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(
                {
                    "figure_dir": str(root / "figures"),
                    "publication_audit_dir": "results/experiment_17_publication_audit",
                    "source_figures": {
                        "solver_validation": "results/figures/stage04_convergence_runtime/put/price_error_vs_grid.png",
                        "risk_surface": "results/figures/stage05_application_risk_surfaces/put/put_T0p25_sig0p2_r0p05_q0/price_surface.png",
                        "application_heatmap": "results/figures/stage05_application_risk_surfaces/put/exercise_fraction_heatmap.png",
                    },
                },
                fh,
            )

        summary = publication_figures.run(str(config_path))

        self.assertEqual(summary["figures"], 9)
        manifest_path = root / "figures" / "figure_manifest.csv"
        self.assertTrue(manifest_path.exists())
        with manifest_path.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), 9)
        for row in rows:
            self.assertTrue(Path(row["png_path"]).exists(), row["figure_id"])
            self.assertTrue(Path(row["pdf_path"]).exists(), row["figure_id"])
            self.assertTrue(row["source_artifact"])


if __name__ == "__main__":
    unittest.main()
