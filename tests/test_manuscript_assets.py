import re
from pathlib import Path
import unittest


class ManuscriptAssetsTest(unittest.TestCase):
    def test_manuscript_referenced_figures_exist_and_are_manifested(self):
        manuscript = Path("paper/manuscript.tex")
        manifest = Path("results/figures/publication/figure_manifest.csv")
        self.assertTrue(manuscript.exists())
        self.assertTrue(manifest.exists())
        text = manuscript.read_text(encoding="utf-8")
        manifest_text = manifest.read_text(encoding="utf-8")
        figures = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text)
        self.assertGreaterEqual(len(figures), 6)
        for figure in figures:
            figure_path = Path(figure)
            self.assertTrue(figure_path.exists(), figure)
            self.assertIn(figure_path.name, manifest_text)

    def test_supplement_and_final_report_point_to_manuscript_package(self):
        supplement = Path("paper/supplementary_material.tex")
        final_report = Path("reports/final_report.tex")
        self.assertTrue(supplement.exists())
        self.assertIn("publication evidence audit", supplement.read_text(encoding="utf-8").lower())
        report_text = final_report.read_text(encoding="utf-8").lower()
        self.assertIn("manuscript", report_text)
        self.assertIn("supplement", report_text)


if __name__ == "__main__":
    unittest.main()
