import csv
from pathlib import Path
import unittest


FORBIDDEN_POSITIVE = [
    "surrogate replaces psor",
    "workflow replaces psor",
    "one network solves",
    "one neural network solves",
    "gamma surrogate",
    "better than the benchmark",
]

FORBIDDEN_MATH_PROCESS_CLAIMS = [
    "new validation evidence",
    "improves solver accuracy",
    "proves convergence",
    "publication-grade solver",
]


class VisualClaimLanguageTest(unittest.TestCase):
    def test_manifest_captions_do_not_overclaim(self):
        manifest = Path("results/figures/report_enrichment/figure_manifest.csv")
        self.assertTrue(manifest.exists())
        with manifest.open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        for row in rows:
            caption = row["caption"].lower()
            for phrase in FORBIDDEN_POSITIVE:
                self.assertNotIn(phrase, caption, row["figure_id"])

    def test_math_process_captions_are_explanatory_not_new_evidence(self):
        manifest = Path("results/figures/report_enrichment/figure_manifest.csv")
        self.assertTrue(manifest.exists())
        with manifest.open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        math_rows = [
            row for row in rows
            if row["figure_id"].startswith("stage01_math_notes_")
            or row["figure_id"].startswith("stage02_solver_validation_")
        ]
        self.assertGreaterEqual(len(math_rows), 20)
        for row in math_rows:
            caption = row["caption"].lower()
            for phrase in FORBIDDEN_MATH_PROCESS_CLAIMS:
                self.assertNotIn(phrase, caption, row["figure_id"])


if __name__ == "__main__":
    unittest.main()
