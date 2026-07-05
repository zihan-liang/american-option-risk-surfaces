from pathlib import Path
import re
import unittest


REQUIRED_STAGE_SUBSECTIONS = [
    "Purpose",
    "Method",
    "Mathematical Formulation",
    "Implementation Details",
    "Validation and Diagnostics",
    "Results",
    "Failure Modes and Limitations",
    "Conclusion and Next Step",
]

PAPER_FACING_FILES = [
    Path("reports/final_report.tex"),
    Path("paper/manuscript.tex"),
    Path("paper/manuscript_outline.tex"),
    Path("paper/supplementary_material.tex"),
]


class ReportEditorialConsistencyTest(unittest.TestCase):
    def test_style_and_metric_sources_exist(self):
        self.assertTrue(Path("docs/report_style_guide.md").exists())
        self.assertTrue(Path("docs/verified_metric_ledger.md").exists())

    def test_stage_reports_preserve_required_structure(self):
        for path in sorted(Path("reports").glob("stage*.tex")):
            text = path.read_text(encoding="utf-8")
            for title in REQUIRED_STAGE_SUBSECTIONS:
                self.assertIn(f"\\subsection{{{title}}}", text, f"{path}: {title}")

    def test_paper_facing_sources_do_not_print_machine_precision_decimals(self):
        excessive_precision = re.compile(r"(?<![A-Za-z0-9_])0\.\d{7,}(?![A-Za-z0-9_])")
        for path in PAPER_FACING_FILES:
            text = path.read_text(encoding="utf-8")
            matches = excessive_precision.findall(text)
            self.assertEqual([], matches, f"{path}: {matches[:5]}")

    def test_continuation_premium_notation_is_consistent(self):
        files = [
            Path("reports/stage01_math_notes.tex"),
            Path("reports/stage03_boundary_greeks_diagnostics.tex"),
            Path("reports/stage06_surrogate_dataset.tex"),
            Path("reports/stage08_positive_premium_surrogate.tex"),
            Path("paper/manuscript.tex"),
            Path("reports/final_report.tex"),
        ]
        for path in files:
            text = path.read_text(encoding="utf-8")
            self.assertNotRegex(
                text,
                r"\bP\(S,\\tau\)\s*=\s*U\(S,\\tau\)-\\Phi\(S\)",
                f"{path}: use C for continuation premium",
            )


if __name__ == "__main__":
    unittest.main()
