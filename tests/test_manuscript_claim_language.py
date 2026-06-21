import importlib.util
from pathlib import Path
import unittest


_SPEC = importlib.util.spec_from_file_location(
    "stage15_publication_audit_language",
    Path("experiments/17_publication_evidence_audit.py"),
)
publication_audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(publication_audit)


class ManuscriptClaimLanguageTest(unittest.TestCase):
    def test_scan_allows_restricted_context_but_flags_positive_overclaims(self):
        safe = "The workflow does not replace PSOR and is not a Greek engine."
        unsafe = "The workflow replaces PSOR as a universal American option solver."

        rows = publication_audit.scan_claim_language(
            {"safe.tex": safe, "unsafe.tex": unsafe},
            ["replaces PSOR", "universal American option solver", "Greek engine"],
        )

        unsafe_rows = [row for row in rows if row["file"] == "unsafe.tex"]
        safe_rows = [row for row in rows if row["file"] == "safe.tex"]
        self.assertTrue(any(row["status"] == "violation" for row in unsafe_rows))
        self.assertTrue(safe_rows)
        self.assertTrue(all(row["status"] == "allowed_restricted_context" for row in safe_rows))

    def test_manuscript_and_supplement_have_no_positive_forbidden_claims(self):
        files = [
            Path("paper/manuscript.tex"),
            Path("paper/supplementary_material.tex"),
            Path("reports/final_report.tex"),
        ]
        for path in files:
            self.assertTrue(path.exists(), path)
        rows = publication_audit.scan_files_for_claim_language(
            files,
            [
                "replaces PSOR",
                "one network solves",
                "Greek engine",
                "publication-grade convergence",
                "Gamma surrogate",
                "universal American option solver",
            ],
        )
        violations = [row for row in rows if row["status"] == "violation"]
        self.assertEqual([], violations)


if __name__ == "__main__":
    unittest.main()
