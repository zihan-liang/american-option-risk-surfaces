from pathlib import Path
import re
import unittest


EXPANDED_DIR = Path("reports/expanded")
STAGE01 = EXPANDED_DIR / "stage01_math_notes_expanded.tex"
STAGE02 = EXPANDED_DIR / "stage02_solver_validation_expanded.tex"
FINAL = EXPANDED_DIR / "final_report_expanded_math.tex"
NOTES = Path("docs/expanded_math_derivation_notes.md")


def expanded_source(main_path: Path) -> str:
    text = main_path.read_text(encoding="utf-8")
    for input_path in re.findall(r"\\input\{([^}]+)\}", text):
        included = Path(f"{input_path}.tex") if not input_path.endswith(".tex") else Path(input_path)
        if included.exists():
            text += "\n" + included.read_text(encoding="utf-8")
    return text


class ExpandedMathReportsTest(unittest.TestCase):
    def test_required_outputs_and_shared_sections_exist(self):
        required = [
            NOTES,
            STAGE01,
            STAGE02,
            FINAL,
            EXPANDED_DIR / "expanded_preamble.tex",
            EXPANDED_DIR / "sections" / "stage01_foundations.tex",
            EXPANDED_DIR / "sections" / "stage02_discretization.tex",
            EXPANDED_DIR / "sections" / "stage03_to_stage11_learning.tex",
        ]
        for path in required:
            self.assertTrue(path.exists(), path)

    def test_stage01_teaches_the_obstacle_and_continuation_premium(self):
        text = expanded_source(STAGE01)
        for phrase in [
            r"\tau=T-t",
            r"U(S,\tau)\geq\Phi(S)",
            r"C(S,\tau)=U(S,\tau)-\Phi(S)",
            r"\begin{readerintuition}",
            r"\begin{commonconfusion}",
        ]:
            self.assertIn(phrase, text)

    def test_stage02_contains_the_full_matrix_and_psor_derivation(self):
        text = expanded_source(STAGE02)
        required_fragments = [
            r"S_i=i\Delta S",
            r"\tau_n=n\Delta\tau",
            r"U^n=(U_1^n,U_2^n,U_3^n)^\top",
            r"a_iU_{i-1}+b_iU_i+c_iU_{i+1}",
            r"A=I-\frac{\Delta\tau}{2}L_h",
            r"B=I+\frac{\Delta\tau}{2}L_h",
            r"x_i^{\mathrm{new}}=\max",
            r"a_1U_0^{n+1}",
            r"c_{M-1}U_M^{n+1}",
        ]
        for fragment in required_fragments:
            self.assertIn(fragment, text)

    def test_final_report_covers_stage03_through_stage11_learning_topics(self):
        text = expanded_source(FINAL)
        for heading in [
            "Stage 3: Boundary Extraction and Greeks",
            "Stage 4: Grid, Domain, and Runtime",
            "Stages 5--6: Risk Surfaces and Dataset Governance",
            "Stages 7--8: Direct Price and Positive Premium",
            "Stages 9--10: Boundary and Delta Diagnostics",
            "Stage 11: Integrated Scenario Screening",
        ]:
            self.assertIn(heading, text)
        for statement in [
            r"\widehat U=\Phi+\widehat C",
            r"\Delta_{\mathrm{put}}\in[-1,0]",
            "regime-level split",
            "six deterministic held-out regimes",
            "does not support a Gamma surrogate",
        ]:
            self.assertIn(statement, text)

    def test_expanded_reports_preserve_claim_scope_and_display_precision(self):
        forbidden_positive_claims = [
            r"(?<!does not )replace(?:s)? CN/PSOR",
            r"(?<!does not )replace(?:s)? PSOR",
            r"one network solves price, boundary, and Greeks",
            r"validated Gamma surrogate",
            r"publication-grade convergence",
        ]
        excessive_precision = re.compile(r"(?<![A-Za-z0-9_])0\.\d{7,}(?![A-Za-z0-9_])")
        for path in [STAGE01, STAGE02, FINAL]:
            text = expanded_source(path)
            self.assertEqual([], excessive_precision.findall(text), path)
            for pattern in forbidden_positive_claims:
                self.assertNotRegex(text, pattern, path)

    def test_existing_concise_reports_are_not_consumed_as_expanded_sources(self):
        for path in [STAGE01, STAGE02, FINAL]:
            text = path.read_text(encoding="utf-8")
            self.assertNotIn(r"\input{reports/stage", text)
            self.assertNotIn(r"\input{reports/final_report", text)


if __name__ == "__main__":
    unittest.main()
