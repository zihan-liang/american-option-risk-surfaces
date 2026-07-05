from pathlib import Path
import unittest


class PersonalBackgroundPositioningTest(unittest.TestCase):
    def test_final_report_frames_student_as_ai_computational_contributor(self):
        text = Path("reports/final_report.tex").read_text(encoding="utf-8").lower()
        for phrase in ["ai/ml", "plotting", "reproducibility", "latex", "experiment"]:
            self.assertIn(phrase, text)
        self.assertNotIn("sole financial-mathematics theorist", text)
        self.assertNotIn("pure financial mathematics expert", text)

    def test_student_contribution_figure_exists(self):
        self.assertTrue(Path("results/figures/report_enrichment/final_report_student_contribution_focus.pdf").exists())


if __name__ == "__main__":
    unittest.main()
