import unittest
from pathlib import Path


class ReadmeReleaseHandoffTests(unittest.TestCase):
    def test_readme_states_final_boundaries_and_reproducibility(self):
        text = Path("README.md").read_text(encoding="utf-8").lower()

        self.assertIn("final project status", text)
        self.assertIn("psor audit", text)
        self.assertIn("does not replace psor", text)
        self.assertIn("does not claim that one network solves", text)
        self.assertIn("release verification", text)

    def test_release_handoff_has_verification_and_continuation_guidance(self):
        text = Path("docs/release_handoff.md").read_text(encoding="utf-8").lower()

        self.assertIn("what is complete", text)
        self.assertIn("what is not claimed", text)
        self.assertIn("how to verify", text)
        self.assertIn("how to continue", text)
        self.assertIn("python experiments/16_release_verification.py", text)


if __name__ == "__main__":
    unittest.main()
