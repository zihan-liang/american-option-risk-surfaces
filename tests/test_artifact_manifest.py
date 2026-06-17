import unittest
from pathlib import Path


class ArtifactManifestTests(unittest.TestCase):
    def test_manifest_separates_source_controlled_and_ignored_artifacts(self):
        text = Path("docs/artifact_manifest.md").read_text(encoding="utf-8").lower()

        self.assertIn("source-controlled artifacts", text)
        self.assertIn("ignored local artifacts", text)
        self.assertIn(".csv", text)
        self.assertIn(".json", text)
        self.assertIn(".tex", text)
        self.assertIn(".pt", text)
        self.assertIn(".npz", text)
        self.assertIn("generated figures", text)
        self.assertIn("build pdf", text)


if __name__ == "__main__":
    unittest.main()
