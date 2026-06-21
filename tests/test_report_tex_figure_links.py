import re
from pathlib import Path
import unittest


class ReportTexFigureLinksTest(unittest.TestCase):
    def _expanded_text(self, tex: Path) -> str:
        text = tex.read_text(encoding="utf-8")
        for include in re.findall(r"\\input\{([^}]+)\}", text):
            include_path = Path(include)
            if include_path.exists():
                text += "\n" + include_path.read_text(encoding="utf-8")
        return text

    def test_all_report_includegraphics_paths_exist(self):
        tex_files = list(Path("reports").glob("stage*.tex")) + [
            Path("reports/final_report.tex"),
            Path("paper/manuscript.tex"),
            Path("paper/supplementary_material.tex"),
        ]
        checked = 0
        for tex in tex_files:
            if not tex.exists():
                continue
            text = self._expanded_text(tex)
            for match in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text):
                checked += 1
                self.assertTrue(Path(match).exists(), f"{tex}: {match}")
        self.assertGreaterEqual(checked, 45)


if __name__ == "__main__":
    unittest.main()
