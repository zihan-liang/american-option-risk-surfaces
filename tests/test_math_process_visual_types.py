from pathlib import Path
import unittest
import uuid

import plotting.report_visuals as report_visuals
from plotting.report_visuals import save_visual


class MathProcessVisualTypesTest(unittest.TestCase):
    def test_matrix_process_visual_types_generate_png_and_pdf(self):
        visual_kinds = [
            "fd_stencil",
            "operator_coefficients",
            "operator_matrix_annotated",
            "cn_matrix_pair",
            "rhs_boundary_adjustment",
            "psor_update_flow",
            "lcp_vector_form",
            "time_marching_grid",
        ]
        for kind in visual_kinds:
            self.assertTrue(
                hasattr(report_visuals, f"save_{kind}"),
                f"{kind} must have an explicit plotting function",
            )
        root = Path("results/test_stage17_math_process_visuals") / uuid.uuid4().hex
        for kind in visual_kinds:
            spec = {
                "kind": kind,
                "title": f"Test {kind}",
                "caption": "Teaching figure for matrix calculations.",
                "source_artifact": "src/solvers/cn_psor.py",
            }
            for suffix in [".png", ".pdf"]:
                out = root / f"{kind}{suffix}"
                save_visual(spec, out)
                self.assertTrue(out.exists(), f"{kind} did not create {suffix}")
                self.assertGreater(out.stat().st_size, 1000, f"{kind} {suffix} is unexpectedly small")


if __name__ == "__main__":
    unittest.main()
