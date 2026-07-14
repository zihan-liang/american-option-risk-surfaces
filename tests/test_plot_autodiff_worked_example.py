"""Regression checks for the concise automatic-differentiation example."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts/plot_autodiff_worked_example.py"


def load_module():
    spec = importlib.util.spec_from_file_location("autodiff_worked_example", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AutodiffWorkedExampleTests(unittest.TestCase):
    def test_worked_example_contains_the_approved_graph_and_local_derivatives(self) -> None:
        self.assertTrue(SCRIPT_PATH.is_file())
        module = load_module()

        self.assertEqual(
            module.NODE_LABELS,
            {
                "x": r"$x=2$",
                "a": r"$a=x^2=4$",
                "b": r"$b=3x=6$",
                "c": r"$c=a+b=10$",
                "f": r"$f=\sin(c)=\sin(10)$",
            },
        )
        self.assertEqual(
            module.LOCAL_DERIVATIVES,
            {
                ("x", "a"): r"$4$",
                ("x", "b"): r"$3$",
                ("a", "c"): r"$1$",
                ("b", "c"): r"$1$",
                ("c", "f"): r"$\cos(10)$",
            },
        )
        self.assertEqual(module.FINAL_GRADIENT, r"$\dfrac{df}{dx}=7\cos(10)$")
