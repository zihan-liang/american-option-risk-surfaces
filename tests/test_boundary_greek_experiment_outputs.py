import json
import importlib.util
import unittest
from pathlib import Path

import numpy as np

_EXPERIMENT_PATH = Path("experiments/02_boundary_greeks_diagnostics.py")
_SPEC = importlib.util.spec_from_file_location("stage03_boundary_greeks", _EXPERIMENT_PATH)
experiment = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(experiment)


class BoundaryGreekExperimentOutputTests(unittest.TestCase):
    def test_small_run_writes_required_stage_outputs(self):
        out_dir = Path("results/test_stage03_boundary_outputs")
        summary = experiment.run_many(
            ["configs/base_put.yaml"],
            output_root=out_dir,
            figure_root=Path("results/figures/test_stage03_boundary_outputs"),
            overrides={"m": 24, "n": 24, "tol": 1e-8},
        )

        self.assertEqual(summary["cases"], 1)
        self.assertTrue((out_dir / "put" / "boundary_curve.csv").exists())
        self.assertTrue((out_dir / "put" / "greek_summary.csv").exists())
        self.assertTrue((out_dir / "put" / "mask_summary.csv").exists())
        self.assertTrue((out_dir / "put" / "surface_greeks.npz").exists())
        self.assertTrue((out_dir / "combined_summary.json").exists())

        with open(out_dir / "put" / "diagnostics_summary.json", encoding="utf-8") as fh:
            diagnostics = json.load(fh)
        self.assertIn("strict_mask_fraction", diagnostics)

        arrays = np.load(out_dir / "put" / "surface_greeks.npz")
        for key in ["spots", "taus", "values", "delta", "gamma", "mask_strict", "mask_near_kink", "mask_near_boundary"]:
            self.assertIn(key, arrays.files)
