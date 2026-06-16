import importlib.util
import csv
import json
import sys
import unittest
from pathlib import Path

import numpy as np

_EXPERIMENT_PATH = Path("experiments/04_application_risk_surfaces.py")
_SPEC = importlib.util.spec_from_file_location("stage05_application", _EXPERIMENT_PATH)
experiment = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = experiment
assert _SPEC.loader is not None
_SPEC.loader.exec_module(experiment)


class ApplicationSweepOutputTests(unittest.TestCase):
    def test_small_run_writes_required_outputs_and_separates_regimes(self):
        out_dir = Path("results/test_stage05_application_clean")
        fig_dir = Path("results/figures/test_stage05_application_clean")
        summary = experiment.run(
            "configs/application_sweep.yaml",
            output_root=out_dir,
            figure_root=fig_dir,
            overrides={
                "m": 24,
                "n": 24,
                "t_values": [0.25],
                "sigma_values": [0.2],
                "r_values": [0.05],
                "q_values": [0.0, 0.06],
                "representative_limit": 2,
                "tol": 1e-8,
            },
        )

        self.assertEqual(summary["total_cases"], 4)
        for name in [
            "risk_metrics.csv",
            "boundary_summary.csv",
            "greek_summary.csv",
            "regime_flags.csv",
            "surface_index.csv",
            "application_summary.json",
            "application_config_manifest.json",
        ]:
            self.assertTrue((out_dir / name).exists(), name)

        with open(out_dir / "application_summary.json", encoding="utf-8") as fh:
            loaded = json.load(fh)
        self.assertEqual(loaded["regime_counts"]["put"], 2)
        self.assertEqual(loaded["regime_counts"]["dividend_call"], 1)
        self.assertEqual(loaded["regime_counts"]["no_dividend_call_control"], 1)

        risk_header = (out_dir / "risk_metrics.csv").read_text(encoding="utf-8").splitlines()[0]
        for column in [
            "regime_class",
            "obstacle_violation",
            "complementarity_error",
            "monotonicity_violation",
            "convexity_violation",
            "runtime_seconds",
            "max_psor_iterations",
        ]:
            self.assertIn(column, risk_header)

        greek_header = (out_dir / "greek_summary.csv").read_text(encoding="utf-8").splitlines()[0]
        self.assertIn("strict_samples", greek_header)
        self.assertIn("gamma_strict_mean_abs", greek_header)

        with open(out_dir / "boundary_summary.csv", newline="", encoding="utf-8") as fh:
            boundary_rows = list(csv.DictReader(fh))
        control_rows = [row for row in boundary_rows if row["regime_class"] == "no_dividend_call_control"]
        self.assertEqual(control_rows[0]["boundary_found"], "0")
        self.assertEqual(control_rows[0]["boundary_samples"], "0")
        self.assertEqual(control_rows[0]["initial_boundary"], "nan")

        arrays = list((out_dir / "surfaces").rglob("*.npz"))
        self.assertGreaterEqual(len(arrays), 1)
        with np.load(arrays[0]) as loaded_npz:
            for key in ["spots", "taus", "values", "delta", "gamma", "boundary", "mask_strict"]:
                self.assertIn(key, loaded_npz.files)


if __name__ == "__main__":
    unittest.main()
