import csv
import importlib.util
import json
import sys
import unittest
from pathlib import Path

_EXPERIMENT_PATH = Path("experiments/03_convergence_domain_runtime.py")
_SPEC = importlib.util.spec_from_file_location("stage04_convergence", _EXPERIMENT_PATH)
experiment = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = experiment
_SPEC.loader.exec_module(experiment)


class ConvergenceExperimentOutputTests(unittest.TestCase):
    def test_small_run_writes_stage04_outputs_with_expected_columns(self):
        out_dir = Path("results/test_stage04_convergence")
        fig_dir = Path("results/figures/test_stage04_convergence")
        summary = experiment.run(
            "configs/convergence.yaml",
            output_root=out_dir,
            figure_root=fig_dir,
            overrides={"grids": [20, 30], "domains": [3.0, 4.0]},
        )

        self.assertEqual(summary["cases"], 2)
        for filename in [
            "grid_convergence.csv",
            "domain_sensitivity.csv",
            "runtime_table.csv",
            "diagnostics_table.csv",
            "convergence_summary.json",
        ]:
            self.assertTrue((out_dir / filename).exists(), filename)

        with open(out_dir / "grid_convergence.csv", newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        self.assertIn("price_rmse", rows[0])
        self.assertIn("boundary_mae", rows[0])
        self.assertIn("delta_strict_rmse", rows[0])
        self.assertIn("gamma_strict_rmse", rows[0])

        with open(out_dir / "convergence_summary.json", encoding="utf-8") as fh:
            data = json.load(fh)
        self.assertIn("max_obstacle_violation", data)
