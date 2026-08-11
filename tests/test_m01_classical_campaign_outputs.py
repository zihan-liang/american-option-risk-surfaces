import csv
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from campaign.registry import get_method_spec
from common import load_config


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CONFIG_IDS = {
    "cn_psor",
    "rannacher_cn_psor",
    "dirk_psor",
    "cn_policy_iteration",
    "cn_penalty_newton",
}


def _load_experiment():
    path = ROOT / "experiments" / "24_classical_oracle.py"
    spec = importlib.util.spec_from_file_location("classical_oracle", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class M01ClassicalCampaignOutputTests(unittest.TestCase):
    def test_config_declares_five_one_factor_variants_without_order_claim(self):
        config = load_config(ROOT / "configs" / "independent_m01_classical.yaml")
        methods = {method["config_id"]: method for method in config["methods"]}

        self.assertEqual(set(methods), EXPECTED_CONFIG_IDS)
        self.assertEqual(set(methods), set(get_method_spec("M01").predeclared_configurations))
        self.assertEqual(config["grid_refinement"]["grids"], [40, 80, 120])
        self.assertEqual(config["smoke"].get("grid_refinement_grids"), [12, 24])
        self.assertEqual(methods["cn_psor"]["lcp_solver"], "psor")
        self.assertEqual(methods["rannacher_cn_psor"]["time_scheme"], "rannacher_cn")
        self.assertEqual(methods["dirk_psor"]["time_scheme"], "sdirk2")
        self.assertEqual(methods["cn_policy_iteration"]["lcp_solver"], "policy_iteration")
        self.assertEqual(methods["cn_penalty_newton"]["lcp_solver"], "penalty_newton")
        scope = get_method_spec("M01").scope
        self.assertNotIn("second-order", scope.lower())
        self.assertIn("L-stable obstacle discretization", scope)

    def test_smoke_run_writes_complete_finite_one_factor_diagnostics(self):
        experiment = _load_experiment()
        with tempfile.TemporaryDirectory() as temp_dir:
            summary = experiment.run(
                str(ROOT / "configs" / "independent_m01_classical.yaml"),
                output_root=temp_dir,
                smoke=True,
            )
            output = Path(temp_dir)
            required = {
                "config_manifest.json",
                "run_metrics.csv",
                "decision.json",
                "artifact_hashes.json",
                "solver_comparison.csv",
                "analytic_controls.csv",
                "grid_refinement.csv",
                "boundary_crosscheck.csv",
                "timing_samples.csv",
                "m01_summary.json",
            }
            self.assertTrue(required.issubset({path.name for path in output.iterdir()}))
            with (output / "solver_comparison.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({row["config_id"] for row in rows}, EXPECTED_CONFIG_IDS)
            self.assertEqual({row["comparison_factor"] for row in rows}, {
                "baseline",
                "time_scheme",
                "lcp_solver",
            })
            self.assertEqual(summary["run_mode"], "smoke_non_evidence")
            self.assertEqual(summary.get("evidence_status"), "diagnostic_non_evidence")
            self.assertEqual(summary["method_id"], "M01")
            self.assertFalse(summary["blind_holdout_accessed"])
            self.assertEqual(
                summary.get("grid_refinement_claim"),
                "diagnostic consistency check only; no convergence order is estimated",
            )
            decision = json.loads((output / "decision.json").read_text(encoding="utf-8"))
            self.assertEqual(decision["method_id"], "M01")
            self.assertEqual(decision["record_count"], 5)
            self.assertEqual(decision["status"], "diagnostic_only")
            self.assertEqual(decision["claim_status"], "smoke_non_evidence")
            with (output / "run_metrics.csv").open(newline="", encoding="utf-8") as handle:
                run_rows = list(csv.DictReader(handle))
            self.assertEqual(len(run_rows), 5)
            self.assertEqual({row["config_id"] for row in run_rows}, EXPECTED_CONFIG_IDS)
            self.assertEqual({row["status"] for row in run_rows}, {"diagnostic_only"})
            self.assertEqual({row["claim_status"] for row in run_rows}, {"smoke_non_evidence"})
            self.assertEqual({row["split"] for row in run_rows}, {"validation_controls"})
            self.assertEqual(len({row["branch"] for row in run_rows}), 1)
            self.assertEqual(len({row["commit"] for row in run_rows}), 1)
            self.assertTrue(run_rows[0]["branch"])
            self.assertTrue(run_rows[0]["commit"])
            manifest = json.loads((output / "config_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["provenance"],
                {"branch": run_rows[0]["branch"], "commit": run_rows[0]["commit"]},
            )
            self.assertEqual(manifest["config"]["evidence_status"], "diagnostic_non_evidence")
            self.assertFalse(manifest["config"]["blind_holdout_accessed"])
            for row in rows:
                for field in (
                    "price_rmse",
                    "delta_rmse",
                    "gamma_rmse",
                    "boundary_rmse",
                    "max_obstacle_violation",
                    "max_lcp_obstacle_violation",
                    "max_lcp_natural_residual",
                    "timing_median_seconds",
                ):
                    self.assertTrue(math.isfinite(float(row[field])), (row["config_id"], field))
            with (output / "grid_refinement.csv").open(newline="", encoding="utf-8") as handle:
                grid_reader = csv.DictReader(handle)
                grid_fields = set(grid_reader.fieldnames or ())
                grid_rows = list(grid_reader)
            grouped_grids = {}
            for row in grid_rows:
                key = (row["scenario_id"], row["config_id"])
                grouped_grids.setdefault(key, []).append(int(row["grid"]))
                self.assertEqual(row["refinement_check"], "diagnostic_only_no_order_claim")
                self.assertTrue(math.isfinite(float(row["grid_spacing"])))
                self.assertEqual(int(row["reference_grid"]), 48)
            self.assertEqual(len(grouped_grids), 10)
            self.assertTrue(all(sorted(grids) == [12, 24] for grids in grouped_grids.values()))
            self.assertNotIn("observed_order", grid_fields)
            hashes = json.loads((output / "artifact_hashes.json").read_text(encoding="utf-8"))
            self.assertEqual(set(hashes["files"]), required - {"artifact_hashes.json"})
            for filename, expected in hashes["files"].items():
                import hashlib

                self.assertEqual(hashlib.sha256((output / filename).read_bytes()).hexdigest(), expected)

    def test_shared_campaign_cli_dispatches_m01_smoke_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [
                    "/Users/liangzihan/Documents/SURF_0159/SURF/.worktrees/american-option-risk-surfaces/.venv-campaign/bin/python",
                    str(ROOT / "experiments" / "26_independent_method_campaign.py"),
                    "--method",
                    "M01",
                    "--phase",
                    "screen",
                    "--config",
                    str(ROOT / "configs" / "independent_m01_classical.yaml"),
                    "--output-root",
                    temp_dir,
                    "--smoke",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            payload = json.loads(result.stdout)
            self.assertEqual(payload["method_id"], "M01")
            self.assertEqual(payload["run_mode"], "smoke_non_evidence")


if __name__ == "__main__":
    unittest.main()
