import csv
import importlib.util
import json
import math
import os
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
        self.assertEqual(config["timing"]["warmups"], 5)
        self.assertEqual(config["timing"]["repetitions"], 30)
        self.assertEqual(config["gates"].get("familywise_alpha"), 0.05)
        negative_regimes = [regime for regime in config["regimes"] if float(regime["r"]) < 0.0]
        self.assertGreaterEqual(len(negative_regimes), 1)
        self.assertTrue(all(str(regime["scenario_id"]).startswith("put_") for regime in negative_regimes))
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
            accounting = summary["resource_accounting"]
            self.assertEqual(accounting["screen_budget_seconds"], 3600)
            self.assertFalse(accounting["budget_exceeded"])
            self.assertEqual(accounting["timing_warmups"], 0)
            self.assertEqual(accounting["timing_repetitions"], 2)
            self.assertEqual(accounting["training_elapsed_seconds"], 0.0)
            self.assertEqual(
                accounting["training_cost_status"], "not_applicable_no_trainable_model"
            )
            self.assertEqual(accounting["memory_measure"], "process_peak_rss")
            for field in (
                "total_screen_elapsed_seconds",
                "deterministic_timing_elapsed_seconds",
                "control_generation_elapsed_seconds",
                "comparison_refinement_elapsed_seconds",
                "memory_mb",
            ):
                self.assertTrue(math.isfinite(float(accounting[field])), field)
                self.assertGreaterEqual(float(accounting[field]), 0.0, field)
            self.assertGreater(float(accounting["memory_mb"]), 0.0)
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
            self.assertEqual(len({row["memory_mb"] for row in run_rows}), 1)
            self.assertGreater(float(run_rows[0]["memory_mb"]), 0.0)
            candidate_rows = [row for row in run_rows if row["config_id"] != "cn_psor"]
            self.assertEqual(len(candidate_rows), 4)
            for row in candidate_rows:
                metrics = json.loads(row["optional_metrics_json"])
                raw = float(metrics["timing_raw_p_value"])
                adjusted = float(metrics["timing_holm_adjusted_p_value"])
                self.assertGreaterEqual(adjusted, raw)
                self.assertIn(metrics["timing_holm_reject"], (0.0, 1.0))
                decision_state = summary["method_decisions"][row["config_id"]]
                self.assertEqual(decision_state["timing_raw_p_value"], raw)
                self.assertEqual(decision_state["timing_holm_adjusted_p_value"], adjusted)
                self.assertIsInstance(decision_state["timing_holm_reject"], bool)
                self.assertIn("all_relevant_hard_lcp_converged", decision_state)
            manifest = json.loads((output / "config_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["provenance"],
                {"branch": run_rows[0]["branch"], "commit": run_rows[0]["commit"]},
            )
            self.assertEqual(manifest["config"]["evidence_status"], "diagnostic_non_evidence")
            self.assertFalse(manifest["config"]["blind_holdout_accessed"])
            self.assertEqual(manifest["config"]["resource_accounting"], accounting)
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
                    if field == "boundary_rmse" and float(row[field]) != float(row[field]):
                        continue
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
            with (output / "analytic_controls.csv").open(newline="", encoding="utf-8") as handle:
                analytic_rows = list(csv.DictReader(handle))
            negative_brennan = [
                row
                for row in analytic_rows
                if row["control"] == "brennan_schwartz_first_step"
                and row["control_status"] == "not_applicable_negative_rate"
            ]
            self.assertEqual(len(negative_brennan), 1)
            self.assertEqual(negative_brennan[0]["applicability_reason"], "negative_interest_rate")
            with (output / "boundary_crosscheck.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                boundary_rows = list(csv.DictReader(handle))
            negative_integral = [
                row
                for row in boundary_rows
                if row["control_status"] == "not_applicable_negative_rate"
            ]
            self.assertEqual(len(negative_integral), 1)
            self.assertEqual(
                negative_integral[0]["applicability_reason"],
                "integral_boundary_requires_positive_rate",
            )
            hashes = json.loads((output / "artifact_hashes.json").read_text(encoding="utf-8"))
            self.assertEqual(set(hashes["files"]), required - {"artifact_hashes.json"})
            for filename, expected in hashes["files"].items():
                import hashlib

                self.assertEqual(hashlib.sha256((output / filename).read_bytes()).hexdigest(), expected)

    def test_promotion_helpers_require_holm_convergence_and_budget(self):
        experiment = _load_experiment()
        status_function = getattr(experiment, "_candidate_run_status", None)
        claim_function = getattr(experiment, "_evidence_status", None)
        self.assertIsNotNone(status_function)
        self.assertIsNotNone(claim_function)
        gates = {
            "smoke": False,
            "baseline": False,
            "no_regression": True,
            "obstacle_ok": True,
            "analytic_ok": True,
            "boundary_ok": True,
            "practical": True,
            "ci_excludes_zero": True,
            "holm_reject": True,
            "hard_lcp_converged": True,
            "budget_exceeded": False,
        }

        status, claim = status_function(**gates)
        self.assertEqual((status.value, claim), ("completed", "candidate_supported"))
        for gate in ("holm_reject", "hard_lcp_converged"):
            blocked = {**gates, gate: False}
            status, claim = status_function(**blocked)
            self.assertNotEqual(status.value, "completed", gate)
            self.assertNotEqual(claim, "candidate_supported", gate)
        status, claim = status_function(**{**gates, "budget_exceeded": True})
        self.assertEqual((status.value, claim), ("resource_blocked", "resource_blocked"))
        self.assertEqual(
            claim_function(smoke=False, supported_config_ids=(), budget_exceeded=False),
            "benchmark_only",
        )
        self.assertEqual(
            claim_function(
                smoke=False,
                supported_config_ids=("dirk_psor",),
                budget_exceeded=False,
            ),
            "candidate_supported",
        )
        self.assertEqual(
            claim_function(smoke=False, supported_config_ids=(), budget_exceeded=True),
            "resource_blocked",
        )

    def test_shared_campaign_cli_dispatches_m01_smoke_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env = os.environ.copy()
            env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + str(ROOT / "experiments")
            result = subprocess.run(
                [
                    sys.executable,
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
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )
            payload = json.loads(result.stdout)
            self.assertEqual(payload["method_id"], "M01")
            self.assertEqual(payload["run_mode"], "smoke_non_evidence")


if __name__ == "__main__":
    unittest.main()
