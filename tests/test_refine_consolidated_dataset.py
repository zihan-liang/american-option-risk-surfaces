import json
import subprocess
import tempfile
import unittest
from pathlib import Path


class RefineConsolidatedDatasetContract(unittest.TestCase):
    def test_real_remote_branches_excludes_symbolic_head(self):
        from analysis.refine_consolidated_dataset import real_remote_branches
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td) / "repo"
            repo.mkdir()
            subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True, stdout=subprocess.DEVNULL)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
            (repo / "a.txt").write_text("x\n", encoding="utf-8")
            subprocess.run(["git", "add", "."], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, stdout=subprocess.DEVNULL)
            sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
            subprocess.run(["git", "update-ref", "refs/remotes/origin/main", sha], cwd=repo, check=True)
            subprocess.run(["git", "symbolic-ref", "refs/remotes/origin/HEAD", "refs/remotes/origin/main"], cwd=repo, check=True)
            self.assertEqual(real_remote_branches(repo), ["main"])

    def test_metric_artifact_filter_rejects_pointwise_and_history(self):
        from analysis.refine_consolidated_dataset import is_metric_artifact
        self.assertTrue(is_metric_artifact("results/standalone/M08/metrics.csv"))
        self.assertTrue(is_metric_artifact("results/x/dimension_summary.csv"))
        self.assertTrue(is_metric_artifact("results/x/decision.json"))
        self.assertFalse(is_metric_artifact("results/x/price_slice_tau_1.csv"))
        self.assertFalse(is_metric_artifact("results/x/training_history.csv"))
        self.assertFalse(is_metric_artifact("results/x/timing_samples.csv"))
        self.assertFalse(is_metric_artifact("results/x/surface_generation.csv"))

    def test_infer_method_from_path_tags_binomial_reference_and_pinn_arms(self):
        from analysis.refine_consolidated_dataset import infer_method
        self.assertEqual(infer_method("results/x/visualization/binomial/price_slice_tau_1.csv"), "BINOMIAL_TREE")
        self.assertEqual(infer_method("results/visualization_exports/x/reference/price_slice_tau_1.csv"), "DIRK_REFERENCE")
        self.assertEqual(infer_method("results/x/arm_c_seed101/scoring/pinn/pinn_metrics_by_regime_seed.csv"), "ARM_C")
        self.assertEqual(infer_method("results/x/arm_d_seed101/scoring/pinn/pinn_metrics_by_regime_seed.csv"), "ARM_D")
        self.assertEqual(infer_method("results/x/d20/FC_HVI/price_slice_tau_1.csv"), "FC_HVI")

    def test_parse_binomial_convergence_preserves_summary_and_pointwise_curve(self):
        from analysis.refine_consolidated_dataset import parse_binomial_convergence, parse_binomial_slice
        meta = {
            "convergence": [
                {"steps": 256, "tau_over_T": 1.0, "rmse_change_from_previous": None, "max_abs_change_from_previous": None},
                {"steps": 512, "tau_over_T": 1.0, "rmse_change_from_previous": 1e-4, "max_abs_change_from_previous": 2e-4},
            ],
            "final_curve_steps": 2048,
        }
        rows = parse_binomial_convergence(meta, {"repo": "r", "branch": "b", "source_commit": "c"})
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["comparison_metric"], "rmse_change_from_previous;max_abs_change_from_previous")
        self.assertIn("0.0001", rows[1]["comparison_value"])
        curve = parse_binomial_slice(
            [{"S_over_K": "0.5", "tau_over_T": "1", "V_over_K": "0.51", "tree_steps": "2048"}],
            {"repo": "r", "branch": "b", "source_commit": "c"},
        )
        self.assertEqual(curve[0]["steps"], 2048)
        self.assertEqual(curve[0]["V_over_K"], "0.51")

    def test_parse_hd2_selection_uses_frozen_gate_thresholds(self):
        from analysis.refine_consolidated_dataset import parse_hd2_selection
        selection = {
            "selection_status": "no_stabilized_candidate",
            "candidate_audit": {
                "HVI-warm-low": {
                    "eligible": False,
                    "physics_score": 0.2,
                    "dimensions": {
                        "2": {"price_ratio": 0.9, "gamma_ratio": 1.05, "boundary_ratio": 1.0, "pde_p95_ratio": 0.5, "fb_p95_ratio": 0.4, "passes": True}
                    },
                }
            },
        }
        gates = {"price_ratio_max": 1.05, "gamma_ratio_max": 1.1, "boundary_ratio_max": 1.1, "pde_p95_ratio_max": 0.75, "fb_p95_ratio_max": 0.75}
        metrics, gate_rows = parse_hd2_selection(selection, gates, {"repo": "r", "branch": "b", "source_commit": "c"})
        self.assertTrue(any(r["metric_name"] == "price_ratio" and r["metric_value"] == 0.9 for r in metrics))
        price_gate = next(r for r in gate_rows if r["metric"] == "price_ratio")
        self.assertEqual(price_gate["threshold"], 1.05)
        self.assertEqual(price_gate["operator"], "<=")
        self.assertTrue(price_gate["passed"])

    def test_audit_detects_empty_duplicate_and_missing_key(self):
        from analysis.refine_consolidated_dataset import audit_rows
        empty = audit_rows("x.csv", [], ["method_id"])
        self.assertEqual(empty["status"], "empty")
        rows = [{"method_id": "A", "metric": "x"}, {"method_id": "A", "metric": "x"}, {"method_id": "", "metric": "y"}]
        audit = audit_rows("x.csv", rows, ["method_id"])
        self.assertEqual(audit["duplicate_rows"], 1)
        self.assertEqual(audit["missing_key_rows"], 1)
        self.assertEqual(audit["status"], "needs_review")

    def test_nested_environment_manifest_is_unwrapped(self):
        import analysis.refine_consolidated_dataset as mod
        self.assertTrue(hasattr(mod, "parse_environment_manifest"))
        obj = {
            "branch": "exp/x",
            "environment": {
                "cuda_available": True,
                "platform": "Windows-10",
                "python": "3.11.9",
                "torch": "2.7.1+cu128",
                "torch_cuda": "12.8",
                "gpu_name": "NVIDIA GeForce RTX 4070 Laptop GPU",
            },
        }
        row = mod.parse_environment_manifest(
            obj,
            {"repo": "r", "branch": "b", "source_commit": "c"},
            "results/hd_pipeline/environment_manifest.json",
            {"method_id": "", "seed": ""},
        )
        self.assertEqual(row["experiment_id"], "hd_pipeline")
        self.assertEqual(row["os"], "Windows-10")
        self.assertEqual(row["python"], "3.11.9")
        self.assertEqual(row["pytorch"], "2.7.1+cu128")
        self.assertEqual(row["cuda"], "12.8")
        self.assertEqual(row["gpu"], "NVIDIA GeForce RTX 4070 Laptop GPU")
        self.assertEqual(row["device"], "cuda")

    def test_semantic_audit_rejects_nearly_empty_environment_rows(self):
        from analysis.refine_consolidated_dataset import audit_rows
        rows = [{
            "experiment_id": "hd_pipeline", "method_id": "", "seed": "", "os": "", "python": "",
            "pytorch": "", "cuda": "", "gpu": "", "cpu": "", "ram": "", "device": "",
        }]
        audit = audit_rows("environment_runs.csv", rows, ["experiment_id"])
        self.assertEqual(audit["status"], "needs_review")
        self.assertIn("environment", audit["notes"].lower())

    def test_semantic_audit_rejects_identifier_names_in_metric_tables(self):
        from analysis.refine_consolidated_dataset import audit_rows
        rows = [{"model_id": "SUP", "dimension": "5", "seed": "101", "metric_name": "dimension", "metric_value": "5"}]
        audit = audit_rows("hd1_metrics.csv", rows, ["model_id", "dimension", "metric_name"])
        self.assertEqual(audit["status"], "needs_review")
        self.assertIn("identifier", audit["notes"].lower())


if __name__ == "__main__":
    unittest.main()
