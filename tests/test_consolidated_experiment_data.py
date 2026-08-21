import csv
import json
import math
import tempfile
import unittest
from pathlib import Path


class ConsolidatedExperimentDataContract(unittest.TestCase):
    def test_public_api_and_rmse_definition(self):
        from analysis.consolidated_experiment_data import price_rmse

        self.assertAlmostEqual(price_rmse([1.0, 2.0], [1.0, 4.0]), math.sqrt(2.0))
        with self.assertRaises(ValueError):
            price_rmse([1.0], [1.0, 2.0])
        with self.assertRaises(ValueError):
            price_rmse([], [])

    def test_output_contract_has_required_tables_and_metric_columns(self):
        from analysis.consolidated_experiment_data import write_empty_output_contract

        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            write_empty_output_contract(out)

            required = {
                "source_artifact_inventory.csv",
                "provenance.csv",
                "experiment_runs.csv",
                "price_metrics.csv",
                "greek_metrics.csv",
                "boundary_metrics.csv",
                "free_boundary_points.csv",
                "price_slices.csv",
                "structural_metrics.csv",
                "compute_metrics.csv",
                "classical_solver_metrics.csv",
                "binomial_convergence.csv",
                "ruixi_arm_metrics.csv",
                "hd1_metrics.csv",
                "hd2_candidate_metrics.csv",
                "hd3_metrics.csv",
                "campaign_method_results.csv",
                "experiment_gates.csv",
                "environment_runs.csv",
                "surface_artifacts.csv",
                "dirk_common_reference_metrics.csv",
                "manifest.json",
                "README.md",
            }
            self.assertTrue(required.issubset({p.name for p in out.iterdir()}))

            with (out / "dirk_common_reference_metrics.csv").open(newline="") as fh:
                header = next(csv.reader(fh))
            for name in (
                "internal_cnpsor_price_rmse",
                "dirk_price_rmse_hd_domain",
                "dirk_price_rmse_shared_domain",
                "reference_type",
                "metric_scope",
                "s_min",
                "s_max",
                "tau_min",
                "tau_max",
                "grid_points",
                "interpolation_method",
                "source_repo",
                "source_branch",
                "source_commit",
                "reference_source",
                "reference_commit",
                "reference_artifact_sha256",
            ):
                self.assertIn(name, header)

            manifest = json.loads((out / "manifest.json").read_text())
            self.assertEqual(manifest["schema_version"], 1)
            self.assertEqual(
                manifest["headline_cross_model_metric"],
                "dirk_price_rmse_shared_domain",
            )
            self.assertEqual(
                manifest["headline_hd_scaling_metric"],
                "dirk_price_rmse_hd_domain",
            )

    def test_shared_domain_is_frozen(self):
        from analysis.consolidated_experiment_data import SHARED_DOMAIN, HD_DOMAIN

        self.assertEqual(SHARED_DOMAIN, (0.5, 1.5, 0.01, 1.0))
        self.assertEqual(HD_DOMAIN, (0.2, 1.5, 0.01, 1.0))


if __name__ == "__main__":
    unittest.main()
