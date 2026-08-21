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
                "source_records.csv",
                "array_artifacts.csv",
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

            with (out / "source_records.csv").open(newline="") as fh:
                source_header = next(csv.reader(fh))
            for name in ("repo", "branch", "path", "record_kind", "record_index", "field", "value", "value_type"):
                self.assertIn(name, source_header)

            with (out / "array_artifacts.csv").open(newline="") as fh:
                array_header = next(csv.reader(fh))
            for name in ("repo", "branch", "path", "array_name", "shape", "dtype", "csv_gz_path", "content_sha256"):
                self.assertIn(name, array_header)

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

    def test_rectilinear_interpolation_is_exact_for_affine_surface_and_refuses_extrapolation(self):
        import numpy as np
        from analysis.consolidated_experiment_data import rectilinear_interpolate

        s = np.array([0.0, 1.0, 2.0])
        tau = np.array([0.0, 1.0])
        values = tau[:, None] * 3.0 + s[None, :] * 2.0 + 5.0
        target_s = np.array([0.25, 1.5])
        target_tau = np.array([0.2, 0.8])
        got = rectilinear_interpolate(s, tau, values, target_s, target_tau)
        expected = target_tau[:, None] * 3.0 + target_s[None, :] * 2.0 + 5.0
        np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-12)

        with self.assertRaises(ValueError):
            rectilinear_interpolate(s, tau, values, np.array([-0.1]), target_tau)
        with self.assertRaises(ValueError):
            rectilinear_interpolate(s, tau, values, target_s, np.array([1.1]))

    def test_json_flattening_and_artifact_classification_are_lossless_for_scalars(self):
        from analysis.consolidated_experiment_data import (
            classify_artifact_path,
            flatten_json_scalars,
        )

        data = {"a": 1, "b": {"c": True, "d": [2.5, "x"]}, "e": None}
        flattened = dict(flatten_json_scalars(data))
        self.assertEqual(flattened["a"], 1)
        self.assertEqual(flattened["b.c"], True)
        self.assertEqual(flattened["b.d[0]"], 2.5)
        self.assertEqual(flattened["b.d[1]"], "x")
        self.assertIsNone(flattened["e"])

        self.assertEqual(classify_artifact_path("results/x/run_metrics.csv"), "metrics_csv")
        self.assertEqual(classify_artifact_path("results/x/surface.npz"), "numeric_array")
        self.assertEqual(classify_artifact_path("results/x/checkpoints/model.pt"), "checkpoint")


if __name__ == "__main__":
    unittest.main()
