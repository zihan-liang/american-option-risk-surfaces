import unittest


class ExtendedSemanticRefinementContract(unittest.TestCase):
    def test_metadata_metric_filter_removes_context_fields_only(self):
        import analysis.semantic_refinement as mod
        self.assertTrue(hasattr(mod, "filter_metadata_metric_rows"))
        rows = [
            {"metric_name": "split=test;region=all:seed", "metric_value": "101"},
            {"metric_name": "split=test;region=all:sample_count", "metric_value": "5203"},
            {"metric_name": "scenario_count", "metric_value": "9"},
            {"metric_name": "decision.json.training_seeds[0]", "metric_value": "101"},
            {"metric_name": "split=test;region=all:rmse", "metric_value": "0.001"},
            {"metric_name": "total_lcp_iterations", "metric_value": "1680"},
            {"metric_name": "boundary_missed_count", "metric_value": "119"},
        ]
        kept, removed = mod.filter_metadata_metric_rows(rows)
        self.assertEqual(removed, 4)
        self.assertEqual(
            [row["metric_name"] for row in kept],
            [
                "split=test;region=all:rmse",
                "total_lcp_iterations",
                "boundary_missed_count",
            ],
        )

    def test_regime_aware_final_headers_extend_greek_and_boundary_tables(self):
        import analysis.semantic_refinement as mod
        self.assertTrue(hasattr(mod, "final_table_header"))
        greek = mod.final_table_header("greek_metrics.csv")
        boundary = mod.final_table_header("boundary_metrics.csv")
        self.assertIn("regime_id", greek)
        self.assertIn("regime_id", boundary)
        self.assertEqual(greek.count("regime_id"), 1)
        self.assertEqual(boundary.count("regime_id"), 1)

    def test_ruixi_greek_rows_are_rebuilt_with_regime_identity(self):
        import analysis.semantic_refinement as mod
        self.assertTrue(hasattr(mod, "rebuild_ruixi_greek_rows"))
        arm_rows = [
            {
                "repo": "Ruixixu-hub/2026-surf-american-risk-surfaces",
                "branch": "main",
                "source_commit": "abc",
                "method_id": "ARM_D",
                "regime_id": "put_T100_s060_r005_q010",
                "seed": "101",
                "metric_name": "split=stress_holdout;region=all:delta_rmse",
                "metric_value": "0.004",
                "reference_type": "DIRK_policy_sinh_M480_N960",
            },
            {
                "repo": "Ruixixu-hub/2026-surf-american-risk-surfaces",
                "branch": "main",
                "source_commit": "abc",
                "method_id": "ARM_D",
                "regime_id": "put_T100_s060_r005_q010",
                "seed": "101",
                "metric_name": "split=stress_holdout;region=all:scaled_gamma_rmse",
                "metric_value": "0.02",
                "reference_type": "DIRK_policy_sinh_M480_N960",
            },
        ]
        rebuilt = mod.rebuild_ruixi_greek_rows(arm_rows)
        self.assertEqual(len(rebuilt), 2)
        self.assertEqual({row["regime_id"] for row in rebuilt}, {"put_T100_s060_r005_q010"})
        self.assertEqual({row["method_id"] for row in rebuilt}, {"ARM_D"})

    def test_ruixi_boundary_rows_are_rebuilt_per_regime(self):
        import analysis.semantic_refinement as mod
        self.assertTrue(hasattr(mod, "rebuild_ruixi_boundary_rows"))
        common = {
            "repo": "Ruixixu-hub/2026-surf-american-risk-surfaces",
            "branch": "main",
            "source_commit": "abc",
            "method_id": "ARM_D",
            "regime_id": "put_T100_s060_r005_q010",
            "seed": "101",
            "reference_type": "DIRK_policy_sinh_M480_N960",
        }
        arm_rows = [
            {**common, "metric_name": "split=stress_holdout;region=all:boundary_reference_found", "metric_value": "120"},
            {**common, "metric_name": "split=stress_holdout;region=all:boundary_predicted_found", "metric_value": "46"},
            {**common, "metric_name": "split=stress_holdout;region=all:boundary_found_rate", "metric_value": "0.38"},
            {**common, "metric_name": "split=stress_holdout;region=all:boundary_conditional_rmse", "metric_value": "0.03"},
            {**common, "metric_name": "split=stress_holdout;region=exercise_region:boundary_found_rate", "metric_value": "0.38"},
        ]
        rebuilt = mod.rebuild_ruixi_boundary_rows(arm_rows)
        self.assertEqual(len(rebuilt), 1)
        row = rebuilt[0]
        self.assertEqual(row["regime_id"], "put_T100_s060_r005_q010")
        self.assertEqual(row["reference_count"], "120")
        self.assertEqual(row["predicted_count"], "46")
        self.assertEqual(row["coverage"], "0.38")
        self.assertEqual(row["conditional_rmse"], "0.03")

    def test_campaign_experiment_runs_are_rebuilt_from_filtered_result_rows(self):
        import analysis.semantic_refinement as mod
        self.assertTrue(hasattr(mod, "rebuild_campaign_experiment_runs"))
        existing = [
            {
                "repo": "zihan-liang/american-option-risk-surfaces",
                "branch": "campaign/independent-methods",
                "source_commit": "config-commit",
                "experiment_family": "CAMPAIGN",
                "experiment_id": "M10",
                "method_id": "M10",
                "status": "recorded",
            },
            {
                "repo": "zihan-liang/american-option-risk-surfaces",
                "branch": "exp/hd3",
                "source_commit": "hd3-commit",
                "experiment_family": "HD3",
                "experiment_id": "hd3_x",
                "method_id": "FC_HVI",
                "status": "completed",
            },
        ]
        filtered_results = [
            {
                "repo": "zihan-liang/american-option-risk-surfaces",
                "branch": "tmp/m10-transfer-20260817",
                "source_commit": "result-commit",
                "method_id": "M10",
                "metric_name": "metrics.csv[0].price_rmse",
                "metric_value": "0.001",
            },
            {
                "repo": "zihan-liang/american-option-risk-surfaces",
                "branch": "campaign/independent-methods",
                "source_commit": "m09-commit",
                "method_id": "M09",
                "metric_name": "decision.json.primary_metric_value",
                "metric_value": "0.1",
            },
        ]
        rebuilt = mod.rebuild_campaign_experiment_runs(existing, filtered_results)
        campaign = {row["method_id"]: row for row in rebuilt if row.get("experiment_family") == "CAMPAIGN"}
        self.assertEqual(set(campaign), {"M09", "M10"})
        self.assertEqual(campaign["M10"]["branch"], "tmp/m10-transfer-20260817")
        self.assertEqual(campaign["M10"]["source_commit"], "result-commit")
        self.assertEqual(campaign["M09"]["branch"], "campaign/independent-methods")
        self.assertTrue(any(row.get("experiment_family") == "HD3" for row in rebuilt))

    def test_hd2_candidate_aggregate_metrics_may_have_blank_dimension(self):
        from analysis.finalize_refinement_audit import audit_semantic_table
        rows = [
            {
                "candidate_id": "HVI-warm-low",
                "dimension": "",
                "metric_name": "physics_score",
                "metric_value": "0.23",
            },
            {
                "candidate_id": "HVI-warm-low",
                "dimension": "",
                "metric_name": "eligible",
                "metric_value": "0",
            },
            {
                "candidate_id": "HVI-warm-low",
                "dimension": "2",
                "metric_name": "price_ratio",
                "metric_value": "0.72",
            },
        ]
        audit = audit_semantic_table("hd2_candidate_metrics.csv", rows)
        self.assertEqual(audit["status"], "ok")
        self.assertEqual(audit["missing_key_rows"], 0)

    def test_hd2_dimension_scoped_metric_still_requires_dimension(self):
        from analysis.finalize_refinement_audit import audit_semantic_table
        rows = [
            {
                "candidate_id": "HVI-warm-low",
                "dimension": "",
                "metric_name": "price_ratio",
                "metric_value": "0.72",
            }
        ]
        audit = audit_semantic_table("hd2_candidate_metrics.csv", rows)
        self.assertEqual(audit["status"], "needs_review")
        self.assertEqual(audit["missing_key_rows"], 1)


if __name__ == "__main__":
    unittest.main()
