import math
import unittest

from campaign.records import MethodRunRecord, RunPhase, RunStatus
from campaign.registry import MethodCategory, MetricDirection, METHOD_REGISTRY, get_method_spec, verify_registry


EXPECTED_METHODS = {
    "M01": ("Classical solvers", MethodCategory.SOLVER_ACCELERATOR),
    "M02": ("Obstacle-aware PINN", MethodCategory.SOLVER_ACCELERATOR),
    "M03": ("Algorithm unrolling", MethodCategory.SOLVER_ACCELERATOR),
    "M04": ("Free-boundary representation", MethodCategory.SOLVER_ACCELERATOR),
    "M05": ("Schrodingerization", MethodCategory.SOLVER_ACCELERATOR),
    "M06": ("Quantum linear systems", MethodCategory.SOLVER_ACCELERATOR),
    "M07": ("Nonlinear quantum reduction", MethodCategory.SOLVER_ACCELERATOR),
    "M08": ("Tensor and low rank", MethodCategory.SURROGATE_COMPRESSION),
    "M09": ("Multi-fidelity", MethodCategory.SURROGATE_COMPRESSION),
    "M10": ("Structured surrogate", MethodCategory.SURROGATE_COMPRESSION),
    "M11": ("Inverse regularization", MethodCategory.RECONSTRUCTION_UQ),
    "M12": ("Diffusion or score model", MethodCategory.RECONSTRUCTION_UQ),
    "M13": ("Adaptive sampling", MethodCategory.SURROGATE_COMPRESSION),
    "M14": ("Hardware acceleration", MethodCategory.SURROGATE_COMPRESSION),
    "M15": ("MC Dropout", MethodCategory.RECONSTRUCTION_UQ),
    "M16": ("Deep ensembles", MethodCategory.RECONSTRUCTION_UQ),
    "M17": ("Conformal prediction", MethodCategory.RECONSTRUCTION_UQ),
    "M18": ("Selective fallback", MethodCategory.RECONSTRUCTION_UQ),
    "M19": ("U-Net/V-Net segmentation", MethodCategory.DIAGNOSTIC_WORKFLOW),
    "M20": ("Transfer learning", MethodCategory.DIAGNOSTIC_WORKFLOW),
    "M21": ("Explainable AI", MethodCategory.DIAGNOSTIC_WORKFLOW),
    "M22": ("Mechanistic audit", MethodCategory.DIAGNOSTIC_WORKFLOW),
    "M23": ("Agentic search", MethodCategory.DIAGNOSTIC_WORKFLOW),
    "M24": ("Formal verification and certificates", MethodCategory.DIAGNOSTIC_WORKFLOW),
}


class CampaignRegistryTests(unittest.TestCase):
    def test_registry_has_exactly_the_24_approved_methods_in_scoreboard_categories(self):
        self.assertEqual(set(METHOD_REGISTRY), set(EXPECTED_METHODS))

        for method_id, (name, category) in EXPECTED_METHODS.items():
            spec = get_method_spec(method_id)
            self.assertEqual(spec.name, name)
            self.assertEqual(spec.category, category)

    def test_each_method_has_a_bounded_independent_campaign_contract(self):
        for method_id in EXPECTED_METHODS:
            spec = get_method_spec(method_id)
            self.assertTrue(spec.baseline)
            self.assertTrue(spec.primary_metric)
            self.assertIsInstance(spec.metric_direction, MetricDirection)
            self.assertGreater(spec.screen_budget_hours, 0)
            self.assertEqual(spec.confirmation_budget_hours, 6)
            self.assertLessEqual(len(spec.predeclared_configurations), 8)
            self.assertGreater(len(spec.predeclared_configurations), 0)
            self.assertTrue(spec.required_outputs)
            self.assertTrue(spec.gates)

        self.assertEqual(get_method_spec("M05").screen_budget_hours, 4)
        self.assertEqual(get_method_spec("M06").screen_budget_hours, 4)
        self.assertEqual(get_method_spec("M07").screen_budget_hours, 4)
        self.assertEqual(get_method_spec("M12").screen_budget_hours, 4)
        self.assertEqual(get_method_spec("M01").screen_budget_hours, 1)
        self.assertEqual(get_method_spec("M14").primary_metric, "end_to_end_speedup")
        self.assertEqual(get_method_spec("M14").metric_direction, MetricDirection.HIGHER_IS_BETTER)
        self.assertEqual(get_method_spec("M17").metric_direction, MetricDirection.TARGET)
        self.assertTrue(any("93--97%" in gate for gate in get_method_spec("M17").gates))
        self.assertTrue(any("1.25x" in gate for gate in get_method_spec("M14").gates))
        self.assertTrue(any("10x" in gate for gate in get_method_spec("M08").gates))

    def test_registry_verification_returns_a_machine_readable_summary(self):
        self.assertEqual(
            verify_registry(),
            {
                "categories": {
                    "diagnostic_workflow": 6,
                    "reconstruction_uq": 6,
                    "solver_accelerator": 7,
                    "surrogate_compression": 5,
                },
                "methods": 24,
                "valid": True,
            },
        )


class MethodRunRecordTests(unittest.TestCase):
    def test_completed_record_requires_finite_primary_and_baseline_metrics(self):
        with self.assertRaisesRegex(ValueError, "primary_metric_value"):
            MethodRunRecord(
                method_id="M01",
                phase=RunPhase.SCREEN,
                status=RunStatus.COMPLETED,
                primary_metric_value=math.nan,
                baseline_metric_value=1.0,
            )

        with self.assertRaisesRegex(ValueError, "baseline_metric_value"):
            MethodRunRecord(
                method_id="M01",
                phase=RunPhase.SCREEN,
                status=RunStatus.COMPLETED,
                primary_metric_value=1.0,
                baseline_metric_value=None,
            )

    def test_optional_metrics_must_be_finite_when_present(self):
        with self.assertRaisesRegex(ValueError, "optional_metrics"):
            MethodRunRecord(
                method_id="M01",
                phase=RunPhase.SCREEN,
                status=RunStatus.DIAGNOSTIC_ONLY,
                optional_metrics={"boundary_rmse": math.inf},
            )

    def test_non_completed_record_preserves_campaign_provenance_without_metrics(self):
        record = MethodRunRecord(
            method_id="M14",
            phase=RunPhase.CONFIRM,
            status=RunStatus.RESOURCE_BLOCKED,
            scenario_count=128,
            seed=101,
            split="holdout",
            branch="feature/campaign-m14",
            commit="abc123",
            timings_seconds={"training": 3.5},
            memory_mb=256.0,
            claim_status="diagnostic_only",
            notes="budget exhausted",
        )

        payload = record.to_dict()
        self.assertEqual(payload["status"], "resource_blocked")
        self.assertEqual(payload["scenario_count"], 128)
        self.assertEqual(payload["timings_seconds"], {"training": 3.5})
