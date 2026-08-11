"""The fixed, independent-method campaign registry."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping


STANDARD_OUTPUTS = (
    "config_manifest.json",
    "run_metrics.csv",
    "decision.json",
    "artifact_hashes.json",
)


class MethodCategory(str, Enum):
    SOLVER_ACCELERATOR = "solver_accelerator"
    SURROGATE_COMPRESSION = "surrogate_compression"
    RECONSTRUCTION_UQ = "reconstruction_uq"
    DIAGNOSTIC_WORKFLOW = "diagnostic_workflow"


class MetricDirection(str, Enum):
    LOWER_IS_BETTER = "lower_is_better"
    HIGHER_IS_BETTER = "higher_is_better"
    TARGET = "target"


@dataclass(frozen=True)
class MethodSpec:
    """A campaign contract for one method, independent of its implementation."""

    method_id: str
    name: str
    scope: str
    category: MethodCategory
    baseline: str
    primary_metric: str
    metric_direction: MetricDirection
    screen_budget_seconds: int
    confirmation_budget_seconds: int
    stochastic: bool
    predeclared_configurations: tuple[str, ...]
    required_outputs: tuple[str, ...]
    gates: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["category"] = self.category.value
        payload["metric_direction"] = self.metric_direction.value
        return payload


def _spec(
    method_id: str,
    name: str,
    scope: str,
    category: MethodCategory,
    baseline: str,
    primary_metric: str,
    metric_direction: MetricDirection,
    configurations: tuple[str, ...],
    gates: tuple[str, ...],
    *,
    stochastic: bool = False,
    screen_budget_seconds: int = 3_600,
    required_outputs: tuple[str, ...] = STANDARD_OUTPUTS,
) -> MethodSpec:
    return MethodSpec(
        method_id=method_id,
        name=name,
        scope=scope,
        category=category,
        baseline=baseline,
        primary_metric=primary_metric,
        metric_direction=metric_direction,
        screen_budget_seconds=screen_budget_seconds,
        confirmation_budget_seconds=21_600,
        stochastic=stochastic,
        predeclared_configurations=configurations,
        required_outputs=required_outputs,
        gates=gates,
    )


METHOD_REGISTRY: Mapping[str, MethodSpec] = {
    "M01": _spec(
        "M01", "Classical solvers",
        "Compare CN, Rannacher-CN, and second-order L-stable DIRK with PSOR, policy iteration, and penalty/Newton one factor at a time.",
        MethodCategory.SOLVER_ACCELERATOR, "CN/PSOR", "wall_time_seconds_at_fixed_accuracy",
        MetricDirection.LOWER_IS_BETTER,
        ("cn_psor", "rannacher_cn_psor", "dirk_psor", "cn_policy_iteration", "cn_penalty_newton"),
        ("Pass grid refinement, European analytic controls, no-dividend American-call equivalence, and applicable independent boundary checks.",),
    ),
    "M02": _spec(
        "M02", "Obstacle-aware PINN",
        "Compare supervised, masked-PDE, separate complementarity, and Fischer--Burmeister losses using the same positive-premium architecture.",
        MethodCategory.SURROGATE_COMPRESSION, "Supervised positive-premium surrogate", "price_rmse",
        MetricDirection.LOWER_IS_BETTER,
        ("supervised", "masked_pde", "separate_complementarity", "fischer_burmeister"),
        ("Obstacle violation is at most 1e-10.", "Price and diagnostic degradation is no more than 2%."),
        stochastic=True,
        required_outputs=STANDARD_OUTPUTS + ("training_history.csv",),
    ),
    "M03": _spec(
        "M03", "Algorithm unrolling",
        "Compare unrolled projected SOR or a learned feasible warm start with tuned PSOR at identical natural-residual tolerance, including cleanup timing.",
        MethodCategory.SOLVER_ACCELERATOR, "Tuned PSOR", "wall_time_seconds_at_fixed_natural_residual",
        MetricDirection.LOWER_IS_BETTER,
        ("unrolled_projected_sor", "learned_feasible_warm_start"),
        ("Match the tuned-PSOR natural-residual tolerance.", "Include ordinary cleanup in timing."),
        stochastic=True,
    ),
    "M04": _spec(
        "M04", "Free-boundary representation",
        "Compare an implicit level-set network and scalar boundary head with price-surface boundary extraction.",
        MethodCategory.SURROGATE_COMPRESSION, "Price-surface boundary extraction", "subgrid_boundary_rmse",
        MetricDirection.LOWER_IS_BETTER,
        ("implicit_level_set", "scalar_boundary_head"),
        ("Use sub-grid boundary RMSE as the primary metric.",),
        stochastic=True,
        required_outputs=STANDARD_OUTPUTS + ("training_history.csv",),
    ),
    "M05": _spec(
        "M05", "Schrodingerization",
        "Solve a frozen continuation-region linear problem and then one live obstacle update.",
        MethodCategory.SOLVER_ACCELERATOR, "Classical discretization floor", "error_vs_classical_discretization_floor",
        MetricDirection.LOWER_IS_BETTER,
        ("frozen_continuation", "one_live_obstacle_update"),
        ("Error stays below the classical discretization floor.",),
        screen_budget_seconds=14_400,
    ),
    "M06": _spec(
        "M06", "Quantum linear systems",
        "Run VQLS on actual N=8,16,32 CN matrices and provide QSVT resource accounting.",
        MethodCategory.SOLVER_ACCELERATOR, "Classical CN linear solve", "linear_system_residual",
        MetricDirection.LOWER_IS_BETTER,
        ("vqls_n8", "vqls_n16", "vqls_n32", "qsvt_resource_accounting"),
        ("Report residual, conditioning, encoding, shots, and readout.", "Do not make a speedup claim."),
        stochastic=True,
        screen_budget_seconds=14_400,
    ),
    "M07": _spec(
        "M07", "Nonlinear quantum reduction",
        "Construct a small occupation-measure/optimal-stopping LP and measure dimensional growth against the direct LCP.",
        MethodCategory.SOLVER_ACCELERATOR, "Direct LCP", "dimension_growth_ratio",
        MetricDirection.LOWER_IS_BETTER,
        ("occupation_measure_lp", "optimal_stopping_lp"),
        ("Treat this as a falsification experiment.",),
        screen_budget_seconds=14_400,
    ),
    "M08": _spec(
        "M08", "Tensor and low rank",
        "Test TT-SVD, TT-cross/completion, POD, and primal-dual reduced basis on V(x,tau,sigma,r,q).",
        MethodCategory.SURROGATE_COMPRESSION, "Full dense surface", "compression_ratio",
        MetricDirection.HIGHER_IS_BETTER,
        ("tt_svd", "tt_cross_completion", "pod", "primal_dual_reduced_basis"),
        ("Achieve at least 10x compression.", "Diagnostic degradation is no more than 2%."),
    ),
    "M09": _spec(
        "M09", "Multi-fidelity",
        "Use complete coarse surfaces plus fewer complete fine surfaces and compare fine-label wall time at matched held-out accuracy.",
        MethodCategory.SURROGATE_COMPRESSION, "Fine-label-only training", "fine_label_wall_time_seconds_at_matched_accuracy",
        MetricDirection.LOWER_IS_BETTER,
        ("coarse_only", "coarse_plus_few_fine", "fine_only"),
        ("Use complete coarse and fine surfaces.", "Do not use undefined local patch solves."),
        stochastic=True,
        required_outputs=STANDARD_OUTPUTS + ("training_history.csv",),
    ),
    "M10": _spec(
        "M10", "Structured surrogate",
        "Compare direct price, positive premium, terminal-scaled premium, and European-anchor correction under matched parameter count and data.",
        MethodCategory.SURROGATE_COMPRESSION, "Direct price surrogate", "price_rmse",
        MetricDirection.LOWER_IS_BETTER,
        ("direct_price", "positive_premium", "terminal_scaled_premium", "european_anchor_correction"),
        ("Match parameter count and data.", "Price and diagnostic degradation is no more than 2%."),
        stochastic=True,
        required_outputs=STANDARD_OUTPUTS + ("training_history.csv",),
    ),
    "M11": _spec(
        "M11", "Inverse regularization",
        "Reconstruct deliberately masked/noisy price surfaces using spline, Tikhonov, total variation, and low-rank baselines.",
        MethodCategory.RECONSTRUCTION_UQ, "Spline, Tikhonov, total variation, and low-rank reconstruction", "gamma_rmse",
        MetricDirection.LOWER_IS_BETTER,
        ("spline", "tikhonov", "total_variation", "low_rank"),
        ("Evaluate kink, Gamma, and boundary preservation.",),
    ),
    "M12": _spec(
        "M12", "Diffusion or score model",
        "Train a compact conditional DDPM for the M11 completion task against deterministic U-Net and low-rank baselines at equal data and compute.",
        MethodCategory.RECONSTRUCTION_UQ, "Deterministic U-Net and low-rank reconstruction", "gamma_rmse",
        MetricDirection.LOWER_IS_BETTER,
        ("conditional_ddpm", "deterministic_unet", "low_rank"),
        ("Use equal data and compute.", "Evaluate kink, Gamma, and boundary preservation."),
        stochastic=True,
        screen_budget_seconds=14_400,
        required_outputs=STANDARD_OUTPUTS + ("training_history.csv",),
    ),
    "M13": _spec(
        "M13", "Adaptive sampling",
        "Compare uniform/Sobol, residual-only, and boundary-only sampling under identical point and wall-time budgets.",
        MethodCategory.SURROGATE_COMPRESSION, "Uniform/Sobol sampling", "price_rmse_at_fixed_budget",
        MetricDirection.LOWER_IS_BETTER,
        ("uniform_sobol", "residual_only", "boundary_only"),
        ("Use identical point and wall-time budgets.", "Do not use a hybrid policy."),
        stochastic=True,
        required_outputs=STANDARD_OUTPUTS + ("training_history.csv",),
    ),
    "M14": _spec(
        "M14", "Hardware acceleration",
        "Compare CPU float64 with a batched Apple-MPS implementation against the float64 oracle.",
        MethodCategory.SOLVER_ACCELERATOR, "CPU float64", "end_to_end_speedup",
        MetricDirection.HIGHER_IS_BETTER,
        ("cpu_float64", "apple_mps_batched"),
        ("Require at least 1.25x end-to-end speedup while matching the float64 oracle.", "Do not generalize to CUDA."),
    ),
    "M15": _spec(
        "M15", "MC Dropout",
        "Use 50 stochastic passes to assess error ranking and OOD detection.",
        MethodCategory.RECONSTRUCTION_UQ, "Deterministic surrogate error ranking", "large_error_ranking_auc",
        MetricDirection.HIGHER_IS_BETTER,
        ("mc_dropout_50_passes",),
        ("Treat variance as an uncalibrated diagnostic, not a 95% interval.",),
        stochastic=True,
        required_outputs=STANDARD_OUTPUTS + ("training_history.csv",),
    ),
    "M16": _spec(
        "M16", "Deep ensembles",
        "Train five independent models and measure large-error detection, disagreement/error correlation, calibration, and fivefold cost.",
        MethodCategory.RECONSTRUCTION_UQ, "Single deterministic model", "large_error_ranking_auc",
        MetricDirection.HIGHER_IS_BETTER,
        ("five_model_ensemble",),
        ("Report large-error detection, disagreement/error correlation, calibration, and fivefold cost.",),
        stochastic=True,
        required_outputs=STANDARD_OUTPUTS + ("training_history.csv",),
    ),
    "M17": _spec(
        "M17", "Conformal prediction",
        "Use split conformal on a fixed deterministic surrogate with complete-scenario scores.",
        MethodCategory.RECONSTRUCTION_UQ, "Nominal 95% split-conformal coverage", "coverage_rate",
        MetricDirection.TARGET,
        ("split_conformal_95",),
        ("Nominal 95% coverage must lie in 93--97%.", "Interval width must be non-vacuous."),
    ),
    "M18": _spec(
        "M18", "Selective fallback",
        "Gate complete scenarios with a fixed residual/domain-distance score independent of M16/M17.",
        MethodCategory.RECONSTRUCTION_UQ, "Random fallback", "accepted_scenario_p99_error",
        MetricDirection.LOWER_IS_BETTER,
        ("residual_score", "domain_distance_score"),
        ("Compare accepted-scenario P95/P99 error and total runtime with random fallback.", "Remain independent of M16/M17."),
    ),
    "M19": _spec(
        "M19", "U-Net/V-Net segmentation",
        "Predict parameter-conditioned exercise masks and compare with a smaller scalar boundary head.",
        MethodCategory.SURROGATE_COMPRESSION, "Smaller scalar boundary head", "boundary_rmse",
        MetricDirection.LOWER_IS_BETTER,
        ("unet", "vnet", "scalar_boundary_head"),
        ("Advance U-Net only if boundary error improves after parameter-count accounting.",),
        stochastic=True,
        required_outputs=STANDARD_OUTPUTS + ("training_history.csv",),
    ),
    "M20": _spec(
        "M20", "Transfer learning",
        "Compare European pretraining followed by American fine-tuning with scratch, frozen-backbone, and correction-head controls under equal American-label budgets.",
        MethodCategory.SURROGATE_COMPRESSION, "Scratch training", "price_rmse_at_equal_american_label_budget",
        MetricDirection.LOWER_IS_BETTER,
        ("scratch", "european_pretrain_finetune", "frozen_backbone", "correction_head"),
        ("Use equal American-label budgets.",),
        stochastic=True,
        required_outputs=STANDARD_OUTPUTS + ("training_history.csv",),
    ),
    "M21": _spec(
        "M21", "Explainable AI",
        "Test gradients/integrated gradients, financially valid LIME/SHAP perturbations, counterfactual boundary distance, and influence diagnostics.",
        MethodCategory.DIAGNOSTIC_WORKFLOW, "Seeded shortcut and failure cases", "seeded_shortcut_detection_rate",
        MetricDirection.HIGHER_IS_BETTER,
        ("gradients_integrated_gradients", "financial_lime_shap", "counterfactual_boundary_distance", "influence"),
        ("Require stable detection of seeded shortcut/failure cases.",),
        stochastic=True,
    ),
    "M22": _spec(
        "M22", "Mechanistic audit",
        "Perform exact 2^7 masked-input interaction decomposition at representative points.",
        MethodCategory.DIAGNOSTIC_WORKFLOW, "Known financial invariances", "invariance_consistency_rate",
        MetricDirection.HIGHER_IS_BETTER,
        ("masked_input_interaction_decomposition",),
        ("Evaluate stability across seeds and consistency with known financial invariances.",),
    ),
    "M23": _spec(
        "M23", "Agentic search",
        "Give three blind Codex agents eight validation-only trials each, then compare with equal-budget random and fixed-grid search before one holdout evaluation.",
        MethodCategory.DIAGNOSTIC_WORKFLOW, "Equal-budget random and fixed-grid search", "holdout_metric_after_validation_selection",
        MetricDirection.LOWER_IS_BETTER,
        ("three_blind_agents", "random_search", "fixed_grid_search"),
        ("Each agent has eight validation-only trials.", "Evaluate each frozen selection once on holdout."),
        stochastic=True,
    ),
    "M24": _spec(
        "M24", "Formal verification and certificates",
        "Check discrete LCP, obstacle, monotonicity, convexity, terminal, boundary, and interval/high-precision certificates on valid and deliberately corrupted surfaces.",
        MethodCategory.DIAGNOSTIC_WORKFLOW, "Deliberately corrupted surfaces", "corruption_detection_rate",
        MetricDirection.HIGHER_IS_BETTER,
        ("discrete_lcp", "shape_constraints", "terminal_boundary", "interval_high_precision"),
        ("Report detection and false-positive rates.",),
    ),
}


def get_method_spec(method_id: str) -> MethodSpec:
    """Return a registered method or raise a helpful error for an unknown id."""
    try:
        return METHOD_REGISTRY[method_id]
    except KeyError as error:
        raise ValueError(f"Unknown campaign method: {method_id}") from error


def list_method_specs() -> tuple[MethodSpec, ...]:
    """Return the stable M01--M24 campaign ordering."""
    return tuple(METHOD_REGISTRY[method_id] for method_id in sorted(METHOD_REGISTRY))


def verify_registry() -> dict[str, Any]:
    """Validate the static registry and return a stable CLI-ready summary."""
    expected_ids = {f"M{index:02d}" for index in range(1, 25)}
    if set(METHOD_REGISTRY) != expected_ids:
        raise ValueError("Campaign registry must contain exactly M01 through M24")

    categories = {category.value: 0 for category in MethodCategory}
    for method_id, spec in METHOD_REGISTRY.items():
        if spec.method_id != method_id:
            raise ValueError(f"Registry key mismatch for {method_id}")
        if not spec.name or not spec.scope or not spec.baseline or not spec.primary_metric:
            raise ValueError(f"Registry entry {method_id} has an incomplete contract")
        if spec.screen_budget_seconds not in {3_600, 14_400} or spec.confirmation_budget_seconds != 21_600:
            raise ValueError(f"Registry entry {method_id} has invalid campaign budgets")
        if not 1 <= len(spec.predeclared_configurations) <= 8:
            raise ValueError(f"Registry entry {method_id} must declare one to eight configurations")
        if not spec.required_outputs or not spec.gates:
            raise ValueError(f"Registry entry {method_id} is missing outputs or gates")
        categories[spec.category.value] += 1

    return {"categories": categories, "methods": len(METHOD_REGISTRY), "valid": True}
