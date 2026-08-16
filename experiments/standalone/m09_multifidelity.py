#!/usr/bin/env python3
"""Corrected M09 runner: validation freezes fractions before ordinary-test evaluation."""
from __future__ import annotations

import copy
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

import m09_multifidelity_legacy as legacy
from m09_multifidelity_legacy import *  # noqa: F401,F403 - preserve tested public runner API
from m09_multifidelity_protocol import build_seeded_mlp, selected_test_keys


def _solve_test_cache(runtime: dict[str, Any], partitions: dict[str, list[dict[str, Any]]], cache: dict[str, Any], surface_rows: list[dict[str, Any]]) -> None:
    """Generate ordinary-test fine oracles only after validation has frozen fractions."""
    for regime in partitions["test"]:
        surface = legacy.solve_surface(runtime["fine_solver"], regime, "fine")
        cache["fine"][str(regime["id"])] = surface
        surface_rows.append({
            "split": "test",
            "regime_id": regime["id"],
            "fidelity": "fine",
            "nodes": int(len(surface["target"])),
            "solve_seconds": surface["solve_seconds"],
        })


def run(config_path: str | Path, *, mode: str | None = None) -> dict[str, Any]:
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    legacy.validate_config(config)
    evidence_mode = str(mode or config["evidence_mode"])
    if evidence_mode not in {"smoke", "canonical"}:
        raise ValueError("mode must be smoke or canonical")

    runtime = copy.deepcopy(config)
    if evidence_mode == "smoke":
        smoke = runtime["smoke"]
        runtime["training_seeds"] = list(smoke["training_seeds"])
        runtime["fine_fractions"] = list(smoke["fine_fractions"])
        runtime["model"]["epochs"] = int(smoke["epochs"])

    out_dir = Path(runtime["output_root"])
    out_dir.mkdir(parents=True, exist_ok=True)
    started_total = time.perf_counter()
    regimes = legacy.build_regimes(runtime)
    if evidence_mode == "smoke":
        regimes = regimes[: int(runtime["smoke"]["max_regimes"])]
    partitions = legacy.split_regimes(runtime, regimes)
    if min(len(partitions[name]) for name in ("train", "validation", "test")) < 1:
        raise RuntimeError("M09 requires nonempty train/validation/test regime partitions")

    train_ids = [str(r["id"]) for r in partitions["train"]]
    val_ids = [str(r["id"]) for r in partitions["validation"]]
    test_ids = [str(r["id"]) for r in partitions["test"]]
    fractions = tuple(float(x) for x in runtime["fine_fractions"])
    subsets = legacy.nested_regime_subsets(train_ids, fractions, selection_seed=int(runtime["selection_seed"]))

    preselection_partitions = {
        "train": partitions["train"],
        "validation": partitions["validation"],
        "test": [],
    }
    cache, surface_rows = legacy.build_surface_cache(runtime["coarse_solver"], runtime["fine_solver"], preselection_partitions)

    coarse_nodes = (int(runtime["coarse_solver"]["m"]) + 1) * (int(runtime["coarse_solver"]["n"]) + 1)
    fine_nodes = (int(runtime["fine_solver"]["m"]) + 1) * (int(runtime["fine_solver"]["n"]) + 1)
    _, _, coarse_regime_key = legacy.concatenate_surfaces(cache["coarse"], train_ids)
    legacy.assert_complete_regime_rows(coarse_regime_key, coarse_nodes, train_ids)
    _, _, fine_regime_key = legacy.concatenate_surfaces(cache["fine"], train_ids)
    legacy.assert_complete_regime_rows(fine_regime_key, fine_nodes, train_ids)

    fine_train_x_all, _, _ = legacy.concatenate_surfaces(cache["fine"], train_ids)
    mean, std = legacy.fit_normalizer(fine_train_x_all)
    val_x, val_y, _ = legacy.concatenate_surfaces(cache["fine"], val_ids)
    val_x = legacy.normalize(val_x, mean, std)
    model_spec = runtime["model"]
    hidden, depth = int(model_spec["hidden"]), int(model_spec["depth"])

    all_history: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    arm_rows: list[dict[str, Any]] = []
    trained: dict[tuple[int, str, float | None], Any] = {}

    for raw_seed in runtime["training_seeds"]:
        seed = int(raw_seed)
        coarse_x, coarse_y, _ = legacy.concatenate_surfaces(cache["coarse"], train_ids)
        coarse_x = legacy.normalize(coarse_x, mean, std)
        coarse_model = build_seeded_mlp(
            legacy.build_mlp,
            seed=seed,
            input_dim=len(legacy.FEATURE_NAMES),
            output_dim=1,
            hidden=hidden,
            depth=depth,
        )
        coarse_model, history, coarse_seconds = legacy.train_model(
            coarse_model, coarse_x, coarse_y, val_x, val_y, model_spec,
            seed=seed, arm="coarse_only", fraction=None, phase="coarse_pretrain",
        )
        all_history.extend(history)
        trained[(seed, "coarse_only", None)] = copy.deepcopy(coarse_model)
        coarse_val, coarse_val_seconds = legacy.evaluate_model(
            coarse_model, cache["fine"], val_ids, mean, std, model_spec, runtime["diagnostics"],
            seed=seed, arm="coarse_only", fraction=None, split_name="validation",
        )
        metric_rows.extend(coarse_val)
        arm_rows.append({
            "seed": seed, "arm": "coarse_only", "fraction": "",
            "validation_price_rmse": legacy._aggregate_price(coarse_val),
            "coarse_pretrain_seconds": coarse_seconds, "fine_tune_seconds": 0.0,
            "training_seconds": coarse_seconds, "validation_eval_seconds": coarse_val_seconds,
            "test_eval_seconds": 0.0, "fine_train_label_seconds": 0.0, "interpolation_seconds": 0.0,
        })

        for fraction in fractions:
            selected_ids = subsets[fraction]
            train_x, train_y, train_regime_key = legacy.concatenate_surfaces(cache["fine"], selected_ids)
            legacy.assert_complete_regime_rows(train_regime_key, fine_nodes, selected_ids)
            train_x = legacy.normalize(train_x, mean, std)

            fine_model = build_seeded_mlp(
                legacy.build_mlp,
                seed=seed,
                input_dim=len(legacy.FEATURE_NAMES),
                output_dim=1,
                hidden=hidden,
                depth=depth,
            )
            fine_model, history, fine_seconds = legacy.train_model(
                fine_model, train_x, train_y, val_x, val_y, model_spec,
                seed=seed, arm="fine_only", fraction=fraction, phase="fine_train",
            )
            all_history.extend(history)
            trained[(seed, "fine_only", fraction)] = fine_model
            fine_val, fine_val_seconds = legacy.evaluate_model(
                fine_model, cache["fine"], val_ids, mean, std, model_spec, runtime["diagnostics"],
                seed=seed, arm="fine_only", fraction=fraction, split_name="validation",
            )
            metric_rows.extend(fine_val)
            label_seconds = legacy._fraction_label_seconds(surface_rows, selected_ids)
            arm_rows.append({
                "seed": seed, "arm": "fine_only", "fraction": fraction,
                "validation_price_rmse": legacy._aggregate_price(fine_val),
                "coarse_pretrain_seconds": 0.0, "fine_tune_seconds": fine_seconds,
                "training_seconds": fine_seconds, "validation_eval_seconds": fine_val_seconds,
                "test_eval_seconds": 0.0, "fine_train_label_seconds": label_seconds, "interpolation_seconds": 0.0,
            })

            multi_model = copy.deepcopy(coarse_model)
            multi_model, history, tune_seconds = legacy.train_model(
                multi_model, train_x, train_y, val_x, val_y, model_spec,
                seed=seed, arm="coarse_plus_few_fine", fraction=fraction, phase="fine_tune",
            )
            all_history.extend(history)
            trained[(seed, "coarse_plus_few_fine", fraction)] = multi_model
            multi_val, multi_val_seconds = legacy.evaluate_model(
                multi_model, cache["fine"], val_ids, mean, std, model_spec, runtime["diagnostics"],
                seed=seed, arm="coarse_plus_few_fine", fraction=fraction, split_name="validation",
            )
            metric_rows.extend(multi_val)
            arm_rows.append({
                "seed": seed, "arm": "coarse_plus_few_fine", "fraction": fraction,
                "validation_price_rmse": legacy._aggregate_price(multi_val),
                "coarse_pretrain_seconds": coarse_seconds, "fine_tune_seconds": tune_seconds,
                "training_seconds": coarse_seconds + tune_seconds, "validation_eval_seconds": multi_val_seconds,
                "test_eval_seconds": 0.0, "fine_train_label_seconds": label_seconds, "interpolation_seconds": 0.0,
            })

    selected_by_seed: dict[int, dict[str, float | None]] = {}
    cost_reductions: list[float] = []
    selection_rows: list[dict[str, Any]] = []
    for seed in [int(x) for x in runtime["training_seeds"]]:
        fine_100 = next(
            row for row in arm_rows
            if row["seed"] == seed and row["arm"] == "fine_only" and float(row["fraction"]) == 1.0
        )
        target = legacy.matched_accuracy_target(
            float(fine_100["validation_price_rmse"]),
            float(runtime["matching"]["relative_price_tolerance"]),
        )
        chosen: dict[str, float | None] = {}
        for arm in ("fine_only", "coarse_plus_few_fine"):
            scores = {
                float(row["fraction"]): float(row["validation_price_rmse"])
                for row in arm_rows if row["seed"] == seed and row["arm"] == arm
            }
            chosen[arm] = legacy.choose_smallest_matching_fraction(scores, target)
        selected_by_seed[seed] = chosen
        baseline_fraction = chosen["fine_only"]
        candidate_fraction = chosen["coarse_plus_few_fine"]
        selection_rows.append({
            "seed": seed, "target_validation_price_rmse": target,
            "fine_only_fraction": baseline_fraction,
            "coarse_plus_few_fine_fraction": candidate_fraction,
        })
        if baseline_fraction is None or candidate_fraction is None:
            continue
        baseline_cost = legacy._fraction_label_seconds(surface_rows, subsets[baseline_fraction])
        candidate_cost = legacy._fraction_label_seconds(surface_rows, subsets[candidate_fraction])
        cost_reductions.append(100.0 * (baseline_cost - candidate_cost) / max(baseline_cost, 1e-15))

    matching_complete = len(cost_reductions) == len(runtime["training_seeds"])
    selected_test_rows: list[dict[str, Any]] = []
    test_eval_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []

    if matching_complete:
        _solve_test_cache(runtime, partitions, cache, surface_rows)

        for seed in [int(x) for x in runtime["training_seeds"]]:
            rows, seconds = legacy.evaluate_model(
                trained[(seed, "coarse_only", None)], cache["fine"], test_ids,
                mean, std, model_spec, runtime["diagnostics"],
                seed=seed, arm="coarse_only", fraction=None, split_name="test",
            )
            metric_rows.extend(rows)
            test_eval_rows.append({"seed": seed, "arm": "coarse_only", "fraction": "", "test_eval_seconds": seconds})

        for seed, arm, fraction in selected_test_keys(selected_by_seed):
            rows, seconds = legacy.evaluate_model(
                trained[(seed, arm, fraction)], cache["fine"], test_ids,
                mean, std, model_spec, runtime["diagnostics"],
                seed=seed, arm=arm, fraction=fraction, split_name="test",
            )
            metric_rows.extend(rows)
            selected_test_rows.extend(rows)
            test_eval_rows.append({"seed": seed, "arm": arm, "fraction": fraction, "test_eval_seconds": seconds})

        cost_point, cost_low, cost_high = legacy.paired_bootstrap_mean_ci(
            cost_reductions,
            repetitions=int(runtime["bootstrap"]["repetitions"]),
            seed=int(runtime["bootstrap"]["seed"]) + 100,
        )
        degradation_summary, degradation_rows = legacy._metric_degradation_summary(
            selected_test_rows, bootstrap=runtime["bootstrap"], diagnostics=runtime["diagnostics"]
        )
        degradation_upper = {
            metric: degradation_summary[f"degradation_ci_upper_pct_{metric}"]
            for metric in legacy.ALL_METRICS
        }
        selected_candidate_obstacle = max(
            float(r["obstacle_violation"])
            for r in selected_test_rows if r["arm"] == "coarse_plus_few_fine"
        )
        passed = legacy.candidate_passes(
            cost_reduction_ci_low_pct=cost_low,
            degradation_ci_upper_pct=degradation_upper,
            obstacle_violation=selected_candidate_obstacle,
            gates=runtime["gates"],
        )

        first_test = cache["fine"][test_ids[0]]["features"]
        for seed, arm, fraction in selected_test_keys(selected_by_seed):
            mean_seconds, std_seconds = legacy._time_inference(
                trained[(seed, arm, fraction)], first_test, mean, std, model_spec, runtime["timing"]
            )
            timing_rows.append({
                "seed": seed, "arm": arm, "fraction": fraction,
                "inference_seconds_mean": mean_seconds, "inference_seconds_std": std_seconds,
            })
    else:
        cost_point = cost_low = cost_high = float("-inf")
        degradation_summary = {}
        degradation_rows = []
        selected_candidate_obstacle = float("inf")
        passed = False

    fixed_eval_seconds = legacy._fixed_eval_label_seconds(surface_rows)
    total_seconds = float(time.perf_counter() - started_total)
    budget_exceeded = evidence_mode == "canonical" and total_seconds > float(runtime["budget_seconds"])
    status = legacy.classify_status(evidence_mode, passed=passed)
    if budget_exceeded:
        status = "resource_blocked"
        passed = False

    manifest = {
        "method_id": "M09", "phase": "screen", "evidence_mode": evidence_mode,
        "branch": legacy._git("branch", "--show-current"),
        "implementation_sha": legacy._git("rev-parse", "HEAD"),
        "python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "torch": __import__("torch").__version__,
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "feature_names": list(legacy.FEATURE_NAMES),
        "model_parameter_count": legacy.model_parameter_count(next(iter(trained.values()))),
        "training_seeds": list(runtime["training_seeds"]),
        "fine_fractions": list(runtime["fine_fractions"]),
        "split_regime_keys": {name: [str(r["id"]) for r in partitions[name]] for name in ("train", "validation", "test")},
        "nested_fine_regime_subsets": {str(k): list(v) for k, v in subsets.items()},
        "coarse_grid": {"m": runtime["coarse_solver"]["m"], "n": runtime["coarse_solver"]["n"]},
        "fine_grid": {"m": runtime["fine_solver"]["m"], "n": runtime["fine_solver"]["n"]},
        "blind_holdout_accessed": False,
        "scope_limitation": "Classical 1D American-put multi-fidelity falsification screen using complete matched CN/PSOR surfaces only; validation freezes fractions before ordinary-test oracle generation/evaluation; no local patch solves and no blind holdout access.",
    }
    legacy._write_json(out_dir / "config_manifest.json", manifest)
    legacy._write_csv(out_dir / "surface_generation.csv", surface_rows)
    legacy._write_csv(out_dir / "metrics.csv", metric_rows + arm_rows + selection_rows + degradation_rows + test_eval_rows + timing_rows)
    legacy._write_csv(out_dir / "training_history.csv", all_history)

    decision = {
        "method_id": "M09", "status": status,
        "claim_status": "diagnostic_only" if status != "candidate_supported" else "candidate_supported",
        "blind_holdout_accessed": False, "matched_accuracy_complete": matching_complete,
        "selected_fractions_by_seed": {str(k): v for k, v in selected_by_seed.items()},
        "fine_label_cost_reduction_pct": cost_point if np.isfinite(cost_point) else None,
        "fine_label_cost_reduction_ci_low_pct": cost_low if np.isfinite(cost_low) else None,
        "fine_label_cost_reduction_ci_high_pct": cost_high if np.isfinite(cost_high) else None,
        "shared_diagnostic_degradation": degradation_summary,
        "selected_candidate_obstacle_violation": selected_candidate_obstacle if np.isfinite(selected_candidate_obstacle) else None,
        "numerical_gate_passed": bool(passed),
        "coarse_label_seconds": float(sum(float(r["solve_seconds"]) for r in surface_rows if r["fidelity"] == "coarse")),
        "fine_eval_label_seconds": fixed_eval_seconds,
        "interpolation_seconds": 0.0, "total_seconds": total_seconds,
        "reason": (
            "Canonical M09 exceeded the frozen one-hour screen budget; no promotion claim is allowed."
            if budget_exceeded else
            "Multi-fidelity met matched held-out accuracy with the required fine-label cost CI and all shared diagnostic gates."
            if passed else
            "Multi-fidelity did not satisfy matched-accuracy fine-label cost and all shared diagnostic gates."
        ),
        "limitations": [
            "The primary cost claim counts fine training-label generation only; validation/test fine-oracle generation is fixed audit overhead and reported separately.",
            "Coarse pretraining uses complete global coarse surfaces and fine tuning uses complete selected fine regimes; no boundary-local patch labels are used.",
            "Known payoff, terminal, and spatial boundary constraints are imposed identically on every learned arm before diagnostics.",
            "Interpolation cost is exactly zero because both fidelities train the same continuous-coordinate MLP; no coarse surface is interpolated into a fine label field.",
            "Ordinary-test fine oracles and model diagnostics are generated only after validation-only matched-accuracy selection is frozen.",
        ],
    }
    legacy._write_json(out_dir / "decision.json", decision)
    artifact_paths = [out_dir / name for name in (
        "config_manifest.json", "surface_generation.csv", "metrics.csv", "training_history.csv", "decision.json"
    )]
    legacy._write_json(out_dir / "artifact_hashes.json", {path.name: legacy._sha256(path) for path in artifact_paths})
    return decision


def main() -> None:
    parser = legacy.argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/standalone/m09_multifidelity.json")
    parser.add_argument("--mode", choices=("smoke", "canonical"), default=None)
    args = parser.parse_args()
    print(json.dumps(run(args.config, mode=args.mode), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
