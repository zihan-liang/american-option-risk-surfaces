import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

from campaign.m03 import (
    REQUIRED_SEEDS,
    build_regime_splits,
    classify_screen_decision,
    run_m03_screen,
    select_fixed_configuration,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_SCRIPT = REPOSITORY_ROOT / "experiments" / "26_independent_method_campaign.py"


def _assert_finite_json(payload):
    if isinstance(payload, dict):
        for value in payload.values():
            _assert_finite_json(value)
    elif isinstance(payload, list):
        for value in payload:
            _assert_finite_json(value)
    elif isinstance(payload, float):
        assert math.isfinite(payload)
    elif isinstance(payload, str):
        assert payload not in {"NaN", "Infinity", "-Infinity"}


def test_regimes_are_split_before_node_expansion_without_leakage():
    ranges = {
        "t": [0.5, 0.5],
        "r": [0.03, 0.03],
        "q": [0.01, 0.01],
        "sigma": [0.25, 0.25],
    }
    splits = build_regime_splits(
        train_count=4,
        validation_count=3,
        generation_seed=1701,
        parameter_ranges=ranges,
    )
    train_ids = {regime["scenario_id"] for regime in splits.train}
    validation_ids = {regime["scenario_id"] for regime in splits.validation}

    assert len(train_ids) == 4
    assert len(validation_ids) == 3
    assert train_ids.isdisjoint(validation_ids)
    assert all(regime["option_type"] == "put" for regime in (*splits.train, *splits.validation))
    assert all(regime["sigma"] == 0.25 for regime in (*splits.train, *splits.validation))


def test_selection_is_one_configuration_fixed_across_all_required_seeds():
    rows = []
    for seed in REQUIRED_SEEDS:
        rows.extend(
            [
                {"config_id": "blend_0p5", "seed": seed, "cleanup_iterations": 10.0},
                {"config_id": "blend_1p0", "seed": seed, "cleanup_iterations": 8.0},
            ]
        )

    selection = select_fixed_configuration(rows, required_seeds=REQUIRED_SEEDS)

    assert selection.config_id == "blend_1p0"
    assert selection.seeds == REQUIRED_SEEDS
    assert set(selection.selected_config_by_seed.values()) == {"blend_1p0"}


def test_budget_exhaustion_is_resource_blocked_without_overriding_smoke_policy():
    canonical = classify_screen_decision(
        smoke=False,
        budget_exhausted=True,
        speed_improved=True,
        diagnostics_pass=True,
        required_available=True,
    )
    smoke = classify_screen_decision(
        smoke=True,
        budget_exhausted=True,
        speed_improved=True,
        diagnostics_pass=True,
        required_available=True,
    )

    assert canonical == ("resource_blocked", "diagnostic_only", "screen_budget_exhausted", False)
    assert smoke == ("diagnostic_only", "diagnostic_only", "smoke_non_evidence", False)


def test_smoke_screen_writes_standard_outputs_hashes_and_never_promotes(tmp_path):
    output_dir = tmp_path / "M03"
    payload = run_m03_screen(
        output_dir=output_dir,
        smoke=True,
        config_overrides={
            "solver": {"m": 8, "n": 3, "max_iter": 2_000},
            "sampling": {"train_regimes": 2, "validation_regimes": 2},
            "network": {"hidden": 8, "depth": 1, "epochs": 1, "batch_size": 64, "patience": 1},
            "timing": {"warmups": 0, "repetitions": 1},
            "statistics": {"bootstrap_iterations": 200},
        },
        provenance={"branch": "exp/m03-algorithm-unrolling", "commit": "test-commit"},
    )

    expected = {
        "artifact_hashes.json",
        "config_manifest.json",
        "decision.json",
        "run_metrics.csv",
        "scenario_metrics.csv",
        "screen_summary.json",
        "selection_metrics.csv",
        "timing_samples.csv",
        "training_history.csv",
    }
    assert {path.name for path in output_dir.iterdir()} == expected
    assert payload["status"] == "diagnostic_only"
    assert payload["evidence"] is False
    assert payload["promotion_eligible"] is False
    assert payload["blind_holdout_accessed"] is False

    manifest = json.loads((output_dir / "config_manifest.json").read_text(encoding="utf-8"))
    decision = json.loads((output_dir / "decision.json").read_text(encoding="utf-8"))
    summary = json.loads((output_dir / "screen_summary.json").read_text(encoding="utf-8"))
    for document in (manifest, decision, summary):
        _assert_finite_json(document)
    assert manifest["provenance"] == {"branch": "exp/m03-algorithm-unrolling", "commit": "test-commit"}
    assert manifest["config"]["actual_counts"]["train_regimes"] == 2
    assert manifest["config"]["actual_counts"]["validation_regimes"] == 2
    assert manifest["config"]["blind_holdout_accessed"] is False
    assert decision["status"] == "diagnostic_only"
    assert decision["blind_holdout_accessed"] is False
    assert decision["evidence"] is False
    assert summary["decision"]["promotion_eligible"] is False
    assert summary["decision"]["reason"] == "smoke_non_evidence"
    costs = summary["costs_seconds"]
    disjoint_cost = sum(
        costs[name]
        for name in ("baseline_selection", "candidate_selection", "label_generation", "preprocessing", "training_total")
    )
    assert disjoint_cost <= costs["elapsed_before_writes"] + 0.01

    with (output_dir / "run_metrics.csv").open(newline="", encoding="utf-8") as handle:
        run_rows = list(csv.DictReader(handle))
    assert len(run_rows) == len(REQUIRED_SEEDS) + 1
    assert {row["config_id"] for row in run_rows} == {summary["selection"]["candidate_config_id"]}
    assert {int(row["seed"]) for row in run_rows if row["seed"]} == set(REQUIRED_SEEDS)
    assert all(row["status"] == "diagnostic_only" for row in run_rows)
    for row in run_rows:
        optional_metrics = json.loads(row["optional_metrics_json"])
        assert optional_metrics["blind_holdout_accessed"] == 0.0
        assert optional_metrics["cleanup_iterations_mean"] >= 0.0
        assert optional_metrics["fallback_count"] >= 0.0
        assert optional_metrics["residual_tolerance"] == manifest["config"]["residual_tolerance"]

    with (output_dir / "training_history.csv").open(newline="", encoding="utf-8") as handle:
        history_rows = list(csv.DictReader(handle))
    assert {int(row["seed"]) for row in history_rows} == set(REQUIRED_SEEDS)

    hashes = json.loads((output_dir / "artifact_hashes.json").read_text(encoding="utf-8"))
    assert set(hashes["files"]) == expected - {"artifact_hashes.json"}
    for filename, expected_digest in hashes["files"].items():
        assert hashlib.sha256((output_dir / filename).read_bytes()).hexdigest() == expected_digest


def test_cli_dispatches_m03_smoke(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            str(CAMPAIGN_SCRIPT),
            "--method",
            "M03",
            "--phase",
            "screen",
            "--smoke",
            "--output-root",
            str(tmp_path),
        ],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["method_id"] == "M03"
    assert payload["status"] == "diagnostic_only"
    assert payload["evidence"] is False
    assert (tmp_path / "M03" / "decision.json").exists()
