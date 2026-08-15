from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def test_baseline_inference_accepts_only_features_and_extracts_from_its_prediction() -> None:
    from campaign.m04 import predict_price_surrogate_boundaries

    class PositivePremiumModel:
        def __call__(self, features):
            return np.maximum(np.asarray(features)[:, 0] + 0.2, 0.0)

    features = np.array([[0.5], [0.7], [0.9]], dtype=np.float64)
    result = predict_price_surrogate_boundaries(
        PositivePremiumModel(),
        feature_rows=features,
        spots=np.array([0.5, 0.7, 0.9]),
        theta=np.array([0.5]),
        strike=1.0,
    )

    assert len(result) == 1
    assert result[0].status == "no_exercise_region"
    assert result[0].value is None


def test_m04_dispatcher_routes_only_the_explicit_screen_path() -> None:
    path = Path("experiments/26_independent_method_campaign.py")
    spec = importlib.util.spec_from_file_location("m04_dispatcher", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    parser = module._build_parser()
    args = parser.parse_args(["--method", "M04", "--phase", "screen", "--smoke", "--config", "configs/independent_m04_free_boundary.yaml"])

    assert args.method == "M04"
    assert args.phase == "screen"
    assert args.smoke is True


def test_m04_config_freezes_complete_regime_counts_and_no_early_stopping() -> None:
    from common import load_config

    config = load_config("configs/independent_m04_free_boundary.yaml")

    assert config["seeds"] == [101, 203, 307, 409, 503]
    assert config["split_counts"] == {"train": 10, "validation": 5, "legacy_test": 5}
    assert config["epochs"] > 0
    assert "patience" not in config
    assert config["timing"] == {"warmups": 5, "repetitions": 30}


def test_output_writer_keeps_all_methods_and_seeds_visible_without_nonfinite_values(tmp_path) -> None:
    from campaign.m04 import write_m04_outputs

    rows = [
        {"method": method, "seed": seed, "split": "validation", "boundary_rmse": 0.1, "status": "defined"}
        for method in ("price_surrogate_boundary_extraction", "scalar_boundary_head", "implicit_level_set")
        for seed in (101, 203, 307, 409, 503)
    ]
    write_m04_outputs(tmp_path, config={"blind_holdout_accessed": False}, run_rows=rows, scenario_rows=rows, profiles=[], timing_rows=[], history=[])

    required = {"config_manifest.json", "run_metrics.csv", "scenario_metrics.csv", "boundary_profiles.csv", "timing_samples.csv", "training_history.csv", "screen_summary.json", "decision.json", "artifact_hashes.json"}
    assert required.issubset({item.name for item in tmp_path.iterdir()})
    assert "NaN" not in (tmp_path / "screen_summary.json").read_text()


def test_train_only_normalization_cannot_be_refit_from_validation_features() -> None:
    from campaign.m04 import TrainingNormalizer

    normalizer = TrainingNormalizer.fit(np.array([[0.0, 2.0], [2.0, 4.0]], dtype=np.float64))
    transformed = normalizer.transform(np.array([[100.0, 200.0]], dtype=np.float64))

    assert normalizer.mean.tolist() == [1.0, 3.0]
    assert transformed[0, 0] == pytest.approx(99.0)


def test_dispatcher_routes_m04_screen_to_its_dedicated_runner(monkeypatch) -> None:
    from campaign import m04

    path = Path("experiments/26_independent_method_campaign.py")
    spec = importlib.util.spec_from_file_location("m04_dispatcher_route", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(m04, "run_m04_screen", lambda **kwargs: {"routed": kwargs["smoke"]})

    result = module.run(SimpleNamespace(method="M04", phase="screen", output_root=None, config="configs/independent_m04_free_boundary.yaml", smoke=True, list_methods=False, verify_all=False))

    assert result == {"routed": True}


def test_m04_direct_entrypoint_delegates_to_the_campaign_runner(monkeypatch) -> None:
    from campaign import m04

    path = Path("experiments/28_m04_free_boundary.py")
    spec = importlib.util.spec_from_file_location("m04_direct_entrypoint", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(m04, "run_m04_screen", lambda **kwargs: {"direct": kwargs["smoke"]})

    assert module.run("configs/independent_m04_free_boundary.yaml", smoke=True) == {"direct": True}
