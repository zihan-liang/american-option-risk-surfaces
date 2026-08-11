import hashlib

import pytest

from campaign.data_access import BlindHoldoutAccessError, CampaignSplits
from campaign.holdout import generate_holdout_regimes


def _commitment_for(seed: int) -> str:
    canonical = f"american-option-risk-surfaces:independent-methods:v1:{seed}\n"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _splits() -> CampaignSplits:
    return CampaignSplits(
        train=({"scenario_id": "train-001"}, {"scenario_id": "train-002"}),
        validation=({"scenario_id": "validation-001"},),
        legacy_test=({"scenario_id": "legacy-001"},),
        public_commitment_sha256=_commitment_for(41),
    )


def test_train_and_validation_scenario_ids_do_not_cross():
    splits = _splits()

    train_ids = {scenario["scenario_id"] for scenario in splits.development("train")}
    validation_ids = {scenario["scenario_id"] for scenario in splits.development("validation")}

    assert train_ids == {"train-001", "train-002"}
    assert validation_ids == {"validation-001"}
    assert train_ids.isdisjoint(validation_ids)


def test_legacy_split_is_exposed_under_its_campaign_name():
    splits = _splits()

    assert splits.development("legacy_test") == ({"scenario_id": "legacy-001"},)
    with pytest.raises(ValueError):
        splits.development("test")


def test_development_and_model_selection_have_no_holdout_payload_before_final_access():
    splits = _splits()

    with pytest.raises(BlindHoldoutAccessError):
        splits.development("blind_holdout")
    with pytest.raises(BlindHoldoutAccessError):
        splits.model_selection("blind_holdout")
    with pytest.raises(AttributeError):
        _ = splits.blind_holdout
    with pytest.raises(AttributeError):
        _ = splits._blind_holdout


def test_final_holdout_access_requires_frozen_configs_and_verified_seed():
    splits = _splits()

    with pytest.raises(BlindHoldoutAccessError):
        splits.final_evaluation(configs_frozen=False, evaluator_seed=41)
    with pytest.raises(BlindHoldoutAccessError):
        splits.final_evaluation(configs_frozen=True, evaluator_seed=42)

    holdout = splits.final_evaluation(configs_frozen=True, evaluator_seed=41)

    assert holdout == generate_holdout_regimes(41)
