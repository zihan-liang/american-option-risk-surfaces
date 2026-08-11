import hashlib

from campaign.holdout import (
    build_holdout_manifest,
    canonical_regime_bytes,
    generate_holdout_regimes,
    verify_public_commitment,
)


def _commitment_for(seed: int) -> str:
    canonical = f"american-option-risk-surfaces:independent-methods:v1:{seed}\n"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def test_public_commitment_uses_the_exact_utf8_line_format():
    seed = 918273
    commitment = _commitment_for(seed)

    assert verify_public_commitment(seed, commitment)
    assert not verify_public_commitment(seed + 1, commitment)


def test_holdout_generator_has_expected_counts_sources_and_put_contract():
    regimes = generate_holdout_regimes(77)

    assert len(regimes) == 128
    assert sum(regime["source"] == "sobol_in_domain" for regime in regimes) == 96
    assert sum(regime["source"] == "stress" for regime in regimes) == 32
    assert {regime["option_type"] for regime in regimes} == {"put"}
    assert {regime["strike"] for regime in regimes} == {1.0}


def test_holdout_generator_is_deterministic_and_changes_in_domain_cases_with_seed():
    first = generate_holdout_regimes(77)
    second = generate_holdout_regimes(77)
    changed = generate_holdout_regimes(78)

    assert first == second
    assert [regime for regime in first if regime["source"] == "sobol_in_domain"] != [
        regime for regime in changed if regime["source"] == "sobol_in_domain"
    ]


def test_holdout_regimes_have_unique_ids_and_parameter_tuples():
    regimes = generate_holdout_regimes(77)

    ids = [regime["scenario_id"] for regime in regimes]
    tuples = [(regime["T"], regime["r"], regime["q"], regime["sigma"], regime["strike"]) for regime in regimes]
    assert len(ids) == len(set(ids)) == 128
    assert len(tuples) == len(set(tuples)) == 128


def test_manifest_records_generator_contract_and_canonical_regime_hash():
    seed = 77
    regimes = generate_holdout_regimes(seed)
    commitment = _commitment_for(seed)

    manifest = build_holdout_manifest(regimes, public_commitment_sha256=commitment)

    assert manifest["generator_version"] == "campaign-holdout-regimes-v1"
    assert manifest["counts"] == {"total": 128, "sobol_in_domain": 96, "stress": 32}
    assert set(manifest["parameter_bounds"]) == {"T", "r", "q", "sigma"}
    assert manifest["public_commitment_sha256"] == commitment
    assert manifest["canonical_regimes_sha256"] == hashlib.sha256(canonical_regime_bytes(regimes)).hexdigest()
