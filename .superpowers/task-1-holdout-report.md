# Task 1C Holdout Report

Status: DONE

Branch: `feature/campaign-holdout`

## Review Round 1

- Removed the `blind_holdout` constructor argument and all stored holdout state from `CampaignSplits`.
- Final evaluation now verifies the supplied seed against the public commitment before deriving regimes with `generate_holdout_regimes(evaluator_seed)`.
- Added tests that both public and private holdout attributes are absent before final access, and that final access exactly matches direct generation from the verified seed.
- Strengthened regime validation to reject missing fields, non-finite numeric values, invalid source labels, and out-of-bounds Sobol parameters while permitting finite stress values outside the in-domain bounds.

RED command for review round 1:

```bash
PYTHONPATH=src /Users/liangzihan/Documents/SURF_0159/SURF/.worktrees/american-option-risk-surfaces/.venv-pilot/bin/python -m pytest tests/test_campaign_leakage.py tests/test_campaign_holdout.py -q
```

Result: failed because `CampaignSplits` still required `blind_holdout`, while malformed regime validation tests also failed as expected.

## Scope

- Added phase-gated campaign split access for `train`, `validation`, `legacy_test`, and `blind_holdout`.
- Added public SHA-256 commitment verification using the required UTF-8 line format.
- Added deterministic 128-regime put holdout generation: 96 scrambled Sobol in-domain regimes and 32 fixed stress regimes.
- Added a canonical regime manifest with generator contract, bounds, counts, public commitment hash, and canonical-regime SHA-256.

## TDD Evidence

RED command (before the campaign package existed):

```bash
PYTHONPATH=src /Users/liangzihan/Documents/SURF_0159/SURF/.worktrees/american-option-risk-surfaces/.venv-pilot/bin/python -m pytest tests/test_campaign_leakage.py tests/test_campaign_holdout.py -q
```

Result: failed during collection with `ModuleNotFoundError: No module named 'campaign'`.

An additional RED check proved that a public `blind_holdout` attribute bypassed the access gate; it failed with `Failed: DID NOT RAISE <class 'AttributeError'>`. The holdout is now stored privately and is returned only through verified final-evaluation access.

## Final Verification

```bash
PYTHONPATH=src /Users/liangzihan/Documents/SURF_0159/SURF/.worktrees/american-option-risk-surfaces/.venv-pilot/bin/python -m pytest tests/test_campaign_leakage.py tests/test_campaign_holdout.py -q
```

Result after review round 1: `21 passed`.

```bash
PYTHONPATH=src PYTHONPYCACHEPREFIX=/private/tmp/american_option_pycache /Users/liangzihan/Documents/SURF_0159/SURF/.worktrees/american-option-risk-surfaces/.venv-pilot/bin/python -m compileall -q src/campaign tests/test_campaign_leakage.py tests/test_campaign_holdout.py
git diff --check
```

Result: both passed. `PYTHONPYCACHEPREFIX` is required in this sandbox because the default macOS bytecode-cache path is not writable.

## Concerns

None. The real holdout seed was neither read nor embedded; only public commitment hashes and non-secret test seeds were used.
