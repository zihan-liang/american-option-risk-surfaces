# Verification Matrix

Use the smallest verification tier that proves the claim being made. Run broader tiers before claiming a branch is complete or release-ready.

## Tier 0: Harness startup

Purpose: confirm the checkout can be used safely by an agent.

```bash
bash scripts/init_harness.sh
```

## Tier 1: Focused edit

Purpose: quick confidence for harness, boundary-head, or Delta-head edits.

```bash
$PY -m compileall -q src tests experiments
$PY -m pytest tests/test_harness_contract.py -q
$PY -m pytest tests/test_boundary_head_model.py tests/test_boundary_head_metrics.py tests/test_boundary_head_experiment_outputs.py -q
$PY -m pytest tests/test_delta_head_model.py tests/test_delta_head_outputs.py -q
```

## Tier 2: Project regression

Purpose: verify ordinary code/documentation changes before commit.

```bash
$PY -m compileall -q src tests experiments
$PY -m pytest tests -q
```

## Tier 3: Release verification

Purpose: verify release-critical artifacts and claim gates.

```bash
$PY experiments/14_claim_audit.py --config configs/claim_audit.yaml
$PY experiments/15_future_extensions_manifest.py --config configs/future_extensions.yaml
$PY experiments/16_release_verification.py
```

## Tier 4: Round-2 evidence audit

Purpose: verify the Round-2 ablation decision remains schema-safe and provenance-safe.

```bash
bash scripts/verify_round2_evidence.sh
```

This tier checks the consolidated Round-2 matrix, required CSV schema, full-scale manifest caps, and the current no-combo decision.

## Tier 5: Paper-readiness screen

Purpose: decide whether the work is ready for a larger paper push.

Required evidence:

- full `tests -q` pass;
- release verification pass;
- claim audit pass;
- verified metric ledger updated;
- all result tables have required schema and provenance manifests;
- no unsupported PSOR replacement, universal-surrogate, or Gamma-surrogate claims.

This tier is a decision review, not just a command.
