# M09 Standalone Multi-Fidelity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether complete coarse CN/PSOR surfaces reduce the fine-label generation cost required to reach matched held-out price accuracy without degrading shared American-option diagnostics.

**Architecture:** Use one identical CPU-float64 MLP for every learned arm. Compare `fine_only`, `coarse_only`, and `coarse_plus_few_fine`; the multi-fidelity arm pretrains the same MLP on all complete coarse training surfaces and fine-tunes it on nested complete fine-regime subsets. Select matched-accuracy fractions on validation data only, then evaluate once on the ordinary test split; never access the sealed blind holdout.

**Tech Stack:** Python 3.12 campaign environment, NumPy, PyTorch, repository CN/PSOR solver and diagnostic utilities, pytest.

## Global Constraints

- Base branch: `campaign/independent-methods` after M08 closure commit `598a434d6876975311ac916f3c3e8de478b4cd61`.
- M09 screen budget is exactly 3600 seconds.
- Canonical learned seeds are exactly `[101,203,307,409,503]`.
- Fine-label fractions are exactly `[0.10,0.25,0.50,1.00]` of complete training regimes.
- Coarse/fine labels are complete matched surfaces; no local patch solves or boundary-only fine labels.
- Canonical execution is CPU float64.
- Hyperparameter/model selection uses validation data only; test data is evaluated after candidate selection.
- Sealed blind holdout access is prohibited and manifest must report `blind_holdout_accessed=false`.
- Promotion requires >=2% fine-label cost reduction with 95% CI excluding the threshold, ordinary diagnostic degradation <=2%, residual degradation <=5%, and hard obstacle violation <=1e-10.
- Smoke mode is always `diagnostic_only`.

---

### Task 1: Freeze pure M09 protocol helpers

**Files:**
- Create: `experiments/standalone/m09_multifidelity_core.py`
- Test: `tests/test_m09_multifidelity.py`
- Create: `configs/standalone/m09_multifidelity.json`

**Interfaces:**
- Produces `validate_config`, `nested_regime_subsets`, `assert_complete_regime_rows`, matched-accuracy selection helpers, paired bootstrap helpers, `candidate_passes`, and `classify_status`.

- [ ] Write focused failing tests for the exact seeds/fractions/grids/gates, nested complete-regime selection, validation-only matching, paired CI, and smoke non-promotion.
- [ ] Run `python -m pytest tests/test_m09_multifidelity.py -q` and verify RED because `m09_multifidelity_core.py` does not yet exist.
- [ ] Implement the smallest pure helper module satisfying those tests.
- [ ] Re-run the focused test file and verify GREEN.

### Task 2: Implement complete coarse/fine surface generation and normalized float64 training

**Files:**
- Create: `experiments/standalone/m09_multifidelity.py`
- Test: `tests/test_m09_multifidelity.py`

**Interfaces:**
- Consumes repository `SolverConfig`, `solve_option`, `delta_gamma`, `boundary_curve`, `fd_operator_matrix`, `time_boundary_values`, and `build_mlp`.
- Produces matched complete surface caches, train-only feature normalization, float64 MLP training, and prediction helpers.

- [ ] Add tests that construct a tiny matched coarse/fine regime and assert same regime key, full `(n+1)*(m+1)` node counts, identical architecture/parameter count across arms, and actual weight transfer from coarse pretraining to fine tuning.
- [ ] Run those tests and verify RED.
- [ ] Implement regime generation, fixed regime split, complete surface construction, normalization fit on training rows only, float64 training with best-validation checkpoint, and weight-transfer fine tuning.
- [ ] Re-run focused tests and verify GREEN.

### Task 3: Implement fine-grid diagnostics and matched-accuracy selection

**Files:**
- Modify: `experiments/standalone/m09_multifidelity.py`
- Test: `tests/test_m09_multifidelity.py`

**Interfaces:**
- Produces per-regime price/Delta/Gamma/boundary/obstacle/dual/complementarity/continuation-residual diagnostics on fine CN/PSOR test surfaces.

- [ ] Add tests for a perfect fine-oracle surface (near-zero reconstruction error and obstacle violation) and validation-only fraction selection.
- [ ] Verify RED.
- [ ] Implement surface reconstruction, CN residual diagnostics, boundary comparison, and per-regime metric rows.
- [ ] Verify GREEN.

### Task 4: Implement timings, evidence artifacts, and campaign gate

**Files:**
- Modify: `experiments/standalone/m09_multifidelity.py`
- Test: `tests/test_m09_multifidelity.py`

**Interfaces:**
- Produces `config_manifest.json`, `metrics.csv`, `training_history.csv`, `decision.json`, and `artifact_hashes.json` under `results/standalone/M09/`.

- [ ] Add tests that smoke cannot promote, manifest sets `blind_holdout_accessed=false`, label-generation timings are separated from fixed validation/test oracle overhead, and all required artifacts are emitted.
- [ ] Verify RED.
- [ ] Implement measured `coarse_label_seconds`, `fine_train_label_seconds`, `fine_eval_label_seconds`, `interpolation_seconds`, `coarse_pretrain_seconds`, `fine_tune_seconds`, `training_seconds`, `inference_seconds`, and `total_seconds`.
- [ ] Compute per-seed matched fractions on validation only; compare selected `coarse_plus_few_fine` against selected `fine_only` on test.
- [ ] Bootstrap paired fine-label cost reduction across model seeds and paired diagnostic degradation across complete test regimes.
- [ ] Apply 2%/2%/5%/1e-10 gates and write decision status.
- [ ] Verify GREEN.

### Task 5: Verification and canonical handoff

**Files:**
- No new files before canonical execution.

- [ ] Run `python -m pytest tests/test_m09_multifidelity.py -q`.
- [ ] Run `python -m compileall experiments/standalone/m09_multifidelity_core.py experiments/standalone/m09_multifidelity.py`.
- [ ] Run relevant campaign tests, then the full suite if the focused tests are clean.
- [ ] Run smoke: `.venv-campaign/bin/python experiments/standalone/m09_multifidelity.py --config configs/standalone/m09_multifidelity.json --mode smoke`.
- [ ] Commit implementation/config/tests/plan as `feat: add standalone M09 multi fidelity` before the canonical run.
- [ ] From the clean implementation commit, run canonical once: `.venv-campaign/bin/python experiments/standalone/m09_multifidelity.py --config configs/standalone/m09_multifidelity.json --mode canonical`.
- [ ] Inspect evidence, close `docs/campaign/m09_closure.md`, then commit only evidence/closure as `results: close M09 multi fidelity`.
