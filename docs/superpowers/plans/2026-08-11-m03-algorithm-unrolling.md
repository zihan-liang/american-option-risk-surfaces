# M03 Algorithm Unrolling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an auditable learned feasible warm-start and PJOR-unrolling screen against tuned PSOR at an identical natural-residual tolerance.

**Architecture:** Numerical and learning primitives live in `src/models/algorithm_unrolling.py`; complete-regime generation, training, validation selection, diagnostics, statistics, and evidence writing live in `src/campaign/m03.py`. The shared campaign CLI dispatches only M03 screen runs and leaves all other method scaffolds unchanged.

**Tech Stack:** Python 3.12, NumPy, SciPy, PyTorch CPU float64, psutil, campaign registry/output/statistics helpers, pytest.

## Global Constraints

- Work only on branch `exp/m03-algorithm-unrolling` in its assigned worktree.
- Do not edit core solver files or access the blind holdout.
- Use seeds `[101, 203, 307, 409, 503]` and complete regimes as split/bootstrap units.
- Use no more than eight predeclared configurations and validation-only selection.
- Canonical screening budget is one hour; only smoke evidence may run before manager review.
- Use `MPLBACKEND=Agg`, `MPLCONFIGDIR=/private/tmp/codex_mplconfig`, and the campaign Python.

---

### Task 1: Numerical And Learning Primitives

**Files:**
- Create: `src/models/algorithm_unrolling.py`
- Test: `tests/test_m03_algorithm_unrolling.py`

**Interfaces:**
- Produces: `natural_residual(...)`, `project_feasible(...)`, `projected_sor_cleanup(...)`, `WarmStartMLP`, `FeatureNormalizer`, and `solve_cn_lcp_surface(...)`.

- [x] **Step 1: Write failing tests** for hard projection, tolerance-matched baseline/candidate cleanup, non-finite prediction fallback, and additive timing/cleanup iteration accounting.
- [x] **Step 2: Run the focused test file and record expected import/behavior failures.**
- [x] **Step 3: Implement the minimum natural-residual PSOR, projected warm-start path, float64 MLP, and result dataclasses.**
- [x] **Step 4: Re-run the focused tests until green, then refactor without changing behavior.**

### Task 2: Complete-Regime Campaign Screen

**Files:**
- Create: `src/campaign/m03.py`
- Create: `configs/m03_algorithm_unrolling.yaml`
- Test: `tests/test_m03_campaign.py`

**Interfaces:**
- Consumes: M03 primitives, `campaign.statistics`, `campaign.io`, `campaign.registry`, `campaign.records`.
- Produces: `M03ScreenConfig`, `build_regime_splits(...)`, `select_fixed_configuration(...)`, `run_m03_screen(...)`, and `validate_m03_artifacts(...)`.

- [x] **Step 1: Write failing tests** proving train/validation regime isolation, fixed candidate selection across all five seeds, standard finite/null-with-reason schemas, hashes, and forced smoke non-promotion.
- [x] **Step 2: Run the tests and record failures caused by the missing campaign module.**
- [x] **Step 3: Implement deterministic regimes, training-only normalization, validation selection, paired regime bootstrap, diagnostic gates, details, and standard output writing.**
- [x] **Step 4: Run campaign tests and inspect all emitted test artifacts for finite JSON/CSV fields.**

### Task 3: Dispatcher Integration

**Files:**
- Modify: `experiments/26_independent_method_campaign.py`
- Test: `tests/test_m03_campaign.py`

**Interfaces:**
- Consumes: `campaign.m03.run_m03_screen`.
- Produces: `--method M03 --phase screen [--smoke]` while preserving existing list, verify, and dry-run behavior.

- [x] **Step 1: Add a failing subprocess test for M03 smoke CLI dispatch.**
- [x] **Step 2: Run it and verify the parser rejects or fails to dispatch `--smoke`.**
- [x] **Step 3: Add the scoped parser flag and M03 screen branch.**
- [x] **Step 4: Run dispatcher and existing campaign contract tests.**

### Task 4: Verification, Smoke Evidence, And Handoff

**Files:**
- Create: `results/independent_method_campaign/M03/*` compact smoke evidence.
- Create: `.superpowers/m03-implementation-report.md`

**Interfaces:**
- Produces: restartable manager-review evidence with commit provenance and `DONE_WITH_CONCERNS` when speed is not reliably demonstrated.

- [x] **Step 1: Run focused tests and commit implementation before generating provenance-bearing smoke outputs.**
- [x] **Step 2: Run only `--smoke`; inspect every generated CSV/JSON and verify every listed SHA-256 hash.**
- [x] **Step 3: Run full pytest, compileall, `git diff --check`, and harness checks under the required environment.**
- [x] **Step 4: Write the report with intended red failures, changed files, commands, metrics, limitations, and exact commit SHAs.**
- [ ] **Step 5: Commit smoke evidence/report and send the manager a compact status notification.**
