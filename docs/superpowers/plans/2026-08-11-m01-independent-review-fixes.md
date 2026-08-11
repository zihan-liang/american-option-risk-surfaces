# M01 Independent Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct every blocking M01 numerical-review finding, finish the five-variant diagnostic smoke path, and leave reproducible numerical and repository-wide verification evidence.

**Architecture:** Keep the finite-difference public API and `SolverResult` compatibility aliases intact. Correct each mathematical issue at its source in the boundary, stage assembly, direct LCP solver, convergence metadata, and summary layers; keep campaign policy and artifact provenance in the M01 runner and shared dispatcher.

**Tech Stack:** Python 3, NumPy, PyYAML-backed project configuration, unittest/pytest, existing campaign record writers.

## Global Constraints

- Work only in `.worktrees/american-option-risk-surfaces/m01-classical-solvers` on `exp/m01-classical-solvers`.
- Preserve the intentional uncommitted M01 runner, config, dispatcher, tests, and smoke artifacts.
- Use `/Users/liangzihan/Documents/SURF_0159/SURF/.worktrees/american-option-risk-surfaces/.venv-campaign/bin/python` with `MPLBACKEND=Agg`.
- Run smoke configurations only; do not run the canonical M01 campaign.
- Do not touch other method worktrees, push, or access the blind holdout.
- Every production behavior change follows a failing regression test and the smallest passing fix.

---

### Task 1: Negative-Rate Put Boundary

**Files:**
- Modify: `src/solvers/boundary_conditions.py`
- Test: `tests/test_solver_time_schemes.py`

**Interfaces:**
- Consumes: `boundary_values(option_type, tau, k, s_max, r, q)` and `solve_option(SolverConfig(...))`.
- Produces: the American-put lower boundary `K * exp(max(-r * tau, 0))` at every stored time level.

- [ ] Add a direct boundary test with a hand-derived negative-rate value and a propagated solver test over every `tau`.
- [ ] Run both tests and confirm they fail because the lower boundary remains `K`.
- [ ] Replace the put constant with the discounted-or-immediate maximum.
- [ ] Rerun the focused tests and commit the boundary fix with its regressions.

### Task 2: SDIRK2 Obstacle Reaction

**Files:**
- Modify: `src/solvers/cn_psor.py`
- Test: `tests/test_solver_time_schemes.py`

**Interfaces:**
- Consumes: the first-stage LCP residual `A Y1 - rhs1`.
- Produces: `lambda1 = (A Y1 - rhs1) / (gamma * dt)` in the `(1-gamma)` second-stage contribution.

- [ ] Add a scalar manufactured-stage test with `L=-1`, obstacle and old value `1`, negative-rate boundary forcing, and a hand-derived final value greater than the obstacle only when `lambda1` is included.
- [ ] Run the test and confirm the old stage assembly returns the obstacle instead.
- [ ] Add the first-stage reaction to the final RHS for American solves; use zero reaction for linear European solves.
- [ ] Rerun time-scheme tests and describe SDIRK2 only as the chosen L-stable obstacle discretization unless a measured order test is added.

### Task 3: One-Sided Brennan-Schwartz

**Files:**
- Modify: `src/solvers/cn_psor.py`
- Modify: `experiments/24_classical_oracle.py`
- Test: `tests/test_lcp_solvers.py`
- Test: `tests/test_m01_classical_campaign_outputs.py`

**Interfaces:**
- Consumes: a tridiagonal M-matrix, monotone put obstacle, and a nonnegative-rate single-boundary regime.
- Produces: obstacle-shifted `UL` factorization, backward triangular solve, forward projected solve, and explicit applicability metadata.

- [ ] Add a positive-rate American-put first-step LCP test that compares against policy iteration at tight natural residual.
- [ ] Add a negative-rate applicability test and run both against the current solver.
- [ ] Implement sufficient structural checks: tridiagonal Z-matrix, positive `UL` pivots, put obstacle shape, and nonnegative supplied rate.
- [ ] Shift by the obstacle, perform the put-direction `UL` sweep, and classify any candidate failing the natural residual as not applicable.
- [ ] Omit non-applicable regimes from Brennan-Schwartz numerical agreement controls and keep their status out of numerical metric gates.
- [ ] Rerun LCP and campaign tests.

### Task 4: Comparable LCP Convergence and Stage Summary

**Files:**
- Modify: `src/solvers/cn_psor.py`
- Modify: `src/diagnostics/solver_checks.py`
- Test: `tests/test_lcp_solvers.py`
- Test: `tests/test_solver_time_schemes.py`

**Interfaces:**
- Consumes: configured `tol` and final natural residual for PSOR, policy iteration, and penalty/Newton.
- Produces: `info["converged"] == (info["natural_residual"] <= tol)` plus independent update/activity diagnostics.

- [ ] Add one-dimensional PSOR and policy tests where the exact residual converges on the last permitted iteration despite unstable update/activity diagnostics.
- [ ] Add a one-dimensional penalty test where the penalized equation is solved but the natural residual remains above tolerance.
- [ ] Add a fabricated-result summary test where the final surface is feasible but a recorded substage obstacle violation is nonzero.
- [ ] Run the tests and confirm the three flag mismatches and missing summary field.
- [ ] Make the natural residual the sole shared convergence flag and retain method-specific diagnostics separately.
- [ ] Add `max_lcp_obstacle_violation` to `solver_summary` and propagate it to campaign comparison rows.

### Task 5: M01 Campaign Diagnostic Path and Refinement Contract

**Files:**
- Modify: `configs/independent_m01_classical.yaml`
- Modify: `experiments/24_classical_oracle.py`
- Modify: `experiments/26_independent_method_campaign.py`
- Modify: `src/campaign/registry.py`
- Test: `tests/test_m01_classical_campaign_outputs.py`
- Update: `results/independent_method_campaign/M01/screen/*`

**Interfaces:**
- Consumes: the five approved one-factor method configurations and smoke overrides.
- Produces: five `diagnostic_only` standard records, `smoke_non_evidence` claims, `blind_holdout_accessed=false`, finite CSV/JSON metrics, hashes, and at least two smoke refinement levels marked `diagnostic_only_no_order_claim`.

- [ ] Extend the campaign test to assert all five exact variants, five diagnostic records, non-evidence claims, false holdout access, finite metrics, provenance, complete hashes, and explicit refinement grouping/labels.
- [ ] Run it and observe the missing refinement/provenance assertions fail.
- [ ] Add two smoke refinement grids, grid spacing/reference metadata, no-order-claim wording, and strict method/config validation in the dispatcher/runner.
- [ ] Replace unsupported registry wording about global second order with the narrower L-stable obstacle-discretization description.
- [ ] Rerun the in-process and shared-CLI smoke tests.
- [ ] Generate compact checked-in smoke artifacts through the dispatcher and inspect every JSON/CSV value and status.

### Task 6: Verification, Review, Reports, and Commits

**Files:**
- Create: `.superpowers/m01-fix-report.md`
- Include all intended modified source, config, test, plan, and compact artifact files in logical commits.

**Interfaces:**
- Produces: exact command evidence, numerical probe values, commit SHAs, changed-file inventory, residual concerns, and a concise manager handoff.

- [ ] Run focused solver/campaign tests with `MPLBACKEND=Agg`.
- [ ] Run full repository pytest with the campaign Python and `MPLBACKEND=Agg`.
- [ ] Run `compileall`, `git diff --check`, artifact finite/provenance inspection, and `git status` review.
- [ ] Request an independent code review and address any blocking finding.
- [ ] Write `.superpowers/m01-fix-report.md`, make logical commits without pushing, and send the manager the SHAs, exact tests, probes, remaining concerns, and changed files.
