# M07 Standalone Nonlinear Quantum Reduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether finite-horizon optimal-stopping LP representations reduce the American problem or instead expand its dimension.

**Architecture:** For each frozen `(N_S,N_T)` pair, construct one sparse finite-state log-GBM Markov chain and solve the same American put by backward dynamic programming, a Bellman-inequality `optimal_stopping_lp`, and a flow-conservation `occupation_measure_lp`. Measure actual sparse LP dimensions/nonzeros and compare values and timing against the direct recursion.

**Tech Stack:** Python, NumPy, SciPy sparse matrices, `scipy.optimize.linprog(method="highs")`; no QPU, quantum simulator, new dependency, or blind-holdout access.

## Global Constraints

- Canonical grids: `(4,4),(8,8),(16,16),(32,32)`.
- Put model: `K=T=1`, `r=0.05`, `q=0`, `sigma=0.2`, log-domain `[-1.5,1.5]`, `x0=0`.
- Markov rows must be finite, nonnegative, and row-stochastic; reflected endpoint mass is disclosed.
- Direct recursion: `V_T=g`, `V_t=max(g,beta P V_{t+1})`.
- Primary metric: `(LP variables+constraints)/N_S`; secondary denominator `N_S*N_T` is also reported.
- LP/direct price error must be at most `max(1e-10, direct_refinement_error)` using a `2N_S,2N_T` direct refinement.
- Dimensional reduction requires at least one formulation to have primary ratio `<=1` at every canonical size.
- Canonical timing uses exactly five warm-ups and 30 repetitions; M07 budget is four hours.
- Smoke is always `diagnostic_only`; canonical failed equivalence or dimension is `failed_metric_gate`.
- Classical falsification only: never call the experiment a quantum run or quantum speedup.

### Task 1: Direct Markov stopping control
- [ ] Write failing stochastic-matrix and Bellman-recursion tests.
- [ ] Implement `build_markov_chain`, `backward_dynamic_programming`, and `initial_price`.
- [ ] Rerun focused tests to green.

### Task 2: Two LP formulations
- [ ] Write failing value-equivalence tests for `optimal_stopping_lp` and `occupation_measure_lp`.
- [ ] Implement Bellman obstacle/continuation inequalities and occupation/stopping flow equalities.
- [ ] Use HiGHS with nonnegative bounds and rerun tests.

### Task 3: Accounting, gates, and artifacts
- [ ] Test exact variables, constraints, nonzeros, both dimension ratios, timing guards, hashes, and smoke no-promotion behavior.
- [ ] Implement actual sparse-matrix accounting, refinement tolerance, five/30 timing guard, status classification, provenance, and `config_manifest.json`, `metrics.csv`, `decision.json`, `artifact_hashes.json`.
- [ ] Run focused tests, compileall, diff hygiene, then full tests before the implementation commit.

### Task 4: Canonical closure
- [ ] Commit implementation as `feat: add standalone M07 nonlinear quantum reduction`.
- [ ] Run canonical screen once from that exact implementation SHA.
- [ ] Inspect finite metrics, all eight formulation/grid rows, hashes, and `blind_holdout_accessed=false`.
- [ ] Write `docs/campaign/m07_closure.md` preserving a negative result if the dimension gate fails.
- [ ] Commit only compact evidence/closure as `results: close M07 nonlinear quantum reduction`, rerun verification, and push.

## Run Command

```bash
PYTHONPATH=src /Users/liangzihan/Documents/SURF_0159/SURF/.worktrees/american-option-risk-surfaces/.venv-campaign/bin/python experiments/standalone/m07_nonlinear_quantum_reduction.py --config configs/standalone/m07_nonlinear_quantum_reduction.json
```

## Stop Conditions

Stop before canonical execution for invalid transition probabilities, LP/direct mismatch on tiny controls, failing focused tests, non-finite required metrics, wrong canonical timing, provenance mismatch, budget excess, or any blind-holdout access.
