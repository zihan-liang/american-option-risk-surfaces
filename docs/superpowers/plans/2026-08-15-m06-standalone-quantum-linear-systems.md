# M06 Standalone Quantum Linear Systems Implementation Plan

> **For agentic workers:** Implement this file task-by-task with TDD. Do not
> broaden the experiment or claim quantum speedup.

**Goal:** Run an exact-statevector VQLS diagnostic on actual frozen American
put CN policy systems of dimension 8, 16, and 32, then audit QSVT-style input,
encoding, conditioning, and output costs.

**Architecture:** One standalone NumPy script reconstructs a mid-maturity CN
system from the repository CN/PSOR oracle, freezes the oracle active set, and
uses a real Ry/CZ hardware-efficient ansatz. Adam minimizes the normalized
global VQLS alignment cost. The exact solution is used only for evaluation.
Pauli coefficients and QSVT resource proxies are measured separately.

**Tech stack:** Python, NumPy, existing `solvers.cn_psor`; no Qiskit, QPU,
network service, or new dependency.

## Frozen Protocol

- Dimensions: `[8, 16, 32]`; seeds: `[101, 203, 307, 409, 503]`.
- Predeclared optimizer configurations: three layers/120 iterations and five
  layers/300 iterations. This bounded sensitivity check separates a shallow
  ansatz failure from a more persistent scaling failure.
- One normalized American put and one mid-maturity CN policy system per size.
- Exact expectation values only; executed shots are zero. Report an explicit
  finite-shot cost projection separately.
- Record normalized linear residual, relative solution errors, fidelity,
  condition number, optimizer evaluations, Pauli term count and truncation
  error, state-loading amplitudes, full-readout values, and QSVT degree proxy.
- Numerical relevance requires solution RMSE no larger than one quarter of the
  matched medium/fine CN discretization floor. For stochastic VQLS, the
  dimension-level decision uses the median solution RMSE across all five
  seeds, and every tested dimension must pass. Overall status remains
  `diagnostic_only` even when that gate passes.
- A dense solve may provide an evaluation target but never a timing baseline.
  No speedup statement is allowed because state preparation, circuit execution,
  shots, tomography, and repeated obstacle updates are not executed end to end.

## Files And Commands

- Create `experiments/standalone/m06_quantum_linear_systems.py`.
- Create `configs/standalone/m06_quantum_linear_systems.json`.
- Test with `tests/test_m06_quantum_linear_systems.py`.
- Run with:
  `PYTHONPATH=src .venv-campaign/bin/python experiments/standalone/m06_quantum_linear_systems.py --config configs/standalone/m06_quantum_linear_systems.json`.
- Required outputs under `results/standalone/M06/`: `config_manifest.json`,
  `metrics.csv`, `resource_accounting.csv`, `decision.json`, and
  `artifact_hashes.json`.

## Acceptance And Handoff

Focused and full tests, compileall, finite-value checks, artifact hashes, clean
Git provenance, and claim-language review must pass. A future ChatGPT repair
should read this plan, the standalone script, its test, and the compact result
files only; the full Deep Research archive is unnecessary unless the scientific
scope changes.
