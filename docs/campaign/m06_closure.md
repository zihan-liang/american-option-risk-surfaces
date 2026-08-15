# M06 Quantum Linear Systems Closure

## Decision

M06 is closed as `failed_metric_gate` with claim status `diagnostic_only`.
Exact-statevector VQLS passes the numerical-relevance gate only at dimension 8.
Neither of the two predeclared optimizer configurations passes dimensions 16
and 32, so no quantum-solver or acceleration claim is supported.

## Numerical Evidence

The decision uses median solution RMSE over five seeds and requires it to be no
larger than one quarter of the matched CN discretization floor.

| Configuration | N=8 | N=16 | N=32 |
|---|---:|---:|---:|
| `depth3_iter120` | `9.8072e-5` | `1.1405e-3` | `1.1875e-2` |
| `depth5_iter300` | `4.8598e-5` | `1.0869e-3` | `1.1872e-2` |
| Gate | `2.9397e-4` | `4.2407e-4` | `9.1748e-5` |

The deeper, longer configuration improves N=8 but does not materially repair
N=16 or N=32. This bounded sensitivity check makes a simple under-training
explanation less plausible, while not proving that every VQLS ansatz would
fail.

## Resource Audit

- Retained Pauli terms grow from 32 to 80 to 192 for N=8, 16, and 32.
- The corresponding projected LCU shot executions at 10,000 shots per
  expectation are 10.24 million, 64 million, and 368.64 million.
- Full classical readout requires 8, 16, and 32 values; exact-statevector runs
  execute zero shots.
- The QSVT degree proxies are 8, 8, and 10. They are
  `kappa*log(1/error)` scale indicators, not circuit counts.
- The correct classical comparator remains an O(N) tridiagonal/banded solve;
  dense inversion timing is not used as a baseline.

## Limits

This experiment uses oracle-frozen active sets and a global VQLS cost. It does
not execute finite-shot optimization, a QPU, local-cost VQLS, coherent state
loading, tomography, repeated obstacle updates, Greeks, or a complete maturity
march. These omissions and the failed scaling gate keep M06 out of the
candidate set. Blind holdout access is `false`.
