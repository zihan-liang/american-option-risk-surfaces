# M07 Closure — Nonlinear Quantum Reduction

## Decision

M07 closes as `failed_metric_gate` with `claim_status=diagnostic_only`.

Both classical LP formulations reproduced the finite-state American-put stopping value within the frozen equivalence tolerance at every canonical grid pair, but neither reduced dimension relative to the simultaneous direct-LCP unknown count. The best formulation was `optimal_stopping_lp`, whose maximum canonical `dimension_growth_ratio` was 96 at `(N_S,N_T)=(32,32)`; `occupation_measure_lp` reached 98 at the same size.

The negative conclusion is not an artifact of using only the single-step LCP denominator. With the secondary full-horizon denominator `N_S*N_T`, the largest-grid ratios were 3.0 for `optimal_stopping_lp` and 3.0625 for `occupation_measure_lp`. Thus the LP lift still uses substantially more variables-plus-constraints than the aggregate direct value-node count.

## Canonical Evidence

- Implementation commit: `9a862f2e9ccd6a9d56c9d8baa85a334ed4ac0780`.
- Frozen grids: `(4,4)`, `(8,8)`, `(16,16)`, `(32,32)`.
- Formulations: `optimal_stopping_lp`, `occupation_measure_lp`.
- Value-equivalence gate: passed for all eight formulation/grid rows.
- Dimension-reduction gate: failed for both formulations.
- Primary metric: `best_formulation_max_dimension_growth_ratio = 96.0`.
- Largest occupation-measure ratio: `98.0`.
- Blind holdout accessed: `false`.
- Canonical screen wall time recorded by the runner: approximately 1.18 s, within the four-hour M07 budget.

The LP solve timings were also slower than direct backward dynamic programming on these small one-factor chains, increasingly so at larger sizes. That timing signal is diagnostic rather than the primary scientific gate because HiGHS overhead and implementation details materially affect tiny problems.

## Scientific Interpretation

The experiment supports a negative result for the claimed reduction: an exact finite-horizon LP representation exists and preserves the stopping value, but the tested formulations lift the time-state problem into a larger optimization problem rather than compressing the direct American stopping recursion. This is consistent with treating occupation-measure/QLP as a structural resource audit rather than evidence of a useful quantum American-option solver.

No quantum circuit, quantum simulator, QPU, quantum LP backend, or quantum-speedup experiment was executed. The result therefore falsifies only the dimensional-reduction premise for these classical finite-state LP formulations under the frozen M07 protocol.

## Limitations

- One normalized one-factor American put tuple was tested: `K=T=1`, `r=0.05`, `q=0`, `sigma=0.2`.
- The finite Markov chain uses a fixed log-price domain and reflects outgoing endpoint probability mass.
- The direct comparator is the exact backward recursion for the same finite-state chain; it is the direct discrete stopping/LCP control for this experiment, not a timing replacement for the repository CN/PSOR oracle.
- Equivalence tolerance is `max(1e-10, direct_refinement_error)` from a `2N_S,2N_T` direct-chain refinement.
- SciPy HiGHS timings should not be extrapolated to quantum central-path complexity or to high-dimensional American models.

## Closure

M07 is retained as valid negative campaign evidence. It is not a candidate reduction and supplies no basis for a quantum speedup or CN/PSOR replacement claim.
