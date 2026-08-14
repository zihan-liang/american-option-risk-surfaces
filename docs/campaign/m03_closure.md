# M03 Algorithm-Unrolling Closure

## Decision

M03 is closed as `failed_metric_gate`. The selected candidate is a five-step
unrolled projected Jacobi over-relaxation (PJOR) prefix followed by mandatory
ordinary PSOR cleanup. It is not an unrolled SOR solver. CN/PSOR remains the
audited benchmark and label generator.

The candidate provides a reproducible timing improvement on the screened local
CPU workloads, but it fails the frozen diagnostic non-regression gates. It is
therefore retained as a useful negative result and is not promoted.

## Accepted Evidence

- Canonical screen with 20 complete training regimes, 10 complete validation
  regimes, and seeds `[101, 203, 307, 409, 503]`.
- Clean committed provenance:
  `campaign/independent-methods` at
  `46c2afb3ad21a97fe831a886a1d7cbb683dc7621`.
- Eight predeclared choices: four baseline PSOR relaxation parameters, two MLP
  warm-start blends, and two PJOR depths.
- 150,000 training and 75,000 validation grid-point targets, represented by
  2,000 and 1,000 complete LCP systems respectively.
- Five timing warm-ups and 30 repetitions for each measured workload.
- Total screen time `592.8520457090344` seconds, below the 3,600-second budget.
- No blind-holdout access, no fallback, no non-finite metric, and matching
  artifact hashes.

## Candidate Comparison

Validation selection chose `unrolled_projected_jacobi_k5`. Its mean total solve
time is `0.10599975418671966` seconds, compared with
`0.12024586206185632` seconds for tuned PSOR. The relative improvement is
`11.847482841287496%`; the paired scenario-bootstrap interval for the time
difference is `[-0.016284800972399534, -0.012103205174937223]` seconds.

Both PJOR depths are faster than the learned MLP warm starts during validation
selection. The five-step PJOR candidate also reduces mean cleanup iterations
from `1387.9` to `1164.5`. Its five learned float64 relaxation parameters occupy
40 bytes per seed. Candidate inference and every ordinary PSOR cleanup iteration
are included in total timing.

## Failed Gates

The candidate passes the natural-residual and hard-obstacle gates, reaching a
maximum natural residual of `9.994505223431815e-11` with zero obstacle
violation. It fails the frozen 2% non-regression limits for price, Delta,
strike-scaled Gamma, boundary, continuation-region, and complementarity
diagnostics. Relative worsening ranges from about 11.6% to 79.4%, with the
boundary percentage larger because its baseline is near zero.

All affected absolute errors are small, between approximately `1e-13` and
`1e-8`, but the protocol did not predeclare an absolute numerical-floor
exception. The gate is therefore not weakened after observing the result. This
distinction should be reported: M03 demonstrates a scoped speed signal, not a
supported accuracy-preserving improvement.

## Reproducibility

Compact canonical evidence is stored under
`results/independent_method_campaign/M03/`. The evidence records the accurate
PJOR name, all eight configuration choices, corrected grid-point and LCP-system
counts, complete MLP and PJOR training histories, LF-only CSV files, and the
clean code commit used for the run.
