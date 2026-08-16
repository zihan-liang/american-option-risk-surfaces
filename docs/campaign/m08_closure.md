# M08 Closure — Tensor and Low-Rank Compression

## Decision

- Status: `failed_metric_gate`
- Claim status: `diagnostic_only`
- Numerical gate passed: `false`
- Passing test configurations: none
- Blind holdout accessed: `false`
- Canonical implementation SHA: `e4841b6674e828bb3db453f650aff84276aab8b9`

## Canonical result

The canonical M08 screen completed within the predeclared one-hour screening budget.

All four tested families achieved substantial storage compression for at least
one tested configuration:

- TT-SVD
- TT completion
- POD
- primal-dual reduced basis

However, no frozen validation winner passed the complete test gate. Price,
Delta, Gamma, exercise-boundary, obstacle/complementarity, and CN/LCP residual
degradation were materially above the predeclared tolerances.

The result therefore constitutes negative evidence for promotion, not a
software or resource failure.

## Interpretation

The experiment indicates that storage compression itself is not the limiting
factor. The tested low-rank representations compress the structured American
option tensor strongly, but the moving exercise boundary and derivative /
complementarity diagnostics are substantially more sensitive to reconstruction
error than price compression alone would suggest.

TT-SVD, POD, and primal-dual RB are oracle compressibility baselines with
complete-tensor access. TT completion alone restricts fitting observations to
complete training regimes and was evaluated using all five campaign seeds.

The primal-dual reduced-basis experiment uses discrete CN/LCP residual
snapshots for its dual basis but is not a certified reduced
variational-inequality solver.

No post-reconstruction obstacle repair was applied before diagnostics.

No quantum hardware, quantum algorithm, or QTT PDE-solver claim is made.

## Audit note

For families whose candidate configurations both produced infinite gate ratios
because of hard diagnostic failures, validation winner selection degenerates
to the deterministic tie/order rule. This does not affect the M08 conclusion:
the alternative configurations also miss the predeclared metric gates by large
margins.

## Campaign disposition

M08 is closed as `failed_metric_gate`.

No confirmation-stage allocation is warranted.
