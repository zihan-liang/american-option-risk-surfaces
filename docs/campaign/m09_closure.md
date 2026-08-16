# M09 Closure — Multi-Fidelity Surrogate

## Decision

- Status: `failed_metric_gate`
- Claim status: `diagnostic_only`
- Numerical gate passed: `false`
- Blind holdout accessed: `false`
- Canonical implementation SHA: `198b9bef9c7f7de22997ba5ea7b986c2a0e50c47`
- Canonical runtime: 41.939708 seconds

## Canonical result

The canonical M09 screen completed within the predeclared one-hour screening
budget.

The multi-fidelity arm achieved a mean fine-training-label cost reduction of
30.0721%, with a paired 95% bootstrap confidence interval of
[10.0240%, 50.1202%]. The cost-reduction gate therefore passed.

Validation-only matched-accuracy selection produced:

- seed 101: fine_only 100%, coarse_plus_few_fine 50%
- seed 203: fine_only 100%, coarse_plus_few_fine 100%
- seed 307: fine_only 100%, coarse_plus_few_fine 50%
- seed 409: fine_only 100%, coarse_plus_few_fine 50%
- seed 503: fine_only 100%, coarse_plus_few_fine 100%

Thus coarse pretraining reduced the required fine-label fraction for three of
five campaign seeds, while two seeds required the full fine-label set.

However, the selected multi-fidelity models did not satisfy the complete shared
diagnostic gate. The paired 95% upper degradation bounds were:

- price RMSE: 2.1790% (limit 2%)
- Delta RMSE: 10.9400% (limit 2%)
- Gamma RMSE: 16.5784% (limit 2%)
- boundary RMSE: 27.4684% (limit 2%)
- obstacle violation: 0.0% degradation, selected maximum violation 0.0
- dual violation: 6.2453% (limit 5%)
- complementarity: 41.9768% (limit 5%)
- continuation residual: 75.7461% (limit 5%)

The method therefore fails the promotion gate despite passing the primary
fine-label-cost criterion.

## Interpretation

The canonical result supports the narrower conclusion that complete coarse
surfaces can reduce the amount of fine CN/PSOR training data required to reach
the predeclared validation price-accuracy target.

It does not support promotion of this multi-fidelity construction as a
shared-diagnostic-preserving surrogate. Price accuracy alone is insufficient:
Greek, exercise-boundary, and complementarity/PDE-residual behavior remains
materially less stable.

The mean test price degradation was negative (-13.1016%), indicating better
mean price RMSE for the selected multi-fidelity candidates, but its paired 95%
upper bound was 2.1790%, slightly above the 2% gate. Several other diagnostic
failures were substantially larger and independently prevent promotion.

No local fine patches, active boundary sampling, or local PDE solves were used.
Coarse and fine labels were complete matched regime surfaces. Interpolation
cost was zero because both fidelities trained the same continuous-coordinate
MLP.

No sealed blind holdout was accessed.

## Audit note

The first smoke run exposed two software/protocol issues before canonical
execution: model construction occurred before application of the declared
PyTorch seed, and ordinary-test evaluations were performed before
validation-only fraction selection was frozen.

These issues were corrected in commit
`198b9bef9c7f7de22997ba5ea7b986c2a0e50c47` before the sole canonical run and
are documented in `docs/campaign/m09_precanonical_fix.md`. No canonical result
was produced under the superseded implementation.

This creates one documented pre-canonical corrective commit beyond the nominal
implementation/results two-commit pattern. Published history was preserved
rather than force-rewritten.

## Campaign disposition

M09 is closed as `failed_metric_gate`.

The cost-reduction result is retained as useful diagnostic evidence, but no
six-hour confirmation-stage allocation is warranted and M09 should not be
promoted.

Proceed to M10.
