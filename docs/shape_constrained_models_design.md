# Shape-Constrained Models Design Note

Status: future work only.

## Motivation

The positive continuation-premium model enforces the obstacle, but it does not guarantee every financial shape property. Direct-price baselines showed obstacle and shape violations, so future models should prefer structural constraints over penalty-only training where feasible.

## Candidate Constraints

- Hard obstacle enforcement through continuation-premium parameterization.
- Option-specific Delta bounds.
- Monotonicity constraints with respect to spot.
- Convexity audits or convex architecture components where practical.
- No-arbitrage consistency checks across maturities and dividends.

## Required Validation

Any shape-constrained model would need to report:

- obstacle violation;
- monotonicity violation;
- convexity violation;
- Delta-bound violation;
- boundary-band price error;
- regime-level validation and test metrics;
- comparison with the Stage 7 baseline and Stage 8 positive-premium model.

## Non-Claim Boundary

This is not an implemented architecture. It is a future design path for reducing violations observed in baseline models.
