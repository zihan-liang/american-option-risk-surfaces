# Uncertainty Quantification Design Note

Status: future work only.

## Motivation

The current price, boundary, and Delta heads output point estimates. Uncertainty quantification could help decide when PSOR spot checks are required for a new regime.

## Candidate Methods

- Deep ensembles for price, boundary, and Delta heads.
- Conformal prediction on regime-level calibration splits.
- Quantile regression for bounded price and Delta diagnostics.
- Bootstrap resampling at the regime level.

## Required Validation

Any UQ claim would require:

- empirical coverage by split and option type;
- interval width summaries;
- calibration error;
- coverage inside boundary and kink masks;
- PSOR-trigger precision and recall on fresh regimes.

## Non-Claim Boundary

No uncertainty intervals are produced by the current workflow. This note only defines a future validation path.
