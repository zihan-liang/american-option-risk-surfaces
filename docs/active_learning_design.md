# Active Learning Design Note

Status: future work only.

## Motivation

The current surrogate dataset is built from a fixed PSOR sweep. Active learning could reduce label cost by selecting new regimes where the current modular workflow is least reliable.

## Candidate Acquisition Signals

- High disagreement between surrogate variants.
- Large boundary-band price error on audited cases.
- High uncertainty if a future UQ module exists.
- Regime proximity to diagnostic-failure regions from the application sweep.
- Large obstacle, shape, or Delta-bound risk in a candidate diagnostic pass.

## Required Workflow

1. Define a candidate regime pool.
2. Score candidates without using hidden PSOR labels.
3. Select a batch of regimes.
4. Run PSOR with full diagnostics.
5. Add only diagnostic-passing regimes to clean labels.
6. Compare label efficiency against random sampling.

## Required Validation

Active learning can support a claim only if it reduces held-out error per added PSOR regime compared with a matched random baseline. Failed acquisitions must remain in an audit table.

## Non-Claim Boundary

No active-learning result is part of the current project. This design note should not be cited as evidence that active learning improves the existing surrogate.
