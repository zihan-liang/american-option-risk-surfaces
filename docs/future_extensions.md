# Future Extensions Roadmap

This document records future work only. None of these extensions are completed contributions of the current project, and none should be used to strengthen the Stage 12 claim matrix without new validation.

## Current Claim Boundary

The completed project supports a validated CN/PSOR benchmark within tested regimes, audited American put and dividend-paying American call risk surfaces, a positive-premium price surrogate, direct boundary diagnostics, a bounded supervised Delta head, and an integrated scenario-screening workflow with PSOR audit.

The completed project does not support claims about active learning, uncertainty quantification, shape-constrained architectures, Gamma surrogates, alternative dynamics, or market calibration.

## Extension Roadmap

| Extension | Limitation addressed | Required validation before any claim |
| --- | --- | --- |
| Larger regime envelope | Current labels exclude controls and diagnostic-failure regimes and cover a finite synthetic sweep. | New PSOR sweeps with obstacle, complementarity, shape, grid, domain, runtime, and failure-regime audits. |
| Active learning for PSOR labels | Current dataset uses a predefined grid rather than adaptive label selection. | Error reduction per added PSOR regime against random sampling and retention of failed acquisitions. |
| Uncertainty quantification | Current surrogate outputs are point estimates. | Empirical coverage, interval width, calibration error, and PSOR spot-check trigger quality. |
| Shape-constrained models | Positive premium enforces the obstacle only; direct baselines violate shape properties. | Obstacle, monotonicity, convexity, Delta-bound, boundary-band, and held-out-regime audits. |
| Improved Gamma treatment | Gamma is delicate near payoff kinks and free boundaries. | Masked and unmasked Gamma convergence, smoothing sensitivity, and high-resolution reference comparisons. |
| Rannacher/payoff smoothing | Current validation identifies nonsmooth payoff and uniform-grid effects. | Before-after solver validation, grid convergence, boundary stability, and Greek diagnostics. |
| Alternative dynamics | Current formulation is one-factor Black-Scholes. | New VI or optimal-stopping formulation, benchmark solver, analytic/theorem checks, and full diagnostics. |
| Market calibration | Current regimes are synthetic and controlled. | Data provenance, calibration error, out-of-sample pricing, boundary plausibility, and PSOR audit. |

## Writing Guidance

Use cautious language:

- "could be investigated"
- "would require"
- "future work only"
- "not validated in the current project"
- "requires a new benchmark and claim audit"

Avoid completed-result language:

- avoid "improves"
- avoid "achieves"
- avoid "solves"
- avoid "is validated"
- avoid "replaces PSOR"

## Decision Rule

An extension can move from roadmap to project claim only after it has:

1. A mathematical or experimental specification.
2. A benchmark or source-of-truth label process.
3. Diagnostics at least as strict as the current CN/PSOR workflow.
4. Reproducible outputs, logs, configs, and a claim-audit row.
5. A report section that lists failure modes and rejected claims.
