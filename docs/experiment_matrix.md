# Experiment Matrix

| ID | Script | Claim Tested | Required Gate |
| --- | --- | --- | --- |
| 01 | `experiments/01_solver_validation.py` | CN/PSOR benchmark is credible | Analytic/theorem checks pass |
| 02 | `experiments/02_boundary_greeks_diagnostics.py` | Boundaries and Greeks can be extracted with diagnostics | Obstacle, complementarity, shape checks recorded |
| 03 | `experiments/03_convergence_domain_runtime.py` | Solver is stable under grid/domain changes | Price/boundary changes are small |
| 04 | `experiments/04_application_risk_surfaces.py` | Stress regimes create meaningful risk maps | Put/dividend-call/control regimes are separated; failures are reported and flagged |
| 05 | `experiments/05_surrogate_dataset.py` | Dataset is traceable to validated PSOR labels | Clean regimes only; exclusions audited; regime split and masks exist |
| 06-13 | Surrogate scripts | Modular acceleration is useful only if constraints hold | No obstacle/Delta-bound claims without metrics |
