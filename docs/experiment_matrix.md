# Experiment Matrix

| ID | Script | Claim Tested | Required Gate |
| --- | --- | --- | --- |
| 01 | `experiments/01_solver_validation.py` | CN/PSOR benchmark is credible | Analytic/theorem checks pass |
| 02 | `experiments/02_boundary_greeks_diagnostics.py` | Boundaries and Greeks can be extracted with diagnostics | Obstacle, complementarity, shape checks recorded |
| 03 | `experiments/03_convergence_domain_runtime.py` | Solver is stable under grid/domain changes | Price/boundary changes are small |
| 04 | `experiments/04_application_risk_surfaces.py` | Stress regimes create meaningful risk maps | Put/dividend-call/control regimes are separated; failures are reported and flagged |
| 05 | `experiments/05_surrogate_dataset.py` | Dataset is traceable to validated PSOR labels | Clean regimes only; exclusions audited; regime split and masks exist |
| 06 | `experiments/06_surrogate_baselines.py` | Ordinary direct-price baselines expose limitations | Regime-level evaluation includes obstacle, shape, boundary-band, and runtime violations |
| 07.5 | `docs/literature_review.md` | Literature supports cautious project positioning | Raw Deep Research markers removed; BibTeX compiles; claims are mapped to references |
| 08 | `experiments/07_positive_premium_tuning.py` | Positive-premium price surrogate enforces obstacle consistency | Zero obstacle violation; validation/test price and premium errors reported; no Greek or boundary claim |
| 09 | `experiments/09_boundary_head.py` | Direct boundary heads provide scoped diagnostic boundary estimates | Finite boundary labels only; put/call metrics reported separately; no price-threshold proxy |
| 10 | `experiments/10_option_specific_call_boundary_refinement.py` | Dividend-call boundary refinement is useful only if tested separately | Call-only labels counted; no-boundary/control cases not fabricated; weak regimes retained |
| 11 | `experiments/11_delta_validation.py` | Price surrogate is not automatically a Greek engine | Autograd and finite-difference Delta audited under strict Greek masks; no final Delta claim from price model |
| 12 | `experiments/12_supervised_delta_head.py` | Bounded supervised Delta head provides scoped Delta diagnostics | Strict-mask put/call metrics reported; Delta-bound violation is zero or numerical roundoff |
| 13 | `experiments/13_integrated_application_workflow.py` | Modular workflow is useful only with PSOR audit checks | Fresh PSOR cases, price/boundary/Delta metrics, speed, and violations reported together |
| 14 | `experiments/14_claim_audit.py` | Final claims are manuscript-safe and evidence-backed | Claim matrix maps each claim to artifacts and metrics; forbidden-language scan and artifact checks pass |
| 15 | `experiments/15_future_extensions_manifest.py` | Future work is separated from completed claims | Every extension is `future_work_only`; validation requirements and language scan are recorded |
