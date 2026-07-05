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
| 16 | `experiments/16_release_verification.py` | Repository handoff is reproducible and packaging-safe | Required files, claim summaries, future-work summaries, and stale tracked generated artifacts are checked |
| 17 | `experiments/17_publication_evidence_audit.py` | Journal-style manuscript claims are scoped to available evidence | Publication gap matrix demotes weak boundary and six-case integrated workflow claims |
| 18 | `experiments/18_publication_figures.py` | Manuscript figures are source-traceable | Figure manifest records PNG/PDF outputs and source artifact for every paper figure |
| 19 | `experiments/19_report_visualization_enrichment.py` | Stage reports and supplement are visually evidence-linked | Report figure manifest records caption, source artifact, PNG, and PDF for every enrichment figure |
| 19a | `experiments/19_report_visualization_enrichment.py` | Mathematical matrix process is explainable through diagrams | Stage 1 and Stage 2 include stencil, operator, CN matrix, RHS, LCP, and PSOR teaching figures |
| 21 | `experiments/21_pinn_inspired_premium.py` | Future-extension PINN-inspired premium regularizer is scoped and PSOR-supervised | PSOR labels remain the benchmark; PDE residual is masked to PSOR continuation points outside boundary/kink/maturity exclusions; no PSOR replacement claim |
| 22 | `experiments/22_pinn_campaign.py` | Future-extension PINN-inspired method campaign is evaluated against the matched supervised surrogate baseline | Multi-seed matched-initialization runs; PSOR remains the benchmark; success gates require price/residual/obstacle/boundary diagnostics; no PSOR replacement claim |
| 23 | `experiments/22_pinn_campaign.py --config configs/pinn_strong_terminal_campaign.yaml` | Strong terminal-PINN regularization can beat the matched supervised surrogate only if residual gates also pass | Multi-seed matched-initialization terminal sweeps; success gates require price gain, zero obstacle violations, residual non-regression, and boundary non-regression; PSOR remains the benchmark |
