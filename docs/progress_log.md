# Progress Log

Use this file to record dated progress, commands, metrics, and decisions.

## Template

- Date:
- Stage:
- Command:
- Key metrics:
- Decision:
- Failure modes observed:
- Next step:

## 2026-06-13

- Stage: CN/PSOR solver validation audit.
- Commands:
  - `python -m unittest discover -s tests -p test_solver_audit.py`
  - `python experiments/01_solver_validation.py --config configs/solver_validation.yaml`
- Key metrics:
  - Smoke grid 120 European put max error: `2.781746966190046e-4`.
  - Smoke grid 120 no-dividend American call max error: `2.781740085198575e-4`.
  - Refined grid 240 European put max error: `6.937489127123442e-5`.
  - Refined grid 240 no-dividend American call max error: `6.937471924713112e-5`.
  - American put obstacle violation: `0.0`.
  - American put complementarity error: `4.0088005617245605e-12`.
  - Dividend-paying American call obstacle violation: `0.0`.
  - Dividend-paying American call complementarity error: `5.2696412383000194e-12`.
- Decision: solver-validation gate passes for continuing classical risk-surface experiments, but not yet as publication-grade final accuracy evidence.
- Failure modes observed: finite-difference error is still visible at moderate grids; `rannacher_steps` is configured but not implemented; boundary values move in grid increments because boundary extraction uses the current uniform grid.
- Next step: use validated solver settings for boundary/Greek diagnostics and keep grid/domain/runtime evidence attached to any reported surfaces.

## 2026-06-15

- Stage: Boundary extraction, Delta/Gamma, and diagnostics.
- Commands:
  - `python -m unittest discover -s tests`
  - `python -m compileall -q src tests experiments`
  - `python experiments/02_boundary_greeks_diagnostics.py --config configs/base_put.yaml`
  - `python experiments/02_boundary_greeks_diagnostics.py --config configs/base_call_dividend.yaml`
  - `python experiments/02_boundary_greeks_diagnostics.py --both`
- Key metrics:
  - Put obstacle violation: `0.0`.
  - Put complementarity error: `4.0088005617245605e-12`.
  - Put boundary at final time level: `0.8K`, status `boundary_found`.
  - Put strict Greek mask fraction: `0.9669421487603306`.
  - Dividend-call obstacle violation: `0.0`.
  - Dividend-call complementarity error: `5.2696412383000194e-12`.
  - Dividend-call boundary at final time level: `1.3333333333333333K`, status `boundary_found`.
  - Dividend-call strict Greek mask fraction: `0.9670104501058671`.
- Decision: Stage 3 passes for classical boundary and Greek diagnostics, with masked reporting required for Gamma.
- Failure modes observed: Gamma spikes near payoff kink and free boundary; boundary locations remain grid-resolution limited; Rannacher smoothing is still not active.
- Next step: run grid/domain sensitivity and runtime studies for price, boundary, Delta, and Gamma before surrogate dataset construction.

## 2026-06-15

- Stage: Grid/domain sensitivity and runtime for price, boundary, Delta, and Gamma.
- Commands:
  - `python -m unittest discover -s tests`
  - `python -m compileall -q src tests experiments`
  - `python experiments/03_convergence_domain_runtime.py --config configs/convergence.yaml`
- Key metrics:
  - Cases: American put and dividend-paying American call.
  - Grid reference: `M=N=240`, `Smax=4K`.
  - Grid set: `[40, 80, 120, 160, 240]`.
  - Domain set: `[3K, 4K, 5K]`.
  - Maximum obstacle violation across runs: `0.0`.
  - Maximum complementarity error across runs: `6.1288898628566304e-12`.
  - Maximum monotonicity violation across runs: `1.2988727874697525e-11`.
  - Maximum convexity violation across runs: `2.597745574939505e-11`.
  - Put grid-160 price RMSE vs reference: `4.720526685137331e-05`.
  - Put grid-160 boundary MAE vs reference: `0.017552083333333343`.
  - Put grid-160 strict Delta RMSE: `0.00046138667440877613`.
  - Put grid-160 strict Gamma RMSE: `0.05211017561299287`.
  - Dividend-call grid-160 price RMSE vs reference: `3.745913490534023e-05`.
  - Dividend-call grid-160 boundary MAE vs reference: `0.036822916666666674`.
  - Dividend-call grid-160 strict Delta RMSE: `0.0005393221866860245`.
  - Dividend-call grid-160 strict Gamma RMSE: `0.04616593897247874`.
  - Runtime at grid 240: put `0.8492777999999817s`, dividend call `1.1688343000000714s`.
- Decision: Stage 4 passes for continuing to classical application risk-surface experiments, but not for publication-grade convergence claims.
- Failure modes observed: boundary errors remain grid-resolution limited; Gamma convergence is weaker than price and Delta; grid-240 is an internal engineering reference, not an external benchmark; Rannacher smoothing is still inactive.
- Next step: generate American put and dividend-paying American call application risk surfaces with diagnostics attached to every surface.

## 2026-06-15

- Stage: Application risk-surface experiments.
- Commands:
  - `python -m unittest discover -s tests`
  - `python -m compileall -q src tests experiments`
  - `python experiments/04_application_risk_surfaces.py --config configs/application_sweep.yaml`
- Key metrics:
  - Total solved regimes: `288`.
  - Regime counts: put `144`, dividend call `108`, no-dividend call control `36`.
  - Saved representative surfaces: `12`, with four per regime class.
  - Surface run id: `run_20260615_102826`.
  - Maximum obstacle violation: `0.0`.
  - Maximum complementarity error: `1.6812843948429075e-09`.
  - Maximum monotonicity violation: `1.2279622587102262e-10`.
  - Maximum convexity violation: `0.006996739106769435`.
  - Delicate regimes: `32`.
  - Diagnostic failures: `32`.
- Decision: Stage 5 passes as an audited application risk-surface generation stage, but not as an unrestricted surrogate-label gate.
- Failure modes observed: high-volatility, long-maturity regimes trigger convexity warnings; no-dividend call controls correctly report no genuine boundary; strict-mask Gamma remains delicate.
- Next step: construct surrogate datasets only from diagnostic-passing regimes or refined reruns of flagged regimes.

## 2026-06-15

- Stage: Surrogate dataset construction.
- Commands:
  - `python -m unittest discover -s tests`
  - `python -m compileall -q src tests experiments`
  - `python experiments/05_surrogate_dataset.py --config configs/surrogate_dataset.yaml`
- Key metrics:
  - Clean regimes included: `223`.
  - Excluded regimes: `65`.
  - Dataset samples: `3264943`.
  - Feature dimension: `7`.
  - Label dimension: `5`.
  - Train/validation/test regimes: `156/33/34`.
  - Train/validation/test samples: `2283996/483153/497794`.
  - Strict Greek mask fraction: `0.9669724708823401`.
  - Exercise indicator fraction: `0.5410535497863209`.
- Decision: Stage 6 passes for beginning baseline surrogate experiments on clean, diagnostic-passing regimes.
- Failure modes observed: 65 regimes are excluded from clean labels, including no-dividend controls and Stage 5 diagnostic-warning regimes; Gamma remains mask-dependent.
- Next step: train baseline surrogate models and report failures honestly before introducing positive-premium architectures.

## 2026-06-16

- Stage: Baseline surrogate models and violation audit.
- Commands:
  - `python -m unittest discover -s tests -p "test_*baseline*.py"`
  - `python experiments/06_surrogate_baselines.py --config configs/baseline_surrogates.yaml`
- Key metrics:
  - Models evaluated: `4`.
  - Train/validation/test sample caps: `250000/100000/250000`.
  - Best validation model: `mlp_direct_price`.
  - Best validation RMSE: `0.028088274965102702`.
  - Direct MLP test RMSE: `0.02821120011211636`.
  - Direct MLP test max obstacle violation: `0.13376212120056152`.
  - Penalty-1 MLP test RMSE: `0.037631195606665294`.
  - Penalty-1 MLP test max obstacle violation: `0.21475380659103394`.
  - Penalty-10 MLP test RMSE: `0.05071797117236048`.
  - Penalty-10 MLP test max obstacle violation: `0.12164252996444702`.
  - Linear baseline test RMSE: `0.6652961027440175`.
  - Linear baseline test max obstacle violation: `1.8467483822943174`.
- Decision: Stage 7 passes as a baseline violation audit, not as a final surrogate or consistency claim.
- Failure modes observed: direct-price baselines still violate the obstacle and shape diagnostics; obstacle penalties reduce neither all violations nor RMSE reliably; no baseline is a Greek engine or boundary solver.
- Next step: introduce the positive continuation-premium price surrogate only after preserving this baseline-failure evidence.

## 2026-06-16

- Stage: Literature review integration and citation audit.
- Inputs:
  - Raw source material: `deep-research-report.md`.
  - Cleaned project-facing review: `docs/literature_review.md`.
  - Bibliography: `paper/references.bib`.
- Key conclusions:
  - Classical VI/LCP/free-boundary literature supports the mathematical and diagnostic framing.
  - CN/PSOR is supported as an auditable benchmark, not as a novel solver.
  - Gamma near payoff kinks and free boundaries must be reported cautiously with masks or restricted regions.
  - Direct-price neural networks are not automatically obstacle- or shape-preserving.
  - Positive continuation premium is literature-inspired and obstacle-preserving by construction, but its accuracy and robustness remain future experimental claims.
  - Boundary and Delta heads should remain separate diagnostic tasks.
- Decision: Stage 7.5 passes as a claim-support and citation-integration stage once the manuscript outline compiles with BibTeX and no raw Deep Research markers remain.
- Failure modes observed: some Deep Research references support project claims only indirectly; those claims are now worded cautiously.
- Next step: start the positive continuation-premium price surrogate stage without overstating literature support.

## 2026-06-16

- Stage: Positive continuation-premium price surrogate.
- Commands:
  - `python -m unittest discover -s tests -p "test_positive_premium*.py"`
  - `python -m unittest discover -s tests -p "test_boundary_distance_profiles_outputs.py"`
  - `python experiments/07_positive_premium_tuning.py --config configs/positive_premium.yaml`
  - `python experiments/08_boundary_distance_profiles.py --config configs/boundary_distance_profiles.yaml`
- Key metrics:
  - Models evaluated: `2`.
  - Best aggregate validation model: `positive_premium_linear`.
  - Best aggregate validation RMSE: `0.018758900679095955`.
  - Selected neural model: `positive_premium_mlp`.
  - Selected neural validation RMSE: `0.019988299678999184`.
  - Selected neural test RMSE: `0.021800903980790293`.
  - Maximum obstacle violation: `0.0`.
  - MLP test boundary-band RMSE: `0.006198628647556628`.
  - MLP test strict-mask RMSE: `0.02068613786457916`.
  - MLP near-kink test RMSE: `0.08439293712841753`.
- Decision: Stage 8 passes for obstacle-consistent price/premium modeling, with explicit limitation that the neural MLP does not beat the linear sanity baseline on aggregate RMSE.
- Failure modes observed: near-kink error is large; the price/premium model is not a boundary, Delta, or Gamma model.
- Next step: train and audit a direct boundary diagnostic head separately.

## 2026-06-16

- Stage: Direct boundary diagnostic head.
- Commands:
  - `python -m unittest discover -s tests -p "test_boundary_head*.py"`
  - `python -m unittest discover -s tests -p "test_call_boundary_refinement_outputs.py"`
  - `python experiments/09_boundary_head.py --config configs/boundary_head.yaml`
  - `python experiments/10_option_specific_call_boundary_refinement.py --config configs/call_boundary_refinement.yaml`
- Key metrics:
  - Finite boundary-labelled samples: `3264943`.
  - Pooled train/validation/test caps: `150000/75000/150000`.
  - Pooled MLP validation boundary RMSE: `0.29858054314764565`.
  - Pooled MLP test boundary RMSE: `0.3513049808491616`.
  - Pooled MLP test boundary MAE: `0.20874504246145487`.
  - Pooled MLP test put boundary RMSE: `0.34210722137234034`.
  - Pooled MLP test call boundary RMSE: `0.36400930907931744`.
  - Finite call boundary-labelled samples: `1449459`.
  - Call-only MLP validation boundary RMSE: `0.3885842951998814`.
  - Call-only MLP test boundary RMSE: `0.6382774801816014`.
  - Call-only linear test boundary RMSE: `0.5812933595799677`.
- Decision: Stage 9 passes as a scoped diagnostic boundary-estimation audit, not as an exact free-boundary solver.
- Failure modes observed: call-only MLP validation performance does not translate into better test RMSE than the call-only linear baseline; boundary labels are repeated across grid nodes and should not be interpreted as independent regime-level observations.
- Next step: validate Delta separately and consider a bounded supervised Delta head only after recording price-autograd limitations.

## 2026-06-16

- Stage: Delta validation and bounded supervised Delta head.
- Commands:
  - `python -m unittest discover -s tests -p "test_delta*.py"`
  - `python experiments/11_delta_validation.py --config configs/delta_validation.yaml`
  - `python experiments/12_supervised_delta_head.py --config configs/delta_head.yaml`
- Key metrics:
  - Strict Greek mask samples: `3157110`.
  - Price-model autograd Delta test RMSE: `0.09770389345203662`.
  - Price-model autograd Delta test MAE: `0.03618789399536279`.
  - Price-model finite-difference Delta test RMSE: `0.09771959227659963`.
  - Price-model derivative max Delta-bound violation: about `0.0595`.
  - Bounded linear Delta test RMSE: `0.1682303771492566`.
  - Bounded supervised MLP Delta validation RMSE: `0.032378324958276646`.
  - Bounded supervised MLP Delta test RMSE: `0.033240113189454475`.
  - Bounded supervised MLP Delta test MAE: `0.014036984504667716`.
  - Bounded supervised MLP max Delta-bound violation: `0.0`.
- Decision: Stage 10 passes as a bounded supervised Delta diagnostic stage, not as evidence that the price surrogate is a Greek engine.
- Failure modes observed: price-model derivatives have nonzero Delta-bound violations; strict-mask capped metrics contain no near-boundary samples, so boundary-zone Delta remains a limitation.
- Next step: build an integrated modular workflow with PSOR audit checks, keeping price, boundary, and Delta claims separate.

## 2026-06-16

- Stage: Integrated modular workflow with PSOR audit.
- Commands:
  - `python -m unittest discover -s tests -p "test_integrated_workflow*.py"`
  - `python experiments/13_integrated_application_workflow.py --config configs/integrated_workflow.yaml`
- Key metrics:
  - Fresh PSOR audit cases: `6`.
  - Mean price RMSE: `0.009187410333670234`.
  - Mean boundary MAE: `0.15389696844866121`.
  - Mean strict Delta RMSE: `0.02998043947678844`.
  - Maximum price obstacle violation: `0.0`.
  - Maximum Delta-bound violation: `0.0`.
  - Mean speedup: `87.52488631464541`.
- Decision: Stage 11 passes as modular scenario screening with PSOR audit, not as a PSOR replacement.
- Failure modes observed: boundary errors remain the weakest part of the integrated workflow; only six deterministic held-out regimes are evaluated; Gamma is not included.
- Next step: perform final claim audit and report/manuscript preparation.

## 2026-06-16

- Stage: Final claim audit, report, and manuscript preparation.
- Commands:
  - `python -m unittest discover -s tests -p "test_claim_*.py"`
  - `python -m unittest discover -s tests -p "test_final_report_assets.py"`
  - `python experiments/14_claim_audit.py --config configs/claim_audit.yaml`
- Key outputs:
  - `results/tables/final_claim_matrix.csv`
  - `results/experiment_14_claim_audit/claim_matrix.csv`
  - `results/experiment_14_claim_audit/claim_audit_summary.json`
  - `results/experiment_14_claim_audit/forbidden_language_scan.csv`
  - `results/experiment_14_claim_audit/missing_artifacts.csv`
  - `reports/final_report.tex`
- Key metrics:
  - Claims audited: `10`.
  - Forbidden-language violations: `0`.
  - Missing required artifacts: `0`.
  - Decision: `manuscript_safe_claims_only`.
- Decision: Stage 12 passes as evidence synthesis and final-report preparation, with no new solver runs, dataset generation, or model training.
- Failure modes retained: solver evidence remains engineering-scale, Gamma is not part of the surrogate workflow, boundary estimates remain weakest, and integrated workflow evidence covers six fresh PSOR audit cases.
- Next step: document future extensions separately without mixing speculative extensions into completed project claims.

## 2026-06-17

- Stage: Future extensions manifest and non-claim roadmap.
- Commands:
  - `python -m unittest discover -s tests -p "test_future_extensions*.py"`
  - `python experiments/15_future_extensions_manifest.py --config configs/future_extensions.yaml`
- Key outputs:
  - `docs/future_extensions.md`
  - `docs/active_learning_design.md`
  - `docs/uq_design.md`
  - `docs/shape_constrained_models_design.md`
  - `results/experiment_15_future_extensions/future_extensions.csv`
  - `results/experiment_15_future_extensions/future_extension_validation_matrix.csv`
  - `results/experiment_15_future_extensions/future_extension_language_scan.csv`
  - `results/experiment_15_future_extensions/future_extensions_summary.json`
  - `reports/stage13_future_extensions.tex`
- Key metrics:
  - Future extensions documented: `8`.
  - Invalid extension rows: `0`.
  - Language violations: `0`.
  - Claim status: `future_work_only`.
- Decision: Stage 13 passes as a roadmap-only stage. No future extension is treated as current project evidence.
- Failure modes retained: every extension requires a new validation gate before it can affect the claim matrix.
- Next step: future implementation should begin with a new project plan for the selected extension.

## 2026-06-17

- Stage: Final repository freeze and reproducibility handoff.
- Commands:
  - `python -m unittest discover -s tests -p "test_release_verification.py"`
  - `python -m unittest discover -s tests -p "test_readme_release_handoff.py"`
  - `python -m unittest discover -s tests -p "test_artifact_manifest.py"`
- Key outputs:
  - `README.md`
  - `docs/release_handoff.md`
  - `docs/artifact_manifest.md`
  - `experiments/16_release_verification.py`
  - `results/experiment_16_release_verification/release_verification_summary.json`
- Decision: Stage 14 adds the release-handoff machinery. The final release gate remains blocked until stale tracked generated manuscript artifacts are removed locally from Git tracking.
- Failure modes retained:
  - `manuscript_outline.aux`
  - `manuscript_outline.log`
  - `manuscript_outline.pdf`
  - `paper/manuscript_outline.pdf`
- Next step: remove the stale tracked generated files locally, rerun `python experiments/16_release_verification.py`, and commit the release freeze.

## 2026-06-17

- Stage: Journal-standard final report upgrade.
- Commands:
  - `python -m unittest discover -s tests -p "test_publication*.py"`
  - `python -m unittest discover -s tests -p "test_manuscript*.py"`
  - `python experiments/17_publication_evidence_audit.py --config configs/publication_evidence.yaml`
  - `python experiments/18_publication_figures.py --config configs/publication_figures.yaml`
- Key outputs:
  - `paper/manuscript.tex`
  - `paper/supplementary_material.tex`
  - `docs/journal_readiness_gap_audit.md`
  - `results/experiment_17_publication_audit/publication_gap_matrix.csv`
  - `results/experiment_17_publication_audit/publication_audit_summary.json`
  - `results/figures/publication/figure_manifest.csv`
- Key metrics:
  - Publication claims audited: `10`.
  - Main-text-ready claims: `5`.
  - Supplement-only or limited claims: `5`.
  - Publication figures generated: `9`.
  - Integrated fresh PSOR audit cases remain `6` against a target of `34`.
- Decision: Stage 15 upgrades the final deliverable into a manuscript-plus-supplement package, but integrated workflow and boundary claims remain explicitly limited.
- Failure modes retained: boundary-head performance is weak, integrated audit breadth is limited, Gamma is diagnostic-only, and PSOR remains required for new-regime trust.

## 2026-06-17

- Stage: Visualization enrichment for every report.
- Commands:
  - `python experiments/19_report_visualization_enrichment.py --config configs/report_visualization_enrichment.yaml`
  - `python -m unittest discover -s tests -p "test_report_visualization*.py"`
  - `python -m unittest discover -s tests -p "test_report_tex_figure_links.py"`
  - `python -m unittest discover -s tests -p "test_visual_claim_language.py"`
  - `python -m unittest discover -s tests -p "test_personal_background_positioning.py"`
- Key outputs:
  - `configs/report_visualization_enrichment.yaml`
  - `src/plotting/report_visuals.py`
  - `experiments/19_report_visualization_enrichment.py`
  - `results/figures/report_enrichment/figure_manifest.csv`
  - `reports/visual_blocks/`
- Key metrics:
  - Reports with enrichment plan: `14`.
  - Evidence-linked enrichment figures generated: `58`.
  - Caption language violations in focused scan: `0`.
- Decision: Stage 16 improves visual communication, report readability, and student contribution positioning without changing the underlying claim matrix.
- Failure modes retained: richer visuals do not make the neural surrogates better than CN/PSOR, do not remove boundary weakness, and do not expand Gamma or PSOR-replacement claims.

## 2026-06-17

- Stage: Mathematical process and matrix-calculation visualization upgrade.
- Commands:
  - `python experiments/19_report_visualization_enrichment.py --config configs/report_visualization_enrichment.yaml`
  - `python -m unittest discover -s tests -p "test_math_process_visual_types.py"`
  - `python -m unittest discover -s tests -p "test_report_visualization_manifest.py"`
  - `python -m unittest discover -s tests -p "test_visual_claim_language.py"`
- Key outputs:
  - New explicit visual kinds in `src/plotting/report_visuals.py`.
  - Expanded Stage 1 and Stage 2 figure plan in `configs/report_visualization_enrichment.yaml`.
  - Regenerated `results/figures/report_enrichment/figure_manifest.csv`.
- Decision: Stage 17 is an explanation upgrade only. It improves understanding of PDE-to-matrix translation, CN matrices, LCP form, and PSOR projection without changing solver behavior or scientific claims.
- Failure modes retained: visual teaching figures are not new accuracy evidence and do not affect surrogate-performance claims.
