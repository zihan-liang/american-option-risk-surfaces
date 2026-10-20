# PINN Experiment Report Visualization Brief

Target report:
`reports/pinn_experiment_report.tex`.

Reference paper:
`/Users/liangzihan/Documents/SURF_0159/SURF/semniar/PINN.pdf`.

Claim boundary for every figure: CN/PSOR remains the benchmark and label
generator. The Stage 23 claim is that `terminal_pde_weight_5` improves over the
matched supervised neural baseline on the capped American put campaign across
price, masked residual, boundary, and strict-Delta diagnostics. Do not imply
that the neural surrogate replaces PSOR.

## What the PINN Paper Figures Communicate

The paper's important figures are not decorative architecture sketches. They
are evidence layouts:

- Fig. 1 and Fig. A.6: a predicted solution field over time and space, marked
  with training/collocation data, plus exact-versus-prediction slices at a few
  times. Scientific purpose: show that the learned function is coherent over
  the whole domain and not just accurate at observed points.
- Fig. 2 and Fig. A.7: solution fields with two snapshot times and slice
  comparisons at the initial/final snapshots. Scientific purpose: show
  interpolation or propagation across a large time interval.
- Fig. 3: a physical domain and the subset of the domain used for
  spatio-temporal training data. Scientific purpose: make the data-support
  assumption visible.
- Fig. 4: predicted-versus-exact pressure fields plus a compact identified-PDE
  table. Scientific purpose: compare a latent reconstructed field against a
  reference and report whether the learned physics coefficients match the
  known equation.
- Fig. 5, Fig. B.8, and Fig. B.9: solution fields, sparse/noisy data points,
  exact-versus-data slices, and a learned-equation table. Scientific purpose:
  show robustness of PDE discovery under sparse snapshots or noise.
- Appendix tables A.1-A.5 and B.6-B.9: sensitivity to data count,
  collocation/snapshot settings, architecture, time gap, and noise. Scientific
  purpose: demonstrate that the method was tested across knobs rather than on
  a single cherry-picked setting.

## Recommended Figures

| Priority | PINN-paper figure idea | Scientific meaning | Our-project analogy | Data/source files needed | Recommended output | Suggested caption |
|---|---|---|---|---|---|---|
| must-have | Solution field with data/collocation support, as in Fig. 1 and Fig. A.6. | Show where the PDE regularizer is meaningful in the domain. | American put PSOR continuation premium heatmap with the free boundary, strike line, and PSOR-derived masked PDE residual support. This explains why residuals are evaluated away from exercise, boundary, kink, and near-maturity points. | `results/experiment_04_application_risk_surfaces/surfaces/put_T1_sig0p2_r0p05_q0.npz`; mask policy in `docs/pinn_inspired_premium_experiment_plan.md`; Stage 23 config `configs/pinn_strong_terminal_campaign.yaml`. | `reports/figures/pinn_psor_premium_mask_panel.pdf` and `.png`; source `scripts/plot_pinn_visualization_prototypes.py`. | Representative American put PSOR surface used to explain the masked continuation residual. The left panel shows normalized continuation premium \(C/K\) with the PSOR free boundary; the right panel shows where the auxiliary Black-Scholes residual is evaluated after continuation, boundary, kink, and maturity exclusions. |
| must-have | Sensitivity/ablation evidence, like the appendix tables. | Show the Stage 23 conclusion comes from a method sweep, not a single run. | Method-by-metric heatmap for all Stage 23 terminal-scaled methods, reporting relative change versus the matched supervised baseline for price RMSE, masked residual, boundary RMSE, and strict-Delta RMSE, with candidate success counts. | `results/experiment_23_pinn_strong_terminal/pinn_campaign_summary.json`; generated CSV `results/figures/pinn_experiment_report/stage23_metric_tradeoff.csv`. | `reports/figures/pinn_stage23_metric_tradeoff.pdf` and `.png`; source `scripts/plot_pinn_visualization_prototypes.py`. | Stage 23 terminal-scaled methods compared with the matched supervised neural baseline. Entries are relative metric changes; negative values are improvements because all displayed metrics are errors. Counts at left report capped candidate-success seeds out of three. |
| useful | Optimization or convergence trace, a common PINN diagnostic even though the paper emphasizes result panels more than loss curves. | Show whether the best method improves validation behavior while carrying a real PDE penalty. | Training curves for supervised baseline and `terminal_pde_weight_5`: validation premium RMSE across seeds plus supervised/PDE training loss components. | `results/experiment_23_pinn_strong_terminal/pinn_campaign_training_history.csv`; generated CSV `results/figures/pinn_experiment_report/stage23_training_curves.csv`. | `reports/figures/pinn_stage23_training_curves.pdf` and `.png`; source `scripts/plot_pinn_visualization_prototypes.py`. | Stage 23 capped put training traces across seeds 21-23. The best method reduces validation premium RMSE while training with a nonzero masked PDE loss; the curves are optimization diagnostics, not independent solver evidence. |
| must-have if prediction exports are available; otherwise useful follow-up | Exact-versus-prediction slices at representative times, as in Fig. 1, Fig. 2, Fig. A.6, and Fig. A.7. | Show local fidelity at fixed time slices and expose where a surface metric hides structure. | For a representative test put regime, plot PSOR price or premium slices at three maturities against the supervised baseline and `terminal_pde_weight_5`. Mark the free boundary crossing and payoff kink. | Requires per-point prediction export from Stage 23, either by extending `experiments/22_pinn_campaign.py` or adding a lightweight evaluation script that loads the trained/retrained models and writes a representative grid. Existing aggregate summaries alone are not enough. | Recommended: `reports/figures/pinn_stage23_price_premium_slices.pdf` and `.png`; source should live beside `scripts/plot_pinn_experiment_report.py` or a new export script. | PSOR reference and neural surrogate slices for a representative American put test regime. The Stage 23 best method should be compared with the matched supervised baseline at fixed maturities; the figure should state that PSOR provides the reference labels and boundary. |
| must-have if prediction exports are available; otherwise useful follow-up | Predicted-versus-exact field comparison, as in Fig. 4. | Locate model errors spatially instead of reporting only aggregate RMSE. | Two-row field panel for supervised baseline and `terminal_pde_weight_5`: price error or premium error heatmaps under the same color scale, plus optional masked residual magnitude heatmaps in the continuation region. | Requires per-point predictions and residual values on a representative grid. Existing files provide aggregate residual summaries only. | Recommended: `reports/figures/pinn_stage23_error_residual_maps.pdf` and `.png`. | Spatial price-error and masked-residual diagnostics for the capped American put surrogate comparison. The residual panels are shown only on the PSOR-derived continuation mask and should not be read as a global obstacle-problem residual. |
| useful | Interface/snapshot markers, as in Fig. 2 and Fig. 5. | Make the nonsmooth interface explicit. | Free-boundary overlay: PSOR boundary \(B(\tau)/K\), optional supervised-baseline boundary proxy, and optional `terminal_pde_weight_5` boundary proxy. If predictions are unavailable, use a simpler boundary-RMSE bar/point panel from aggregate metrics. | Boundary reference from Stage 04 surfaces or Stage 05 metadata; candidate proxy requires prediction export. Aggregate fallback uses `pinn_campaign_metrics.csv`. | Recommended with exports: `reports/figures/pinn_stage23_boundary_overlay.pdf`; aggregate fallback: `reports/figures/pinn_stage23_boundary_metric_panel.pdf`. | Early-exercise boundary diagnostic for the capped put campaign. The PSOR curve is the reference; neural curves, if shown, are diagnostic threshold proxies rather than independently solved free boundaries. |
| useful | Physical/training-domain support plot, as in Fig. 3. | Tell readers what samples the model saw and what samples the residual used. | Scatter or binned density of capped train/validation/test put samples in \((\log(S/K), \tau/T)\), with PDE-mask samples highlighted. | `data/processed/surrogate_dataset.npz`; `results/experiment_05_surrogate_dataset/dataset_schema.json`; Stage 23 caps in `configs/pinn_strong_terminal_campaign.yaml`. | Recommended: `reports/figures/pinn_stage23_sample_mask_support.pdf`. | Capped put sample support for the Stage 23 experiment. Highlighted points identify PSOR continuation samples eligible for the auxiliary residual after boundary, kink, and maturity exclusions. |
| optional | Learned-equation table, as in Fig. 4, Fig. 5, Fig. B.8, and Fig. B.9. | In the paper this validates PDE discovery. | For this project, do not present an identified-PDE table because the method is not discovering Black-Scholes coefficients. A compact residual-formula box or workflow schematic is safer. | Equation and mask policy from `reports/pinn_experiment_report.tex` and `src/models/pinn_residual.py`. | Optional: `reports/figures/pinn_masked_residual_workflow.pdf`. | Masked residual workflow for the PINN-inspired premium surrogate. CN/PSOR supplies labels and masks; the neural model remains a PSOR-supervised surrogate. |

## Prototype Outputs Generated

The following prototypes were generated from existing repo artifacts only:

- `reports/figures/pinn_psor_premium_mask_panel.pdf`
- `reports/figures/pinn_psor_premium_mask_panel.png`
- `reports/figures/pinn_stage23_metric_tradeoff.pdf`
- `reports/figures/pinn_stage23_metric_tradeoff.png`
- `reports/figures/pinn_stage23_training_curves.pdf`
- `reports/figures/pinn_stage23_training_curves.png`
- `results/figures/pinn_experiment_report/stage23_metric_tradeoff.csv`
- `results/figures/pinn_experiment_report/stage23_training_curves.csv`

These are ready for the report writer to inspect and either include directly or
adapt. The slice, spatial-error, and boundary-overlay figures should wait until
per-point prediction exports are available; otherwise they would overstate what
the committed artifacts currently support.

## Recommended Report Placement

1. Put `pinn_psor_premium_mask_panel` after the masked residual formulation.
   It visually explains the mask before metrics appear.
2. Put `pinn_stage23_metric_tradeoff` in the Results section near
   Table 1/Table 2, either replacing or complementing the existing relative
   change bar chart.
3. Put `pinn_stage23_training_curves` after the Stage 23 interpretation
   paragraph if space allows. It is a useful diagnostic but less central than
   the mask panel and method-metric heatmap.
4. Treat slice/error/boundary overlays as the next upgrade once prediction
   exports are available.

## Suggested Handoff Text

Image drawer complete; please forward this brief to report writer.

The report writer should review
`docs/pinn_report_visualization_brief.md` and the generated prototypes under
`reports/figures/`. The two most important additions are
`pinn_psor_premium_mask_panel` and `pinn_stage23_metric_tradeoff`. The training
curve figure is useful but secondary. Do not add exact-versus-prediction slices
or residual maps unless Stage 23 per-point prediction exports are created.
