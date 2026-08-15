# ChatGPT Method Packets: M07--M24

Each packet is subordinate to `chatgpt_experiment_manual.md`. The expected
outcome is a hypothesis only. Do not tune toward it.

## M07 Nonlinear Quantum Reduction

- **Research:** `/Users/liangzihan/Documents/SURF_0159/SURF/deep_research/7.md`.
- **Question:** Does an occupation-measure or optimal-stopping LP reduce the
  American LCP, or does its dimension grow faster than the direct problem?
- **Experiment:** Build finite-state/time American-put Markov chains at
  `(N_S,N_T)=(4,4),(8,8),(16,16),(32,32)`. Solve by backward dynamic
  programming/direct LCP and by a flow-conservation occupation-measure LP.
  Include both `occupation_measure_lp` and `optimal_stopping_lp` formulations.
- **Metrics:** price error, variables, constraints, nonzeros, solve time, and
  `dimension_growth_ratio=(LP variables+constraints)/(direct LCP unknowns)`.
- **Gate:** LP value must match the direct value within discretization error;
  dimensional growth must not undermine the claimed reduction.
- **Limits:** This is a classical falsification experiment, not a quantum run.
- **Expected outcome:** likely negative due time-state occupation growth.

## M08 Tensor And Low Rank

- **Research:** `/Users/liangzihan/Documents/SURF_0159/SURF/deep_research/8.md`.
- **Question:** Can the structured risk tensor be compressed at least 10x
  without damaging price, Greeks, obstacle residual, or exercise boundary?
- **Experiment:** Generate a structured CN/PSOR tensor
  `V(x,tau,sigma,r,q)` from complete regimes. Compare TT-SVD, a documented
  TT-cross/completion approximation, POD, and primal-dual reduced basis.
  Predeclare rank/tolerance configurations, using validation regimes only.
- **Integrity:** Do not rename generic matrix completion as TT-cross. Use a
  tested tensor-train implementation or mark that variant unavailable.
- **Metrics:** stored scalars, compression ratio, reconstruction time, price,
  Delta, Gamma and boundary RMSE, obstacle/complementarity diagnostics.
- **Gate:** compression at least 10x and every shared diagnostic degrades at
  most 2% (residual at most 5%).
- **Expected outcome:** POD/TT may compress smooth continuation regions; Gamma
  and the moving boundary are likely limiting diagnostics.

## M09 Multi-Fidelity

- **Research:** `/Users/liangzihan/Documents/SURF_0159/SURF/deep_research/9.md`.
- **Question:** Can complete coarse surfaces reduce fine-label cost at matched
  held-out accuracy?
- **Experiment:** Compare `fine_only`, `coarse_only`, and
  `coarse_plus_few_fine` using identical surrogate architecture and five seeds.
  Coarse and fine labels must each be complete surfaces from matched regimes;
  interpolation and fine-tuning costs are included. Use predeclared fine-label
  fractions such as 10%, 25%, 50%, and 100% without local patch solves.
- **Metrics:** fine-label generation seconds at matched price accuracy, all
  Greek/boundary/residual diagnostics, training and total time.
- **Gate:** at least 2% fine-label cost reduction with CI excluding zero and no
  shared-diagnostic regression.
- **Expected outcome:** plausible positive result at moderate accuracy.

## M10 Structured Surrogate

- **Research:** `/Users/liangzihan/Documents/SURF_0159/SURF/deep_research/10.md`.
- **Experiment:** Under identical data, optimizer, parameter count, epochs and
  five seeds, compare direct price, positive premium, terminal-scaled premium,
  and European-anchor correction. Reconstruct the same American price before
  every metric. Use train-only normalization and regime-level splits.
- **Metrics:** price/Delta/Gamma/boundary RMSE, obstacle and masked residual,
  training/inference time, parameter count.
- **Gate:** at least 2% price improvement with CI excluding zero and no more
  than allowed diagnostic degradation.
- **Expected outcome:** terminal scaling or European anchoring may improve
  price, but residual/Greek trade-offs may prevent promotion.

## M11 Inverse Regularization

- **Research:** `/Users/liangzihan/Documents/SURF_0159/SURF/deep_research/11.md`
  and `/Users/liangzihan/Documents/SURF_0159/SURF/deep_research/11_2.md`.
- **Experiment:** From complete oracle surfaces, predeclare random and
  boundary-focused masks plus noise levels. Reconstruct with spline,
  Tikhonov, total variation, and low-rank methods. Fit hyperparameters on
  validation regimes only; preserve the untouched complete targets.
- **Metrics:** observed/unobserved price RMSE, full and boundary-neighborhood
  Gamma RMSE, kink error, boundary RMSE, obstacle violation and runtime.
- **Gate:** primary Gamma improvement must be statistically supported without
  worsening price/boundary/obstacle gates.
- **Expected outcome:** Tikhonov may be smooth but blur Gamma; TV may preserve
  kinks; no universal winner is assumed.

## M12 Diffusion Or Score Model

- **Research:** `/Users/liangzihan/Documents/SURF_0159/SURF/deep_research/12.md`.
- **Experiment:** Use exactly the M11 masks, noise, splits and complete targets.
  Compare compact conditional DDPM, deterministic U-Net, and low-rank baseline
  at equal data and declared compute budget. Five seeds; serialize training.
- **Metrics:** M11 metrics plus sampling variance, training time, inference time
  per completed surface and memory.
- **Gate:** DDPM must improve Gamma with supported uncertainty while preserving
  price, boundary and obstacle diagnostics at matched compute.
- **Limits:** Samples are reconstructions, not calibrated posterior draws.
- **Expected outcome:** likely negative on this small deterministic task.

## M13 Adaptive Sampling

- **Research:** `/Users/liangzihan/Documents/SURF_0159/SURF/deep_research/13.md`
  and `/Users/liangzihan/Documents/SURF_0159/SURF/deep_research/13_2.md`.
- **Experiment:** Compare only uniform/Sobol, residual-only, and boundary-only
  policies. Use identical total points, complete train regimes, architecture,
  wall-time ceiling and five seeds. Sampling scores come from training data or
  current model residuals; validation targets never influence point selection.
- **Metrics:** price/Greek/boundary/residual errors at fixed point and wall-time
  budgets; sampled-point distributions and preprocessing time.
- **Gate:** at least 2% practical improvement with CI and no regression gates.
- **Limits:** No hybrid policy in this campaign.
- **Expected outcome:** specialized samplers may improve their target metric but
  may lose elsewhere.

## M14 Hardware Acceleration

- **Research:** `/Users/liangzihan/Documents/SURF_0159/SURF/deep_research/14.md`.
- **Experiment:** Compare canonical CPU float64 with a batched Apple-MPS
  implementation on identical complete workloads. Include transfer,
  preprocessing, fallback and synchronization. Use five warm-ups and 30 timed
  runs for each supported batch/grid size. Verify every output against CPU
  float64 before timing claims.
- **Metrics:** end-to-end speedup, price/Greek/boundary errors, residual,
  obstacle, memory and fallback frequency.
- **Gate:** at least 1.25x end-to-end speedup while matching the oracle and all
  diagnostics.
- **Limits:** Report Apple MPS only; do not infer RTX/CUDA behavior.
- **Expected outcome:** small grids may be slower; large batches may benefit.

## M15 MC Dropout

- **Research:** `/Users/liangzihan/Documents/SURF_0159/SURF/deep_research/15.md`.
- **Experiment:** Train the fixed surrogate with dropout using five seeds.
  At evaluation make exactly 50 stochastic passes on in-domain, stress, and
  predeclared OOD regimes. Keep deterministic mean prediction as accuracy
  comparator.
- **Metrics:** disagreement/error Spearman correlation, large-error ranking
  AUROC, OOD AUROC, risk-coverage curve, price/Greek accuracy and 50-pass cost.
- **Gate:** improve error ranking with CI and preserve deterministic accuracy.
- **Limits:** Dropout variance is uncalibrated; never label it a 95% interval.
- **Expected outcome:** useful ranking signal is possible but calibration is
  unlikely without additional methods.

## M16 Deep Ensembles

- **Research:** `/Users/liangzihan/Documents/SURF_0159/SURF/deep_research/16.md`.
- **Experiment:** Train exactly five independent instances of the same
  architecture and data protocol; compare with one fixed deterministic model.
  Do not reuse ensemble members from M15.
- **Metrics:** large-error/OOD AUROC, disagreement-error correlation,
  calibration curves, risk coverage, accuracy, five-model training and
  inference cost.
- **Gate:** statistically improve error detection without accuracy regression;
  disclose fivefold nominal cost.
- **Expected outcome:** stronger uncertainty ranking than MC Dropout at much
  higher cost.

## M17 Conformal Prediction

- **Research:** `/Users/liangzihan/Documents/SURF_0159/SURF/deep_research/17.md`.
- **Experiment:** Freeze one deterministic surrogate before calibration. Split
  complete regimes into train, calibration and test. Build 95% split-conformal
  intervals from predeclared complete-scenario nonconformity scores; never tune
  on test coverage.
- **Metrics:** marginal scenario coverage, width, normalized width, conditional
  diagnostic slices and coverage under stress/OOD (clearly non-guaranteed).
- **Gate:** test coverage 93--97% with finite, non-vacuous width and no leakage.
- **Limits:** Claim marginal exchangeable coverage only.
- **Expected outcome:** nominal in-domain coverage should be achievable, with
  weak OOD behavior.

## M18 Selective Fallback

- **Research:** `/Users/liangzihan/Documents/SURF_0159/SURF/deep_research/18.md`.
- **Experiment:** Freeze residual-only and domain-distance scores independently
  of M16/M17. Rank complete scenarios, route rejected scenarios to CN/PSOR, and
  compare retention levels with random fallback using identical accepted counts.
- **Metrics:** accepted-scenario P95/P99 errors, coverage/retention, total time,
  oracle fallback time and risk-coverage curves.
- **Gate:** supported tail-error reduction at useful retention and lower total
  cost than solving every scenario with CN/PSOR.
- **Expected outcome:** domain distance may help OOD; residual score may better
  identify local failures.

## M19 U-Net/V-Net Segmentation

- **Research:** `/Users/liangzihan/Documents/SURF_0159/SURF/deep_research/19.md`.
- **Experiment:** Predict complete parameter-conditioned exercise masks using
  compact 1-D U-Net and V-Net models, then compare with a smaller scalar
  boundary head. Use identical complete regimes, five seeds, fixed epochs and
  report exact parameter counts. Convert masks to boundaries with one shared
  monotonic, sub-grid extractor and record invalid/no-boundary cases.
- **Metrics:** boundary RMSE/cells, mask IoU/F1, monotonicity failures, parameter
  count, training/inference time.
- **Gate:** U-Net advances only if boundary error improves after explicit
  parameter-count accounting; invalid-mask rate may not worsen.
- **Expected outcome:** scalar head may remain competitive in one dimension.

## M20 Transfer Learning

- **Research:** `/Users/liangzihan/Documents/SURF_0159/SURF/deep_research/20.md`.
- **Experiment:** Generate European pretraining labels analytically or with the
  accepted equality solver. Compare scratch, European-pretrain/fine-tune,
  frozen backbone and correction head at identical American-label budgets and
  architecture. Use complete American regimes, five seeds and predeclared
  budgets such as 10%, 25%, 50%, 100%.
- **Metrics:** price/Greek/boundary/residual errors versus American-label count,
  pretraining/fine-tuning/label time and parameter counts.
- **Gate:** supported improvement at equal American-label cost and no shared
  diagnostic regression; include European-label cost separately.
- **Expected outcome:** potential benefit at low American-label budgets.

## M21 Explainable AI

- **Research:** `/Users/liangzihan/Documents/SURF_0159/SURF/deep_research/21.md`.
- **Experiment:** Create controlled models with seeded shortcut features and
  seeded failure regions. Evaluate gradients/integrated gradients,
  financially valid LIME/SHAP perturbations, counterfactual boundary distance,
  and influence diagnostics. Perturbations must remain valid complete financial
  regimes and preserve declared dependencies.
- **Metrics:** shortcut/failure detection rate, rank stability across seeds,
  false-positive rate, invariance checks and compute cost.
- **Gate:** stable detection of known seeded cases across five seeds without
  systematic attribution to forbidden/noncausal artifacts.
- **Limits:** Explanations are diagnostic associations, not causality.
- **Expected outcome:** gradient methods may be stable locally; unconstrained
  perturbation methods should fail validity checks.

## M22 Mechanistic Audit

- **Research prerequisite:** Create `docs/research/m22_primary_source_audit.md`
  from primary sources before canonical execution. If browsing is unavailable,
  return `research_blocked`.
- **Experiment:** Freeze seven inputs and a baseline/masking convention. At
  representative continuation, boundary and exercise points, evaluate all
  `2^7` masked subsets and compute the exact Möbius/Harsanyi interaction
  decomposition. Repeat across five trained seeds.
- **Metrics:** reconstruction identity error, interaction rank stability,
  invariance consistency, sign consistency and seeded-failure detection.
- **Gate:** exact decomposition must reconstruct predictions numerically and
  stable dominant interactions must agree with tested financial invariances.
- **Limits:** Masking convention materially changes interpretation; disclose it.
- **Expected outcome:** useful audit, not a predictive improvement method.

## M23 Agentic Search

- **Research prerequisite:** Create `docs/research/m23_primary_source_audit.md`.
- **Experiment:** Three genuinely independent blind coding agents receive the
  same frozen search space, validation data and exactly eight trials each.
  Compare their best validation choices with equal-budget random and fixed-grid
  search. Persist every trial, prompt hash, agent/model identity and cost.
- **Metrics:** best validation metric after 24 trials, trial failures, compute,
  configuration diversity and frozen-selection provenance.
- **Gate:** no method accesses blind holdout. Freeze one selection per search
  strategy for later manager-controlled final evaluation.
- **Stop:** If three independent agents or auditable trial budgets are
  unavailable, record `resource_blocked`; never imitate three agents with one
  conversation.
- **Expected outcome:** unknown; workflow cost may exceed any search benefit.

## M24 Formal Verification And Certificates

- **Research prerequisite:** Create `docs/research/m24_primary_source_audit.md`.
- **Experiment:** Build discrete certificates for LCP residual,
  obstacle/complementarity, monotonicity in spot where applicable, convexity,
  terminal payoff, physical boundaries, and selected interval/high-precision
  recomputations. Test valid oracle surfaces and seeded corruptions: local price
  dip, obstacle breach, boundary shift, Gamma sign flip, terminal corruption,
  NaN/Inf and residual perturbation.
- **Metrics:** per-corruption detection rate, false-positive rate on valid
  surfaces, localization accuracy, runtime and unavailable-certificate reasons.
- **Gate:** detect every targeted corruption family at its declared magnitude
  with low false-positive rate; never apply an inapplicable theorem silently.
- **Expected outcome:** strong diagnostic candidate if applicability conditions
  are encoded correctly.
