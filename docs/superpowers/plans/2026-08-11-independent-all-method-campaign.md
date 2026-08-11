# Independent All-Method Experiment Campaign

**Status:** approved and in progress

**Integration branch:** `campaign/independent-methods`

**Integration worktree:** `/Users/liangzihan/Documents/SURF_0159/SURF/.worktrees/american-option-risk-surfaces/campaign-independent-methods`

**Common benchmark:** one-dimensional Black--Scholes American put

**Combination experiments:** explicitly out of scope

## Objective

Test all 24 candidate methods independently under one auditable protocol before considering combinations. Every method must produce a reproducible screening decision. Negative, inconclusive, invalid, and resource-limited outcomes remain evidence and must appear in the final bilingual report.

## Global Constraints

1. Use the current single-asset American put as the common benchmark. Allow only small method-specific tensor, quantum, reconstruction, and European-to-American transfer pilots.
2. Run locally on the Apple M5 machine with 16 GB memory. Serialize compute-heavy runs.
3. Use CPU float64 as the canonical environment. Use Apple MPS only for M14 and compare it against the float64 oracle.
4. Treat Stages 21--23 and older pilot branches as historical context only. They do not enter the final scorecard unless rerun through this campaign harness.
5. Do not combine methods in this campaign. Any combination appears only as a future recommendation.
6. Every method receives a separate branch, worktree, commit, result directory, verification record, and manager review.
7. The manager independently checks artifacts and claims. Weak or inconsistent work is returned for correction before integration.
8. Every delegated task must notify the manager when it is complete.

## Shared Statistical Protocol

- Stochastic screening seeds: `101, 203, 307, 409, 503`.
- Confirmation seeds: add `607, 709, 811, 907, 1009`.
- Neural screening data: 150,000 training points and 75,000 validation points, sampled by complete regimes.
- Neural architecture: hidden width 128, depth 3, 30 epochs, batch size 8192, patience 8, CPU float64.
- Hyperparameter cap: eight predeclared configurations per method.
- Selection data: validation only.
- Uncertainty unit: complete parameter regime, never individual grid point.
- Report paired mean differences, standard deviations, scenario-bootstrap 95% confidence intervals, and Holm correction within each method family.
- Hard-obstacle methods must have obstacle violation at most `1e-10`.
- Price, Delta, Gamma, boundary, continuation, and complementarity diagnostics may not worsen by more than 2%; residual diagnostics may not worsen by more than 5%.
- An improvement claim requires at least a 2% practical gain and a paired 95% confidence interval excluding zero.
- Deterministic timing uses five warm-ups and 30 measured repetitions. Record label generation, preprocessing, training, inference, fallback, and memory separately.
- Screening budget: one hour per method; M05--M07 and M12 may use four hours. Confirmation budget: six hours per promoted method.
- Budget exhaustion produces `resource_blocked`, never a fabricated metric failure.

## Blind Holdout

The existing dataset test split is renamed conceptually to `legacy_test` because Stages 21--23 have already influenced it. Before method development, commit a SHA-256 commitment to a secret holdout seed in `configs/independent_campaign_holdout_commitment.json`. Keep the seed outside Git. After all method configurations are frozen, reveal the seed and generate 128 fresh put regimes: 96 Sobol in-domain cases and 32 stress cases. Evaluate each frozen method once.

## Wave 0: Campaign Foundation

1. Import the reviewed harness commits.
2. Add a registry containing 24 `MethodSpec` entries, budgets, baselines, primary metrics, categories, and gates.
3. Add `MethodRunRecord`, canonical CSV/JSON schemas, artifact hashing, provenance, and decision-state validation.
4. Add scenario-level paired bootstrap, Holm correction, timing helpers, promotion logic, and explicit failed/resource-blocked handling.
5. Add train/validation/legacy-test/holdout access controls and tests proving selection code cannot access the holdout.
6. Add CLI `experiments/26_independent_method_campaign.py --method Mxx --phase screen|confirm|final`.
7. Standard method outputs: `config_manifest.json`, `run_metrics.csv`, optional `training_history.csv`, `decision.json`, and `artifact_hashes.json`.

## Wave 1: Oracle and Numerical Methods

### M01 Classical solvers

Extend `SolverConfig` with explicit time-scheme and LCP-solver selections. Implement CN, Rannacher-CN, and second-order L-stable DIRK; implement PSOR, policy iteration, and penalty/Newton. Replace PSOR-specific result fields with generic LCP iteration fields while retaining compatibility properties. Compare one factor at a time: CN/PSOR, Rannacher-CN/PSOR, DIRK/PSOR, CN/policy, and CN/penalty. Validate with grid refinement, European analytic controls, no-dividend American-call equivalence, Brennan--Schwartz where applicable, and an independent integral-equation boundary check.

### M02 Obstacle-aware PINN

Compare supervised, masked-PDE, separate complementarity, and Fischer--Burmeister losses using the same positive-premium architecture.

### M03 Algorithm unrolling

Compare unrolled projected SOR or a learned feasible warm start with tuned PSOR at identical natural-residual tolerance. Include ordinary cleanup in timing.

### M04 Free-boundary representation

Compare an implicit level-set network and scalar boundary head with price-surface boundary extraction. Primary metric: sub-grid boundary RMSE.

## Wave 2: Quantum and Compression

### M05 Schrodingerization

Solve a frozen continuation-region linear problem and then one live obstacle update. Pass only if error stays below the classical discretization floor.

### M06 Quantum linear systems

Run VQLS on actual `N=8,16,32` CN matrices and provide QSVT resource accounting. Report residual, conditioning, encoding, shots, and readout. Make no speedup claim.

### M07 Nonlinear quantum reduction

Construct a small occupation-measure/optimal-stopping LP and measure dimensional growth against the direct LCP. Treat this as a falsification experiment.

### M08 Tensor and low rank

Test TT-SVD, TT-cross/completion, POD, and primal-dual reduced basis on `V(x,tau,sigma,r,q)`. Require at least 10x compression with no more than 2% diagnostic degradation.

### M09 Multi-fidelity

Use complete coarse surfaces plus fewer complete fine surfaces. Compare fine-label wall time needed to reach the same held-out accuracy. Do not use undefined local patch solves.

## Wave 3: Surrogates, Reconstruction, Sampling, and Hardware

### M10 Structured surrogate

Compare direct price, positive premium, terminal-scaled premium, and European-anchor correction under matched parameter count and data.

### M11 Inverse regularization

Reconstruct deliberately masked/noisy price surfaces using spline, Tikhonov, total variation, and low-rank baselines. Evaluate kink, Gamma, and boundary preservation.

### M12 Diffusion or score model

Train a compact conditional DDPM for the M11 completion task. Compare with a deterministic U-Net and low-rank baseline at equal data and compute.

### M13 Adaptive sampling

Compare uniform/Sobol, residual-only, and boundary-only sampling under identical point and wall-time budgets. Do not use a hybrid policy.

### M14 Hardware acceleration

Compare CPU float64 with a batched Apple-MPS implementation. Require at least 1.25x end-to-end speedup while matching the float64 oracle. Do not generalize to CUDA.

## Wave 4: Uncertainty and Selective Computation

### M15 MC Dropout

Use 50 stochastic passes to assess error ranking and OOD detection. Treat variance as an uncalibrated diagnostic, not a 95% interval.

### M16 Deep ensembles

Train five independent models. Measure large-error detection, disagreement/error correlation, calibration, and fivefold cost.

### M17 Conformal prediction

Use split conformal on a fixed deterministic surrogate with complete-scenario scores. Nominal 95% coverage must lie in 93--97% with non-vacuous interval width.

### M18 Selective fallback

Gate complete scenarios with a fixed residual/domain-distance score independent of M16/M17. Compare accepted-scenario P95/P99 error and total runtime with random fallback.

## Wave 5: Segmentation, Transfer, Explainability, and Workflow

### M19 U-Net/V-Net segmentation

Predict parameter-conditioned exercise masks and compare with a smaller scalar boundary head. Advance U-Net only if boundary error improves after parameter-count accounting.

### M20 Transfer learning

Compare European pretraining followed by American fine-tuning with scratch, frozen-backbone, and correction-head controls under equal American-label budgets.

### M21 Explainable AI

Test gradients/integrated gradients, financially valid LIME/SHAP perturbations, counterfactual boundary distance, and influence diagnostics. Success requires stable detection of seeded shortcut/failure cases.

### M22 Mechanistic audit

Perform exact `2^7` masked-input interaction decomposition at representative points. Evaluate stability across seeds and consistency with known financial invariances.

### M23 Agentic search

Give three blind Codex agents eight validation-only trials each. Compare the best validation result with equal-budget random and fixed-grid search, then evaluate once on holdout.

### M24 Formal verification and certificates

Check discrete LCP, obstacle, monotonicity, convexity, terminal, boundary, and interval/high-precision certificates on valid and deliberately corrupted surfaces. Report detection and false-positive rates.

## Stage 27 Final Evaluation

1. Freeze every method configuration and record its config hash and commit.
2. Reveal and verify the committed holdout seed.
3. Generate the 96 Sobol and 32 stress regimes.
4. Evaluate every frozen method exactly once.
5. Produce four scoreboards: solver/accelerator, surrogate/compression, reconstruction/UQ, and diagnostic/workflow.
6. Report rankings only within compatible categories; provide Pareto fronts, pass/fail gates, confidence intervals, cost, reproducibility, and negative results.

## Final Report

The report writer drafts an English-first, matching-Chinese scientific narrative. The image drawer creates code-generated figures for the protocol, method taxonomy, confidence-interval forests, accuracy/runtime Pareto fronts, UQ risk-coverage, tensor ranks, quantum resource growth, and full gate heatmap. Captions stay in LaTeX. The report writer integrates them into:

- `reports/independent_methods_comparison_report.tex`
- `reports/independent_methods_comparison_report.pdf`

The manager verifies every table against CSV/JSON evidence and inspects every rendered page. No overlap, clipping, unreadable bilingual layout, or unsupported claim may remain.

## Required Verification

- Unit tests for time schemes, LCP solvers, residuals, compatibility, analytic controls, and theorem controls.
- Contract tests for all 24 registry entries, schemas, finite metrics, seeds, budgets, provenance, and artifact hashes.
- Leakage tests for the blind holdout and scenario split boundaries.
- Reproducibility tests for fixed seeds and paired samples.
- Statistical tests for bootstrap, Holm correction, promotion, failures, and resource blocking.
- Method-specific smoke tests for tensor, VQLS, diffusion, conformal, fallback, XAI, agent budgets, and certificates.
- Full `pytest`, `compileall`, `git diff --check`, harness verification, LaTeX compilation, citation/claim scan, PDF text extraction, and page-by-page visual inspection.

## Completion Rule

The campaign is complete only when all 24 methods have a reviewed independent decision, the blind final evaluation is finished once, all evidence is traceable to committed configs and code, and the bilingual report has passed manager inspection.
