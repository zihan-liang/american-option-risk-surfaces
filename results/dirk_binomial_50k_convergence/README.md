# DIRK vs CRR50000 convergence result upload

This commit publishes the convergence-rate result artifacts for the matched American-put audit.

Files under `results/dirk_binomial_50k_convergence/` contain the canonical-centered CRR50000-relative convergence tables and a compact summary of the observed RMSE orders.

Files under `results/dirk_binomial_50k_clean_isolation/` contain the clean-isolation error table, frozen/post-hoc temporal order decision, and run-status evidence.

Headline result: spatial self-convergence at fixed N=3840 gives overall RMSE orders 2.004463269782384 and 2.006830691980684. Frozen temporal self-convergence at fixed M=1920 gives 1.0761152751698817, 1.2842628988657108, and 1.5750699763070575. Post-hoc temporal extensions give 2.758276447906639 and 1.5731379810654837, so stable second-order temporal convergence is not claimed.

The frozen M=1920,N=60 point did not converge at the strict 1e-12 Policy Iteration gate and remains recorded in `run_metrics.csv` and `convergence_decision.json`.
