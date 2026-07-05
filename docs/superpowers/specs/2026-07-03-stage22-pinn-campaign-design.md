# Stage 22 PINN-Inspired Campaign Design

## Goal

Stage 22 tests whether PINN-inspired additions can improve the current supervised positive-premium surrogate while preserving CN/PSOR as the benchmark and label generator. The target is not to beat PSOR as a solver. The target is to beat or match the supervised neural baseline against PSOR labels while improving at least one diagnostic that matters for free-boundary-aware risk surfaces.

## Claim Boundary

This is future-extension work. Any successful result must be described as a scoped neural-surrogate diagnostic, not as a replacement for CN/PSOR. If no method improves the baseline, the correct conclusion is that the tested PINN-inspired additions did not help under the audited configuration.

Allowed wording:

- "PINN-inspired regularization was tested as a future-extension diagnostic."
- "The best configuration improved the supervised surrogate on the capped Stage 22 metric gate."
- "CN/PSOR remains the benchmark and source of labels."

Forbidden wording:

- "The PINN replaces PSOR."
- "The PINN solves American options."
- "The PINN is better than the numerical benchmark."

## Baseline

Use the current Stage 21 matched-initialization supervised positive-premium model as the baseline. Every variant must start from the same initial weights as the supervised baseline for a given seed.

The baseline model reconstructs normalized price by

```text
p_hat = payoff_over_K + premium_hat
```

where `premium_hat >= 0` by the positive-premium head.

## Methods To Test

Stage 22 must test individual methods first, then combinations.

Individual methods:

1. `pde_weight_sweep`: fixed PDE weights in `{0, 1e-6, 1e-5, 1e-4, 1e-3}`.
2. `pde_ramp`: supervised warm start with `lambda_pde = 0`, followed by a ramp to the configured final weight.
3. `terminal_premium`: hard terminal premium head using `premium_hat = theta * softplus(f(x))`, where `theta = tau/T`.
4. `residual_sampling`: sample PDE batches with a residual-aware policy after warm-up, while still using the PSOR mask.
5. `delta_loss`: add a low-weight Delta/Sobolev-style loss on strict Greek-mask samples.

Combination methods:

1. `terminal_plus_ramp`.
2. `terminal_plus_ramp_plus_residual_sampling`.
3. `best_combo`: the best safe combination selected by validation score from individual and two-component runs.

## Mask Policy

The PDE residual mask must continue to come from PSOR labels, not from the neural model. For the first Stage 22 campaign, the PDE loss applies only where all are true:

```text
PSOR continuation gap > gap_epsilon
not near_boundary
not near_kink
tau_over_T > tau_min_for_pde
option_scope matches
```

No PDE residual may be applied in the exercise region, near the free boundary, near the payoff kink, or near maturity.

## Residual

The log-space residual remains:

```text
R = (1 / T) * p_theta
    - 0.5 * sigma^2 * p_xx
    - (r - q - 0.5 * sigma^2) * p_x
    + r * p
```

The `1 / T` factor is required because the dataset uses `theta = tau/T`.

## Experiment Matrix

Default Stage 22 run:

- option scope: put
- seeds: `21`, `22`, `23`
- train cap: `12000`
- validation cap: `4000`
- test cap: `4000`
- hidden: `64`
- depth: `2`
- epochs: `8`
- warm start epochs for ramped variants: `4`
- strict Greek mask required for Delta loss

The first implementation may keep these caps small to protect iteration speed. If a method looks promising, a later run can increase caps and seeds.

## Success Gate

For each method, compare against the matched supervised baseline for the same seed. A method is marked `candidate_success` only if:

1. test price RMSE is lower by at least `0.5%`, or validation price RMSE is lower by at least `0.5%` and test price RMSE is not worse by more than `0.1%`;
2. max obstacle violation remains zero or numerical roundoff;
3. boundary-band RMSE is not worse by more than `1%`;
4. masked residual mean absolute value is not worse by more than `5%`;
5. if Delta loss is enabled, strict-mask Delta RMSE is reported.

A method is marked `diagnostic_success` if price RMSE is neutral but masked residual mean absolute value, boundary-band RMSE, or strict Delta RMSE improves by at least `2%` without violating the other gates.

If no method passes these gates, Stage 22 should commit a negative result clearly.

## Outputs

Write lightweight outputs under:

```text
results/experiment_22_pinn_campaign/
```

Required files:

- `pinn_campaign_config_manifest.json`
- `pinn_campaign_metrics.csv`
- `pinn_campaign_residual_metrics.csv`
- `pinn_campaign_delta_metrics.csv`
- `pinn_campaign_summary.json`
- `pinn_campaign_training_history.csv`

Optional local-only artifacts:

- model checkpoints
- figures
- large prediction arrays

Do not commit optional large artifacts unless the artifact policy is changed explicitly.

## Tests

Required test coverage:

1. terminal premium head gives zero premium at `theta = 0`;
2. ramp schedule returns zero PDE weight during warm-up and reaches the final weight at the configured epoch;
3. residual-aware sampler never samples outside the PSOR PDE mask;
4. campaign ranking prefers a method that passes the success gate over one that improves price while violating residual or obstacle gates;
5. Stage 22 dry-run writes required CSV/JSON outputs.

## Delegation Requirement

Every worker thread asked to implement or revise Stage 22 must notify this thread when finished. The completion message must include:

- branch name;
- commit hashes;
- files changed;
- commands/tests run;
- summary of best method and metrics;
- whether any method passed `candidate_success` or `diagnostic_success`;
- remaining unrelated dirty files.
