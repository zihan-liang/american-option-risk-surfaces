# PINN-Inspired Positive-Premium Experiment Plan

This Stage 21 experiment is a future-extension probe, not a replacement for
CN/PSOR. It follows the Deep Research recommendation to use Option B: keep
PSOR as the benchmark and label generator, train the neural model mainly from
PSOR labels, and add a Black-Scholes residual only as a masked auxiliary
regularizer.

## Scope

- Default to American puts for the first run.
- Reuse the Stage 05 surrogate dataset and Stage 07 positive-premium model
  pattern.
- Preserve the existing CN/PSOR solver and diagnostics.
- Commit only source, config, tests, docs, and lightweight CSV/JSON summaries.
  Model checkpoints, processed datasets, figures, PDFs, and logs remain local.

## Mathematical Design

The model reconstructs normalized price from a nonnegative premium:

```text
p_hat = payoff_over_K + premium_hat
```

The auxiliary residual is evaluated in log-moneyness coordinates with
`theta = tau / T`:

```text
R = (1 / T) * p_theta
    - 0.5 * sigma^2 * p_xx
    - (r - q - 0.5 * sigma^2) * p_x
    + r * p
```

The `1 / T` factor is required because the dataset uses normalized maturity
time. The residual is never enforced globally.

## Mask Policy

The PDE mask comes from PSOR labels, not from the neural model:

```text
gap_PSOR = price_PSOR_over_K - payoff_over_K
mask = gap_PSOR > gap_epsilon
       and not near_boundary
       and not near_kink
       and tau_over_T > tau_min_for_pde
       and is_put
```

If this mask has no valid samples after caps and filtering, the experiment
writes an insufficient-data summary instead of training a misleading physics
model.

## Outputs And Claim Boundary

The experiment writes reproducible summaries under
`results/experiment_21_pinn_inspired_premium/`:

- `pinn_config_manifest.json`
- `pinn_metrics.csv`
- `pinn_residual_metrics.csv`
- `pinn_summary.json`
- optional `pinn_training_history.csv`

Any report wording should say that this is a PINN-inspired future extension
that evaluates whether a masked physics regularizer improves surrogate
diagnostics while preserving obstacle and boundary consistency. It must not
claim that the neural model solves American options or replaces PSOR.

## Initial Small Run

The default Stage 21 config ran on capped put samples from the existing
surrogate dataset. It produced lightweight CSV/JSON summaries under
`results/experiment_21_pinn_inspired_premium/`.

The small run is diagnostic only. The PINN-inspired model preserved zero
obstacle violation on the capped test slice and reported masked continuation
residual summaries outside boundary/kink/maturity exclusions, but the matched
ablation did not improve the supervised baseline on this run. Test price RMSE
was `0.0927620625` for the supervised model and `0.0928312998` for the
PINN-inspired model; test residual mean-absolute value was `0.0459431366` for
the supervised model and `0.0461946940` for the PINN-inspired model. The
boundary threshold proxy also did not find stable crossings. These outputs are
not strong enough to promote the extension into a main project claim; they are
an implementation and diagnostic baseline for future ablations.
