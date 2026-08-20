# HD3 Matched American-Put Comparison

## Status and claim boundary

This is a **post-hoc, single-seed diagnostic comparison**. It does not replace the original HD3 experiment and it is not blind-holdout or random-seed robustness evidence.

- Seed: `101`
- Canonical dimensions: `10, 20, 50`
- Optional stress dimension: `100`
- Models: `FC_HVI`, `DEEPSETS_HVI`
- Frozen objective: `HVI-warm-low`, 50% supervised warm start, final physics weight `0.01`, fixed optimizer mode, two Hutchinson probes.

## Matched reduced contract

The target comparison contract matches Ruixi Xu's low-dimensional American-put regime `put_T100_s060_r005_q010`:

- American put
- `K = 1`
- `T = 1`
- `r = 0.05`
- effective geometric-basket `q_G = 0.10`
- effective geometric-basket `sigma_G = 0.60`

The equal-weight equicorrelated geometric basket uses `rho = 0.80`. For each dimension `d`, individual-asset parameters are computed as

`a_d = rho + (1-rho)/d`

`sigma_d = sigma_G / sqrt(a_d)`

`q_d = q_G - 0.5 * (sigma_d^2 - sigma_G^2)`.

The implementation verifies the mapping by calling the repository's existing `effective_geometric_basket_parameters` helper and requires agreement with `(sigma_G, q_G) = (0.60, 0.10)` to tight floating-point tolerance.

Expected mappings are approximately:

| d | asset sigma | asset q |
|---:|---:|---:|
| 10 | 0.6625891564 | 0.0604878049 |
| 20 | 0.6666666667 | 0.0577777778 |
| 50 | 0.6691496051 | 0.0561194030 |
| 100 | 0.6699834367 | 0.0555610973 |

## Fair one-dimensional visualization slice

For cross-method `V(S)` figures, the high-dimensional model is evaluated on the symmetric diagonal

`S_1 = S_2 = ... = S_d = S`.

On this slice the geometric basket is exactly `G = S`, so the high-dimensional prediction solves the same reduced pricing object as the matched one-dimensional benchmark. Low-dimensional PINNs also use a slice: their time coordinate is fixed while stock price varies.

## Numerical references

The training/evaluation oracle remains the existing high-dimensional reduced CN/PSOR oracle. Because the effective reduced contract is identical across canonical dimensions, the d10/d20/d50 oracle surfaces are required to agree numerically; preflight stops if they do not.

A Cox-Ross-Rubinstein American-put tree is generated independently for plotting/sanity checks. It is not used for training labels. Successive tree refinements are recorded in `binomial_convergence.json`.

## Training budget

The experiment deliberately changes the financial contract rather than retuning the method. It keeps the completed HD3 architecture, sampling, training-set sizes and step budgets:

- canonical: `6000` Adam steps/model
- stress: `4000` steps/model
- learning rate: `1e-3`
- initial batch size: `512`, with predeclared OOM fallback sizes
- supervised points: `20,000`
- validation points: `8,192`
- test points: `16,384`

Canonical GPU execution is performed on the user's Windows RTX 4070 Laptop GPU. Checkpointing is periodic and the run can be resumed by rerunning the same command.

## Visualization outputs

Finalization exports compact, Git-trackable data rather than the entire runtime/checkpoint tree. For every completed model and oracle it produces:

- `price_slice_tau_1.csv`
- `price_slice_tau_05.csv`
- `price_slice_tau_01.csv`
- `surface.npz`
- `free_boundary.csv`
- `metadata.json`
- `checksums.json`

These outputs are intended to support, without retraining:

1. all-model `V(S)` overlays;
2. free-boundary-neighborhood zooms;
3. multi-maturity price curves;
4. `S*(tau)` overlays and boundary-error plots;
5. exercise/continuation disagreement maps;
6. absolute and signed value-error heatmaps;
7. 3D value surfaces;
8. dimension-scaling and structural diagnostics;
9. accuracy-compute Pareto plots.

## Compute comparison caveat

Compute accounting is recorded, but raw wall-clock ratios against Ruixi's Arm C/D campaign must be described as an **engineering accuracy-compute comparison**. Ruixi's reported campaign trains many regime-specific one-dimensional networks, whereas this matched run trains six canonical high-dimensional networks. A raw total-time ratio is not automatically a pure algorithmic speedup.
