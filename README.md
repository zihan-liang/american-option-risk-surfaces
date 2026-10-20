# Scientific Machine Learning and Numerical Methods in Derivative Pricing

Free-boundary American-option risk surfaces with classical LCP solvers,
physics-informed neural networks and high-dimensional hybrid
variational-inequality models.

SURF 2026 project (SURF-2026-0159), Department of Foundational Mathematics,
School of Mathematics and Physics, Xi'an Jiaotong-Liverpool University (XJTLU).

**Team:** Zihan Liang, Qimu Luo, Xiaoya Wu, Ruixi Xu
**Supervisors:** Dr. Ren Zhao, Dr. Yiru Ye
**Poster:** [docs/poster/surf-2026-poster.pdf](docs/poster/surf-2026-poster.pdf)

## Overview

Early exercise turns American-option pricing under the Black–Scholes model into a
variational inequality (VI): the option value, the exercise (free) boundary and the
price sensitivities have to be resolved together. This project compares
Crank–Nicolson (CN) linear complementarity problem (LCP) solvers, builds a
high-accuracy DIRK–Policy Iteration–sinh-grid reference, and evaluates
low-dimensional obstacle PINNs and high-dimensional hybrid-VI neural models against
that reference and against a 50,000-step binomial tree. The main finding is that
building structure into the neural model improves price accuracy, while reliable
free-boundary recovery remains harder than price approximation.

## Model problem

For asset price `S`, time to maturity `tau`, rate `r`, dividend yield `q` and
volatility `sigma`, with `L_BS V = 1/2 sigma^2 S^2 V_SS + (r - q) S V_S - r V`, an
American option with payoff `Phi(S)` satisfies

```
min(V - Phi, V_tau - L_BS V) = 0,   V(S, 0) = Phi(S).
```

The outputs of interest are the value `V(S, tau)`, the free boundary `S*(tau)`, and
the Greeks `Delta = V_S` and `Gamma = V_SS`. The regimes studied are the American
put, the dividend-paying American call, and the no-dividend call as a control
(no early exercise).

## Methods

**Numerical PDE methods**

- *CN–LCP.* Crank–Nicolson discretisation turns the VI into one LCP per time step.
- *CN–Projected SOR.* The classical benchmark: a relaxed Gauss–Seidel update that is
  projected back onto the payoff obstacle at every node.
- *CN–Brennan–Schwartz (projected LU).* Uses LU factorisation of the tridiagonal CN
  system with projected back-substitution.
- *High-accuracy reference: DIRK–Policy Iteration–sinh grid.* Combines L-stable,
  second-order diagonally implicit Runge–Kutta time stepping, stage-wise LCP policy
  iteration, and a strike-concentrated sinh-mapped spatial grid.

**Physics-informed and hybrid neural methods**

- *Vanilla soft-LCP PINN.* Learns `V` directly. Terminal, boundary, obstacle and
  complementarity conditions enter the loss as soft penalties.
- *Structure-aware ETC+FB PINN.* Enforces the terminal condition exactly through
  `V = V_lift + s N(x, s)` and encodes complementarity with a smoothed
  Fischer–Burmeister function.
- *High-dimensional hybrid VI (HVI), d = 10, 20, 50.* `V = Phi + s softplus(N)`
  guarantees `V >= Phi`. It is trained on reference prices, with a Fischer–Burmeister
  residual enforcing the PDE and the exercise constraint. There are two variants:
  - *FC-HVI*: a fully connected network on the state and time.
  - *DeepSets-HVI*: mean-pooled per-asset embeddings, which make the model invariant
    to asset ordering.

**Validation.** Results are checked against a 50,000-step binomial (CRR) tree as an
independent classical benchmark.

## Key results

All numbers below are taken from the final poster.

| Result | Setting | Value |
| --- | --- | --- |
| LCP solver efficiency | Same CN–LCP discretisation, `1e-12` residual tolerance | Median runtime `2.44e-1 s` (CN–PSOR) vs `9.38e-3 s` (projected LU): **26.0x speedup** |
| Price accuracy vs 50k-step binomial tree | American put, `K=1, T=1, r=0.05, q=0.10, sigma=0.60` | CN+PSOR: RMSE `5.01e-5`, max error `2.71e-4`. DIRK+Policy+sinh: RMSE `2.70e-6`, max error `1.01e-5`. **18.5x lower RMSE** |
| Free-boundary accuracy vs binomial tree | Same contract | Boundary RMSE `9.40e-3` (CN–PSOR) vs `2.39e-3` (DIRK) |

Refinement study for the DIRK–Policy–sinh reference (American put,
`K=1, T=0.25, r=0.01, q=0, sigma=0.20`). The observed orders are about 1.9–2.5,
which is consistent with second-order behaviour.

| Temporal (M, N) | Price RMSE | Gamma RMSE | | Spatial (M, N) | Price RMSE | Gamma RMSE |
| --- | --- | --- | --- | --- | --- | --- |
| (240, 120) | `6.35e-8` | `2.75e-5` | | (120, 960) | `1.10e-5` | `1.92e-3` |
| (240, 240) | `1.47e-8` | `5.18e-6` | | (240, 960) | `2.64e-6` | `3.93e-4` |
| (240, 480) | `3.00e-9` | `1.37e-6` | | (480, 960) | `4.70e-7` | `1.03e-4` |

Neural models: price RMSE against the high-accuracy reference on the common domain
`S/K in [0.5, 1.5]`, `tau/T in [0.01, 1]`.

| Model | Price RMSE |
| --- | --- |
| Vanilla PINN | `2.79e-2` |
| Structure-aware PINN (ETC+FB) | `1.05e-3` |
| FC-HVI (d = 50) | `2.31e-3` |
| DeepSets-HVI (d = 50) | `4.00e-3` |

The structure-aware PINN reduces price RMSE by a factor of 26.7 relative to the
vanilla PINN and has the lowest error of the neural models. The d = 50 HVI models
stay at the `1e-3` RMSE scale but do not beat the structure-aware PINN in this
comparison. The free boundary is much harder to recover: the vanilla soft-LCP PINN
detects only 0.6% of boundary crossings, compared with 28.2% for ETC+FB.

The protocol caveats that go with these comparisons are recorded in
[docs/final_poster_result_claims.md](docs/final_poster_result_claims.md) and in
`results/consolidated_csv/README.md`.

## Final project status

The project went through two phases.

1. **CN/PSOR benchmark and modular surrogates** (experiments `01`–`23`,
   `reports/`, `paper/`). This phase covers VI/LCP notes, CN/PSOR validation,
   boundary and Greek diagnostics, risk surfaces for 288 regimes, a positive
   continuation-premium price surrogate, boundary and bounded Delta heads, and an
   integrated workflow with PSOR audit. Across 6 held-out audit regimes, the
   integrated workflow reached a mean price RMSE of `0.0092`, had no obstacle or
   Delta-bound violations, and gave a mean speedup of 87.5x over PSOR.
2. **Independent method campaign and final comparison** (experiments `24`–`28`,
   `standalone/`, `hd*`, `reference_audits/`, `analysis/`). This phase covers the
   classical LCP screen, obstacle PINNs, high-dimensional HVI models, the CRR
   50,000-step audit, and the consolidated cross-repository dataset behind the
   poster.

Scope and claim boundaries:

- The neural workflow does not replace PSOR or the classical solvers. They remain
  the audited reference.
- The project does not claim that one network solves price, boundary and Greeks
  together, and it makes no Gamma-accuracy claim for any surrogate.
- Accurate prices do not imply accurate free boundaries (see the free-boundary
  results above).

## Companion repository

The DIRK–Policy Iteration–sinh-grid reference solver and the low-dimensional PINN
runs were developed in the team's companion repository,
[Ruixixu-hub/2026-surf-american-risk-surfaces](https://github.com/Ruixixu-hub/2026-surf-american-risk-surfaces).
Results exported from it are included in `results/consolidated_csv/`. The DIRK/CRR50k
contract tests (`tests/test_dirk_binomial_50k_convergence.py`) import its
`american_risk_surfaces` package at commit `5f63fa7b12f704c3577c72f9d7b7e5146e78c1ca`.
If that package is not on `PYTHONPATH`, these tests are skipped.

## Repository layout

```
src/            Library code: solvers (CN/PSOR, policy iteration, Brennan–Schwartz,
                CRR tree), diagnostics, models (PINNs, HVI, surrogate heads),
                campaign utilities, HD pipeline, plotting
experiments/    Entry-point scripts: stages 01–28, standalone/ methods,
                hd*_ high-dimensional runs, reference_audits/ (DIRK vs CRR50k)
configs/        YAML/JSON experiment configurations
analysis/       Consolidated-dataset builder and final-comparison figure scripts
results/        Committed result tables, manifests and compact figures
                (consolidated_csv/ holds the cross-repository dataset)
reports/        LaTeX stage reports, the PINN experiment report and generated figures
paper/          Manuscript and supplementary material sources
docs/           Methodology notes, campaign closures, design notes, poster
scripts/        Plotting, artifact bootstrap, Windows run helpers, stage-1 plots
tests/          pytest / unittest suite
```

## Installation

Python 3.10+ (CI uses 3.11).

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

## Running the tests

```bash
export PYTHONPATH=src:experiments MPLBACKEND=Agg
bash scripts/bootstrap_artifacts.sh   # regenerates ignored report figures and data/processed
git checkout -- results               # the bootstrap also rewrites a few tracked summaries
python -m pytest
```

A few report and manuscript tests check figures that are generated rather than
committed, so they fail until `scripts/bootstrap_artifacts.sh` has run. To run the
DIRK/CRR50k contracts, clone the companion repository at the pinned commit and add
its `src/` directory to `PYTHONPATH`:

```bash
git clone https://github.com/Ruixixu-hub/2026-surf-american-risk-surfaces.git ../ruixi
git -C ../ruixi checkout 5f63fa7b12f704c3577c72f9d7b7e5146e78c1ca
PYTHONPATH=../ruixi/src:. python -m pytest tests/test_dirk_binomial_50k_convergence.py
```

## Running the experiments

Run all commands from the repository root with `PYTHONPATH=src:experiments`.

```bash
# Independent method campaign: list methods, run the classical LCP screen (M01)
python experiments/26_independent_method_campaign.py --list-methods
python experiments/26_independent_method_campaign.py --method M01 --phase screen \
    --smoke --config configs/independent_m01_classical.yaml --output-root /tmp/m01

# 50,000-step binomial benchmark convergence (preflight; use --phase all for the full run)
python experiments/binomial_benchmark_convergence.py --phase preflight --result-dir /tmp/binomial

# Final matched high-dimensional HVI comparison (d = 10, 20, 50); "status" reads committed results
python experiments/hd3_matched_american_put_boundary_complete.py --phase status

# Stage-1 HD plots from committed results (written to results/hd3_*/plots_stage1/)
python scripts/plots/stage1/run_stage1_all.py
```

The full high-dimensional training runs were executed on a Windows GPU workstation.
The helper scripts are `scripts/run_hd3_matched_boundary_complete_windows.ps1` and
`scripts/run_binomial_benchmark_windows.ps1`, with instructions under `docs/campaign/`
and `docs/binomial_benchmark_convergence_windows.md`.

## Release verification

```bash
python experiments/16_release_verification.py
```

This checks that the required documents and results are present and that the claim
language stays within the audited scope.
