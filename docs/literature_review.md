# Literature Review and Claim Support

This document integrates the raw Deep Research output in `docs/notes/deep-research-report.md` into a project-facing literature review. The raw file is retained as source material, but its internal citation markers are not final citations. The citable bibliography for the manuscript is maintained in `paper/references.bib`.

## Executive Summary

- The classical literature strongly supports formulating American option pricing as an optimal stopping, variational inequality, obstacle, and linear complementarity problem. This supports the project's VI/LCP framing.
- Crank-Nicolson finite differences with PSOR is a reasonable one-dimensional Black-Scholes American option benchmark, but it should be described as a classical benchmark, not as a new solver.
- Obstacle violation and complementarity residuals are core numerical diagnostics because they directly test the discrete LCP structure.
- Grid refinement, domain sensitivity, and runtime studies are necessary because truncation, mesh size, and nonsmooth payoffs affect price, boundary, and Greek reliability.
- Payoff kinks and free boundaries make Gamma especially delicate. Greek reporting should use masks or carefully defined regions of interest.
- Neural option-pricing surrogates are well established as accelerators or emulators, but ordinary direct-price networks do not automatically preserve obstacle, monotonicity, convexity, or no-arbitrage structure.
- The positive continuation-premium parameterization is supported indirectly by early-exercise premium and obstacle decompositions. It should be presented as a literature-inspired design, not as a settled standard architecture.
- Boundary and Delta heads should be described as modular diagnostic tasks. The literature supports separate stopping-rule, boundary, and derivative supervision more strongly than a one-network-solves-everything claim.
- Regime-level evaluation is more meaningful than pointwise random splitting for this project because random point splits can overstate generalization across parameter regimes.
- The project's strongest positioning is as a reproducible research-engineering study: validated CN/PSOR labels, diagnostic-aware risk surfaces, audited baseline failures, and modular constraint-aware surrogate components.

## American Option Formulation and Free Boundaries

American options are naturally described as optimal stopping problems. In PDE language, this leads to a variational inequality or obstacle problem: in the continuation region the value solves a Black-Scholes type PDE, while in the exercise region it equals the payoff. The interface between the two regions is the early-exercise free boundary.

Jaillet, Lamberton, and Lapeyre provide a classical variational-inequality treatment of American option pricing, while Huang and Pang connect option pricing to linear complementarity formulations. These sources support the project's use of the VI/LCP language for solver design and diagnostics.

For American puts, the free boundary is not just a plotting artifact. Work by Kim, Carr-Jarrow-Myneni, Peskir, and Chen-Chadam shows that early-exercise premium and boundary characterizations are central mathematical objects. For dividend-paying American calls, early exercise can become optimal; Evans, Kuske, and Keller provide useful boundary analysis near expiry for assets with dividends.

Project use: cite this literature when introducing the mathematical formulation, the continuation/exercise decomposition, and the reason for extracting early-exercise boundaries as independent diagnostics.

Claim limit: do not claim this project introduces a new formulation of American options. It uses established VI/LCP/free-boundary theory as the numerical and ML diagnostic foundation.

## CN/PSOR and Finite-Difference Benchmarks

Crank-Nicolson is a standard time discretization for parabolic option-pricing PDEs. After finite-difference discretization of the American option variational inequality, the problem becomes a discrete LCP. PSOR is a classical projected iterative method for such LCPs.

The project uses CN/PSOR as a benchmark because it is auditable, reproducible, and compatible with obstacle and complementarity diagnostics. This is the correct role for the solver: a trusted numerical label generator and audit standard, not a novel pricing method.

Boundary truncation is also not a minor implementation detail. Kangro and Nicolaides show that far-field boundary conditions affect Black-Scholes PDE error, supporting explicit domain-sensitivity studies. Pooley, Vetzal, and Forsyth, together with Giles and Carter, support caution around nonsmooth payoffs and Rannacher-style damping, especially for derivative quantities.

Project use: cite this literature in the benchmark method section, the grid/domain/runtime stage reports, and the solver-validation discussion.

Claim limit: the current CN/PSOR implementation can be called validated for the tested engineering envelope, but not publication-grade across all regimes without further refinement and stronger reference comparisons.

## Diagnostics: Obstacle, Complementarity, Shape, Grid, Domain, Runtime

Because the discrete American option problem is an LCP, the obstacle condition and complementarity residual are definition-level checks. A low price RMSE alone cannot show that the numerical surface solves the right constrained problem.

Shape diagnostics also matter. The no-arbitrage and constraint-aware ML literature shows that monotonicity, convexity, and positivity are not automatically preserved by generic networks. Dugas et al., Yang-Zheng-Hospedales, and Cao-Liu-Zhai are useful support for the idea that constraints should be encoded or audited explicitly.

Scientific computing and SciML benchmark literature support the project's broader diagnostic posture: report more than aggregate RMSE, record runtime and configurations, preserve negative results, and avoid hiding failure regimes.

Project use: cite these sources when explaining why each solver and surrogate output is accompanied by obstacle, complementarity, monotonicity, convexity, grid, domain, runtime, and failure-regime tables.

Claim limit: shape diagnostics in this project are numerical audits, not formal no-arbitrage proofs over a continuous domain.

## Boundary Extraction and Greeks

Early-exercise boundary extraction is supported by both classical American option theory and numerical practice. Boundary behavior near expiry can be delicate, especially for American puts and dividend-paying calls, so boundary curves should be reported with status flags and grid-resolution caveats.

Delta and Gamma can be approximated from finite-difference price surfaces, but American-style Greeks are more delicate than prices. In particular, in 't Hout discusses undesirable convergence behavior for CN approximations of Delta and Gamma for American-style options, even with damping. Pooley, Vetzal, and Forsyth also show that nonsmooth payoff features can pollute hedging parameters.

Project use: cite this literature when justifying strict Greek masks, kink masks, boundary masks, and the decision not to treat unmasked Gamma near the free boundary as uniformly reliable.

Claim limit: do not claim the price surface automatically provides reliable Greeks everywhere. Greeks require separate validation and masked reporting.

## Neural Surrogates for Option Pricing

Neural networks have a long history in derivative pricing and hedging. Hutchinson, Lo, and Poggio are an early example of learning-network-based derivative pricing. Ruf and Wang provide a broad review of neural networks for option pricing and hedging. Liu, Oosterlee, and Bohte are especially relevant to this project because they train neural networks as fast emulators of established numerical methods.

This literature supports using neural networks as accelerators or surrogates after a trusted numerical benchmark exists. It does not support replacing PSOR audits altogether, and it does not imply that a direct-price neural network satisfies the American option obstacle or shape properties.

Project use: cite this literature when introducing the surrogate workflow and when explaining why the dataset is generated from diagnostic-passing CN/PSOR outputs.

Claim limit: do not claim ML replaces the benchmark. The safe claim is that ML approximates benchmark outputs within tested regimes and remains subject to diagnostic audits.

## Constraint-Aware and No-Arbitrage ML

Generic neural networks are flexible but not automatically financially consistent. Shape-restricted or no-arbitrage-aware architectures exist because unconstrained networks may violate economically important properties. Dugas et al. support incorporating functional knowledge into neural networks, Yang-Zheng-Hospedales support rationality-by-design for option pricing networks, and Cao-Liu-Zhai support no-arbitrage constraints in neural option valuation.

The positive continuation-premium design planned for this project follows this broader principle. Since an American option value satisfies `V >= payoff`, writing `V_hat = payoff + nonnegative_premium_hat` is a hard obstacle-preserving parameterization. The literature supports the decomposition idea indirectly through early-exercise premium theory, but it does not establish this exact neural architecture as a standard solution.

Project use: cite these sources to motivate hard constraints over soft penalties and to explain why Stage 7 baseline penalty models are not enough.

Claim limit: positive continuation-premium is a reasonable, literature-inspired design whose empirical behavior must still be tested.

## Modular Surrogates: Price, Boundary, and Delta as Separate Tasks

The literature supports modularity more strongly than a single unified claim. Deep optimal stopping work learns stopping rules directly. Early-exercise boundary literature treats the boundary as a separate mathematical object. Differential machine learning work argues that derivative information benefits from direct supervision or derivative-aware training.

This supports the project's modular plan:

- a positive continuation-premium price surrogate for obstacle-consistent price/premium estimates;
- a direct boundary diagnostic head for scoped boundary estimates;
- a bounded supervised Delta head for Delta diagnostics.

Project use: cite Becker-Cheridito-Jentzen, Huge-Savine, Frandsen-Pedersen-Poulsen, and in 't Hout when explaining why the price network is not automatically a boundary or Greek engine.

Claim limit: the modules should be evaluated separately and scoped to tested regimes. Do not claim one neural network solves price, boundary, and Greeks.

## Claim-to-Literature Map

| Project claim | Literature support | Assessment |
| --- | --- | --- |
| CN/PSOR is an appropriate benchmark. | Jaillet-Lamberton-Lapeyre; Huang-Pang; standard finite-difference/LCP literature. | Strong support for benchmark use, not novelty. |
| Obstacle and complementarity diagnostics are necessary. | VI/LCP formulation literature. | Strong support. |
| Grid/domain/runtime studies are necessary. | Kangro-Nicolaides; Pooley-Vetzal-Forsyth; Giles-Carter; scientific benchmarking literature. | Strong support. |
| Gamma near the free boundary must be treated cautiously. | in 't Hout; Pooley-Vetzal-Forsyth. | Strong support. |
| Direct-price neural surrogates are not automatically constraint-preserving. | Dugas et al.; Yang-Zheng-Hospedales; Cao-Liu-Zhai. | Strong support. |
| Positive continuation-premium is reasonable for obstacle consistency. | Kim; Carr-Jarrow-Myneni; Peskir; obstacle decomposition. | Indirect support. Must be tested experimentally. |
| Boundary estimation should be separate from price regression. | Boundary/free-boundary theory; deep optimal stopping. | Moderate to strong support. |
| Delta from a price surrogate requires separate validation. | Differential ML; American Greek approximation literature. | Strong support. |
| Regime-level splits are more meaningful than random point splits. | Ruf-Wang; leakage and SciML benchmark literature. | Moderate support by analogy; strong enough for project protocol. |

## Writing Guidance

Safe wording:

- "This project uses the classical VI/LCP/free-boundary formulation of American options to organize a reproducible numerical benchmark and downstream ML diagnostics."
- "CN/PSOR is used as an auditable benchmark label generator, not as a novel solver contribution."
- "The literature motivates obstacle, complementarity, shape, grid, domain, runtime, boundary, and Greek diagnostics as necessary evidence beyond aggregate price error."
- "The positive continuation-premium surrogate is a literature-inspired hard-constraint parameterization that must be evaluated empirically."
- "Boundary and Delta are treated as separate diagnostic tasks rather than automatic byproducts of a price network."

Avoid:

- "We introduce a new American option solver."
- "One neural network solves price, boundary, and Greeks."
- "The direct price surrogate is automatically obstacle-consistent."
- "Autograd Delta from the price surrogate is reliable without separate validation."
- "ML replaces PSOR."
- "Positive continuation-premium is already established in the literature as the standard American option neural architecture."

Recommended final positioning:

"This project is a reproducible research-engineering study centered on a validated CN/PSOR American option benchmark. It audits free boundaries, obstacle and complementarity conditions, and Greek reliability before evaluating modular, constraint-aware neural surrogate components."
