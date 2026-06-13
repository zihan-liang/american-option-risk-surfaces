# Free-Boundary-Aware Risk Surfaces for American Options

This repository implements a reproducible computational finance workflow for American option risk surfaces:

1. derive and document the variational inequality / LCP formulation;
2. validate a Crank-Nicolson / PSOR finite-difference benchmark;
3. compute diagnostics, early-exercise boundaries, Delta, Gamma, grid/domain sensitivity, and runtime;
4. generate supervised datasets only after solver validation;
5. train modular neural surrogates for price/premium, boundary diagnostics, and bounded Delta diagnostics.

The neural models are accelerators and diagnostic heads. They do not replace the PSOR benchmark, and the price surrogate is not treated as an automatic Greek engine.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
$env:PYTHONPATH="src"
python -m pytest tests -q
python -m compileall -q src tests experiments
python experiments/01_solver_validation.py --config configs/solver_validation.yaml
```

## Repository Layout

- `src/solvers/`: Black-Scholes formulas, CN/PSOR solver, boundary extraction, Greeks.
- `src/diagnostics/`: obstacle, complementarity, monotonicity, convexity, runtime metrics.
- `src/models/`: modular PyTorch models for price/premium, boundary, and Delta heads.
- `src/data/`: dataset construction and regime splits.
- `src/plotting/`: reusable plotting helpers.
- `experiments/`: reproducible stage scripts.
- `configs/`: YAML experiment configurations.
- `docs/`, `reports/`, `paper/`: planning, reporting, and manuscript artifacts.
