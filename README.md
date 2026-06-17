# Free-Boundary-Aware Risk Surfaces for American Options

This repository contains a reproducible computational-finance workflow for American option risk surfaces. The project builds a validated Crank-Nicolson/PSOR benchmark first, then uses that benchmark to generate audited labels for modular machine-learning diagnostics.

## Final Project Status

The completed workflow covers:

1. mathematical VI/LCP formulation notes;
2. CN/PSOR solver validation;
3. boundary, Delta, Gamma, shape, grid/domain, and runtime diagnostics;
4. American put and dividend-paying American call risk-surface experiments;
5. clean surrogate dataset construction from diagnostic-passing PSOR labels;
6. baseline direct-price surrogate violation audit;
7. positive continuation-premium price surrogate;
8. direct boundary diagnostic head;
9. bounded supervised Delta head;
10. integrated modular workflow with PSOR audit;
11. final claim audit, final report, manuscript outline, and future-extension roadmap.

## Claim Boundaries

Allowed project claims are intentionally scoped:

- CN/PSOR is the audited benchmark and label generator within tested regimes.
- The positive-premium price surrogate enforces `price >= payoff` by construction on evaluated samples.
- Direct boundary heads provide scoped supervised boundary diagnostics.
- The bounded Delta head provides scoped strict-mask Delta diagnostics.
- The integrated workflow supports scenario screening with PSOR audit.

The workflow does not replace PSOR. The project does not claim that one network solves price, boundary, and Greeks. The price surrogate is not a Greek engine, the boundary head is not an exact free-boundary solver, and future-extension notes are not completed results.

## Main Outputs

- Final claim matrix: `results/tables/final_claim_matrix.csv`
- Final report source: `reports/final_report.tex`
- Manuscript outline: `paper/manuscript_outline.tex`
- Literature review: `docs/literature_review.md`
- Future roadmap: `docs/future_extensions.md`
- Release handoff: `docs/release_handoff.md`
- Artifact manifest: `docs/artifact_manifest.md`

Generated PDFs are built into `build/` locally and are not treated as source artifacts.

## Reproducibility Commands

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt

$env:PYTHONPATH="src;experiments"
$env:PYTHONPYCACHEPREFIX="$env:TEMP\american_option_pycache"

python -m unittest discover -s tests
python -m compileall -q src tests experiments
python experiments/14_claim_audit.py --config configs/claim_audit.yaml
python experiments/15_future_extensions_manifest.py --config configs/future_extensions.yaml
python experiments/16_release_verification.py
```

## Report Builds

```powershell
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build reports/final_report.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build reports/stage13_future_extensions.tex
pdflatex -interaction=nonstopmode -halt-on-error -jobname=manuscript_outline_release -output-directory=build paper/manuscript_outline.tex
bibtex build/manuscript_outline_release
pdflatex -interaction=nonstopmode -halt-on-error -jobname=manuscript_outline_release -output-directory=build paper/manuscript_outline.tex
pdflatex -interaction=nonstopmode -halt-on-error -jobname=manuscript_outline_release -output-directory=build paper/manuscript_outline.tex
Copy-Item -LiteralPath build\manuscript_outline_release.pdf -Destination build\manuscript_outline.pdf -Force
```

## Repository Layout

- `src/solvers/`: Black-Scholes formulas, CN/PSOR solver, boundary extraction, Greeks.
- `src/diagnostics/`: obstacle, complementarity, monotonicity, convexity, runtime metrics.
- `src/models/`: modular PyTorch models for price/premium, boundary, and Delta heads.
- `src/data/`: dataset construction and regime splits.
- `experiments/`: reproducible stage scripts and release verification.
- `configs/`: YAML experiment configurations.
- `docs/`, `reports/`, `paper/`: project governance, reports, and manuscript artifacts.

## Release Verification

Run `python experiments/16_release_verification.py` before handoff. It checks required docs/results, Stage 12 and Stage 13 claim hygiene, and whether stale generated manuscript artifacts are still tracked.
