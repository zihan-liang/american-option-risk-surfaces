# Release Handoff

## What Is Complete

The repository now contains a full reproducible workflow for the SURF project:

- mathematical formulation notes for the American option VI/LCP;
- validated CN/PSOR benchmark and diagnostics;
- American put and dividend-paying American call risk-surface studies;
- clean PSOR-labelled surrogate dataset;
- direct-price baseline violation audit;
- positive-premium price/premium surrogate;
- direct boundary diagnostic head;
- bounded supervised Delta head;
- integrated modular workflow with PSOR audit;
- final claim audit, report outline, literature review, and future-extension roadmap.

## What Is Not Claimed

The workflow does not replace PSOR. The project does not claim that one network solves price, boundary, and Greeks. The price surrogate is not a Greek engine. The boundary head is not an exact free-boundary solver. The Delta head is not a Gamma model. Future-extension notes are not completed results.

## How To Verify

Run the release checks from the repository root:

```powershell
$env:PYTHONPATH="src;experiments"
$env:PYTHONPYCACHEPREFIX="$env:TEMP\american_option_pycache"
python -m unittest discover -s tests
python -m compileall -q src tests experiments
python experiments/14_claim_audit.py --config configs/claim_audit.yaml
python experiments/15_future_extensions_manifest.py --config configs/future_extensions.yaml
python experiments/16_release_verification.py
```

Build report PDFs locally:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build reports/final_report.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build reports/stage13_future_extensions.tex
pdflatex -interaction=nonstopmode -halt-on-error -jobname=manuscript_outline_release -output-directory=build paper/manuscript_outline.tex
bibtex build/manuscript_outline_release
pdflatex -interaction=nonstopmode -halt-on-error -jobname=manuscript_outline_release -output-directory=build paper/manuscript_outline.tex
pdflatex -interaction=nonstopmode -halt-on-error -jobname=manuscript_outline_release -output-directory=build paper/manuscript_outline.tex
Copy-Item -LiteralPath build\manuscript_outline_release.pdf -Destination build\manuscript_outline.pdf -Force
```

## How To Continue

Future work should begin with a new project plan and a new claim gate. Use `docs/future_extensions.md` to choose an extension, then create a separate validation plan before implementing code. Active learning, uncertainty quantification, shape-constrained models, Gamma surrogates, alternative dynamics, and market calibration require new evidence before they can appear in the claim matrix.

## Local Artifact Policy

Large `.pt`, `.npz`, generated figures, and build PDF artifacts are local outputs. Do not commit them unless a future release explicitly changes the artifact policy.

## Required Git Cleanup Before Final Release

The release verifier flags stale generated manuscript artifacts if they remain tracked by Git. Remove them from tracking locally before the final release commit:

```powershell
Remove-Item -LiteralPath .git\index.lock -Force -ErrorAction SilentlyContinue
git rm -- manuscript_outline.aux manuscript_outline.log manuscript_outline.pdf paper/manuscript_outline.pdf
python experiments/16_release_verification.py
```

The generated manuscript PDF should be rebuilt into `build/manuscript_outline.pdf` instead of committed from the repository root or `paper/`.
