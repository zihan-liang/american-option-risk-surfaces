# Reproducibility Checklist

- [ ] Record Python version and package versions.
- [x] Run `python -m unittest discover -s tests` for the solver-validation stage.
- [x] Run `python -m compileall -q src tests experiments` for the solver-validation stage.
- [ ] Store each experiment config beside outputs or in manifests.
- [ ] Record random seed for datasets and surrogate training.
- [ ] Keep generated `.npz`, `.pt`, plots, and logs out of Git unless explicitly curated.
- [ ] Link every paper figure to a script, config, and source data file.
- [x] Re-run solver validation after changing numerical settings.

## Solver Validation Audit Commands

```powershell
$env:PYTHONPATH="src"
$env:PYTHONPYCACHEPREFIX="$env:TEMP\american_option_pycache"
python -m unittest discover -s tests
python -m compileall -q src tests experiments
python experiments/01_solver_validation.py --config configs/solver_validation.yaml
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build reports/stage02_solver_validation.tex
```
