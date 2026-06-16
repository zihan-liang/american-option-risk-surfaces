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

## Stage 3 Boundary/Greek Diagnostics Commands

```powershell
$env:PYTHONPATH="src;experiments"
$env:PYTHONPYCACHEPREFIX="$env:TEMP\american_option_pycache"
python -m unittest discover -s tests
python -m compileall -q src tests experiments
python experiments/02_boundary_greeks_diagnostics.py --config configs/base_put.yaml
python experiments/02_boundary_greeks_diagnostics.py --config configs/base_call_dividend.yaml
python experiments/02_boundary_greeks_diagnostics.py --both
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build reports/stage03_boundary_greeks_diagnostics.tex
```

Stage 3 output contract:

- [x] Boundary CSV includes `tau`, `boundary`, `status`, `exercise_nodes`, and `continuation_nodes`.
- [x] NPZ includes `spots`, `taus`, `values`, `delta`, `gamma`, and named masks.
- [x] Greek summaries distinguish all nodes, strict mask, payoff kink band, boundary band, and near-zero spot band.
- [x] Figures are generated under `results/figures/stage03_boundary_greeks/`.

## Stage 4 Grid/Domain/Runtime Commands

```powershell
$env:PYTHONPATH="src;experiments"
$env:PYTHONPYCACHEPREFIX="$env:TEMP\american_option_pycache"
python -m unittest discover -s tests
python -m compileall -q src tests experiments
python experiments/03_convergence_domain_runtime.py --config configs/convergence.yaml
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build reports/stage04_grid_domain_runtime.tex
```

Stage 4 output contract:

- [x] `grid_convergence.csv` contains price, boundary, strict Delta, and strict Gamma errors for both option cases.
- [x] `domain_sensitivity.csv` contains put and dividend-call domain checks over `Smax in {3K,4K,5K}`.
- [x] `runtime_table.csv` contains runtime and PSOR iteration counts.
- [x] `diagnostics_table.csv` contains obstacle, complementarity, monotonicity, and convexity diagnostics for every grid/domain run.
- [x] `convergence_config_manifest.json` records the configuration used to generate the Stage 4 outputs.
- [x] Figures are generated under `results/figures/stage04_convergence_runtime/`.

## Stage 5 Application Risk-Surface Commands

```powershell
$env:PYTHONPATH="src;experiments"
$env:PYTHONPYCACHEPREFIX="$env:TEMP\american_option_pycache"
python -m unittest discover -s tests
python -m compileall -q src tests experiments
python experiments/04_application_risk_surfaces.py --config configs/application_sweep.yaml
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build reports/stage05_application_risk_surfaces.tex
```

Stage 5 output contract:

- [x] `risk_metrics.csv` includes diagnostics, runtime, PSOR iterations, and exercise fractions for every regime.
- [x] `boundary_summary.csv` separates genuine `boundary_found` levels from no-boundary controls.
- [x] `greek_summary.csv` reports strict-mask Delta/Gamma summaries.
- [x] `regime_flags.csv` records delicate and diagnostic-failure regimes.
- [x] `surface_index.csv` records the selected representative `.npz` files.
- [x] `application_config_manifest.json` records the sweep configuration and `surface_run_id`.
- [x] Figures are generated under `results/figures/stage05_application_risk_surfaces/`.

## Stage 6 Surrogate Dataset Commands

```powershell
$env:PYTHONPATH="src;experiments"
$env:PYTHONPYCACHEPREFIX="$env:TEMP\american_option_pycache"
python -m unittest discover -s tests
python -m compileall -q src tests experiments
python experiments/05_surrogate_dataset.py --config configs/surrogate_dataset.yaml
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build reports/stage06_surrogate_dataset.tex
```

Stage 6 output contract:

- [x] `data/processed/surrogate_dataset.npz` contains features, labels, masks, split labels, regime keys, source-stage labels, and solver-config hashes.
- [x] `dataset_manifest.json` records source Stage 5 files, solver settings, sample counts, split counts, and schema names.
- [x] `split_table.csv` uses regime-level train/validation/test splits with no leakage.
- [x] `excluded_regimes.csv` records every control or diagnostic-failure regime excluded from clean labels.
- [x] `dataset_schema.json`, `label_summary.csv`, `mask_summary.csv`, and `dataset_audit.json` are generated.
- [x] Label-distribution figures are generated under `results/figures/stage06_surrogate_dataset/`.

## Stage 7 Baseline Surrogate Commands

```powershell
$env:PYTHONPATH="src;experiments"
$env:PYTHONPYCACHEPREFIX="$env:TEMP\american_option_pycache"
python -m unittest discover -s tests
python -m compileall -q src tests experiments
python experiments/06_surrogate_baselines.py --config configs/baseline_surrogates.yaml
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build reports/stage07_baseline_surrogates.tex
```

Stage 7 output contract:

- [x] `baseline_metrics.csv` reports RMSE, MAE, max error, and obstacle violation by split and option type.
- [x] `baseline_by_regime.csv` keeps regime-level errors and does not hide weak regimes.
- [x] `violation_table.csv` reports obstacle, boundary-band, monotonicity, convexity, train time, and inference throughput.
- [x] `training_history.csv` records MLP training curves and penalty weights.
- [x] `baseline_summary.json` records the selected validation baseline and sample caps.
- [x] Model artifacts are written under `results/models/` and kept out of Git by ignore rules.
- [x] Figures are generated under `results/figures/stage07_baseline_surrogates/`.

## Stage 7.5 Literature Review Integration Commands

```powershell
$env:PYTHONPYCACHEPREFIX="$env:TEMP\american_option_pycache"
python -m compileall -q src tests experiments
python -m unittest discover -s tests
Select-String -Path docs/literature_review.md,paper/manuscript_outline.tex -Pattern "turn[0-9]|TODO|TBD"
pdflatex -interaction=nonstopmode -halt-on-error -jobname=manuscript_outline_stage75 -output-directory=build paper/manuscript_outline.tex
bibtex build/manuscript_outline_stage75
pdflatex -interaction=nonstopmode -halt-on-error -jobname=manuscript_outline_stage75 -output-directory=build paper/manuscript_outline.tex
pdflatex -interaction=nonstopmode -halt-on-error -jobname=manuscript_outline_stage75 -output-directory=build paper/manuscript_outline.tex
Copy-Item -LiteralPath build\manuscript_outline_stage75.pdf -Destination build\manuscript_outline.pdf -Force
```

Stage 7.5 output contract:

- [x] `deep-research-report.md` is preserved as raw source material.
- [x] `docs/literature_review.md` is English, cleaned, and free of Deep Research citation markers.
- [x] `paper/references.bib` contains the core classical, ML, constraint-aware, and reproducibility references.
- [x] `paper/manuscript_outline.tex` includes literature-backed citation anchors and a BibTeX bibliography.
- [x] `docs/claim_audit.md` records cautious claim wording for positive continuation premium, boundary heads, and Delta heads.

## Stage 8 Positive Premium Surrogate Commands

```powershell
$env:PYTHONPATH="src;experiments"
$env:PYTHONPYCACHEPREFIX="$env:TEMP\american_option_pycache"
python -m unittest discover -s tests
python -m compileall -q src tests experiments
python experiments/07_positive_premium_tuning.py --config configs/positive_premium.yaml
python experiments/08_boundary_distance_profiles.py --config configs/boundary_distance_profiles.yaml
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build reports/stage08_positive_premium_surrogate.tex
```

Stage 8 output contract:

- [x] `positive_premium_metrics.csv` reports validation/test price and premium errors by split and option type.
- [x] `positive_premium_by_regime.csv` retains regime-level errors.
- [x] `positive_premium_violation_table.csv` records zero obstacle violation plus boundary-band and strict-mask errors.
- [x] `positive_premium_training_history.csv` records MLP training history and linear sanity baseline row.
- [x] `positive_premium_summary.json` records the best aggregate validation model and selected neural artifact.
- [x] `boundary_distance_metrics.csv` reports near-boundary, away-boundary, near-kink, and strict-mask price errors.
- [x] `results/models/selected_final_price_premium.pt` stores the selected neural positive-premium model.
