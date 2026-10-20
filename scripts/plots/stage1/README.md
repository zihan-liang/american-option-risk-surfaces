# Stage 1 final HD plotting

This package renders the HD-only figures from the committed final results in:

`results/hd3_matched_american_put_boundary_complete/`

It does **not** perform training and does **not** modify canonical result files.

## Scientific scope

Stage 1 intentionally uses the HD experiment's internal CN/PSOR reference for the price/Greek RMSE plots. Do not present those RMSE values as the final cross-repository Arm C / Arm D ranking. The later common-reference comparison must rescore all methods against the same Ruixi DIRK + Policy Iteration + sinh-grid high-accuracy reference.

The free-boundary coverage plot uses the frozen primary criterion already stored in `run_metrics.csv`; it does not reconstruct or fabricate missing boundaries.

## Requirements

Python 3.11 is recommended. Required plotting packages:

```powershell
python -m pip install pandas matplotlib
```

If using the existing CUDA/PINN virtual environment, no GPU is required for plotting.

## Run every Stage 1 figure

From the repository root:

```powershell
python .\scripts\plots\stage1\run_stage1_all.py
```

## Output

Outputs are written to:

`results/hd3_matched_american_put_boundary_complete/plots_stage1/`

Expected figures, each in PNG and PDF:

1. `F_hd_price_rmse_vs_dimension`
2. `F_hd_reduced_delta_rmse_vs_dimension`
3. `F_hd_reduced_gamma_rmse_vs_dimension`
4. `F_hd_pde_p95_vs_dimension`
5. `F_hd_training_time_vs_dimension`
6. `F_hd_parameter_count_vs_dimension`
7. `F_hd_permutation_consistency_vs_dimension`
8. `F_hd_transverse_invariance_vs_dimension`
9. `F_hd_boundary_coverage`
10. `F_hd_accuracy_compute_internal`
11. `F_binomial_convergence`

The runner also writes normalized plotting tables and calls `verify_stage1_outputs.py` to verify that all 11 figures exist in both formats and that the plotting tables are non-empty and finite where expected.

## Run individual steps

```powershell
python .\scripts\plots\stage1\build_stage1_tables.py
python .\scripts\plots\stage1\plot_hd_scaling.py
python .\scripts\plots\stage1\plot_hd_boundary_and_compute.py
python .\scripts\plots\stage1\plot_binomial_convergence.py
python .\scripts\plots\stage1\verify_stage1_outputs.py
```
