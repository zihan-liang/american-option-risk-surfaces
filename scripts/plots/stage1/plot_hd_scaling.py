from __future__ import annotations

import pandas as pd

from common_plot_utils import PLOT_OUT, SUMMARY_CSV, plot_metric_vs_dimension


def main() -> None:
    df = pd.read_csv(SUMMARY_CSV)

    plot_metric_vs_dimension(
        df,
        y_col="price_rmse_internal",
        y_label="Price RMSE (internal CN/PSOR reference)",
        out_name="F_hd_price_rmse_vs_dimension",
    )
    plot_metric_vs_dimension(
        df,
        y_col="reduced_delta_rmse_internal",
        y_label=r"Reduced $\Delta_G$ RMSE (internal CN/PSOR reference)",
        out_name="F_hd_reduced_delta_rmse_vs_dimension",
    )
    plot_metric_vs_dimension(
        df,
        y_col="reduced_gamma_rmse_internal",
        y_label=r"Reduced $\Gamma_G$ RMSE (internal CN/PSOR reference)",
        out_name="F_hd_reduced_gamma_rmse_vs_dimension",
    )
    plot_metric_vs_dimension(
        df,
        y_col="trusted_pde_p95",
        y_label="Trusted PDE residual P95",
        out_name="F_hd_pde_p95_vs_dimension",
    )
    plot_metric_vs_dimension(
        df,
        y_col="training_seconds",
        y_label="Training time (s)",
        out_name="F_hd_training_time_vs_dimension",
    )
    plot_metric_vs_dimension(
        df,
        y_col="parameter_count",
        y_label="Trainable parameters",
        out_name="F_hd_parameter_count_vs_dimension",
    )
    plot_metric_vs_dimension(
        df,
        y_col="permutation_consistency_rmse",
        y_label="Permutation-consistency RMSE",
        out_name="F_hd_permutation_consistency_vs_dimension",
        log_y=True,
    )
    plot_metric_vs_dimension(
        df,
        y_col="transverse_invariance_rmse",
        y_label="Transverse-invariance RMSE",
        out_name="F_hd_transverse_invariance_vs_dimension",
    )

    print(f"HD scaling figures written to: {PLOT_OUT}")


if __name__ == "__main__":
    main()
