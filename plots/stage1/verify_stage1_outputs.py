from __future__ import annotations

from pathlib import Path

import pandas as pd

from common_plot_utils import PLOT_OUT, SUMMARY_CSV, BINOMIAL_CSV


FIGURE_NAMES = [
    "F_hd_price_rmse_vs_dimension",
    "F_hd_reduced_delta_rmse_vs_dimension",
    "F_hd_reduced_gamma_rmse_vs_dimension",
    "F_hd_pde_p95_vs_dimension",
    "F_hd_training_time_vs_dimension",
    "F_hd_parameter_count_vs_dimension",
    "F_hd_permutation_consistency_vs_dimension",
    "F_hd_transverse_invariance_vs_dimension",
    "F_hd_boundary_coverage",
    "F_hd_accuracy_compute_internal",
    "F_binomial_convergence",
]


def main() -> None:
    summary = pd.read_csv(SUMMARY_CSV)
    if len(summary) != 6:
        raise RuntimeError(f"Expected six HD summary rows, found {len(summary)}")
    if summary.isna().any().any():
        raise RuntimeError("HD Stage 1 summary unexpectedly contains NaN values")

    binomial = pd.read_csv(BINOMIAL_CSV)
    if binomial.empty:
        raise RuntimeError("Binomial convergence plotting table is empty")
    if binomial[["tau_over_T", "steps", "max_abs_change_from_previous"]].isna().any().any():
        raise RuntimeError("Binomial plotting table contains unexpected NaN values")

    missing: list[str] = []
    empty: list[str] = []
    for stem in FIGURE_NAMES:
        for suffix in (".png", ".pdf"):
            path = PLOT_OUT / f"{stem}{suffix}"
            if not path.exists():
                missing.append(str(path))
            elif path.stat().st_size == 0:
                empty.append(str(path))

    if missing:
        raise RuntimeError("Missing Stage 1 outputs:\n" + "\n".join(missing))
    if empty:
        raise RuntimeError("Empty Stage 1 outputs:\n" + "\n".join(empty))

    print(f"Verified {len(FIGURE_NAMES)} figures in PNG+PDF format.")
    print(f"Output directory: {PLOT_OUT}")


if __name__ == "__main__":
    main()
