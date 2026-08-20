from __future__ import annotations

import pandas as pd

from common_plot_utils import (
    BINOMIAL_CSV,
    PLOT_OUT,
    RESULT_ROOT,
    SUMMARY_CSV,
    ensure_dir,
    load_json,
    read_run_metrics,
    require_columns,
)


def build_hd_summary() -> pd.DataFrame:
    df = read_run_metrics().copy()

    summary = pd.DataFrame(
        {
            "dimension": df["dimension"].astype(int),
            "model": df["model"],
            "price_rmse_internal": df["normalized_price_rmse"].astype(float),
            "reduced_delta_rmse_internal": df["reduced_delta_rmse"].astype(float),
            "reduced_gamma_rmse_internal": df["reduced_gamma_rmse"].astype(float),
            "trusted_pde_p95": df["trusted_pde_p95"].astype(float),
            "permutation_consistency_rmse": df["permutation_consistency_rmse"].astype(float),
            "transverse_invariance_rmse": df["transverse_invariance_rmse"].astype(float),
            "training_seconds": df["training_seconds"].astype(float),
            "parameter_count": df["parameter_count"].astype(int),
            "peak_cuda_gib": df["peak_cuda_gib"].astype(float),
            "boundary_coverage": df["boundary_coverage"].astype(float),
            "reference_boundary_count": df["reference_boundary_count"].astype(int),
            "predicted_boundary_count": df["predicted_boundary_count"].astype(int),
            "matched_boundary_count": df["matched_boundary_count"].astype(int),
        }
    )

    if not ((summary["boundary_coverage"] >= 0.0) & (summary["boundary_coverage"] <= 1.0)).all():
        raise ValueError("boundary_coverage must lie in [0,1]")

    return summary.sort_values(["model", "dimension"]).reset_index(drop=True)


def build_binomial_convergence() -> pd.DataFrame:
    path = RESULT_ROOT / "binomial_convergence.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing binomial convergence file: {path}")

    payload = load_json(path)
    if not isinstance(payload, dict) or "convergence" not in payload:
        raise ValueError("binomial_convergence.json must contain a 'convergence' array")

    df = pd.DataFrame(payload["convergence"])
    require_columns(
        df,
        [
            "tau_over_T",
            "steps",
            "max_abs_change_from_previous",
            "rmse_change_from_previous",
        ],
        "binomial convergence records",
    )

    final_steps = int(payload.get("final_curve_steps", -1))
    if final_steps != 2048:
        raise ValueError(f"Final cached CRR curve must use 2048 steps, found {final_steps}")

    # The first refinement in each maturity has no previous level, so its
    # convergence difference is intentionally null and is excluded from plots.
    out = df.dropna(subset=["max_abs_change_from_previous"]).copy()
    out["tau_over_T"] = out["tau_over_T"].astype(float)
    out["steps"] = out["steps"].astype(int)
    out["max_abs_change_from_previous"] = out["max_abs_change_from_previous"].astype(float)
    out["rmse_change_from_previous"] = out["rmse_change_from_previous"].astype(float)
    return out.sort_values(["tau_over_T", "steps"]).reset_index(drop=True)


def main() -> None:
    ensure_dir(PLOT_OUT)

    hd = build_hd_summary()
    hd.to_csv(SUMMARY_CSV, index=False)
    print(f"Wrote: {SUMMARY_CSV}")

    binomial = build_binomial_convergence()
    binomial.to_csv(BINOMIAL_CSV, index=False)
    print(f"Wrote: {BINOMIAL_CSV}")


if __name__ == "__main__":
    main()
