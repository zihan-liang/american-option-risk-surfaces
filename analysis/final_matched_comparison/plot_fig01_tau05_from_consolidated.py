"""Plot the revised Fig01 directly from consolidated immutable experiment arrays.

This script is intentionally post-processing only. It compares the stored
high-accuracy DIRK reference, Arm C, Arm D, FC-HVI d20, and DeepSets-HVI d20
on the shared American-put contract at tau/T = 0.5 and S/K in [0.5, 1.5].
The rendered figure is the poster-specific single-panel Fig01 revision.
"""
from __future__ import annotations

import gzip
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
ARRAYS = ROOT / "results" / "consolidated_csv" / "arrays"
OUTPUT = ROOT / "figures" / "final_matched_comparison" / "poster"

TAU_TARGET = 0.5
S_MIN = 0.5
S_MAX = 1.5

COLORS = {
    "reference": "#202020",
    "arm_c": "#A23B72",
    "arm_d": "#2A9D8F",
    "fc": "#2878B5",
    "deepsets": "#E68A2E",
}

# Consolidated-array identifiers from array_artifacts.csv.
LOW_DIM_IDS = {
    "reference": "afc1aa3ab21f5e09",
    "arm_c": "ba4871fe2762f15d",
    "arm_d": "0a4a7530da60aab1",
}
HD_IDS = {
    "fc": "fd142e14b9491876",
    "deepsets": "fdae846ad052f43a",
}


def _read_array(prefix: str, name: str) -> np.ndarray:
    path = ARRAYS / f"{prefix}__{name}.csv.gz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with gzip.open(path, "rt") as handle:
        return np.loadtxt(handle, delimiter=",")


def _low_dim_surface(prefix: str) -> dict[str, np.ndarray]:
    s = _read_array(prefix, "S_over_K").reshape(-1)
    tau = _read_array(prefix, "tau_over_T").reshape(-1)
    values = _read_array(prefix, "V_over_K")
    values = np.asarray(values, dtype=float).reshape(tau.size, s.size)
    return {"S": s, "tau": tau, "V": values}


def _hd_surface(prefix: str) -> dict[str, np.ndarray]:
    s_flat = _read_array(prefix, "S_over_K").reshape(-1)
    tau_flat = _read_array(prefix, "tau_over_T").reshape(-1)
    v_flat = _read_array(prefix, "V_over_K").reshape(-1)
    s = np.unique(s_flat)
    tau = np.unique(tau_flat)
    if s.size * tau.size != v_flat.size:
        raise ValueError(f"Non-rectangular HD surface for {prefix}")
    order = np.lexsort((s_flat, tau_flat))
    values = v_flat[order].reshape(tau.size, s.size)
    return {"S": s, "tau": tau, "V": values}


def _time_slice(surface: dict[str, np.ndarray], tau_target: float) -> np.ndarray:
    tau = surface["tau"]
    values = surface["V"]
    upper = int(np.searchsorted(tau, tau_target, side="left"))
    if upper == 0:
        return values[0].copy()
    if upper < tau.size and np.isclose(tau[upper], tau_target):
        return values[upper].copy()
    if upper >= tau.size:
        return values[-1].copy()
    lower = upper - 1
    w = (tau_target - tau[lower]) / (tau[upper] - tau[lower])
    return (1.0 - w) * values[lower] + w * values[upper]


def _on_common_s(surface: dict[str, np.ndarray], s_common: np.ndarray) -> np.ndarray:
    return np.interp(s_common, surface["S"], _time_slice(surface, TAU_TARGET))


def main() -> int:
    reference = _low_dim_surface(LOW_DIM_IDS["reference"])
    arm_c = _low_dim_surface(LOW_DIM_IDS["arm_c"])
    arm_d = _low_dim_surface(LOW_DIM_IDS["arm_d"])
    fc = _hd_surface(HD_IDS["fc"])
    deepsets = _hd_surface(HD_IDS["deepsets"])

    s = reference["S"]
    mask = (s >= S_MIN) & (s <= S_MAX)
    s_plot = s[mask]

    series = [
        ("High-Accuracy DIRK Reference", reference, "reference", 2.8),
        ("Arm C (seed 101)", arm_c, "arm_c", 2.0),
        ("Arm D (seed 101)", arm_d, "arm_d", 2.0),
        ("FC-HVI d20", fc, "fc", 2.0),
        ("DeepSets-HVI d20", deepsets, "deepsets", 2.0),
    ]

    # 12 x 7 gives a larger single-panel poster figure without changing aspect
    # ratio aggressively; PDF remains vector and PNG is exported at 360 dpi.
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    for label, surface, color_key, linewidth in series:
        ax.plot(
            s_plot,
            _on_common_s(surface, s_plot),
            label=label,
            color=COLORS[color_key],
            linewidth=linewidth,
        )

    ax.set_xlim(S_MIN, S_MAX)
    ax.set_xlabel(r"$S/K$", fontsize=14)
    ax.set_ylabel(r"$V/K$", fontsize=14)
    ax.set_title(r"Price Slice at $\tau/T=0.5$", fontsize=15, pad=10)
    ax.tick_params(labelsize=12)
    ax.grid(True, color="#D8D8D8", linewidth=0.65, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95, fontsize=10.5)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    pdf = OUTPUT / "fig01_price_slice_tau05.pdf"
    png = OUTPUT / "fig01_price_slice_tau05.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=360, bbox_inches="tight")
    plt.close(fig)

    # Lightweight numerical audit to make silent data mixups obvious in CI.
    ref_values = _on_common_s(reference, s_plot)
    print(f"wrote {pdf}")
    print(f"wrote {png}")
    print(f"points={s_plot.size}, S/K=[{s_plot[0]:.6g}, {s_plot[-1]:.6g}], tau/T={TAU_TARGET}")
    for label, surface, _, _ in series[1:]:
        err = _on_common_s(surface, s_plot) - ref_values
        print(f"{label}: signed_err_min={err.min():.6e}, signed_err_max={err.max():.6e}, rmse={np.sqrt(np.mean(err**2)):.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
