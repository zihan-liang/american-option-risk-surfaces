from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from common_plot_utils import BINOMIAL_CSV, PLOT_OUT, save_figure


def main() -> None:
    df = pd.read_csv(BINOMIAL_CSV)

    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    for tau in sorted(df["tau_over_T"].unique(), reverse=True):
        group = df[df["tau_over_T"] == tau].sort_values("steps")
        ax.plot(group["steps"], group["max_abs_change_from_previous"], marker="o")
        ax.annotate(
            rf"$\tau/T={tau:g}$",
            (group["steps"].iloc[-1], group["max_abs_change_from_previous"].iloc[-1]),
            xytext=(7, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ticks = sorted(int(v) for v in df["steps"].unique())
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(v) for v in ticks])
    ax.set_xlabel("CRR tree steps")
    ax.set_ylabel("Max. absolute change from previous refinement")
    ax.margins(x=0.18)

    save_figure(fig, PLOT_OUT / "F_binomial_convergence")
    print(f"Binomial convergence figure written to: {PLOT_OUT}")


if __name__ == "__main__":
    main()
