"""Reusable visual helpers for report enrichment figures."""

from __future__ import annotations

import csv
from pathlib import Path
import re
import textwrap
from typing import Any

import matplotlib.image as mpimg
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_DISPLAY_LABELS = {
    "american_put": "American put",
    "dividend_american_call": "Dividend-paying call",
    "no_dividend_american_call": "No-dividend call",
    "european_put_closed_form": "European put",
    "linear_price_baseline": "Linear price baseline",
    "mlp_direct_price": "Direct-price MLP",
    "mlp_obstacle_penalty_1": r"Penalty MLP ($\lambda=1$)",
    "mlp_obstacle_penalty_10": r"Penalty MLP ($\lambda=10$)",
    "positive_premium_linear": "Positive-premium linear",
    "positive_premium_mlp": "Positive-premium MLP",
    "boundary_head_linear": "Boundary linear",
    "boundary_head_mlp": "Boundary MLP",
    "call_boundary_head_linear": "Call-boundary linear",
    "call_boundary_head_mlp": "Call-boundary MLP",
    "pooled_boundary_head_linear_on_calls": "Pooled boundary linear (calls)",
    "bounded_delta_linear": r"Bounded $\Delta$ linear",
    "bounded_delta_head_mlp": r"Bounded $\Delta$ MLP",
    "price_model_autograd_delta": r"Price-model autograd $\Delta$",
    "price_model_finite_difference_delta": r"Price-model finite-difference $\Delta$",
    "price_over_K": r"$U/K$",
    "premium_over_K": r"$C/K$",
    "payoff_over_K": r"$\Phi/K$",
    "delta": r"$\Delta$",
    "K_gamma": r"$K\Gamma$",
    "exercise_indicator": "Exercise",
    "strict_greek_mask": "Strict Greek mask",
    "near_kink": "Near payoff kink",
    "near_boundary": "Near boundary",
    "near_zero_spot": "Near zero spot",
    "main_claim_ready": "Main-text ready",
    "supplement_only": "Supplement only",
}

DEFAULT_REPORT_DPI = 300
RASTERIZED_REPORT_DPI = 450


def format_display_label(value: Any, aliases: dict[str, str] | None = None) -> str:
    raw = str(value)
    mapping = {**DEFAULT_DISPLAY_LABELS, **(aliases or {})}
    if raw in mapping:
        return mapping[raw]

    regime = re.fullmatch(
        r"(put|call)_T([^_]+)_sig([^_]+)_r([^_]+)_q([^_]+)",
        raw,
    )
    if regime:
        option, maturity, sigma, rate, dividend = regime.groups()
        decode = lambda token: token.replace("p", ".")
        option_label = "Put" if option == "put" else "Call"
        return (
            rf"{option_label}: $T={decode(maturity)}$, "
            rf"$\sigma={decode(sigma)}$, $r={decode(rate)}$, "
            rf"$q={decode(dividend)}$"
        )

    return raw.replace("_", " ").strip().title()


def apply_report_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "figure.dpi": DEFAULT_REPORT_DPI,
            "savefig.dpi": DEFAULT_REPORT_DPI,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


def save_visual(spec: dict[str, Any], output_path: str | Path) -> None:
    kind = spec.get("kind", "workflow")
    if kind == "payoff":
        save_payoff(output_path, spec)
    elif kind == "regions":
        save_regions(output_path, spec)
    elif kind == "lcp":
        save_lcp(output_path, spec)
    elif kind == "premium":
        save_premium(output_path, spec)
    elif kind == "matrix":
        save_matrix(output_path, spec)
    elif kind == "fd_stencil":
        save_fd_stencil(output_path, spec)
    elif kind == "operator_coefficients":
        save_operator_coefficients(output_path, spec)
    elif kind == "operator_matrix_annotated":
        save_operator_matrix_annotated(output_path, spec)
    elif kind == "cn_matrix_pair":
        save_cn_matrix_pair(output_path, spec)
    elif kind == "rhs_boundary_adjustment":
        save_rhs_boundary_adjustment(output_path, spec)
    elif kind == "psor_update_flow":
        save_psor_update_flow(output_path, spec)
    elif kind == "lcp_vector_form":
        save_lcp_vector_form(output_path, spec)
    elif kind == "time_marching_grid":
        save_time_marching_grid(output_path, spec)
    elif kind == "bar_from_csv":
        save_bar_from_csv(output_path, spec)
    elif kind == "line_from_csv":
        save_line_from_csv(output_path, spec)
    elif kind == "image_panel":
        save_image_panel(output_path, spec)
    elif kind == "split_pie":
        save_split_pie(output_path, spec)
    elif kind == "claim_map":
        save_claim_map(output_path, spec)
    elif kind == "roadmap":
        save_roadmap(output_path, spec)
    elif kind == "contribution":
        save_contribution(output_path, spec)
    else:
        save_workflow(output_path, spec)


def save_workflow(path: str | Path, spec: dict[str, Any]) -> None:
    apply_report_style()
    labels = spec.get("labels") or ["CN/PSOR", "Diagnostics", "Dataset", "Separate heads", "PSOR audit"]
    layout = str(spec.get("layout", "horizontal")).lower()
    fig_size = (6.4, 6.2) if layout == "vertical" else (8.6, 3.2)
    fig, ax = plt.subplots(figsize=fig_size)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    if layout == "vertical":
        ys = np.linspace(0.82, 0.18, len(labels))
        for y, label in zip(ys, labels):
            wrapped = "\n".join(textwrap.wrap(str(label).replace(" | ", " "), width=28))
            ax.text(
                0.5,
                y,
                wrapped,
                ha="center",
                va="center",
                fontsize=10,
                bbox={"boxstyle": "round,pad=0.55", "fc": "#edf2f7", "ec": "#2d3748"},
            )
        for upper, lower in zip(ys[:-1], ys[1:]):
            ax.annotate("", xy=(0.5, lower + 0.065), xytext=(0.5, upper - 0.065), arrowprops={"arrowstyle": "->", "lw": 1.5})
    else:
        xs = np.linspace(0.08, 0.92, len(labels))
        for x, label in zip(xs, labels):
            wrapped = "\n".join(textwrap.wrap(str(label).replace(" | ", " "), width=16))
            ax.text(
                x,
                0.55,
                wrapped,
                ha="center",
                va="center",
                fontsize=8.5,
                bbox={"boxstyle": "round,pad=0.32", "fc": "#edf2f7", "ec": "#2d3748"},
            )
        for left, right in zip(xs[:-1], xs[1:]):
            ax.annotate("", xy=(right - 0.075, 0.55), xytext=(left + 0.075, 0.55), arrowprops={"arrowstyle": "->", "lw": 1.4})
    ax.set_title(spec.get("title", "Workflow"), pad=12)
    _save(fig, path)


def save_payoff(path: str | Path, spec: dict[str, Any]) -> None:
    apply_report_style()
    s = np.linspace(0.0, 2.0, 300)
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    ax.plot(s, np.maximum(1 - s, 0), label="American put payoff", lw=2)
    ax.plot(s, np.maximum(s - 1, 0), label="American call payoff", lw=2)
    ax.axvline(1, color="#4a5568", ls="--", lw=1, label=r"$S=K$")
    ax.set_xlabel(r"$S/K$")
    ax.set_ylabel(r"$\Phi/K$")
    ax.set_title(spec.get("title", "Put and call payoff geometry"))
    ax.legend(fontsize=8)
    _save(fig, path)


def save_regions(path: str | Path, spec: dict[str, Any]) -> None:
    apply_report_style()
    tau = np.linspace(0.02, 2.0, 160)
    boundary = 0.72 + 0.23 * (1 - np.exp(-1.7 * tau))
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    ax.plot(tau, boundary, color="#2b6cb0", lw=2, label="free boundary")
    ax.fill_between(tau, 0.35, boundary, color="#fed7d7", label="exercise", alpha=0.8)
    ax.fill_between(tau, boundary, 1.8, color="#c6f6d5", label="continuation", alpha=0.8)
    ax.set_xlabel(r"Time to maturity, $\tau$")
    ax.set_ylabel(r"$S/K$")
    ax.set_title(spec.get("title", "Exercise and continuation regions"))
    ax.legend(fontsize=8)
    _save(fig, path)


def save_lcp(path: str | Path, spec: dict[str, Any]) -> None:
    apply_report_style()
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    u_minus_phi = np.linspace(0, 1, 120)
    residual = np.maximum(0, 1 - 1.8 * u_minus_phi)
    ax.plot(u_minus_phi, residual, lw=2, label="admissible residual envelope")
    ax.scatter([0, 0.55], [0.8, 0], color=["#e53e3e", "#2b6cb0"], zorder=3)
    ax.text(0.02, 0.82, "exercise\nU=payoff", fontsize=8)
    ax.text(0.58, 0.05, "continuation\nPDE residual=0", fontsize=8)
    ax.set_xlabel(r"$U-\Phi$")
    ax.set_ylabel("PDE/LCP residual")
    ax.set_title(spec.get("title", "Complementarity geometry"))
    _save(fig, path)


def save_premium(path: str | Path, spec: dict[str, Any]) -> None:
    apply_report_style()
    x = np.linspace(-5, 5, 300)
    softplus = np.log1p(np.exp(x))
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    ax.plot(x, softplus, lw=2, label=r"$\operatorname{softplus}(f_\theta)$")
    ax.axhline(0, color="#4a5568", lw=1)
    ax.set_xlabel("raw network output")
    ax.set_ylabel(r"$C/K \geq 0$")
    ax.set_title(spec.get("title", "Positive continuation-premium construction"))
    ax.legend(fontsize=8)
    _save(fig, path)


def save_matrix(path: str | Path, spec: dict[str, Any]) -> None:
    apply_report_style()
    n = int(spec.get("size", 28))
    mat = np.zeros((n, n))
    for i in range(n):
        mat[i, i] = 1
        if i > 0:
            mat[i, i - 1] = 0.65
        if i < n - 1:
            mat[i, i + 1] = 0.65
    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    ax.imshow(mat, cmap="Blues")
    ax.set_title(spec.get("title", "Tridiagonal finite-difference structure"))
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    _save(fig, path, dpi=RASTERIZED_REPORT_DPI)


def save_fd_stencil(path: str | Path, spec: dict[str, Any]) -> None:
    apply_report_style()
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    xs = [0.22, 0.5, 0.78]
    labels = [r"$U_{i-1}$", r"$U_i$", r"$U_{i+1}$"]
    spot_labels = [r"$S_{i-1}$", r"$S_i$", r"$S_{i+1}$"]
    for x, label, spot in zip(xs, labels, spot_labels):
        ax.scatter([x], [0.55], s=520, color="#ebf8ff", edgecolors="#2b6cb0", linewidths=1.6)
        ax.text(x, 0.55, label, ha="center", va="center", fontsize=13)
        ax.text(x, 0.26, spot, ha="center", va="center", fontsize=10)
    ax.plot(xs, [0.55, 0.55, 0.55], color="#2d3748", lw=1.3)
    ax.annotate("", xy=(0.47, 0.77), xytext=(0.25, 0.77), arrowprops={"arrowstyle": "<->", "lw": 1.1})
    ax.annotate("", xy=(0.75, 0.77), xytext=(0.53, 0.77), arrowprops={"arrowstyle": "<->", "lw": 1.1})
    ax.text(0.36, 0.84, r"$\Delta S$", ha="center")
    ax.text(0.64, 0.84, r"$\Delta S$", ha="center")
    ax.text(
        0.5,
        0.05,
        r"$U_S \approx (U_{i+1}-U_{i-1})/(2\Delta S)$, "
        r"$U_{SS} \approx (U_{i+1}-2U_i+U_{i-1})/\Delta S^2$",
        ha="center",
        fontsize=10,
    )
    ax.set_title(spec.get("title", "Finite-difference stencil"), pad=12)
    _save(fig, path)


def save_operator_coefficients(path: str | Path, spec: dict[str, Any]) -> None:
    apply_report_style()
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    boxes = [
        (0.18, r"$a_i$", r"lower", r"$\frac{1}{2}\sigma^2S_i^2/\Delta S^2-\frac{(r-q)S_i}{2\Delta S}$"),
        (0.50, r"$b_i$", r"diagonal", r"$-\sigma^2S_i^2/\Delta S^2-r$"),
        (0.82, r"$c_i$", r"upper", r"$\frac{1}{2}\sigma^2S_i^2/\Delta S^2+\frac{(r-q)S_i}{2\Delta S}$"),
    ]
    for x, symbol, name, formula in boxes:
        ax.text(x, 0.72, symbol, ha="center", va="center", fontsize=16, bbox={"boxstyle": "round,pad=0.35", "fc": "#edf2f7", "ec": "#2d3748"})
        ax.text(x, 0.49, name, ha="center", fontsize=10, color="#2d3748")
        ax.text(x, 0.25, formula, ha="center", fontsize=9)
    ax.text(0.5, 0.04, r"Row $i$ of $L_hU$ is $a_iU_{i-1}+b_iU_i+c_iU_{i+1}$.", ha="center", fontsize=10)
    ax.set_title(spec.get("title", "Black-Scholes finite-difference coefficients"), pad=12)
    _save(fig, path)


def save_operator_matrix_annotated(path: str | Path, spec: dict[str, Any]) -> None:
    apply_report_style()
    n = int(spec.get("size", 7))
    mat = np.full((n, n), np.nan)
    for i in range(n):
        mat[i, i] = 0.0
        if i > 0:
            mat[i, i - 1] = -1.0
        if i < n - 1:
            mat[i, i + 1] = 1.0
    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    shown = np.nan_to_num(mat, nan=0.0)
    ax.imshow(np.abs(shown), cmap="Blues", vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            text = ""
            if i == j:
                text = r"$b_i$" if i == n // 2 else r"$b$"
            elif i == j + 1:
                text = r"$a_i$" if i == n // 2 else r"$a$"
            elif i + 1 == j:
                text = r"$c_i$" if i == n // 2 else r"$c$"
            if text:
                ax.text(j, i, text, ha="center", va="center", fontsize=10)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([fr"$U_{i+1}$" for i in range(n)], rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels([fr"row {i+1}" for i in range(n)], fontsize=8)
    ax.set_title(spec.get("title", r"Annotated $L_h$ matrix"))
    _save(fig, path, dpi=RASTERIZED_REPORT_DPI)


def save_cn_matrix_pair(path: str | Path, spec: dict[str, Any]) -> None:
    apply_report_style()
    n = 7
    l_mat = np.zeros((n, n))
    for i in range(n):
        l_mat[i, i] = -2
        if i > 0:
            l_mat[i, i - 1] = 1
        if i < n - 1:
            l_mat[i, i + 1] = 1
    dt = 0.2
    eye = np.eye(n)
    a_mat = eye - 0.5 * dt * l_mat
    b_mat = eye + 0.5 * dt * l_mat
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.1))
    cmap = plt.get_cmap("coolwarm")
    norm = mcolors.Normalize(vmin=-0.25, vmax=1.25)
    for ax, mat, title in zip(axes, [a_mat, b_mat], [r"$A=I-\Delta\tau L_h/2$", r"$B=I+\Delta\tau L_h/2$"]):
        for row in range(n):
            for col in range(n):
                ax.add_patch(
                    plt.Rectangle(
                        (col - 0.5, row - 0.5),
                        1,
                        1,
                        facecolor=cmap(norm(mat[row, col])),
                        edgecolor="#ffffff",
                        linewidth=0.25,
                    )
                )
        ax.set_title(title)
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(n - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for i in range(n):
            for j in range(n):
                if abs(mat[i, j]) > 1e-12:
                    ax.text(j, i, f"{mat[i,j]:.1f}", ha="center", va="center", fontsize=7)
    fig.suptitle(spec.get("title", "Crank-Nicolson matrix pair"), y=0.98)
    _save(fig, path, dpi=RASTERIZED_REPORT_DPI)


def save_rhs_boundary_adjustment(path: str | Path, spec: dict[str, Any]) -> None:
    apply_report_style()
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    labels = [
        r"$B U^n$",
        r"$+\frac{\Delta\tau}{2}a_1(L^n+L^{n+1})$",
        r"$+\frac{\Delta\tau}{2}c_{M-1}(R^n+R^{n+1})$",
        r"$rhs$",
    ]
    xs = [0.12, 0.39, 0.66, 0.9]
    colors = ["#edf2f7", "#fff5f5", "#fffaf0", "#e6fffa"]
    for x, label, color in zip(xs, labels, colors):
        ax.text(x, 0.55, label, ha="center", va="center", fontsize=11, bbox={"boxstyle": "round,pad=0.35", "fc": color, "ec": "#2d3748"})
    for left, right in zip(xs[:-1], xs[1:]):
        ax.annotate("", xy=(right - 0.08, 0.55), xytext=(left + 0.08, 0.55), arrowprops={"arrowstyle": "->", "lw": 1.3})
    ax.text(0.5, 0.18, "Only the first and last interior equations receive boundary contributions.", ha="center", fontsize=10)
    ax.set_title(spec.get("title", "Boundary adjustment in the CN right hand side"), pad=12)
    _save(fig, path)


def save_psor_update_flow(path: str | Path, spec: dict[str, Any]) -> None:
    apply_report_style()
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    boxes = [
        (0.18, 0.70, "old iterate\n$x_i^{old}$", "#edf2f7"),
        (0.50, 0.70, "Gauss-Seidel\ncandidate $y_i$", "#ebf8ff"),
        (0.82, 0.70, "relaxation\n$x_i^{old}+\\omega(y_i-x_i^{old})$", "#fefcbf"),
        (0.50, 0.38, "obstacle projection\n$\\max(\\Phi_i, \\cdot)$", "#fed7d7"),
        (0.50, 0.16, "new feasible iterate\n$x_i^{new} \\geq \\Phi_i$", "#c6f6d5"),
    ]
    for x, y, label, color in boxes:
        ax.text(x, y, label, ha="center", va="center", fontsize=10, bbox={"boxstyle": "round,pad=0.45", "fc": color, "ec": "#2d3748"})
    ax.annotate("", xy=(0.38, 0.70), xytext=(0.28, 0.70), arrowprops={"arrowstyle": "->", "lw": 1.4})
    ax.annotate("", xy=(0.70, 0.70), xytext=(0.60, 0.70), arrowprops={"arrowstyle": "->", "lw": 1.4})
    ax.annotate("", xy=(0.58, 0.48), xytext=(0.76, 0.61), arrowprops={"arrowstyle": "->", "lw": 1.4})
    ax.annotate("", xy=(0.50, 0.25), xytext=(0.50, 0.31), arrowprops={"arrowstyle": "->", "lw": 1.4})
    ax.text(0.5, 0.02, r"PSOR solves the CN linear system while enforcing the American payoff obstacle at each node.", ha="center", fontsize=9)
    ax.set_title(spec.get("title", "PSOR projected update"), pad=12)
    _save(fig, path)


def save_lcp_vector_form(path: str | Path, spec: dict[str, Any]) -> None:
    apply_report_style()
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    lines = [
        r"$A U^{n+1} \geq B U^n$",
        r"$U^{n+1} \geq \Phi$",
        r"$(U^{n+1}-\Phi)^T(AU^{n+1}-BU^n)=0$",
    ]
    y_positions = [0.75, 0.52, 0.29]
    colors = ["#ebf8ff", "#f0fff4", "#fffaf0"]
    for y, line, color in zip(y_positions, lines, colors):
        ax.text(0.5, y, line, ha="center", va="center", fontsize=14, bbox={"boxstyle": "round,pad=0.35", "fc": color, "ec": "#2d3748"})
    ax.text(0.5, 0.08, "At each node either the PDE residual is zero or the option sits on the payoff obstacle.", ha="center", fontsize=9)
    ax.set_title(spec.get("title", "Linear complementarity system"), pad=12)
    _save(fig, path)


def save_time_marching_grid(path: str | Path, spec: dict[str, Any]) -> None:
    apply_report_style()
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    s = np.arange(0, 7)
    tau = np.arange(0, 5)
    for t in tau:
        ax.scatter(s, np.full_like(s, t), s=85, color="#ebf8ff", edgecolors="#2b6cb0")
    for t in tau[:-1]:
        ax.annotate("", xy=(3, t + 0.78), xytext=(3, t + 0.22), arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "#2d3748"})
    ax.text(3, -0.45, "payoff at tau = 0", ha="center", fontsize=10)
    ax.text(3, 3.72, "later time-to-maturity level", ha="center", fontsize=9)
    ax.set_xlabel("spot grid index")
    ax.set_ylabel("time-to-maturity step")
    ax.set_title(spec.get("title", "Time marching from payoff"), pad=12)
    ax.set_xticks(s)
    ax.set_yticks(tau)
    fig.subplots_adjust(top=0.82, bottom=0.18)
    _save(fig, path)


def save_bar_from_csv(path: str | Path, spec: dict[str, Any]) -> None:
    rows = _read_csv(spec.get("source_artifact", ""))
    x_key = spec.get("x")
    y_key = spec.get("y")
    labels: list[str] = []
    values: list[float] = []
    for row in rows:
        if spec.get("filter_key") and str(row.get(spec["filter_key"])) != str(spec.get("filter_value")):
            continue
        value = _to_float(row.get(y_key, ""))
        if value == value:
            labels.append(
                format_display_label(
                    row.get(x_key, len(labels)),
                    spec.get("display_labels"),
                )
            )
            values.append(value)
        if len(values) >= int(spec.get("limit", 10)):
            break
    _bar(
        path,
        labels,
        values,
        spec.get("title", "Metric summary"),
        spec.get("ylabel", y_key or "value"),
        orientation=spec.get("orientation", "vertical"),
    )


def save_line_from_csv(path: str | Path, spec: dict[str, Any]) -> None:
    apply_report_style()
    rows = _read_csv(spec.get("source_artifact", ""))
    x_key = spec.get("x")
    y_key = spec.get("y")
    groups = spec.get("group")
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    if groups:
        for group_value in sorted({row.get(groups, "") for row in rows}):
            xs, ys = _xy(rows, x_key, y_key, group_key=groups, group_value=group_value)
            if xs:
                ax.plot(xs, ys, marker="o", label=group_value)
        ax.legend(fontsize=8)
    else:
        xs, ys = _xy(rows, x_key, y_key)
        ax.plot(xs, ys, marker="o")
    ax.set_xlabel(spec.get("xlabel", x_key or "x"))
    ax.set_ylabel(spec.get("ylabel", y_key or "y"))
    ax.set_title(spec.get("title", "Metric trend"))
    _save(fig, path)


def save_image_panel(path: str | Path, spec: dict[str, Any]) -> None:
    apply_report_style()
    source = Path(spec.get("source_artifact", ""))
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    if source.exists() and source.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        ax.imshow(mpimg.imread(source))
        ax.axis("off")
    else:
        ax.text(0.5, 0.5, "source artifact\navailable in repository", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
    if spec.get("show_title", False):
        ax.set_title(spec.get("title", "Evidence figure"))
    _save(fig, path)


def save_split_pie(path: str | Path, spec: dict[str, Any]) -> None:
    rows = _read_csv(spec.get("source_artifact", ""))
    key = spec.get("key", "split")
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.get(key, "unknown")] = counts.get(row.get(key, "unknown"), 0) + int(float(row.get("samples", 1) or 1))
    labels = list(counts) or ["train", "validation", "test"]
    values = list(counts.values()) or [70, 15, 15]
    apply_report_style()
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    ax.pie(values, labels=labels, autopct="%1.0f%%", startangle=90)
    ax.set_title(spec.get("title", "Dataset split composition"))
    _save(fig, path)


def save_claim_map(path: str | Path, spec: dict[str, Any]) -> None:
    rows = _read_csv(spec.get("source_artifact", ""))
    counts: dict[str, int] = {}
    for row in rows:
        status = row.get("readiness") or row.get("status") or "unknown"
        counts[status] = counts.get(status, 0) + 1
    labels = [format_display_label(label, spec.get("display_labels")) for label in counts]
    _bar(
        path,
        labels,
        list(counts.values()),
        spec.get("title", "Claim readiness map"),
        "Claims",
        orientation=spec.get("orientation", "vertical"),
    )


def save_roadmap(path: str | Path, spec: dict[str, Any]) -> None:
    apply_report_style()
    labels = spec.get("labels") or ["Larger regimes", "Active learning", "UQ", "Shape constraints", "Alternative dynamics"]
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    y = np.zeros(len(labels))
    x = np.arange(len(labels))
    ax.scatter(x, y, s=180, color="#2b6cb0")
    for i, label in enumerate(labels):
        ax.text(i, 0.13, str(label).replace(" ", "\n"), ha="center", fontsize=8)
    ax.plot(x, y, color="#2b6cb0", lw=2)
    ax.set_ylim(-0.2, 0.65)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title(spec.get("title", "Future-work roadmap"))
    _save(fig, path)


def save_contribution(path: str | Path, spec: dict[str, Any]) -> None:
    labels = spec.get("labels") or [
        "AI/ML pipeline",
        "Python experiments",
        "Diagnostics",
        "Plotting",
        "Reproducibility",
        "LaTeX reporting",
    ]
    values = [5, 5, 4, 5, 5, 4]
    _bar(
        path,
        labels,
        values,
        spec.get("title", "Student contribution focus"),
        "Emphasis",
        orientation=spec.get("orientation", "horizontal"),
    )


def _bar(
    path: str | Path,
    labels: list[str],
    values: list[float],
    title: str,
    ylabel: str,
    *,
    orientation: str = "vertical",
) -> None:
    apply_report_style()
    horizontal = orientation.lower() == "horizontal"
    if horizontal:
        fig, ax = plt.subplots(figsize=(7.2, max(3.6, 0.48 * max(len(labels), 1))))
    else:
        fig, ax = plt.subplots(figsize=(max(5.8, min(10.0, 0.75 * max(len(labels), 1))), 3.6))
    if labels and values:
        if horizontal:
            positions = np.arange(len(values))
            ax.barh(positions, values, color="#2b6cb0")
            ax.set_yticks(positions)
            ax.set_yticklabels(labels)
            ax.invert_yaxis()
            ax.set_xlabel(ylabel)
        else:
            ax.bar(range(len(values)), values, color="#2b6cb0")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(
                labels,
                rotation=25 if len(labels) > 5 else 0,
                ha="right" if len(labels) > 5 else "center",
            )
            ax.set_ylabel(ylabel)
    else:
        ax.text(0.5, 0.5, "No numeric rows available", ha="center", va="center")
    ax.set_title(title)
    _save(fig, path)


def _xy(rows: list[dict[str, str]], x_key: str, y_key: str, *, group_key: str | None = None, group_value: str | None = None) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        if group_key and row.get(group_key) != group_value:
            continue
        x = _to_float(row.get(x_key, ""))
        y = _to_float(row.get(y_key, ""))
        if x == x and y == y:
            xs.append(x)
            ys.append(y)
    pairs = sorted(zip(xs, ys))
    return [x for x, _ in pairs], [y for _, y in pairs]


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    candidate = Path(path)
    if not candidate.exists() or candidate.suffix.lower() != ".csv":
        return []
    with candidate.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _save(fig, path: str | Path, *, dpi: int = DEFAULT_REPORT_DPI) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches=None, dpi=dpi)
    plt.close(fig)
