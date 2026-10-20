"""Dependency-free CI renderer for the revised Fig01 at tau/T=0.5.

Reads the same consolidated immutable arrays as the Matplotlib plotter and
writes a static SVG for GitHub review.  No experiment code is rerun.
"""
from __future__ import annotations

import bisect
import csv
import gzip
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARRAYS = ROOT / "results" / "consolidated_csv" / "arrays"
OUTPUT = ROOT / "results" / "figures" / "final_matched_comparison" / "poster"

TAU_TARGET = 0.5
S_MIN = 0.5
S_MAX = 1.5

LOW_DIM_IDS = {
    "reference": "afc1aa3ab21f5e09",
    "arm_c": "ba4871fe2762f15d",
    "arm_d": "0a4a7530da60aab1",
}
HD_IDS = {
    "fc": "fd142e14b9491876",
    "deepsets": "fdae846ad052f43a",
}
SERIES = [
    ("High-Accuracy DIRK Reference", "reference", "#202020", 3.0),
    ("Arm C (seed 101)", "arm_c", "#A23B72", 2.1),
    ("Arm D (seed 101)", "arm_d", "#2A9D8F", 2.1),
    ("FC-HVI d20", "fc", "#2878B5", 2.1),
    ("DeepSets-HVI d20", "deepsets", "#E68A2E", 2.1),
]


def read_flat(prefix: str, name: str) -> list[float]:
    path = ARRAYS / f"{prefix}__{name}.csv.gz"
    if not path.is_file():
        raise FileNotFoundError(path)
    values: list[float] = []
    with gzip.open(path, "rt", newline="") as handle:
        for row in csv.reader(handle):
            for cell in row:
                cell = cell.strip()
                if cell:
                    values.append(float(cell))
    return values


def rows_from_flat(values: list[float], n_tau: int, n_s: int) -> list[list[float]]:
    if len(values) != n_tau * n_s:
        raise ValueError(f"surface size {len(values)} != {n_tau}*{n_s}")
    return [values[i * n_s : (i + 1) * n_s] for i in range(n_tau)]


def load_low(prefix: str) -> tuple[list[float], list[float], list[list[float]]]:
    s = read_flat(prefix, "S_over_K")
    tau = read_flat(prefix, "tau_over_T")
    v = rows_from_flat(read_flat(prefix, "V_over_K"), len(tau), len(s))
    return s, tau, v


def load_hd(prefix: str) -> tuple[list[float], list[float], list[list[float]]]:
    s_flat = read_flat(prefix, "S_over_K")
    tau_flat = read_flat(prefix, "tau_over_T")
    v_flat = read_flat(prefix, "V_over_K")
    if not (len(s_flat) == len(tau_flat) == len(v_flat)):
        raise ValueError("HD flattened arrays have inconsistent lengths")
    s = sorted(set(s_flat))
    tau = sorted(set(tau_flat))
    lookup = {(t, x): v for x, t, v in zip(s_flat, tau_flat, v_flat)}
    if len(lookup) != len(s) * len(tau):
        raise ValueError("HD surface is not rectangular")
    rows = [[lookup[(t, x)] for x in s] for t in tau]
    return s, tau, rows


def interpolate_time(tau: list[float], rows: list[list[float]], target: float) -> list[float]:
    j = bisect.bisect_left(tau, target)
    if j == 0:
        return list(rows[0])
    if j < len(tau) and math.isclose(tau[j], target, rel_tol=0.0, abs_tol=1e-12):
        return list(rows[j])
    if j >= len(tau):
        return list(rows[-1])
    lo = j - 1
    w = (target - tau[lo]) / (tau[j] - tau[lo])
    return [(1.0 - w) * a + w * b for a, b in zip(rows[lo], rows[j])]


def interp1(x_src: list[float], y_src: list[float], x: float) -> float:
    j = bisect.bisect_left(x_src, x)
    if j == 0:
        return y_src[0]
    if j >= len(x_src):
        return y_src[-1]
    if math.isclose(x_src[j], x, rel_tol=0.0, abs_tol=1e-12):
        return y_src[j]
    lo = j - 1
    w = (x - x_src[lo]) / (x_src[j] - x_src[lo])
    return (1.0 - w) * y_src[lo] + w * y_src[j]


def on_grid(surface: tuple[list[float], list[float], list[list[float]]], grid: list[float]) -> list[float]:
    s, tau, rows = surface
    y = interpolate_time(tau, rows, TAU_TARGET)
    return [interp1(s, y, x) for x in grid]


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def main() -> int:
    surfaces = {
        "reference": load_low(LOW_DIM_IDS["reference"]),
        "arm_c": load_low(LOW_DIM_IDS["arm_c"]),
        "arm_d": load_low(LOW_DIM_IDS["arm_d"]),
        "fc": load_hd(HD_IDS["fc"]),
        "deepsets": load_hd(HD_IDS["deepsets"]),
    }
    ref_s = surfaces["reference"][0]
    grid = [x for x in ref_s if S_MIN - 1e-12 <= x <= S_MAX + 1e-12]
    if not grid:
        raise ValueError("empty shared plotting grid")
    curves = {key: on_grid(surface, grid) for key, surface in surfaces.items()}

    all_y = [v for values in curves.values() for v in values]
    ymin, ymax = min(all_y), max(all_y)
    pad = max(0.02 * (ymax - ymin), 1e-4)
    ymin -= pad
    ymax += pad

    W, H = 1200, 700
    L, R, T, B = 100, 45, 55, 85
    PW, PH = W - L - R, H - T - B

    def sx(x: float) -> float:
        return L + (x - S_MIN) / (S_MAX - S_MIN) * PW

    def sy(y: float) -> float:
        return T + (ymax - y) / (ymax - ymin) * PH

    out: list[str] = []
    out.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
    out.append('<rect width="100%" height="100%" fill="white"/>')
    out.append('<style>text{font-family:Arial,Helvetica,sans-serif;fill:#202020}.tick{font-size:15px}.label{font-size:19px}.title{font-size:22px;font-weight:600}.legend{font-size:15px}</style>')

    for i in range(11):
        x = S_MIN + 0.1 * i
        px = sx(x)
        out.append(f'<line x1="{px:.2f}" y1="{T}" x2="{px:.2f}" y2="{T+PH}" stroke="#D8D8D8" stroke-width="1" opacity="0.8"/>')
        out.append(f'<text class="tick" x="{px:.2f}" y="{T+PH+27}" text-anchor="middle">{x:.1f}</text>')
    for i in range(6):
        y = ymin + (ymax - ymin) * i / 5
        py = sy(y)
        out.append(f'<line x1="{L}" y1="{py:.2f}" x2="{L+PW}" y2="{py:.2f}" stroke="#D8D8D8" stroke-width="1" opacity="0.8"/>')
        out.append(f'<text class="tick" x="{L-14}" y="{py+5:.2f}" text-anchor="end">{y:.2f}</text>')

    out.append(f'<line x1="{L}" y1="{T}" x2="{L}" y2="{T+PH}" stroke="#333" stroke-width="1.4"/>')
    out.append(f'<line x1="{L}" y1="{T+PH}" x2="{L+PW}" y2="{T+PH}" stroke="#333" stroke-width="1.4"/>')

    for label, key, color, width in SERIES:
        pts = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in zip(grid, curves[key]))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"/>')

    out.append(f'<text class="title" x="{W/2:.1f}" y="31" text-anchor="middle">Price Slice at τ/T=0.5</text>')
    out.append(f'<text class="label" x="{L+PW/2:.1f}" y="{H-24}" text-anchor="middle">S/K</text>')
    out.append(f'<text class="label" x="28" y="{T+PH/2:.1f}" text-anchor="middle" transform="rotate(-90 28 {T+PH/2:.1f})">V/K</text>')

    box_w, box_h = 300, 151
    box_x, box_y = L + PW - box_w - 14, T + 14
    out.append(f'<rect x="{box_x}" y="{box_y}" width="{box_w}" height="{box_h}" rx="5" fill="white" stroke="#BBBBBB" opacity="0.95"/>')
    for i, (label, _, color, width) in enumerate(SERIES):
        yy = box_y + 25 + i * 26
        out.append(f'<line x1="{box_x+15}" y1="{yy}" x2="{box_x+55}" y2="{yy}" stroke="{color}" stroke-width="{width}"/>')
        out.append(f'<text class="legend" x="{box_x+65}" y="{yy+5}">{esc(label)}</text>')

    out.append('</svg>')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    svg = OUTPUT / "fig01_price_slice_tau05.svg"
    svg.write_text("\n".join(out), encoding="utf-8")

    ref = curves["reference"]
    log_lines = [
        f"wrote {svg}",
        f"points={len(grid)} S/K=[{grid[0]:.6g},{grid[-1]:.6g}] tau/T={TAU_TARGET}",
        f"svg_bytes={svg.stat().st_size}",
    ]
    for label, key, _, _ in SERIES[1:]:
        err = [a - b for a, b in zip(curves[key], ref)]
        rmse = math.sqrt(sum(e * e for e in err) / len(err))
        log_lines.append(f"{label}: signed_err_min={min(err):.6e} signed_err_max={max(err):.6e} rmse={rmse:.6e}")
    log = OUTPUT / "fig01_render.log"
    log.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(log.read_text(encoding="utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
