#!/usr/bin/env python
"""Build the Figure 1 Perseus repair-and-audit workflow diagram."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon


COLORS = {
    "ink": "#1F2933",
    "muted": "#667085",
    "line": "#AAB4C0",
    "panel": "#F7F9FB",
    "panel_edge": "#D7DEE8",
    "blue": "#EAF2F8",
    "blue_edge": "#416C9B",
    "green": "#E6F1EC",
    "green_edge": "#2C7A68",
    "violet": "#EEEAF5",
    "violet_edge": "#7256A3",
    "amber": "#F7EFE2",
    "amber_edge": "#B07D3F",
    "red": "#F4E5E8",
    "red_edge": "#B95C68",
    "gray": "#F2F4F7",
    "gray_edge": "#8B97A6",
}


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L", "DejaVu Serif", "serif"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "font.size": 7.6,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def rounded_box(
    ax: mpl.axes.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str,
    face: str,
    edge: str,
    title_size: float = 7.7,
    body_size: float = 6.6,
    lw: float = 0.8,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.010,rounding_size=0.018",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.014,
        y + h - 0.036,
        title,
        ha="left",
        va="top",
        fontsize=title_size,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        x + 0.014,
        y + h - 0.083,
        body,
        ha="left",
        va="top",
        fontsize=body_size,
        color=COLORS["muted"],
        linespacing=1.17,
    )


def decision_diamond(
    ax: mpl.axes.Axes,
    cx: float,
    cy: float,
    w: float,
    h: float,
    title: str,
    body: str,
) -> None:
    points = [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)]
    patch = Polygon(points, closed=True, facecolor=COLORS["amber"], edgecolor=COLORS["amber_edge"], linewidth=0.85)
    ax.add_patch(patch)
    ax.text(cx, cy + 0.026, title, ha="center", va="center", fontsize=7.6, fontweight="bold", color=COLORS["ink"])
    ax.text(cx, cy - 0.034, body, ha="center", va="center", fontsize=6.6, color=COLORS["muted"], linespacing=1.06)


def arrow(
    ax: mpl.axes.Axes,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    color: str = "line",
    lw: float = 0.9,
    rad: float = 0.0,
    style: str = "-|>",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle=style,
            mutation_scale=8.5,
            linewidth=lw,
            color=COLORS[color],
            shrinkA=1.2,
            shrinkB=1.2,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def label(ax: mpl.axes.Axes, x: float, y: float, text: str, color: str = "muted", size: float = 6.6) -> None:
    ax.text(x, y, text, ha="center", va="center", fontsize=size, color=COLORS[color])


def chip(ax: mpl.axes.Axes, x: float, y: float, w: float, text: str, edge: str, face: str = "white") -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        0.050,
        boxstyle="round,pad=0.006,rounding_size=0.012",
        linewidth=0.65,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + 0.025, text, ha="center", va="center", fontsize=6.4, color=COLORS["ink"])


def build_framework(result_dir: Path, output_stem: str) -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(7.25, 3.55), constrained_layout=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.012, 0.970, "Perseus: validation-gated repair contract", ha="left", va="top", fontsize=8.8, fontweight="bold", color=COLORS["ink"])
    ax.text(
        0.988,
        0.970,
        "Frozen estimator -> selective repair -> auditable deployment",
        ha="right",
        va="top",
        fontsize=6.8,
        color=COLORS["muted"],
    )
    ax.plot([0.012, 0.988], [0.925, 0.925], color=COLORS["panel_edge"], linewidth=0.65)

    rounded_box(
        ax,
        0.030,
        0.680,
        0.145,
        0.155,
        "Drifted workload",
        "STATS target\ncolumn / table / value",
        COLORS["gray"],
        COLORS["gray_edge"],
    )
    rounded_box(
        ax,
        0.215,
        0.680,
        0.145,
        0.155,
        "Frozen MSCN",
        "1000-bit samples\nweights unchanged",
        COLORS["blue"],
        COLORS["blue_edge"],
    )
    rounded_box(
        ax,
        0.400,
        0.680,
        0.155,
        0.155,
        "Residual field",
        "drift signatures\nlog-cardinality repair",
        COLORS["violet"],
        COLORS["violet_edge"],
    )
    decision_diamond(
        ax,
        0.660,
        0.758,
        0.155,
        0.168,
        "Gate?",
        "tail improves\nvs. raw + scalar",
    )
    rounded_box(
        ax,
        0.820,
        0.745,
        0.145,
        0.118,
        "Deploy",
        "approved repair\nrecord policy",
        COLORS["green"],
        COLORS["green_edge"],
        body_size=6.6,
    )
    rounded_box(
        ax,
        0.820,
        0.575,
        0.145,
        0.118,
        "Fallback",
        "route unstable drift\nkeep safer baseline",
        COLORS["red"],
        COLORS["red_edge"],
        body_size=6.6,
    )

    arrow(ax, 0.177, 0.758, 0.213, 0.758)
    arrow(ax, 0.362, 0.758, 0.397, 0.758)
    arrow(ax, 0.557, 0.758, 0.581, 0.758)
    arrow(ax, 0.738, 0.780, 0.818, 0.800, color="green_edge")
    arrow(ax, 0.736, 0.735, 0.818, 0.633, color="red_edge")
    label(ax, 0.790, 0.832, "pass", color="green_edge")
    label(ax, 0.787, 0.675, "fail", color="red_edge")

    band = FancyBboxPatch(
        (0.022, 0.365),
        0.956,
        0.175,
        boxstyle="round,pad=0.010,rounding_size=0.014",
        linewidth=0.6,
        edgecolor=COLORS["panel_edge"],
        facecolor=COLORS["panel"],
    )
    ax.add_patch(band)
    ax.text(0.045, 0.515, "Permissioned repair tuple", ha="left", va="top", fontsize=7.4, fontweight="bold", color=COLORS["ink"])
    ax.text(
        0.045,
        0.476,
        "<x, C, G, F, B, R> separates target evidence, candidates, gate, fallback, DBMS boundary, and replay record.",
        ha="left",
        va="top",
        fontsize=6.6,
        color=COLORS["muted"],
    )
    chip(ax, 0.055, 0.394, 0.135, "x: drift signature", COLORS["gray_edge"])
    chip(ax, 0.210, 0.394, 0.135, "C: repair candidates", COLORS["violet_edge"])
    chip(ax, 0.365, 0.394, 0.135, "G: held-out gate", COLORS["amber_edge"])
    chip(ax, 0.520, 0.394, 0.135, "F: fallback", COLORS["red_edge"])
    chip(ax, 0.675, 0.394, 0.135, "B: DBMS probe", COLORS["blue_edge"])
    chip(ax, 0.830, 0.394, 0.105, "R: replay", COLORS["green_edge"])

    arrow(ax, 0.477, 0.678, 0.460, 0.540, color="violet_edge", rad=0.12)
    arrow(ax, 0.660, 0.675, 0.610, 0.540, color="amber_edge", rad=-0.10)

    y0 = 0.125
    h = 0.160
    rounded_box(
        ax,
        0.035,
        y0,
        0.205,
        h,
        "Estimator evidence",
        "Column P95/P99: -90.5/-95.1%\nTable P95/P99: -99.7/-98.8%\nValue drift: routed",
        "white",
        COLORS["green_edge"],
        title_size=7.2,
        body_size=6.35,
    )
    rounded_box(
        ax,
        0.275,
        y0,
        0.205,
        h,
        "Executable labels",
        "DuckDB COUNT(*) audits\n13050 raw-tail cases\nbitmap reconstruction checked",
        "white",
        COLORS["blue_edge"],
        title_size=7.2,
        body_size=6.35,
    )
    rounded_box(
        ax,
        0.515,
        y0,
        0.205,
        h,
        "Optimizer boundary",
        "PG13 estimate-map callback\n105123 estimates / 4528 groups\nheld-out subset: 0.79x median",
        "white",
        COLORS["amber_edge"],
        title_size=7.2,
        body_size=6.35,
    )
    rounded_box(
        ax,
        0.755,
        y0,
        0.205,
        h,
        "External support",
        "SSCard shipped-result audit\n34 passing budget rows\nhook replay checked",
        "white",
        COLORS["gray_edge"],
        title_size=7.2,
        body_size=6.35,
    )

    arrow(ax, 0.122, 0.394, 0.138, 0.286, color="line", lw=0.72)
    arrow(ax, 0.432, 0.394, 0.378, 0.286, color="line", lw=0.72)
    arrow(ax, 0.742, 0.394, 0.618, 0.286, color="line", lw=0.72)
    arrow(ax, 0.883, 0.394, 0.858, 0.286, color="line", lw=0.72)

    ax.text(
        0.500,
        0.050,
        "Core message: repair is allowed only after held-out tail evidence, and optimizer entry remains audited rather than assumed.",
        ha="center",
        va="bottom",
        fontsize=6.6,
        color=COLORS["ink"],
        bbox={
            "facecolor": "#F9FAFB",
            "edgecolor": COLORS["panel_edge"],
            "boxstyle": "round,pad=0.22,rounding_size=0.02",
            "linewidth": 0.5,
        },
    )

    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    fig_dir = result_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / output_stem
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=600)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("paper/workload_drift_residual_field/results"),
        help="Directory containing the figures/ output directory.",
    )
    parser.add_argument("--output-stem", default="paper_framework")
    args = parser.parse_args()
    build_framework(args.result_dir, args.output_stem)


if __name__ == "__main__":
    main()
