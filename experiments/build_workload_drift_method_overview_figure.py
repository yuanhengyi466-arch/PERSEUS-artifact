#!/usr/bin/env python
"""Build the manuscript method-overview figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


COLORS = {
    "ink": "#1F2933",
    "muted": "#667085",
    "line": "#BFC9D4",
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
    "band": "#F7F9FB",
    "band_edge": "#E1E7EE",
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
            "font.size": 7.2,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def box(
    ax: mpl.axes.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str,
    face: str,
    edge: str,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=0.95,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x + 0.018, y + h - 0.054, title, ha="left", va="top", fontsize=7.0, fontweight="bold", color=COLORS["ink"])
    ax.text(x + 0.018, y + h - 0.108, body, ha="left", va="top", fontsize=5.7, color=COLORS["muted"], linespacing=1.22)


def arrow(ax: mpl.axes.Axes, x0: float, y0: float, x1: float, y1: float) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=9.0,
            linewidth=0.9,
            color=COLORS["line"],
            shrinkA=1.0,
            shrinkB=1.0,
        )
    )


def small_chip(ax: mpl.axes.Axes, x: float, y: float, w: float, text: str, edge: str) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        0.082,
        boxstyle="round,pad=0.009,rounding_size=0.018",
        linewidth=0.8,
        edgecolor=edge,
        facecolor="white",
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + 0.041, text, ha="center", va="center", fontsize=5.7, color=COLORS["ink"])


def build_method_overview(result_dir: Path, output_stem: str) -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(7.25, 2.55), constrained_layout=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.015,
        0.965,
        "a",
        ha="left",
        va="top",
        fontsize=8.3,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        0.042,
        0.965,
        "Validation-gated residual transport adapts a frozen cardinality estimator without changing its architecture.",
        ha="left",
        va="top",
        fontsize=7.5,
        fontweight="bold",
        color=COLORS["ink"],
    )

    xs = [0.025, 0.215, 0.405, 0.595, 0.785]
    w = 0.16
    y = 0.54
    h = 0.285
    box(
        ax,
        xs[0],
        y,
        w,
        h,
        "Frozen estimator",
        "MSCN checkpoint\nSTATS encodings\nbaseline prediction",
        COLORS["blue"],
        COLORS["blue_edge"],
    )
    box(
        ax,
        xs[1],
        y,
        w,
        h,
        "Target evidence",
        "labeled subset A\nquery signatures\nlog residuals",
        COLORS["green"],
        COLORS["green_edge"],
    )
    box(
        ax,
        xs[2],
        y,
        w,
        h,
        "Residual transport",
        "signature-tail shift\nresidual field\nclipped targets",
        COLORS["violet"],
        COLORS["violet_edge"],
    )
    box(
        ax,
        xs[3],
        y,
        w,
        h,
        "Validation gate",
        "P90/P95/P99 gains\nsplit stability\nbaseline comparator",
        COLORS["amber"],
        COLORS["amber_edge"],
    )
    box(
        ax,
        xs[4],
        y,
        w,
        h,
        "Action",
        "repair structural drift\nreject value drift\nfallback policy",
        COLORS["red"],
        COLORS["red_edge"],
    )

    for i in range(4):
        arrow(ax, xs[i] + w + 0.012, y + h / 2, xs[i + 1] - 0.012, y + h / 2)

    band = FancyBboxPatch(
        (0.02, 0.14),
        0.955,
        0.245,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        linewidth=0.55,
        edgecolor=COLORS["band_edge"],
        facecolor=COLORS["band"],
        zorder=-1,
    )
    ax.add_patch(band)
    ax.text(0.04, 0.35, "Audit trail and system boundary", ha="left", va="top", fontsize=6.6, fontweight="bold", color=COLORS["ink"])
    ax.text(
        0.04,
        0.31,
        "Estimator repair is evaluated separately from optimizer claims, then audited through executable label checks and planner probes.",
        ha="left",
        va="top",
        fontsize=5.65,
        color=COLORS["muted"],
    )

    chip_y = 0.174
    small_chip(ax, 0.055, chip_y, 0.18, "bitmap reproduction", COLORS["blue_edge"])
    small_chip(ax, 0.285, chip_y, 0.18, "comparators + ablations", COLORS["violet_edge"])
    small_chip(ax, 0.515, chip_y, 0.18, "DuckDB label checks", COLORS["green_edge"])
    small_chip(ax, 0.745, chip_y, 0.18, "PostgreSQL probes", COLORS["amber_edge"])

    arrow(ax, xs[3] + w / 2, y - 0.012, 0.5, 0.39)
    ax.text(
        0.985,
        0.05,
        "All adaptation decisions are reproducible from local CSV artifacts.",
        ha="right",
        va="bottom",
        fontsize=5.5,
        color=COLORS["muted"],
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
    parser.add_argument("--output-stem", default="method_overview")
    args = parser.parse_args()
    build_method_overview(args.result_dir, args.output_stem)


if __name__ == "__main__":
    main()
