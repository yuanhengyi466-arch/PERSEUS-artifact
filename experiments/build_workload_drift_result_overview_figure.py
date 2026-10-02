#!/usr/bin/env python
"""Build the manuscript result-overview figure."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter


DRIFTS = ["column", "table", "value"]
DRIFT_LABELS = {"column": "Column", "table": "Table", "value": "Value"}

EXPANDED_METHODS = ["isotonic", "ridge-residual", "random-forest", "residual-field", "gated-residual-field"]
ABLATION_VARIANTS = [
    "prediction-only",
    "structure-only",
    "no-prediction-scale",
    "no-table-column",
    "no-clipping",
    "no-value",
]

METHOD_LABELS = {
    "signature-tail": "Signature",
    "residual-field": "Residual",
    "gated-residual-field": "Gated",
    "isotonic": "Isotonic",
    "ridge-residual": "Ridge",
    "random-forest": "Random forest",
}

ABLATION_LABELS = {
    "prediction-only": "Prediction only",
    "structure-only": "Structure only",
    "no-prediction-scale": "No prediction scale",
    "no-table-column": "No table/column",
    "no-clipping": "No clipping",
    "no-value": "No value bins",
}

COLORS = {
    "ink": "#1F2933",
    "muted": "#6B7280",
    "grid": "#E5EAF1",
    "axis": "#9AA4B2",
    "panel": "#F8FAFC",
    "baseline": "#C9D2DD",
    "column": "#3E6F9E",
    "table": "#2F806D",
    "value": "#8D6AA8",
    "residual": "#2F806D",
    "pass": "#237A66",
    "fail": "#B85D67",
}

LABEL_BOX = {"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 0.7}

METHOD_COLORS = {
    "isotonic": "#9CA7B3",
    "ridge-residual": "#73859B",
    "random-forest": "#4F81AD",
    "residual-field": "#2C7A68",
    "gated-residual-field": "#155B50",
}

REPAIR_MAP = LinearSegmentedColormap.from_list(
    "repair_map",
    ["#C96972", "#F1E9E2", "#C6DEB7", "#2F806D"],
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def value(row: dict[str, str], key: str) -> float:
    return float(row[key])


def rows_by(rows: list[dict[str, str]], *keys: str) -> dict[tuple[str, ...], dict[str, str]]:
    return {tuple(row[key] for key in keys): row for row in rows}


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L", "DejaVu Serif", "serif"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "font.size": 7.4,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.62,
            "axes.labelcolor": COLORS["ink"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "legend.frameon": False,
        }
    )


def clean_axes(ax: mpl.axes.Axes, grid: str | None = None) -> None:
    ax.set_facecolor(COLORS["panel"])
    ax.spines["left"].set_color(COLORS["ink"])
    ax.spines["bottom"].set_color(COLORS["ink"])
    ax.tick_params(axis="both", labelsize=7.0, length=2.3, width=0.52, pad=2.0)
    if grid:
        ax.grid(axis=grid, color=COLORS["grid"], linewidth=0.50, alpha=0.9)
        ax.set_axisbelow(True)


def compact_count_tick(x: float, _pos: int | None = None) -> str:
    if x >= 1_000_000:
        return f"{x / 1_000_000:g}M"
    if x >= 1000:
        return f"{x / 1000:g}k"
    return f"{x:g}"


def panel(ax: mpl.axes.Axes, label: str, title: str) -> None:
    ax.text(
        -0.08,
        1.075,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.2,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        0.5,
        -0.265,
        title,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.4,
        fontweight="bold",
        color=COLORS["ink"],
    )


def compact_percent(x: float) -> str:
    return f"{x:.0f}" if abs(x) >= 10 else f"{x:.1f}"


def luminance(rgba: tuple[float, float, float, float]) -> float:
    r, g, b, _a = rgba
    return 0.299 * r + 0.587 * g + 0.114 * b


def plot_tail_repair(ax: mpl.axes.Axes, default_rows: list[dict[str, str]]) -> None:
    data = rows_by(default_rows, "drift", "method")
    y_positions = np.arange(len(DRIFTS))[::-1]
    xmax = 0.0
    xmin = float("inf")
    label_offsets = {
        "column": (12, -18, "top"),
        "table": (11, -12, "top"),
        "value": (11, 10, "bottom"),
    }

    for yi, drift in zip(y_positions, DRIFTS):
        row = data[(drift, "residual-field")]
        before = value(row, "base_p99")
        after = value(row, "method_p99")
        reduction = 100 * value(row, "p99_reduction")
        passed = row["passed"].lower() == "true"
        color = COLORS["pass"] if passed else COLORS["fail"]
        xmax = max(xmax, before, after)
        xmin = min(xmin, before, after)

        ax.plot([before, after], [yi, yi], color="#DCE2EA", linewidth=5.6, solid_capstyle="round", zorder=1)
        ax.scatter(before, yi, s=48, marker="s", color=COLORS["baseline"], edgecolor="white", linewidth=0.65, zorder=3)
        ax.scatter(after, yi, s=52, marker="o", color=color, edgecolor="white", linewidth=0.75, zorder=4)

        label = f"{abs(reduction):.0f}% lower" if reduction >= 0 else f"{abs(reduction):.0f}% worse"
        status = "pass" if passed else "reject"
        dx, dy, va = label_offsets[drift]
        ax.annotate(
            f"{label} | {status}",
            xy=(after, yi),
            xytext=(dx, dy),
            textcoords="offset points",
            va=va,
            fontsize=6.5,
            color=color,
            bbox=LABEL_BOX,
            clip_on=False,
            zorder=6,
        )

    ax.set_xscale("log")
    ax.set_xlim(max(30, xmin / 2.5), xmax * 1.5)
    ax.xaxis.set_major_locator(FixedLocator([100, 1000, 10000, 100000, 1000000]))
    ax.xaxis.set_major_formatter(FuncFormatter(compact_count_tick))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks(y_positions, [DRIFT_LABELS[d] for d in DRIFTS])
    ax.set_xlabel("P99 q-error, log scale", fontsize=7.1)
    clean_axes(ax, grid="x")
    panel(ax, "a", "Residual field sharply repairs structural tails")


def plot_comparator_matrix(ax: mpl.axes.Axes, expanded_rows: list[dict[str, str]]) -> None:
    data = rows_by(expanded_rows, "method", "drift")
    norm = Normalize(vmin=-30, vmax=100)
    y_positions = np.arange(len(EXPANDED_METHODS))[::-1]

    for y, method in zip(y_positions, EXPANDED_METHODS):
        for x, drift in enumerate(DRIFTS):
            row = data[(method, drift)]
            reduction = 100 * value(row, "p99_reduction")
            passed = row["passed"].lower() == "true"
            face = REPAIR_MAP(norm(reduction))
            edge = COLORS["ink"] if passed else "#FFFFFF"
            linewidth = 0.9 if passed else 0.35
            cell = FancyBboxPatch(
                (x - 0.38, y - 0.33),
                0.76,
                0.66,
                boxstyle="round,pad=0.015,rounding_size=0.075",
                facecolor=face,
                edgecolor=edge,
                linewidth=linewidth,
                zorder=2,
            )
            ax.add_patch(cell)
            text_color = "white" if luminance(face) < 0.55 else COLORS["ink"]
            ax.text(x, y, compact_percent(reduction), ha="center", va="center", fontsize=7.1, color=text_color, zorder=3)

    ax.set_xlim(-0.55, len(DRIFTS) - 0.45)
    ax.set_ylim(-0.6, len(EXPANDED_METHODS) - 0.4)
    ax.set_xticks(range(len(DRIFTS)), [DRIFT_LABELS[d] for d in DRIFTS])
    ax.set_yticks(y_positions, [METHOD_LABELS[m] for m in EXPANDED_METHODS])
    ax.set_facecolor(COLORS["panel"])
    ax.tick_params(axis="both", labelsize=7.1, length=0, pad=2.2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for x in np.arange(0.5, len(DRIFTS), 1):
        ax.axvline(x, color=COLORS["grid"], linewidth=0.55, zorder=0)
    for y in np.arange(0.5, len(EXPANDED_METHODS), 1):
        ax.axhline(y, color=COLORS["grid"], linewidth=0.55, zorder=0)
    panel(ax, "b", "Expanded post-hoc comparators (P99 reduction)")
    ax.text(0.98, 1.035, "outline = gate pass", transform=ax.transAxes, fontsize=6.2, color=COLORS["muted"], ha="right", va="bottom")


def plot_budget(ax: mpl.axes.Axes, budget_rows: list[dict[str, str]]) -> None:
    rows = [row for row in budget_rows if row["method"] == "residual-field"]
    colors = {"column": COLORS["column"], "table": COLORS["table"], "value": COLORS["value"]}
    min_x = float("inf")
    max_x = 0

    for drift in DRIFTS:
        drift_rows = sorted([row for row in rows if row["drift"] == drift], key=lambda row: int(row["adaptation_rows"]))
        xs = [int(row["adaptation_rows"]) for row in drift_rows]
        p95 = [100 * value(row, "p95_reduction") for row in drift_rows]
        p99 = [100 * value(row, "p99_reduction") for row in drift_rows]
        min_x = min(min_x, min(xs))
        max_x = max(max_x, max(xs))
        ax.plot(xs, p95, color=colors[drift], linewidth=1.55, marker="o", markersize=3.1, zorder=3)
        ax.plot(xs, p99, color=colors[drift], linewidth=1.05, linestyle=(0, (3.2, 2.2)), marker="s", markersize=2.6, alpha=0.74, zorder=2)

        label_y = p95[-1]
        offset = {"column": 11, "table": -12, "value": -13}[drift]
        va = "bottom" if offset > 0 else "top"
        ax.annotate(
            DRIFT_LABELS[drift],
            xy=(xs[-1], label_y),
            xytext=(11, offset),
            textcoords="offset points",
            fontsize=6.6,
            color=colors[drift],
            ha="left",
            va=va,
            bbox=LABEL_BOX,
            clip_on=False,
            zorder=5,
        )

    ax.axhline(0, color=COLORS["axis"], linewidth=0.75)
    ax.set_xscale("log")
    ax.set_xlim(max(1, min_x / 1.25), max_x * 1.55)
    ax.xaxis.set_major_locator(FixedLocator([100, 250, 500, 1000, 2000]))
    ax.xaxis.set_major_formatter(FuncFormatter(compact_count_tick))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(-62, 108)
    ax.set_xlabel("Labeled target rows", fontsize=7.1)
    ax.set_ylabel("Reduction (%)", fontsize=7.1)
    clean_axes(ax, grid="y")
    panel(ax, "c", "Budget sensitivity")
    ax.legend(
        handles=[
            Line2D([0], [0], color=COLORS["ink"], linewidth=1.15, label="P95"),
            Line2D([0], [0], color=COLORS["ink"], linewidth=1.05, linestyle=(0, (3.2, 2.2)), label="P99"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.14, 1.08),
        ncols=2,
        fontsize=6.2,
        handlelength=1.65,
        columnspacing=0.85,
        borderaxespad=0.0,
    )


def plot_ablation_loss(ax: mpl.axes.Axes, ablation_rows: list[dict[str, str]]) -> None:
    data = rows_by(ablation_rows, "variant", "drift")
    full = {drift: 100 * value(data[("full", drift)], "p99_reduction") for drift in ["column", "table"]}
    y = np.arange(len(ABLATION_VARIANTS))[::-1]
    height = 0.28

    for idx, drift in enumerate(["column", "table"]):
        losses = [full[drift] - 100 * value(data[(variant, drift)], "p99_reduction") for variant in ABLATION_VARIANTS]
        offset = height / 1.75 if drift == "column" else -height / 1.75
        ax.barh(
            y + offset,
            losses,
            height=height,
            color=COLORS[drift],
            edgecolor="white",
            linewidth=0.45,
            label=DRIFT_LABELS[drift],
            alpha=0.94,
        )

    ax.axvline(0, color=COLORS["axis"], linewidth=0.75)
    ax.set_yticks(y, [ABLATION_LABELS[v] for v in ABLATION_VARIANTS])
    ax.set_ylim(-0.55, len(ABLATION_VARIANTS) - 0.05)
    ax.set_xlim(-4, 44)
    ax.set_xlabel("P99 reduction loss vs full model (pp)", fontsize=7.1)
    clean_axes(ax, grid="x")
    panel(ax, "d", "Ablation stress test on structural drift")
    ax.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, 0.03),
        ncols=2,
        fontsize=6.6,
        handlelength=1.0,
        columnspacing=1.0,
        borderaxespad=0.0,
    )


def build_overview(result_dir: Path, output_stem: str) -> None:
    set_style()
    default_rows = read_rows(result_dir / "default_method_comparison.csv")
    expanded_rows = read_rows(result_dir / "expanded_method_comparison.csv")
    budget_rows = read_rows(result_dir / "budget_sweep.csv")
    ablation_rows = read_rows(result_dir / "feature_ablation.csv")

    fig = plt.figure(figsize=(7.25, 4.62), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.04], width_ratios=[1.08, 1.0], hspace=0.70, wspace=0.40)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    plot_tail_repair(ax_a, default_rows)
    plot_comparator_matrix(ax_b, expanded_rows)
    plot_budget(ax_c, budget_rows)
    plot_ablation_loss(ax_d, ablation_rows)

    fig.subplots_adjust(left=0.085, right=0.985, top=0.915, bottom=0.145)

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
        help="Directory containing result CSVs and the figures/ output directory.",
    )
    parser.add_argument("--output-stem", default="result_overview")
    args = parser.parse_args()
    build_overview(args.result_dir, args.output_stem)


if __name__ == "__main__":
    main()
