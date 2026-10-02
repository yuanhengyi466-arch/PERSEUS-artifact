#!/usr/bin/env python
"""Build a compact manuscript figure for external and optimizer-boundary evidence."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


DRIFTS = ["column", "table", "value"]
DRIFT_LABELS = {"column": "Column", "table": "Table", "value": "Value"}

COLORS = {
    "ink": "#1F2933",
    "muted": "#6B7280",
    "grid": "#E6E9EF",
    "axis": "#9AA3AF",
    "fill": "#E8EDF2",
    "green": "#2C7A68",
    "green_light": "#9EC8B7",
    "amber": "#B07D3F",
    "red": "#B95C68",
    "red_light": "#E4A5AD",
    "blue": "#436C9B",
    "blue_light": "#B7C9DF",
    "panel": "#F7F9FB",
}

LABEL_BOX = {"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 0.6}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "axes.labelcolor": COLORS["ink"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "legend.frameon": False,
        }
    )


def panel(ax: mpl.axes.Axes, label: str, title: str) -> None:
    ax.set_facecolor(COLORS["panel"])
    ax.text(
        -0.095,
        1.10,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.3,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        0.5,
        -0.30,
        title,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.7,
        fontweight="bold",
        color=COLORS["ink"],
    )


def clean_axes(ax: mpl.axes.Axes, grid: str | None = None) -> None:
    ax.spines["left"].set_color(COLORS["ink"])
    ax.spines["bottom"].set_color(COLORS["ink"])
    ax.tick_params(axis="both", labelsize=7.0, length=2.2, width=0.55, pad=1.8)
    if grid:
        ax.grid(axis=grid, color=COLORS["grid"], linewidth=0.55, alpha=0.9)
        ax.set_axisbelow(True)


def parse_case_count(text: str) -> tuple[int, int]:
    passed, total = text.split("/")
    return int(passed), int(total)


def parse_ratio(text: str) -> float | None:
    value = str(text or "").strip().replace("\\times", "").replace("$", "").replace("x", "")
    if value in {"", "--"}:
        return None
    return float(value)


def plot_stress_inset(ax: mpl.axes.Axes, rows: list[dict[str, str]]) -> None:
    stress = next((row for row in rows if row.get("scope") == "Column PG stress"), None)
    if not stress:
        return
    cost = parse_ratio(stress.get("median_ratio"))
    best = parse_ratio(stress.get("best_median_ratio"))
    if cost is None or best is None:
        return

    callout = (
        f"Stress Q={int(stress['query_groups'])}\n"
        f"cost {cost:.2f}x  best {best:.2f}x\n"
        f"{stress.get('plan_change', '--')} plan changes"
    )
    ax.text(
        0.965,
        0.925,
        callout,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.3,
        linespacing=1.18,
        color=COLORS["ink"],
        bbox={
            "facecolor": "white",
            "edgecolor": COLORS["axis"],
            "linewidth": 0.45,
            "alpha": 0.93,
            "boxstyle": "round,pad=0.22,rounding_size=0.02",
        },
        zorder=8,
    )


def plot_external_check(ax: mpl.axes.Axes, rows: list[dict[str, str]]) -> None:
    labels = [row["dataset"].replace("_", "-") for row in rows]
    passed = []
    total = []
    p99_ranges = []
    for row in rows:
        p, t = parse_case_count(row["passing_cases"])
        passed.append(p)
        total.append(t)
        p99_ranges.append(row["p99_range"].replace("\\%", "%"))

    y = np.arange(len(rows))[::-1]
    rates = [p / t for p, t in zip(passed, total)]
    ax.barh(y, [1.0] * len(rows), color=COLORS["fill"], height=0.48, edgecolor="none", zorder=1)
    ax.barh(y, rates, color=COLORS["green"], height=0.48, edgecolor="white", linewidth=0.6, zorder=2)

    for yi, p, t, rate, p99 in zip(y, passed, total, rates, p99_ranges):
        ax.text(rate + 0.02, yi + 0.12, f"{p}/{t}", ha="left", va="center", fontsize=7.0, color=COLORS["ink"])
        ax.text(rate + 0.02, yi - 0.14, f"P99 red. {p99}", ha="left", va="center", fontsize=6.3, color=COLORS["muted"])

    ax.set_xlim(0, 1.08)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Passing estimator/repair settings", fontsize=7.2)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0], ["0", "25%", "50%", "75%", "100%"])
    clean_axes(ax, grid="x")
    panel(ax, "a", "External estimator/data check")


def plot_optimizer_probe(
    ax: mpl.axes.Axes,
    deploy_rows: list[dict[str, str]],
    exact_rows: list[dict[str, str]],
) -> None:
    by_key = {(row["drift"], row["method"]): row for row in deploy_rows}
    exact_by_key = {(row["drift"], row["method"]): row for row in exact_rows}
    x = np.arange(len(DRIFTS))
    width = 0.34
    cost = [float(by_key[(drift, "cost")]["median_ratio_vs_default"]) for drift in DRIFTS]
    oracle = [float(exact_by_key[(drift, "oracle-best")]["median_ratio_vs_default"]) for drift in DRIFTS]

    ax.axhline(1.0, color=COLORS["axis"], linewidth=0.75, linestyle=(0, (3, 2)), zorder=1)
    ax.bar(x - width / 2, cost, width=width, color=COLORS["green"], edgecolor="white", linewidth=0.6, label="cost selected", zorder=2)
    ax.bar(x + width / 2, oracle, width=width, color=COLORS["amber"], edgecolor="white", linewidth=0.6, label="oracle upper bound", zorder=2)

    for xi, c, o in zip(x, cost, oracle):
        ax.text(
            xi - width / 2,
            c + 0.055,
            f"{c:.2f}x" if c != 1.0 else "1.00x",
            ha="center",
            va="bottom",
            fontsize=6.7,
            color=COLORS["ink"],
            bbox=LABEL_BOX,
            clip_on=False,
            zorder=5,
        )
        ax.text(
            xi + width / 2,
            o + 0.055,
            f"{o:.2f}x",
            ha="center",
            va="bottom",
            fontsize=6.7,
            color=COLORS["ink"],
            bbox=LABEL_BOX,
            clip_on=False,
            zorder=5,
        )

    ax.set_ylim(0, 1.26)
    ax.set_xticks(x, [DRIFT_LABELS[d] for d in DRIFTS])
    ax.set_ylabel("Median runtime / default", fontsize=7.2)
    clean_axes(ax, grid="y")
    panel(ax, "b", "PostgreSQL multi-subplan probe")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.46), ncols=2, fontsize=6.4, handlelength=1.0, columnspacing=1.25)


def plot_optimizer_queries(ax: mpl.axes.Axes, rows: list[dict[str, str]], stress_rows: list[dict[str, str]]) -> None:
    cost_rows = [row for row in rows if row["selector"] == "cost"]
    rng = np.random.default_rng(7)
    ax.axhline(1.0, color=COLORS["axis"], linewidth=0.75, linestyle=(0, (3, 2)), zorder=1)
    tick_labels = []
    for xi, drift in enumerate(DRIFTS):
        values = np.array([float(row["selected_ratio_vs_default"]) for row in cost_rows if row["drift"] == drift])
        sample = values
        if len(sample) > 520:
            sample = rng.choice(sample, size=520, replace=False)
        jitter = rng.uniform(-0.11, 0.11, size=len(sample))
        color = COLORS["green"] if drift in {"column", "table"} else COLORS["red"]
        light = COLORS["green_light"] if drift in {"column", "table"} else COLORS["red_light"]
        if len(values):
            q1, median, q3 = np.percentile(values, [25, 50, 75])
            lo, hi = np.percentile(values, [5, 95])
            ax.vlines(xi, lo, hi, color=light, linewidth=4.5, alpha=0.55, zorder=2)
            ax.add_patch(
                mpl.patches.Rectangle(
                    (xi - 0.15, q1),
                    0.30,
                    q3 - q1,
                    facecolor=light,
                    edgecolor=color,
                    linewidth=0.75,
                    alpha=0.48,
                    zorder=3,
                )
            )
        ax.scatter(
            xi + jitter,
            sample,
            s=9,
            color=color,
            edgecolor="none",
            linewidth=0.0,
            alpha=0.32,
            zorder=4,
        )
        if len(values):
            median = float(np.median(values))
            ax.plot([xi - 0.19, xi + 0.19], [median, median], color=COLORS["ink"], linewidth=1.15, zorder=5)
            tick_labels.append(f"{DRIFT_LABELS[drift]}\nmed. {median:.2f}x")
        else:
            tick_labels.append(DRIFT_LABELS[drift])
    ax.set_xlim(-0.45, len(DRIFTS) - 0.55)
    ax.set_ylim(0.0, 2.8)
    ax.set_xticks(range(len(DRIFTS)), tick_labels)
    ax.set_ylabel("Runtime / default", fontsize=7.2)
    clean_axes(ax, grid="y")
    plot_stress_inset(ax, stress_rows)
    panel(ax, "c", "Per-query cost-selected ratios")


def build_boundary_evidence(
    result_dir: Path,
    output_stem: str,
    deploy_summary: Path | None = None,
    deploy_selection: Path | None = None,
    exact_summary: Path | None = None,
) -> None:
    set_style()
    sscard_rows = read_rows(result_dir / "sscard_external_generalization.csv")
    exact_rows = read_rows(exact_summary or (result_dir / "postgres_multisubplan_injection_summary.csv"))
    deploy_rows = read_rows(deploy_summary or (result_dir / "postgres_multisubplan_injection_summary.csv"))
    selection_rows = read_rows(deploy_selection or (result_dir / "postgres_multisubplan_injection_selection.csv"))
    stress_rows = read_rows(result_dir / "optimizer_tail_stress.csv")

    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.55), constrained_layout=False)
    plot_external_check(axes[0], sscard_rows)
    plot_optimizer_probe(axes[1], deploy_rows, exact_rows)
    plot_optimizer_queries(axes[2], selection_rows, stress_rows)
    cost_counts = {
        drift: sum(
            1
            for row in selection_rows
            if row.get("selector") == "cost" and row.get("drift") == drift
        )
        for drift in DRIFTS
    }
    count_text = ", ".join(f"{DRIFT_LABELS[d]} {cost_counts[d]}" for d in DRIFTS)
    stress = next((row for row in stress_rows if row.get("scope") == "Column PG stress"), None)
    stress_text = f"; stress Q={stress['query_groups']}" if stress else ""
    fig.text(
        0.06,
        0.030,
        f"Panels are generated from local CSV summaries; lower runtime ratios are better; oracle bars are upper bounds; panel c executable queries: {count_text}{stress_text}.",
        fontsize=6.0,
        color=COLORS["muted"],
    )
    fig.subplots_adjust(left=0.07, right=0.985, top=0.86, bottom=0.36, wspace=0.35)

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
        help="Directory containing generated CSV summaries and the figures/ output directory.",
    )
    parser.add_argument("--output-stem", default="boundary_evidence")
    parser.add_argument("--deploy-summary", type=Path, default=None)
    parser.add_argument("--deploy-selection", type=Path, default=None)
    parser.add_argument("--exact-summary", type=Path, default=None)
    args = parser.parse_args()
    build_boundary_evidence(
        args.result_dir,
        args.output_stem,
        deploy_summary=args.deploy_summary,
        deploy_selection=args.deploy_selection,
        exact_summary=args.exact_summary,
    )


if __name__ == "__main__":
    main()
