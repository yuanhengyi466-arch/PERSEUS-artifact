import argparse
import csv
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research_plan"


DRIFTS = {
    "column": {
        "metrics": RESEARCH / "workload_drift_mscn_windows_bitmap" / "seed_1" / "metrics.json",
        "no_bitmap_metrics": RESEARCH / "workload_drift_mscn_windows" / "seed_1" / "metrics.json",
        "default_budget": 2000,
        "min_eval_rows": 5000,
        "budgets": [100, 250, 500, 1000, 2000],
    },
    "table": {
        "metrics": RESEARCH / "workload_drift_mscn_windows_table_bitmap" / "seed_1" / "metrics.json",
        "no_bitmap_metrics": RESEARCH / "workload_drift_mscn_windows_table" / "seed_1" / "metrics.json",
        "default_budget": 2000,
        "min_eval_rows": 5000,
        "budgets": [250, 500, 1000, 2000],
    },
    "value": {
        "metrics": RESEARCH / "workload_drift_mscn_windows_value_bitmap" / "seed_1" / "metrics.json",
        "no_bitmap_metrics": RESEARCH / "workload_drift_mscn_windows_value" / "seed_1" / "metrics.json",
        "default_budget": 500,
        "min_eval_rows": 1000,
        "budgets": [100, 250, 500, 1000],
    },
}


METHOD_LABELS = {
    "signature-tail": "Signature-tail",
    "residual-field": "Residual-field",
    "gated-residual-field": "Gated residual-field",
    "isotonic": "Isotonic",
    "ridge-residual": "Ridge residual",
    "random-forest": "Random forest",
}

DEFAULT_METHODS = [
    ("signature-tail", "full"),
    ("isotonic", "prediction-only"),
    ("ridge-residual", "full"),
    ("random-forest", "full"),
    ("residual-field", "full"),
    ("gated-residual-field", "full"),
]

ABLATION_PRESETS = [
    "full",
    "prediction-only",
    "structure-only",
    "no-value",
    "no-prediction-scale",
    "no-table-column",
]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def run_gate(
    drift: str,
    method: str,
    budget: int,
    output: Path,
    force: bool,
    feature_preset: str = "full",
    extra_args: list[str] | None = None,
) -> dict:
    if output.exists() and not force:
        return read_json(output)
    spec = DRIFTS[drift]
    cmd = [
        sys.executable,
        "experiments/run_workload_drift_tail_residual_transport.py",
        "--method",
        method,
        "--baseline-metrics",
        str(spec["metrics"].relative_to(ROOT)),
        "--output-json",
        str(output.relative_to(ROOT)),
        "--adaptation-rows",
        str(budget),
        "--feature-preset",
        feature_preset,
        "--seeds",
        "1,2,3",
        "--min-eval-rows",
        str(spec["min_eval_rows"]),
    ]
    if extra_args:
        cmd.extend(extra_args)
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
    if not output.exists():
        raise RuntimeError(
            f"gate did not write {output} (exit={proc.returncode})\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    data = read_json(output)
    data["runner_returncode"] = proc.returncode
    data["runner_stdout_tail"] = proc.stdout[-4000:]
    data["runner_stderr_tail"] = proc.stderr[-4000:]
    write_json(output, data)
    return data


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}"


def pass_cell(value: bool) -> str:
    return "\\passcell{yes}" if value else "no"


def green_cell(text: str, enabled: bool = True) -> str:
    return f"\\passcell{{{text}}}" if enabled else text


def best_cell(text: str, is_best: bool) -> str:
    return f"\\bestcell{{{text}}}" if is_best else text


def reduction_cell(text: str, value: float, best_value=None) -> str:
    if value <= 0:
        return text
    if best_value is not None and best_value > 0 and value == best_value:
        return best_cell(text, True)
    return green_cell(text)


TABLE_DARK = "#2b2b2b"
GRID_COLOR = "#E7ECEF"
PALETTE = {
    "mscn": "#BFC7CE",
    "signature-tail": "#9FC7DF",
    "residual-field": "#A9D18E",
    "residual-field-edge": "#5B8F54",
    "gated-residual-field": "#C7DAF0",
    "isotonic": "#C9C9C9",
    "ridge-residual": "#9FC7DF",
    "random-forest": "#B7D7A8",
    "negative": "#D8A0A6",
    "neutral": "#D9D9D9",
}


def apply_plot_style(ax) -> None:
    ax.spines["left"].set_color(TABLE_DARK)
    ax.spines["bottom"].set_color(TABLE_DARK)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(width=0.7, length=3.0, color=TABLE_DARK)


def median_summary(report: dict, key: str) -> dict:
    rows = report["split_reports"]
    return {
        metric: float(np.median([row[key][metric] for row in rows]))
        for metric in ["median", "p90", "p95", "p99", "mean"]
    }


def collect_results(output_dir: Path, force: bool) -> dict:
    sweep_dir = output_dir / "gate_sweep"
    reports = {}
    rows = []
    for drift, spec in DRIFTS.items():
        for method in ["signature-tail", "residual-field"]:
            for budget in spec["budgets"]:
                out = sweep_dir / f"{drift}_{method}_{budget}.json"
                report = run_gate(drift, method, budget, out, force)
                reports[(drift, method, budget)] = report
                rows.append(
                    {
                        "drift": drift,
                        "method": method,
                        "adaptation_rows": budget,
                        "passed": report["passed"],
                        "passed_seed_count": report["passed_seed_count"],
                        "seed_count": report["seed_count"],
                        "p90_reduction": report["aggregate_reductions"]["p90"],
                        "p95_reduction": report["aggregate_reductions"]["p95"],
                        "p99_reduction": report["aggregate_reductions"]["p99"],
                        "mean_reduction": report["aggregate_reductions"]["mean"],
                        "p95_vs_global": report["aggregate_reductions_vs_global_calibration"]["p95"],
                        "p99_vs_global": report["aggregate_reductions_vs_global_calibration"]["p99"],
                        "innovation_score": report["innovation_score"]["score"],
                        "decision": report["innovation_score"]["decision"],
                    }
                )
    return {"reports": reports, "rows": rows}


def collect_expanded_results(output_dir: Path, force: bool, jobs: int) -> dict:
    expanded_dir = output_dir / "expanded"
    tasks = []
    for drift, spec in DRIFTS.items():
        budget = spec["default_budget"]
        for method, feature_preset in DEFAULT_METHODS:
            name = f"{drift}_{method}_{feature_preset}_{budget}.json"
            tasks.append(
                {
                    "kind": "method",
                    "drift": drift,
                    "method": method,
                    "feature_preset": feature_preset,
                    "budget": budget,
                    "output": expanded_dir / "methods" / name,
                }
            )
        for feature_preset in ABLATION_PRESETS:
            name = f"{drift}_residual-field_{feature_preset}_{budget}.json"
            tasks.append(
                {
                    "kind": "ablation",
                    "drift": drift,
                    "method": "residual-field",
                    "feature_preset": feature_preset,
                    "budget": budget,
                    "output": expanded_dir / "ablations" / name,
                }
            )
        no_clip_name = f"{drift}_residual-field_no_clip_{budget}.json"
        tasks.append(
            {
                "kind": "ablation",
                "drift": drift,
                "method": "residual-field",
                "feature_preset": "full",
                "budget": budget,
                "output": expanded_dir / "ablations" / no_clip_name,
                "label": "no-clipping",
                "extra_args": ["--residual-clip-lower", "0", "--residual-clip-upper", "100"],
            }
        )

    results = []
    max_workers = max(1, jobs)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_map = {
            pool.submit(
                run_gate,
                task["drift"],
                task["method"],
                task["budget"],
                task["output"],
                force,
                task["feature_preset"],
                task.get("extra_args"),
            ): task
            for task in tasks
        }
        for future in as_completed(future_map):
            task = future_map[future]
            report = future.result()
            row = report_row(
                task["drift"],
                task["method"],
                task.get("label", task["feature_preset"]),
                task["budget"],
                report,
            )
            row["kind"] = task["kind"]
            row["output_json"] = str(task["output"])
            results.append(row)
    results.sort(key=lambda row: (row["kind"], row["drift"], row["method"], row["variant"]))
    method_rows = [row for row in results if row["kind"] == "method"]
    ablation_rows = [row for row in results if row["kind"] == "ablation"]
    return {"rows": results, "method_rows": method_rows, "ablation_rows": ablation_rows}


def report_row(drift: str, method: str, variant: str, budget: int, report: dict) -> dict:
    base = median_summary(report, "baseline_qerror")
    glob = median_summary(report, "global_calibration_qerror")
    meth = median_summary(report, "method_qerror")
    fallback_used = 0
    gate_passed = 0
    gate_rows = 0
    if report["split_reports"]:
        fallback_used = sum(1 for item in report["split_reports"] if item.get("fallback_used"))
        gate_passed = sum(1 for item in report["split_reports"] if item.get("gate_passed"))
        gate_rows = int(np.median([item.get("gate_rows", 0) for item in report["split_reports"]]))
    return {
        "drift": drift,
        "method": method,
        "variant": variant,
        "budget": budget,
        "passed": report["passed"],
        "passed_seed_count": report["passed_seed_count"],
        "seed_count": report["seed_count"],
        "fit_rows_median": int(np.median([item.get("fit_rows", budget) for item in report["split_reports"]])),
        "gate_rows_median": gate_rows,
        "gate_passed_seed_count": gate_passed,
        "fallback_seed_count": fallback_used,
        "base_p95": base["p95"],
        "global_p95": glob["p95"],
        "method_p95": meth["p95"],
        "base_p99": base["p99"],
        "global_p99": glob["p99"],
        "method_p99": meth["p99"],
        "p90_reduction": report["aggregate_reductions"]["p90"],
        "p95_reduction": report["aggregate_reductions"]["p95"],
        "p99_reduction": report["aggregate_reductions"]["p99"],
        "mean_reduction": report["aggregate_reductions"]["mean"],
        "p95_vs_global": report["aggregate_reductions_vs_global_calibration"]["p95"],
        "p99_vs_global": report["aggregate_reductions_vs_global_calibration"]["p99"],
        "score": report["innovation_score"]["score"],
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_tables(output_dir: Path, reports: dict, sweep_rows: list[dict]) -> None:
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    baseline_rows = []
    for drift, spec in DRIFTS.items():
        metrics = read_json(spec["metrics"])
        q = metrics["predict_qerror"]
        baseline_rows.append(
            {
                "drift": drift,
                "train_rows": metrics["train_rows"],
                "target_rows": metrics["predict_rows"],
                "sample_feats": metrics["feature_dims"]["sample_feats"],
                "median": q["median"],
                "p90": q["p90"],
                "p95": q["p95"],
                "p99": q["p99"],
                "mean": q["mean"],
            }
        )
    write_csv(output_dir / "baseline_summary.csv", baseline_rows)

    bitmap_rows = []
    for drift, spec in DRIFTS.items():
        no_bitmap = read_json(spec["no_bitmap_metrics"])
        bitmap = read_json(spec["metrics"])
        nb = no_bitmap["predict_qerror"]
        bm = bitmap["predict_qerror"]
        bitmap_rows.append(
            {
                "drift": drift,
                "no_bitmap_p95": nb["p95"],
                "bitmap_p95": bm["p95"],
                "p95_change": (nb["p95"] - bm["p95"]) / nb["p95"],
                "no_bitmap_p99": nb["p99"],
                "bitmap_p99": bm["p99"],
                "p99_change": (nb["p99"] - bm["p99"]) / nb["p99"],
                "no_bitmap_sample_feats": no_bitmap["feature_dims"]["sample_feats"],
                "bitmap_sample_feats": bitmap["feature_dims"]["sample_feats"],
            }
        )
    write_csv(output_dir / "bitmap_effect_summary.csv", bitmap_rows)

    default_rows = []
    for drift, spec in DRIFTS.items():
        for method in ["signature-tail", "residual-field"]:
            report = reports[(drift, method, spec["default_budget"])]
            base = median_summary(report, "baseline_qerror")
            glob = median_summary(report, "global_calibration_qerror")
            meth = median_summary(report, "method_qerror")
            default_rows.append(
                {
                    "drift": drift,
                    "method": method,
                    "budget": spec["default_budget"],
                    "passed": report["passed"],
                    "score": report["innovation_score"]["score"],
                    "base_p95": base["p95"],
                    "global_p95": glob["p95"],
                    "method_p95": meth["p95"],
                    "base_p99": base["p99"],
                    "global_p99": glob["p99"],
                    "method_p99": meth["p99"],
                    "p95_reduction": report["aggregate_reductions"]["p95"],
                    "p99_reduction": report["aggregate_reductions"]["p99"],
                }
            )
    write_csv(output_dir / "default_method_comparison.csv", default_rows)

    baseline_tex = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Strong bitmap MSCN reproduction on STATS workload drift.}",
        "\\label{tab:baseline}",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Drift & Train & Target & Median & P90 & P95 & P99 \\\\",
        "\\midrule",
    ]
    baseline_best = {
        "median": min(row["median"] for row in baseline_rows),
        "p90": min(row["p90"] for row in baseline_rows),
        "p95": min(row["p95"] for row in baseline_rows),
        "p99": min(row["p99"] for row in baseline_rows),
    }
    for row in baseline_rows:
        median_text = f"{row['median']:.2f}"
        p90_text = f"{row['p90']:.1f}"
        p95_text = f"{row['p95']:.1f}"
        p99_text = f"{row['p99']:.1f}"
        baseline_tex.append(
            f"{row['drift'].title()} & {row['train_rows']} & {row['target_rows']} "
            f"& {best_cell(median_text, row['median'] == baseline_best['median'])} "
            f"& {best_cell(p90_text, row['p90'] == baseline_best['p90'])} "
            f"& {best_cell(p95_text, row['p95'] == baseline_best['p95'])} "
            f"& {best_cell(p99_text, row['p99'] == baseline_best['p99'])} \\\\"
        )
    baseline_tex.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    (table_dir / "baseline_summary.tex").write_text("\n".join(baseline_tex), encoding="utf-8")

    bitmap_tex = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Effect of reconstructing MSCN materialized-sample bitmaps. Positive changes mean the bitmap baseline has lower q-error.}",
        "\\label{tab:bitmap-effect}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Drift & No-bitmap P95 & Bitmap P95 & No-bitmap P99 & Bitmap P99 \\\\",
        "\\midrule",
    ]
    for row in bitmap_rows:
        bitmap_p95 = f"{row['bitmap_p95']:.1f}"
        bitmap_p99 = f"{row['bitmap_p99']:.1f}"
        bitmap_tex.append(
            f"{row['drift'].title()} & {row['no_bitmap_p95']:.1f} "
            f"& {best_cell(bitmap_p95, row['bitmap_p95'] < row['no_bitmap_p95'])} "
            f"& {row['no_bitmap_p99']:.1f} "
            f"& {best_cell(bitmap_p99, row['bitmap_p99'] < row['no_bitmap_p99'])} \\\\"
        )
    bitmap_tex.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    (table_dir / "bitmap_effect.tex").write_text("\n".join(bitmap_tex), encoding="utf-8")

    comparison_tex = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Method comparison at the default adaptation budget. Reductions are relative to MSCN on the same held-out target split.}",
        "\\label{tab:method-comparison}",
        "\\begin{tabular}{llrrrrrrr}",
        "\\toprule",
        "Drift & Method & Budget & Pass & MSCN P95 & Method P95 & P95 Red. & MSCN P99 & P99 Red. \\\\",
        "\\midrule",
    ]
    default_best = {}
    for drift in {row["drift"] for row in default_rows}:
        drift_rows = [row for row in default_rows if row["drift"] == drift]
        default_best[drift] = {
            "method_p95": min(row["method_p95"] for row in drift_rows),
            "p95_reduction": max(row["p95_reduction"] for row in drift_rows),
            "p99_reduction": max(row["p99_reduction"] for row in drift_rows),
        }
    for row in default_rows:
        best = default_best[row["drift"]]
        method_p95 = f"{row['method_p95']:.1f}"
        p95_reduction = f"{pct(row['p95_reduction'])}\\%"
        p99_reduction = f"{pct(row['p99_reduction'])}\\%"
        comparison_tex.append(
            f"{row['drift'].title()} & {METHOD_LABELS[row['method']]} & {row['budget']} "
            f"& {pass_cell(row['passed'])} & {row['base_p95']:.1f} "
            f"& {best_cell(method_p95, row['method_p95'] == best['method_p95'])} "
            f"& {reduction_cell(p95_reduction, row['p95_reduction'], best['p95_reduction'])} "
            f"& {row['base_p99']:.1f} "
            f"& {reduction_cell(p99_reduction, row['p99_reduction'], best['p99_reduction'])} \\\\"
        )
    comparison_tex.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    (table_dir / "method_comparison.tex").write_text("\n".join(comparison_tex), encoding="utf-8")

    budget_tex = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Residual-field budget sensitivity. Entries are median reductions over three adaptation splits.}",
        "\\label{tab:budget}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Drift & Rows & P90 Red. & P95 Red. & P99 Red. \\\\",
        "\\midrule",
    ]
    budget_best = {}
    for drift in {row["drift"] for row in sweep_rows if row["method"] == "residual-field"}:
        drift_rows = [
            row for row in sweep_rows if row["method"] == "residual-field" and row["drift"] == drift
        ]
        budget_best[drift] = {
            "p90_reduction": max(row["p90_reduction"] for row in drift_rows),
            "p95_reduction": max(row["p95_reduction"] for row in drift_rows),
            "p99_reduction": max(row["p99_reduction"] for row in drift_rows),
        }
    for row in sweep_rows:
        if row["method"] != "residual-field":
            continue
        best = budget_best[row["drift"]]
        p90_reduction = f"{pct(row['p90_reduction'])}\\%"
        p95_reduction = f"{pct(row['p95_reduction'])}\\%"
        p99_reduction = f"{pct(row['p99_reduction'])}\\%"
        budget_tex.append(
            f"{row['drift'].title()} & {row['adaptation_rows']} "
            f"& {reduction_cell(p90_reduction, row['p90_reduction'], best['p90_reduction'])} "
            f"& {reduction_cell(p95_reduction, row['p95_reduction'], best['p95_reduction'])} "
            f"& {reduction_cell(p99_reduction, row['p99_reduction'], best['p99_reduction'])} \\\\"
        )
    budget_tex.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    (table_dir / "budget_sensitivity.tex").write_text("\n".join(budget_tex), encoding="utf-8")


def write_expanded_tables(output_dir: Path, method_rows: list[dict], ablation_rows: list[dict]) -> None:
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    strong_methods = [
        row
        for row in method_rows
        if row["method"] in {"isotonic", "ridge-residual", "random-forest", "residual-field", "gated-residual-field"}
        and row["variant"] in {"prediction-only", "full"}
    ]
    write_csv(output_dir / "expanded_method_comparison.csv", strong_methods)
    write_csv(output_dir / "feature_ablation.csv", ablation_rows)

    comparison_tex = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Expanded post-hoc adaptation baselines at the default budget. All methods use the same target-label budget and the same held-out splits.}",
        "\\label{tab:expanded-methods}",
        "\\begin{tabular}{llrrrrrr}",
        "\\toprule",
        "Drift & Method & Pass & P95 & P95 Red. & P99 & P99 Red. & vs. Global P99 \\\\",
        "\\midrule",
    ]
    order = {
        "isotonic": 0,
        "ridge-residual": 1,
        "random-forest": 2,
        "residual-field": 3,
        "gated-residual-field": 4,
    }
    expanded_best = {}
    for drift in {row["drift"] for row in strong_methods}:
        drift_rows = [row for row in strong_methods if row["drift"] == drift]
        expanded_best[drift] = {
            "method_p95": min(row["method_p95"] for row in drift_rows),
            "method_p99": min(row["method_p99"] for row in drift_rows),
            "p95_reduction": max(row["p95_reduction"] for row in drift_rows),
            "p99_reduction": max(row["p99_reduction"] for row in drift_rows),
            "p99_vs_global": max(row["p99_vs_global"] for row in drift_rows),
        }
    for row in sorted(strong_methods, key=lambda item: (item["drift"], order[item["method"]])):
        best = expanded_best[row["drift"]]
        method_p95 = f"{row['method_p95']:.1f}"
        p95_reduction = f"{pct(row['p95_reduction'])}\\%"
        method_p99 = f"{row['method_p99']:.1f}"
        p99_reduction = f"{pct(row['p99_reduction'])}\\%"
        p99_vs_global = f"{pct(row['p99_vs_global'])}\\%"
        comparison_tex.append(
            f"{row['drift'].title()} & {METHOD_LABELS[row['method']]} "
            f"& {pass_cell(row['passed'])} "
            f"& {best_cell(method_p95, row['method_p95'] == best['method_p95'])} "
            f"& {reduction_cell(p95_reduction, row['p95_reduction'], best['p95_reduction'])} "
            f"& {best_cell(method_p99, row['method_p99'] == best['method_p99'])} "
            f"& {reduction_cell(p99_reduction, row['p99_reduction'], best['p99_reduction'])} "
            f"& {reduction_cell(p99_vs_global, row['p99_vs_global'], best['p99_vs_global'])} \\\\"
        )
    comparison_tex.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    (table_dir / "expanded_method_comparison.tex").write_text(
        "\n".join(comparison_tex), encoding="utf-8"
    )

    ablation_tex = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Residual-field ablations at the default budget. Reductions are median values over three adaptation splits.}",
        "\\label{tab:ablation}",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Drift & Variant & Pass & P90 Red. & P95 Red. & P99 Red. \\\\",
        "\\midrule",
    ]
    variant_order = {name: idx for idx, name in enumerate(ABLATION_PRESETS + ["no-clipping"])}
    ablation_best = {}
    for drift in {row["drift"] for row in ablation_rows}:
        drift_rows = [row for row in ablation_rows if row["drift"] == drift]
        ablation_best[drift] = {
            "p90_reduction": max(row["p90_reduction"] for row in drift_rows),
            "p95_reduction": max(row["p95_reduction"] for row in drift_rows),
            "p99_reduction": max(row["p99_reduction"] for row in drift_rows),
        }
    for row in sorted(ablation_rows, key=lambda item: (item["drift"], variant_order[item["variant"]])):
        label = row["variant"].replace("-", " ")
        best = ablation_best[row["drift"]]
        p90_reduction = f"{pct(row['p90_reduction'])}\\%"
        p95_reduction = f"{pct(row['p95_reduction'])}\\%"
        p99_reduction = f"{pct(row['p99_reduction'])}\\%"
        ablation_tex.append(
            f"{row['drift'].title()} & {label} & {pass_cell(row['passed'])} "
            f"& {reduction_cell(p90_reduction, row['p90_reduction'], best['p90_reduction'])} "
            f"& {reduction_cell(p95_reduction, row['p95_reduction'], best['p95_reduction'])} "
            f"& {reduction_cell(p99_reduction, row['p99_reduction'], best['p99_reduction'])} \\\\"
        )
    ablation_tex.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    (table_dir / "feature_ablation.tex").write_text("\n".join(ablation_tex), encoding="utf-8")


def plot_results(output_dir: Path, reports: dict, sweep_rows: list[dict]) -> None:
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L", "DejaVu Serif", "serif"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 140,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )

    drifts = list(DRIFTS)
    x = np.arange(len(drifts))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.0, 2.7))
    for offset, method in [(-width / 2, "signature-tail"), (width / 2, "residual-field")]:
        vals = []
        for drift in drifts:
            report = reports[(drift, method, DRIFTS[drift]["default_budget"])]
            vals.append(100.0 * report["aggregate_reductions"]["p95"])
        ax.bar(
            x + offset,
            vals,
            width=width,
            label=METHOD_LABELS[method],
            color=PALETTE[method],
            edgecolor="white",
            linewidth=0.6,
        )
    apply_plot_style(ax)
    ax.axhline(0, color=TABLE_DARK, linewidth=0.8)
    ax.set_xticks(x, [name.title() for name in drifts])
    ax.set_ylabel("P95 q-error reduction (%)")
    ax.legend(ncols=2, loc="upper right")
    ax.set_ylim(min(-10, ax.get_ylim()[0]), 110)
    fig.tight_layout()
    fig.savefig(fig_dir / "p95_method_comparison.pdf")
    fig.savefig(fig_dir / "p95_method_comparison.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    colors = {"column": "#7FB3D5", "table": "#A9D18E", "value": "#C7A6D8"}
    for drift in drifts:
        rows = [
            row for row in sweep_rows if row["method"] == "residual-field" and row["drift"] == drift
        ]
        rows.sort(key=lambda row: row["adaptation_rows"])
        ax.plot(
            [row["adaptation_rows"] for row in rows],
            [100.0 * row["p95_reduction"] for row in rows],
            marker="o",
            linewidth=1.5,
            color=colors[drift],
            label=f"{drift.title()} P95",
        )
        ax.plot(
            [row["adaptation_rows"] for row in rows],
            [100.0 * row["p99_reduction"] for row in rows],
            marker="s",
            linewidth=1.0,
            linestyle="--",
            color=colors[drift],
            alpha=0.75,
            label=f"{drift.title()} P99",
        )
    apply_plot_style(ax)
    ax.axhline(0, color=TABLE_DARK, linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Labeled target adaptation rows")
    ax.set_ylabel("Reduction vs MSCN (%)")
    ax.legend(ncols=3, fontsize=6.5)
    fig.tight_layout()
    fig.savefig(fig_dir / "budget_sensitivity.pdf")
    fig.savefig(fig_dir / "budget_sensitivity.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    labels = []
    base_vals = []
    method_vals = []
    for drift in drifts:
        report = reports[(drift, "residual-field", DRIFTS[drift]["default_budget"])]
        base = median_summary(report, "baseline_qerror")
        meth = median_summary(report, "method_qerror")
        labels.append(drift.title())
        base_vals.append(base["p99"])
        method_vals.append(meth["p99"])
    loc = np.arange(len(labels))
    ax.bar(
        loc - width / 2,
        base_vals,
        width,
        label="MSCN",
        color=PALETTE["mscn"],
        edgecolor="white",
        linewidth=0.6,
    )
    ax.bar(
        loc + width / 2,
        method_vals,
        width,
        label="Residual-field",
        color=PALETTE["residual-field"],
        edgecolor="white",
        linewidth=0.6,
    )
    apply_plot_style(ax)
    ax.set_yscale("log")
    ax.set_xticks(loc, labels)
    ax.set_ylabel("P99 q-error (log scale)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "p99_qerror_residual_field.pdf")
    fig.savefig(fig_dir / "p99_qerror_residual_field.png")
    plt.close(fig)


def plot_expanded_results(output_dir: Path, method_rows: list[dict], ablation_rows: list[dict]) -> None:
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L", "DejaVu Serif", "serif"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 140,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )
    drifts = list(DRIFTS)
    methods = ["isotonic", "ridge-residual", "random-forest", "residual-field", "gated-residual-field"]
    fig, axes = plt.subplots(1, len(drifts), figsize=(7.4, 2.6), sharey=True)
    for ax, drift in zip(axes, drifts):
        rows = {
            row["method"]: row
            for row in method_rows
            if row["drift"] == drift and row["method"] in methods and row["variant"] in {"prediction-only", "full"}
        }
        vals = [100.0 * rows[method]["p99_reduction"] for method in methods]
        method_colors = [
            PALETTE["isotonic"],
            PALETTE["ridge-residual"],
            PALETTE["random-forest"],
            PALETTE["residual-field"],
            PALETTE["gated-residual-field"],
        ]
        colors = [
            method_color if val >= 0 else PALETTE["negative"]
            for val, method_color in zip(vals, method_colors)
        ]
        ax.bar(np.arange(len(methods)), vals, color=colors, edgecolor="white", linewidth=0.6)
        apply_plot_style(ax)
        ax.axhline(0, color=TABLE_DARK, linewidth=0.8)
        ax.set_title(drift.title())
        ax.set_xticks(np.arange(len(methods)), ["Iso", "Ridge", "RF", "Field", "Gated"], rotation=30, ha="right")
    axes[0].set_ylabel("P99 reduction vs MSCN (%)")
    fig.tight_layout()
    fig.savefig(fig_dir / "expanded_p99_method_comparison.pdf")
    fig.savefig(fig_dir / "expanded_p99_method_comparison.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(drifts), figsize=(7.4, 2.6), sharey=True)
    variants = ABLATION_PRESETS + ["no-clipping"]
    for ax, drift in zip(axes, drifts):
        rows = {
            row["variant"]: row
            for row in ablation_rows
            if row["drift"] == drift
        }
        vals = [100.0 * rows[variant]["p95_reduction"] for variant in variants]
        colors = [PALETTE["residual-field"] if val >= 0 else PALETTE["negative"] for val in vals]
        ax.bar(np.arange(len(variants)), vals, color=colors, edgecolor="white", linewidth=0.6)
        apply_plot_style(ax)
        ax.axhline(0, color=TABLE_DARK, linewidth=0.8)
        ax.set_title(drift.title())
        ax.set_xticks(
            np.arange(len(variants)),
            ["Full", "Pred", "Struct", "No val", "No pred", "No tbl/col", "No clip"],
            rotation=35,
            ha="right",
            fontsize=7,
        )
    axes[0].set_ylabel("P95 reduction vs MSCN (%)")
    fig.tight_layout()
    fig.savefig(fig_dir / "feature_ablation_p95.pdf")
    fig.savefig(fig_dir / "feature_ablation_p95.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and summarize Workload-Drift paper experiments.")
    parser.add_argument("--output-dir", default="paper/workload_drift_residual_field/results")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--expanded", action="store_true")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    result = collect_results(output_dir, args.force)
    write_csv(output_dir / "budget_sweep.csv", result["rows"])
    write_tables(output_dir, result["reports"], result["rows"])
    plot_results(output_dir, result["reports"], result["rows"])
    expanded_manifest = {}
    if args.expanded:
        expanded = collect_expanded_results(output_dir, args.force, args.jobs)
        write_csv(output_dir / "expanded_all.csv", expanded["rows"])
        write_expanded_tables(output_dir, expanded["method_rows"], expanded["ablation_rows"])
        plot_expanded_results(output_dir, expanded["method_rows"], expanded["ablation_rows"])
        expanded_manifest = {
            "expanded_rows": len(expanded["rows"]),
            "expanded_all_csv": str(output_dir / "expanded_all.csv"),
            "expanded_method_comparison_csv": str(output_dir / "expanded_method_comparison.csv"),
            "feature_ablation_csv": str(output_dir / "feature_ablation.csv"),
        }

    manifest = {
        "passed": True,
        "scope": "Windows-native no-API Workload-Drift STATS experiments for SIGMOD-style paper",
        "drifts": list(DRIFTS),
        "methods": [
            "MSCN bitmap",
            "global calibration",
            "signature-tail",
            "isotonic",
            "ridge-residual",
            "random-forest",
            "residual-field",
            "gated-residual-field",
        ],
        "budget_sweep_rows": len(result["rows"]),
        "expected_negative_result": "Residual-field does not pass the strict value-drift gate under the strong bitmap MSCN baseline.",
        "outputs": {
            "budget_sweep_csv": str(output_dir / "budget_sweep.csv"),
            "baseline_summary_csv": str(output_dir / "baseline_summary.csv"),
            "bitmap_effect_summary_csv": str(output_dir / "bitmap_effect_summary.csv"),
            "default_method_comparison_csv": str(output_dir / "default_method_comparison.csv"),
            "figures": str(output_dir / "figures"),
            "tables": str(output_dir / "tables"),
        },
        "expanded": expanded_manifest,
    }
    write_json(output_dir / "paper_results_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
