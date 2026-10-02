import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.run_workload_drift_tail_residual_transport import (
    learn_residual_field,
    load_target_rows,
    qerror,
    relative_reduction,
    sha256_array,
    signatures,
    summarize,
    tail_risk_shift,
    apply_residual_field,
)


DRIFTS = {
    "column": {
        "metrics": ROOT / "research_plan" / "workload_drift_mscn_windows_bitmap" / "seed_1" / "metrics.json",
        "fit_rows": 2000,
        "gate_rows": 1000,
    },
    "table": {
        "metrics": ROOT / "research_plan" / "workload_drift_mscn_windows_table_bitmap" / "seed_1" / "metrics.json",
        "fit_rows": 2000,
        "gate_rows": 1000,
    },
    "value": {
        "metrics": ROOT / "research_plan" / "workload_drift_mscn_windows_value_bitmap" / "seed_1" / "metrics.json",
        "fit_rows": 500,
        "gate_rows": 500,
    },
}


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}"


def pass_cell(text: str, passed: bool) -> str:
    return f"\\passcell{{{text}}}" if passed else text


def best_cell(text: str, enabled: bool) -> str:
    return f"\\bestcell{{{text}}}" if enabled else text


def read_metrics(spec: dict) -> tuple[list[list[str]], np.ndarray, np.ndarray, list[list[tuple]], Path]:
    metrics_path = spec["metrics"]
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    target_path = (
        metrics_path.parent
        / "data"
        / metrics["provenance"]["predict_rel"].replace("/", "\\")
    )
    if not target_path.exists():
        target_path = metrics_path.parent / "data" / metrics["provenance"]["predict_rel"]
    predictions = np.loadtxt(metrics["artifacts"]["predictions"], dtype=np.float64)
    rows, labels = load_target_rows(target_path)
    if predictions.shape[0] != labels.shape[0]:
        raise ValueError(f"prediction/label length mismatch: {predictions.shape[0]} != {labels.shape[0]}")
    all_signatures = [signatures(row) for row in rows]
    return rows, labels, predictions, all_signatures, target_path


def evaluate_region(predictions, labels, residual_shifts, global_shift: float, idx: np.ndarray) -> dict:
    baseline_q = summarize(qerror(predictions[idx], labels[idx]))
    global_predictions = predictions[idx] * math.exp(global_shift)
    global_q = summarize(qerror(global_predictions, labels[idx]))
    method_predictions = predictions[idx] * np.exp(residual_shifts)
    method_q = summarize(qerror(method_predictions, labels[idx]))
    reductions = {
        key: relative_reduction(baseline_q[key], method_q[key])
        for key in ["median", "p90", "p95", "p99", "mean"]
    }
    reductions_vs_global = {
        key: relative_reduction(global_q[key], method_q[key])
        for key in ["median", "p90", "p95", "p99", "mean"]
    }
    return {
        "rows": int(idx.shape[0]),
        "baseline_qerror": baseline_q,
        "global_calibration_qerror": global_q,
        "method_qerror": method_q,
        "relative_reductions": reductions,
        "relative_reductions_vs_global_calibration": reductions_vs_global,
    }


def gate_passes(region: dict, args) -> tuple[bool, dict]:
    red = region["relative_reductions"]
    red_global = region["relative_reductions_vs_global_calibration"]
    baseline = region["baseline_qerror"]
    method = region["method_qerror"]
    checks = {
        "p90_reduction": red["p90"] >= args.min_p90_reduction,
        "p95_reduction": red["p95"] >= args.min_p95_reduction,
        "p99_reduction": red["p99"] >= args.min_p99_reduction,
        "beats_global_p95": red_global["p95"] >= args.min_vs_global_p95_reduction,
        "beats_global_p99": red_global["p99"] >= args.min_vs_global_p99_reduction,
        "median_not_much_worse": method["median"] <= baseline["median"] * args.max_median_regression,
        "gate_rows": region["rows"] >= args.min_gate_rows,
    }
    return all(checks.values()), checks


def evaluate_split(drift: str, spec: dict, seed: int, args) -> dict:
    rows, labels, predictions, all_signatures, target_path = read_metrics(spec)
    rng = np.random.default_rng(seed)
    indices = np.arange(labels.shape[0])
    rng.shuffle(indices)

    fit_rows = int(spec["fit_rows"])
    gate_rows = int(spec["gate_rows"])
    if fit_rows + gate_rows + args.min_final_rows > labels.shape[0]:
        raise ValueError(f"not enough target rows for nested split on {drift}")

    fit_idx = indices[:fit_rows]
    gate_idx = indices[fit_rows : fit_rows + gate_rows]
    final_idx = indices[fit_rows + gate_rows :]

    residuals = np.log(labels) - np.log(np.maximum(1.0, predictions))
    global_shift = tail_risk_shift(residuals[fit_idx], args.tail_weight, args.ridge)
    model = learn_residual_field(
        rows,
        predictions,
        residuals,
        fit_idx,
        seed,
        args.residual_clip_lower,
        args.residual_clip_upper,
        "full",
    )
    gate_shifts = apply_residual_field(model, gate_idx)
    final_shifts = apply_residual_field(model, final_idx)

    gate_region = evaluate_region(predictions, labels, gate_shifts, global_shift, gate_idx)
    gate_passed, gate_checks = gate_passes(gate_region, args)
    final_region = evaluate_region(predictions, labels, final_shifts, global_shift, final_idx)

    deployed_qerror = final_region["method_qerror"] if gate_passed else final_region["baseline_qerror"]
    deployed_reductions = final_region["relative_reductions"] if gate_passed else {
        key: 0.0 for key in ["median", "p90", "p95", "p99", "mean"]
    }
    final_checks = {
        "p95_reduction": final_region["relative_reductions"]["p95"] >= args.min_p95_reduction,
        "p99_reduction": final_region["relative_reductions"]["p99"] >= args.min_p99_reduction,
        "median_not_much_worse": (
            final_region["method_qerror"]["median"]
            <= final_region["baseline_qerror"]["median"] * args.max_median_regression
        ),
        "final_rows": final_region["rows"] >= args.min_final_rows,
    }

    return {
        "drift": drift,
        "seed": seed,
        "fit_rows": fit_rows,
        "gate_rows": gate_rows,
        "final_rows": int(final_idx.shape[0]),
        "gate_passed": bool(gate_passed),
        "gate_checks": gate_checks,
        "final_residual_checks": final_checks,
        "action": "deploy residual-field" if gate_passed else "reject residual-field",
        "gate": gate_region,
        "final_test": final_region,
        "deployed_final_qerror": deployed_qerror,
        "deployed_final_reductions": deployed_reductions,
        "global_shift": float(global_shift),
        "feature_dim": int(model["feature_dim"]),
        "vocab_sizes": model["vocab_sizes"],
        "residual_clip_range": model["clip_range"],
        "fit_indices_sha256": sha256_array(fit_idx),
        "gate_indices_sha256": sha256_array(gate_idx),
        "final_indices_sha256": sha256_array(final_idx),
        "target_workload": str(target_path),
    }


def median_metric(split_reports: list[dict], section: str, metric_group: str, metric: str) -> float:
    return float(np.median([item[section][metric_group][metric] for item in split_reports]))


def summarize_drift(drift: str, split_reports: list[dict]) -> dict:
    first = split_reports[0]
    gate_passed = sum(1 for item in split_reports if item["gate_passed"])
    final_p95_red = median_metric(split_reports, "final_test", "relative_reductions", "p95")
    final_p99_red = median_metric(split_reports, "final_test", "relative_reductions", "p99")
    return {
        "drift": drift,
        "fit_rows": first["fit_rows"],
        "gate_rows": first["gate_rows"],
        "final_rows_median": int(np.median([item["final_rows"] for item in split_reports])),
        "gate_passed_seed_count": gate_passed,
        "seed_count": len(split_reports),
        "final_base_p95": median_metric(split_reports, "final_test", "baseline_qerror", "p95"),
        "final_method_p95": median_metric(split_reports, "final_test", "method_qerror", "p95"),
        "final_base_p99": median_metric(split_reports, "final_test", "baseline_qerror", "p99"),
        "final_method_p99": median_metric(split_reports, "final_test", "method_qerror", "p99"),
        "final_p95_reduction": final_p95_red,
        "final_p99_reduction": final_p99_red,
        "action": "deploy" if gate_passed == len(split_reports) else "reject",
        "passed": bool(
            gate_passed == len(split_reports)
            and final_p95_red >= 0.40
            and final_p99_red >= 0.60
        ),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_latex_table(path: Path, rows: list[dict]) -> None:
    lines = [
        "\\begin{table}[!t]",
        "\\centering",
        "\\caption{Nested fit/gate/final-test audit for residual-field transport. The gate split is used only to decide whether to enable the repair; final-test rows are disjoint and used only for reporting.}",
        "\\label{tab:nested-final-test}",
        "\\papertablesetup",
        "\\setlength{\\tabcolsep}{2.0pt}",
        "\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}l|c|c|c c|l@{}}",
        "\\toprule",
        "\\tabheader{Drift} & \\tabheader{Fit/Gate/Test} & \\tabheader{Gate} & \\tabheader{Final Q} & \\tabheader{Final Red.} & \\tabheader{Action} \\\\",
        "\\midrule",
    ]
    for row in rows:
        gate_text = f"{row['gate_passed_seed_count']}/{row['seed_count']}"
        split_text = f"{row['fit_rows']}/{row['gate_rows']}/{row['final_rows_median']}"
        q_text = f"{row['final_method_p95']:.1f}/{row['final_method_p99']:.0f}"
        red_text = f"{pct(row['final_p95_reduction'])}/{pct(row['final_p99_reduction'])}\\%"
        structural = row["drift"] in {"column", "table"}
        lines.append(
            f"{row['drift'].title()} & {split_text} & {pass_cell(gate_text, row['action'] == 'deploy')} "
            f"& {best_cell(q_text, structural and row['action'] == 'deploy')} "
            f"& {best_cell(red_text, structural and row['action'] == 'deploy')} & {row['action']} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular*}", "\\end{table}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a nested fit/gate/final-test residual-field audit.")
    parser.add_argument("--output-dir", default="paper/workload_drift_residual_field/results")
    parser.add_argument("--seeds", default="11,12,13")
    parser.add_argument("--tail-weight", type=float, default=1.5)
    parser.add_argument("--ridge", type=float, default=0.005)
    parser.add_argument("--residual-clip-lower", type=float, default=1.0)
    parser.add_argument("--residual-clip-upper", type=float, default=99.0)
    parser.add_argument("--min-p90-reduction", type=float, default=0.30)
    parser.add_argument("--min-p95-reduction", type=float, default=0.40)
    parser.add_argument("--min-p99-reduction", type=float, default=0.60)
    parser.add_argument("--min-vs-global-p95-reduction", type=float, default=0.25)
    parser.add_argument("--min-vs-global-p99-reduction", type=float, default=0.40)
    parser.add_argument("--max-median-regression", type=float, default=1.20)
    parser.add_argument("--min-gate-rows", type=int, default=500)
    parser.add_argument("--min-final-rows", type=int, default=1000)
    args = parser.parse_args()

    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    by_drift = {}
    summary_rows = []
    for drift, spec in DRIFTS.items():
        split_reports = [evaluate_split(drift, spec, seed, args) for seed in seeds]
        by_drift[drift] = split_reports
        summary_rows.append(summarize_drift(drift, split_reports))

    report = {
        "scope": "nested fit/gate/final-test residual-field audit",
        "seeds": seeds,
        "thresholds": {
            "min_p90_reduction": args.min_p90_reduction,
            "min_p95_reduction": args.min_p95_reduction,
            "min_p99_reduction": args.min_p99_reduction,
            "min_vs_global_p95_reduction": args.min_vs_global_p95_reduction,
            "min_vs_global_p99_reduction": args.min_vs_global_p99_reduction,
            "max_median_regression": args.max_median_regression,
            "min_gate_rows": args.min_gate_rows,
            "min_final_rows": args.min_final_rows,
        },
        "summary": summary_rows,
        "split_reports": by_drift,
        "command": ["experiments/run_workload_drift_nested_final_test.py"],
    }

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "nested_final_test.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(out_dir / "nested_final_test.csv", summary_rows)
    write_latex_table(out_dir / "tables" / "nested_final_test.tex", summary_rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
