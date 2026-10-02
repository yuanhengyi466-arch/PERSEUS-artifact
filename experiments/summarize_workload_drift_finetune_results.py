import argparse
import csv
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research_plan"


DRIFTS = {
    "column": {
        "metrics": RESEARCH / "workload_drift_mscn_windows_bitmap" / "seed_1" / "metrics.json",
        "adaptation_rows": 2000,
        "min_eval_rows": 5000,
    },
    "table": {
        "metrics": RESEARCH / "workload_drift_mscn_windows_table_bitmap" / "seed_1" / "metrics.json",
        "adaptation_rows": 2000,
        "min_eval_rows": 5000,
    },
    "value": {
        "metrics": RESEARCH / "workload_drift_mscn_windows_value_bitmap" / "seed_1" / "metrics.json",
        "adaptation_rows": 500,
        "min_eval_rows": 1000,
    },
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def run_command(cmd: list[str]) -> dict:
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
    return {
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def lr_tag(lr: float) -> str:
    return f"{lr:.0e}".replace("+", "").replace("-", "m")


def run_finetune(
    drift: str,
    output_dir: Path,
    force: bool,
    epochs: int,
    lr: float,
    subdir: str = "finetune",
) -> dict:
    spec = DRIFTS[drift]
    out = (
        output_dir
        / subdir
        / f"{drift}_mscn_finetune_{spec['adaptation_rows']}_e{epochs}_lr{lr_tag(lr)}.json"
    )
    if out.exists() and not force:
        return read_json(out)
    cmd = [
        sys.executable,
        "experiments/run_workload_drift_mscn_finetune.py",
        "--baseline-metrics",
        str(spec["metrics"].relative_to(ROOT)),
        "--output-json",
        str(out.relative_to(ROOT)),
        "--adaptation-rows",
        str(spec["adaptation_rows"]),
        "--seeds",
        "1,2,3",
        "--epochs",
        str(epochs),
        "--lr",
        str(lr),
        "--min-eval-rows",
        str(spec["min_eval_rows"]),
    ]
    result = run_command(cmd)
    if not out.exists():
        raise RuntimeError(f"fine-tune run did not write {out}: {result}")
    data = read_json(out)
    data["runner_returncode"] = result["returncode"]
    data["runner_stdout_tail"] = result["stdout_tail"]
    data["runner_stderr_tail"] = result["stderr_tail"]
    write_json(out, data)
    return data


def median_summary(report: dict, key: str) -> dict:
    return {
        metric: float(np.median([row[key][metric] for row in report["split_reports"]]))
        for metric in ["median", "p90", "p95", "p99", "mean"]
    }


def row_from_report(drift: str, method: str, report: dict) -> dict:
    method_q = median_summary(report, "method_qerror")
    base_q = median_summary(report, "baseline_qerror")
    return {
        "drift": drift,
        "method": method,
        "budget": report["requires_labeled_target_adaptation_rows"]
        if "requires_labeled_target_adaptation_rows" in report
        else DRIFTS[drift]["adaptation_rows"],
        "passed": report["passed"],
        "passed_seed_count": report["passed_seed_count"],
        "seed_count": report["seed_count"],
        "base_p95": base_q["p95"],
        "method_p95": method_q["p95"],
        "base_p99": base_q["p99"],
        "method_p99": method_q["p99"],
        "p90_reduction": report["aggregate_reductions"]["p90"],
        "p95_reduction": report["aggregate_reductions"]["p95"],
        "p99_reduction": report["aggregate_reductions"]["p99"],
        "mean_reduction": report["aggregate_reductions"]["mean"],
    }


def load_existing_posthoc(output_dir: Path) -> list[dict]:
    rows = []
    expanded = output_dir / "expanded_method_comparison.csv"
    if not expanded.exists():
        return rows
    with expanded.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["method"] in {"ridge-residual", "random-forest", "residual-field"}:
                rows.append(
                    {
                        "drift": row["drift"],
                        "method": row["method"],
                        "budget": int(row["budget"]),
                        "passed": row["passed"] == "True",
                        "passed_seed_count": int(row["passed_seed_count"]),
                        "seed_count": int(row["seed_count"]),
                        "base_p95": float(row["base_p95"]),
                        "method_p95": float(row["method_p95"]),
                        "base_p99": float(row["base_p99"]),
                        "method_p99": float(row["method_p99"]),
                        "p90_reduction": float(row["p90_reduction"]),
                        "p95_reduction": float(row["p95_reduction"]),
                        "p99_reduction": float(row["p99_reduction"]),
                        "mean_reduction": float(row["mean_reduction"]),
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}"


def pass_cell(value: bool) -> str:
    return "\\passcell{yes}" if value else "no"


def best_cell(text: str, is_best: bool) -> str:
    return f"\\bestcell{{{text}}}" if is_best else text


def reduction_cell(text: str, value: float, best_value=None) -> str:
    if value <= 0:
        return text
    if best_value is not None and best_value > 0 and value == best_value:
        return best_cell(text, True)
    return f"\\passcell{{{text}}}"


def write_table(output_dir: Path, rows: list[dict]) -> None:
    labels = {
        "ridge-residual": "Ridge residual",
        "random-forest": "Random forest",
        "residual-field": "Residual-field",
        "mscn-finetune": "MSCN fine-tune",
    }
    order = {"ridge-residual": 0, "random-forest": 1, "residual-field": 2, "mscn-finetune": 3}
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Same-budget target-label comparison against MSCN fine-tuning. All methods use identical adaptation/evaluation splits.}",
        "\\label{tab:finetune}",
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "Drift & Method & Pass & Method P95 & P95 Red. & Method P99 & P99 Red. \\\\",
        "\\midrule",
    ]
    best_by_drift = {}
    for drift in {row["drift"] for row in rows}:
        drift_rows = [row for row in rows if row["drift"] == drift]
        best_by_drift[drift] = {
            "method_p95": min(row["method_p95"] for row in drift_rows),
            "p95_reduction": max(row["p95_reduction"] for row in drift_rows),
            "method_p99": min(row["method_p99"] for row in drift_rows),
            "p99_reduction": max(row["p99_reduction"] for row in drift_rows),
        }
    for row in sorted(rows, key=lambda item: (item["drift"], order[item["method"]])):
        best = best_by_drift[row["drift"]]
        method_p95 = f"{row['method_p95']:.1f}"
        p95_reduction = f"{pct(row['p95_reduction'])}\\%"
        method_p99 = f"{row['method_p99']:.1f}"
        p99_reduction = f"{pct(row['p99_reduction'])}\\%"
        lines.append(
            f"{row['drift'].title()} & {labels[row['method']]} & {pass_cell(row['passed'])} "
            f"& {best_cell(method_p95, row['method_p95'] == best['method_p95'])} "
            f"& {reduction_cell(p95_reduction, row['p95_reduction'], best['p95_reduction'])} "
            f"& {best_cell(method_p99, row['method_p99'] == best['method_p99'])} "
            f"& {reduction_cell(p99_reduction, row['p99_reduction'], best['p99_reduction'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    table = output_dir / "tables" / "finetune_comparison.tex"
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text("\n".join(lines), encoding="utf-8")


def write_sensitivity_table(output_dir: Path, rows: list[dict]) -> None:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{MSCN fine-tuning sensitivity. Entries are median P95/P99 reductions over three adaptation splits.}",
        "\\label{tab:finetune-sensitivity}",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Drift & Epochs/LR & P95 Red. & P99 Red. \\\\",
        "\\midrule",
    ]
    best_by_drift = {}
    for drift in {row["drift"] for row in rows}:
        drift_rows = [row for row in rows if row["drift"] == drift]
        best_by_drift[drift] = {
            "p95_reduction": max(row["p95_reduction"] for row in drift_rows),
            "p99_reduction": max(row["p99_reduction"] for row in drift_rows),
        }
    for row in sorted(rows, key=lambda item: (item["drift"], item["epochs"], item["lr"])):
        best = best_by_drift[row["drift"]]
        p95_reduction = f"{pct(row['p95_reduction'])}\\%"
        p99_reduction = f"{pct(row['p99_reduction'])}\\%"
        lines.append(
            f"{row['drift'].title()} & {row['epochs']}/{row['lr']:.0e} "
            f"& {reduction_cell(p95_reduction, row['p95_reduction'], best['p95_reduction'])} "
            f"& {reduction_cell(p99_reduction, row['p99_reduction'], best['p99_reduction'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    table = output_dir / "tables" / "finetune_sensitivity.tex"
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize MSCN fine-tune baselines.")
    parser.add_argument("--output-dir", default="paper/workload_drift_residual_field/results")
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()

    output_dir = ROOT / args.output_dir
    rows = load_existing_posthoc(output_dir)
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = {
            pool.submit(run_finetune, drift, output_dir, args.force, args.epochs, args.lr): drift
            for drift in DRIFTS
        }
        for future in as_completed(futures):
            drift = futures[future]
            rows.append(row_from_report(drift, "mscn-finetune", future.result()))
    rows.sort(key=lambda item: (item["drift"], item["method"]))
    write_csv(output_dir / "finetune_comparison.csv", rows)
    write_table(output_dir, rows)
    sensitivity_rows = []
    if args.sweep:
        variants = [(5, 1e-4), (20, 1e-4), (5, 1e-3), (20, 1e-3)]
        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
            futures = {
                pool.submit(
                    run_finetune,
                    drift,
                    output_dir,
                    args.force,
                    epochs,
                    lr,
                    "finetune_sweep",
                ): (drift, epochs, lr)
                for drift in DRIFTS
                for epochs, lr in variants
            }
            for future in as_completed(futures):
                drift, epochs, lr = futures[future]
                row = row_from_report(drift, "mscn-finetune", future.result())
                row["epochs"] = epochs
                row["lr"] = lr
                sensitivity_rows.append(row)
        sensitivity_rows.sort(key=lambda item: (item["drift"], item["epochs"], item["lr"]))
        write_csv(output_dir / "finetune_sensitivity.csv", sensitivity_rows)
        write_sensitivity_table(output_dir, sensitivity_rows)
    manifest = {
        "passed": True,
        "rows": len(rows),
        "csv": str(output_dir / "finetune_comparison.csv"),
        "table": str(output_dir / "tables" / "finetune_comparison.tex"),
        "epochs": args.epochs,
        "lr": args.lr,
        "sweep_rows": len(sensitivity_rows),
        "sweep_csv": str(output_dir / "finetune_sensitivity.csv") if sensitivity_rows else "",
        "sweep_table": str(output_dir / "tables" / "finetune_sensitivity.tex")
        if sensitivity_rows
        else "",
    }
    write_json(output_dir / "finetune_comparison_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
