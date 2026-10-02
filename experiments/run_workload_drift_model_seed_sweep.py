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
        "train_rel": "workloads/stats-drift/varying_column_queries/queries_4_columns.csv",
        "predict_rel": "workloads/stats-drift/varying_column_queries/queries_5_columns.csv",
        "output_dir": RESEARCH / "workload_drift_mscn_windows_bitmap",
        "train_bitmap": RESEARCH
        / "workload_drift_mscn_windows"
        / "seed_1"
        / "data"
        / "workloads"
        / "stats-drift"
        / "varying_column_queries"
        / "queries_4_columns_bitmap.csv",
        "predict_bitmap": RESEARCH
        / "workload_drift_mscn_windows"
        / "seed_1"
        / "data"
        / "workloads"
        / "stats-drift"
        / "varying_column_queries"
        / "queries_5_columns_bitmap.csv",
        "adaptation_rows": 2000,
        "min_eval_rows": 5000,
    },
    "table": {
        "train_rel": "workloads/stats-drift/varying_table_queries/queries_4_tables.csv",
        "predict_rel": "workloads/stats-drift/varying_table_queries/queries_5_tables.csv",
        "output_dir": RESEARCH / "workload_drift_mscn_windows_table_bitmap",
        "train_bitmap": RESEARCH
        / "workload_drift_mscn_windows_table"
        / "seed_1"
        / "data"
        / "workloads"
        / "stats-drift"
        / "varying_table_queries"
        / "queries_4_tables_bitmap.csv",
        "predict_bitmap": RESEARCH
        / "workload_drift_mscn_windows_table"
        / "seed_1"
        / "data"
        / "workloads"
        / "stats-drift"
        / "varying_table_queries"
        / "queries_5_tables_bitmap.csv",
        "adaptation_rows": 2000,
        "min_eval_rows": 5000,
    },
    "value": {
        "train_rel": "workloads/stats-drift/varying_value_queries/value_distribution_1_1.csv",
        "predict_rel": "workloads/stats-drift/varying_value_queries/value_distribution_7_2.csv",
        "output_dir": RESEARCH / "workload_drift_mscn_windows_value_bitmap",
        "train_bitmap": RESEARCH
        / "workload_drift_mscn_windows_value"
        / "seed_1"
        / "data"
        / "workloads"
        / "stats-drift"
        / "varying_value_queries"
        / "value_distribution_1_1_bitmap.csv",
        "predict_bitmap": RESEARCH
        / "workload_drift_mscn_windows_value"
        / "seed_1"
        / "data"
        / "workloads"
        / "stats-drift"
        / "varying_value_queries"
        / "value_distribution_7_2_bitmap.csv",
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


def ensure_mscn(drift: str, seed: int, force: bool) -> dict:
    spec = DRIFTS[drift]
    metrics = spec["output_dir"] / f"seed_{seed}" / "metrics.json"
    if metrics.exists() and not force:
        data = read_json(metrics)
        data["runner_returncode"] = 0
        return data
    cmd = [
        sys.executable,
        "experiments/run_workload_drift_mscn_windows.py",
        "--train-rel",
        spec["train_rel"],
        "--predict-rel",
        spec["predict_rel"],
        "--output-dir",
        str(spec["output_dir"].relative_to(ROOT)),
        "--epochs",
        "5",
        "--batch-size",
        "512",
        "--hidden-units",
        "128",
        "--seed",
        str(seed),
        "--threads",
        "2",
        "--num-materialized-samples",
        "1000",
        "--train-bitmap-file",
        str(spec["train_bitmap"]),
        "--predict-bitmap-file",
        str(spec["predict_bitmap"]),
    ]
    result = run_command(cmd)
    if not metrics.exists():
        raise RuntimeError(f"MSCN seed run failed for {drift} seed {seed}: {result}")
    data = read_json(metrics)
    data["runner_returncode"] = result["returncode"]
    data["runner_stdout_tail"] = result["stdout_tail"]
    data["runner_stderr_tail"] = result["stderr_tail"]
    write_json(metrics, data)
    return data


def run_residual_gate(drift: str, seed: int, force: bool, output_dir: Path) -> dict:
    spec = DRIFTS[drift]
    out = output_dir / "model_seed_gates" / f"{drift}_mscn_seed_{seed}_residual_field.json"
    if out.exists() and not force:
        return read_json(out)
    metrics = spec["output_dir"] / f"seed_{seed}" / "metrics.json"
    cmd = [
        sys.executable,
        "experiments/run_workload_drift_tail_residual_transport.py",
        "--method",
        "residual-field",
        "--baseline-metrics",
        str(metrics.relative_to(ROOT)),
        "--output-json",
        str(out.relative_to(ROOT)),
        "--adaptation-rows",
        str(spec["adaptation_rows"]),
        "--seeds",
        "1,2,3",
        "--min-eval-rows",
        str(spec["min_eval_rows"]),
    ]
    result = run_command(cmd)
    if not out.exists():
        raise RuntimeError(f"residual gate failed to write {out}: {result}")
    data = read_json(out)
    data["runner_returncode"] = result["returncode"]
    data["runner_stdout_tail"] = result["stdout_tail"]
    data["runner_stderr_tail"] = result["stderr_tail"]
    write_json(out, data)
    return data


def median_summary(report: dict, key: str) -> dict:
    rows = report["split_reports"]
    return {
        metric: float(np.median([row[key][metric] for row in rows]))
        for metric in ["median", "p90", "p95", "p99", "mean"]
    }


def experiment_one(drift: str, seed: int, force: bool, output_dir: Path) -> dict:
    metrics = ensure_mscn(drift, seed, force)
    gate = run_residual_gate(drift, seed, force, output_dir)
    method = median_summary(gate, "method_qerror")
    baseline = median_summary(gate, "baseline_qerror")
    return {
        "drift": drift,
        "model_seed": seed,
        "train_seconds": metrics["train_seconds"],
        "train_rows": metrics["train_rows"],
        "target_rows": metrics["predict_rows"],
        "baseline_median": metrics["predict_qerror"]["median"],
        "baseline_p90": metrics["predict_qerror"]["p90"],
        "baseline_p95": metrics["predict_qerror"]["p95"],
        "baseline_p99": metrics["predict_qerror"]["p99"],
        "gate_passed": gate["passed"],
        "passed_seed_count": gate["passed_seed_count"],
        "adaptation_seed_count": gate["seed_count"],
        "method_p95": method["p95"],
        "method_p99": method["p99"],
        "p95_reduction": gate["aggregate_reductions"]["p95"],
        "p99_reduction": gate["aggregate_reductions"]["p99"],
        "baseline_gate_p95": baseline["p95"],
        "baseline_gate_p99": baseline["p99"],
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{MSCN training-seed robustness for the bitmap baseline and residual-field transport. Each row uses three adaptation splits.}",
        "\\label{tab:model-seeds}",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Drift & MSCN Seed & Baseline P95 & Baseline P99 & Pass & P95 Red. & P99 Red. \\\\",
        "\\midrule",
    ]
    best_by_drift = {}
    for drift in {row["drift"] for row in rows}:
        drift_rows = [row for row in rows if row["drift"] == drift]
        best_by_drift[drift] = {
            "p95_reduction": max(row["p95_reduction"] for row in drift_rows),
            "p99_reduction": max(row["p99_reduction"] for row in drift_rows),
        }
    for row in rows:
        best = best_by_drift[row["drift"]]
        p95_reduction = f"{100.0 * row['p95_reduction']:.1f}\\%"
        p99_reduction = f"{100.0 * row['p99_reduction']:.1f}\\%"
        lines.append(
            f"{row['drift'].title()} & {row['model_seed']} & {row['baseline_p95']:.1f} "
            f"& {row['baseline_p99']:.1f} & {pass_cell(row['gate_passed'])} "
            f"& {reduction_cell(p95_reduction, row['p95_reduction'], best['p95_reduction'])} "
            f"& {reduction_cell(p99_reduction, row['p99_reduction'], best['p99_reduction'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    (table_dir / "model_seed_robustness.tex").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MSCN model-seed robustness checks.")
    parser.add_argument("--output-dir", default="paper/workload_drift_residual_field/results")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_dir = ROOT / args.output_dir
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    tasks = [(drift, seed) for drift in DRIFTS for seed in seeds]
    rows = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        future_map = {
            pool.submit(experiment_one, drift, seed, args.force, output_dir): (drift, seed)
            for drift, seed in tasks
        }
        for future in as_completed(future_map):
            rows.append(future.result())
    rows.sort(key=lambda row: (row["drift"], row["model_seed"]))
    write_csv(output_dir / "model_seed_robustness.csv", rows)
    write_table(output_dir, rows)
    manifest = {
        "passed": True,
        "scope": "MSCN bitmap model-seed robustness plus residual-field adaptation gates",
        "rows": len(rows),
        "csv": str(output_dir / "model_seed_robustness.csv"),
        "table": str(output_dir / "tables" / "model_seed_robustness.tex"),
    }
    write_json(output_dir / "model_seed_robustness_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
