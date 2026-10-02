#!/usr/bin/env python
"""Summarize connected-subplan coverage for the PostgreSQL injection audit."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import run_workload_drift_postgres_multisubplan_injection as multi


def number(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.1f}"


def compute_rows(probe_csv: Path, drifts: list[str], ks: list[int]) -> list[dict]:
    cases = multi.collect_cases(probe_csv, drifts, 0)
    bundles = {}
    rows = []
    for k in ks:
        counts = []
        by_drift = Counter()
        for case in cases:
            key = (case["drift"], int(case["seed"]))
            if key not in bundles:
                bundles[key] = multi.load_target_bundle(*key)
            row = bundles[key]["rows"][int(case["query_index"])]
            subplan_count = len(multi.subplan_subsets(row, k))
            counts.append(subplan_count)
            by_drift[case["drift"]] += subplan_count
        rows.append(
            {
                "max_subplans": "uncapped" if k == 0 else str(k),
                "query_groups": len(cases),
                "subplan_hints": int(sum(counts)),
                "column_subplans": int(by_drift["column"]),
                "table_subplans": int(by_drift["table"]),
                "value_subplans": int(by_drift["value"]),
                "median_subplans_per_query": float(np.median(counts)) if counts else 0.0,
                "p95_subplans_per_query": float(np.quantile(counts, 0.95)) if counts else 0.0,
                "max_subplans_per_query": int(max(counts)) if counts else 0,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_table(path: Path, rows: list[dict]) -> None:
    lines = [
        "\\begin{table}[!t]",
        "\\centering",
        "\\caption{Connected-subplan coverage for the PostgreSQL deployable audit. "
        "The audit uses all eligible query groups from the executable raw-tail set; "
        "\\texttt{max\\_subplans}=32 is equivalent to uncapped connected-subplan enumeration.}",
        "\\label{tab:postgres-subplan-coverage}",
        "\\papertablesetup",
        "\\setlength{\\tabcolsep}{2.4pt}",
        "\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}l r r r r@{}}",
        "\\toprule",
        "\\tabheader{Cap} & \\tabheader{Q groups} & \\tabheader{Hints} & \\tabheader{P95/Q} & \\tabheader{Max/Q} \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['max_subplans']} "
            f"& {int(row['query_groups'])} "
            f"& {int(row['subplan_hints'])} "
            f"& {number(float(row['p95_subplans_per_query']))} "
            f"& {int(row['max_subplans_per_query'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular*}", "\\end{table}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--probe-csv",
        type=Path,
        default=ROOT / "paper" / "workload_drift_residual_field" / "results" / "duckdb_execution_probe.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "paper" / "workload_drift_residual_field" / "results",
    )
    parser.add_argument("--drifts", default="column,table,value")
    parser.add_argument("--caps", default="8,16,32,64,128,0")
    args = parser.parse_args()

    if not args.probe_csv.is_absolute():
        args.probe_csv = ROOT / args.probe_csv
    if not args.output_dir.is_absolute():
        args.output_dir = ROOT / args.output_dir

    drifts = [item.strip() for item in args.drifts.split(",") if item.strip()]
    caps = [int(item.strip()) for item in args.caps.split(",") if item.strip()]
    rows = compute_rows(args.probe_csv, drifts, caps)

    csv_path = args.output_dir / "postgres_subplan_coverage.csv"
    table_path = args.output_dir / "tables" / "postgres_subplan_coverage.tex"
    manifest_path = args.output_dir / "postgres_subplan_coverage_manifest.json"
    write_csv(csv_path, rows)
    write_table(table_path, rows)

    uncapped = next(row for row in rows if row["max_subplans"] == "uncapped")
    k32 = next(row for row in rows if row["max_subplans"] == "32")
    manifest = {
        "passed": k32["subplan_hints"] == uncapped["subplan_hints"],
        "scope": "Connected-subplan coverage audit for PostgreSQL multi-subplan injection",
        "probe_csv": str(args.probe_csv),
        "csv": str(csv_path),
        "table": str(table_path),
        "query_groups": int(uncapped["query_groups"]),
        "uncapped_connected_subplans": int(uncapped["subplan_hints"]),
        "k32_connected_subplans": int(k32["subplan_hints"]),
        "max_subplans_per_query": int(uncapped["max_subplans_per_query"]),
        "criterion": "max_subplans=32 covers the same connected-subplan set as uncapped enumeration",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
