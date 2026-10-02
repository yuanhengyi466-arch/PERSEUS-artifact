#!/usr/bin/env python
"""Summarize local official-entry SSCard reproduction artifacts."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SSCARD_ROOT = ROOT / "baselines" / "SSCard"
RESULT_DIR = ROOT / "paper" / "workload_drift_residual_field" / "results"
DATASETS = ["DBLP_AN", "IMDB_AN", "IMDB_MT", "TPCH_PN", "WIKI"]
EXP_NAME = "SSCard_buc1_h3_l5000_e32_spline_load_L_final"


def parse_key_values(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue
        if line.startswith("Percentile: "):
            key, value = line.rsplit(":", 1)
        else:
            key, value = line.split(":", 1)
        out[key.strip()] = value.strip()
    return out


def parse_seconds(text: str) -> float:
    parts = [float(item) for item in text.split(":")]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0]


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return sum(1 for _ in handle)


def load_dataset(dataset: str) -> dict:
    exp = SSCARD_ROOT / "sscard" / "exp" / dataset / EXP_NAME
    data_dir = SSCARD_ROOT / "datasets" / dataset
    args_path = exp / "args_build.txt"
    build_path = exp / "building_results.txt"
    query_path = exp / "query_results_final.txt"
    analysis_path = exp / "analyses.csv"
    for path in [args_path, build_path, query_path, analysis_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    build = parse_key_values(build_path)
    query = parse_key_values(query_path)
    percentile = ast.literal_eval(query["Percentile: [0.5, 0.7, 0.9, 0.99]"])
    command = args_path.read_text(encoding="utf-8", errors="replace").strip()
    return {
        "dataset": dataset,
        "command": command,
        "train_rows": count_lines(data_dir / f"{dataset}.csv"),
        "test_rows": count_lines(data_dir / f"{dataset}_test.csv") - 1,
        "data_len": int(float(build.get("data len0", 0))),
        "tree_nodes": int(float(build.get("Tree node0", 0))),
        "model_mb": float(build.get("Model size0", 0)),
        "build_seconds": parse_seconds(build["Building time"]),
        "query_seconds": parse_seconds(query["Query time"]),
        "p50_qerror": float(percentile[0]),
        "p90_qerror": float(percentile[2]),
        "p99_qerror": float(percentile[3]),
        "max_qerror": float(query["Max q-error"]),
        "avg_query_ms": float(query["Avg. query time(s)"]) * 1000.0,
        "analysis_queries": max(0, count_lines(analysis_path) - 1),
        "analysis_p99_qerror": float(percentile[3]),
    }


def number(value: float | int, decimals: int = 1) -> str:
    value = float(value)
    if value >= 1000:
        return f"{value:.0f}"
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.{decimals}f}"


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
        "\\caption{Local SSCard official loaded-L query-path reproduction using the repository's \\texttt{sscard/src/run.py} with \\texttt{load\\_L=True}.}",
        "\\label{tab:sscard-official-repro}",
        "\\papertablesetup",
        "\\setlength{\\tabcolsep}{2.0pt}",
        "\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}l|r r|r r r@{}}",
        "\\toprule",
        "\\tabheader{Dataset} & \\tabheader{Train} & \\tabheader{Test} & \\tabheader{Build/load} & \\tabheader{P90/P99 Q} & \\tabheader{Avg ms} \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['dataset'].replace('_', '\\_')} "
            f"& {number(row['train_rows'], 0)} "
            f"& {number(row['test_rows'], 0)} "
            f"& {number(float(row['build_seconds']) / 60.0, 1)}m "
            f"& {number(row['p90_qerror'], 1)}/{number(row['p99_qerror'], 1)} "
            f"& {number(row['avg_query_ms'], 3)} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular*}", "\\end{table}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    args = parser.parse_args()
    if not args.output_dir.is_absolute():
        args.output_dir = ROOT / args.output_dir
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    rows = [load_dataset(dataset) for dataset in datasets]
    csv_path = args.output_dir / "sscard_official_reproduction.csv"
    table_path = args.output_dir / "tables" / "sscard_official_reproduction.tex"
    manifest_path = args.output_dir / "sscard_official_reproduction_manifest.json"
    write_csv(csv_path, rows)
    write_table(table_path, rows)
    manifest = {
        "passed": True,
        "scope": "Local official loaded-L SSCard query-path reproduction summary",
        "datasets": datasets,
        "exp_name": EXP_NAME,
        "csv": str(csv_path),
        "table": str(table_path),
        "source_root": str(SSCARD_ROOT),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
