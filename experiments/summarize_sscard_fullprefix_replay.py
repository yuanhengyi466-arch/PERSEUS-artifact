#!/usr/bin/env python
"""Summarize prefix-preserving SSCard full-hook replay results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "paper" / "workload_drift_residual_field" / "results"
DEFAULT_RUN_DIR = RESULT_DIR / "sscard_dbms_fullprefix_20s"
DEFAULT_CARD_MANIFEST = RESULT_DIR / "sscard_fullprefix_cards" / "sscard_fullprefix_cards_manifest.json"


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - pos) + ordered[upper] * (pos - lower)


def method_label(method: str) -> str:
    return {
        "sscard": "SSCard prefix",
        "pg": "PG prefix",
    }.get(method, method)


def fmt(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "--"
    return f"{value:.{digits}f}"


def summarize(rows: list[dict]) -> tuple[list[dict], dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["method"]].append(row)

    summary = []
    for method, items in sorted(groups.items()):
        completed = [row for row in items if row["status"] == "completed" and row["execution_ms"]]
        exec_ms = [float(row["execution_ms"]) for row in completed]
        q_after = [int(row["query_no_after"]) for row in items if row["query_no_after"].isdigit()]
        summary.append(
            {
                "method": method,
                "label": method_label(method),
                "attempted_queries": len(items),
                "completed": len(completed),
                "timeouts": sum(1 for row in items if row["status"] == "timeout"),
                "errors": sum(1 for row in items if row["status"] == "error"),
                "query_no_end": max(q_after) if q_after else None,
                "median_exec_ms_completed": statistics.median(exec_ms) if exec_ms else None,
                "p90_exec_ms_completed": percentile(exec_ms, 0.90),
            }
        )

    by_query: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        if row["status"] == "completed" and row["execution_ms"]:
            by_query[row["query_file"]][row["method"]] = float(row["execution_ms"])
    ratios = [
        item["sscard"] / item["pg"]
        for item in by_query.values()
        if "sscard" in item and "pg" in item and item["pg"] > 0
    ]
    comparison = {
        "common_completed_queries": len(ratios),
        "median_sscard_vs_pg_ratio": statistics.median(ratios) if ratios else None,
        "p90_sscard_vs_pg_ratio": percentile(ratios, 0.90),
        "sscard_faster_common_completed": sum(1 for value in ratios if value < 1.0),
    }
    return summary, comparison


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_table(path: Path, rows: list[dict], comparison: dict, timeout_ms: int) -> None:
    lines = [
        "\\begin{table}[!t]",
        "\\centering",
        "\\caption{SSCard full-hook replay with prefix-preserving complete card files. The first 416 slots are the shipped SSCard/PostgreSQL files; slots 416--699 use traced PostgreSQL native base-relation fallback so the 79-query hook run no longer reads past the card file.}",
        "\\label{tab:sscard-fullprefix-replay}",
        "\\papertablesetup",
        "\\setlength{\\tabcolsep}{2.2pt}",
        "\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}l r r r r r@{}}",
        "\\toprule",
        "\\tabheader{Card source} & \\tabheader{Q} & \\tabheader{Done} & \\tabheader{T/O} & \\tabheader{End slot} & \\tabheader{Med. ms} \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['label']} & {row['attempted_queries']} & {row['completed']} & "
            f"{row['timeouts']} & {row['query_no_end']} & "
            f"{fmt(row['median_exec_ms_completed'], 1)} \\\\"
        )
    lines.extend(
        [
            "\\midrule",
            "\\multicolumn{6}{@{}p{\\columnwidth}@{}}{\\footnotesize Common completed queries: "
            + f"{comparison['common_completed_queries']}; median SSCard-prefix/PG-prefix ratio: "
            + f"{fmt(comparison['median_sscard_vs_pg_ratio'], 2)}"
            + "$\\times$; SSCard-prefix faster on "
            + f"{comparison['sscard_faster_common_completed']} queries. Timeout: "
            + f"{timeout_ms // 1000}s"
            + ".} \\\\",
            "\\bottomrule",
            "\\end{tabular*}",
            "\\end{table}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--card-manifest", type=Path, default=DEFAULT_CARD_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    args = parser.parse_args()

    for attr in ["run_dir", "card_manifest", "output_dir"]:
        value = getattr(args, attr)
        if not value.is_absolute():
            setattr(args, attr, ROOT / value)

    run_manifest_path = args.run_dir / "sscard_dbms_e2e_run_manifest.json"
    detail_path = args.run_dir / "sscard_dbms_e2e_detail.csv"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    card_manifest = json.loads(args.card_manifest.read_text(encoding="utf-8"))
    rows = read_rows(detail_path)
    summary, comparison = summarize(rows)

    summary_csv = args.output_dir / "sscard_fullprefix_replay_summary.csv"
    table_path = args.output_dir / "tables" / "sscard_fullprefix_replay.tex"
    manifest_path = args.output_dir / "sscard_fullprefix_replay_manifest.json"
    write_csv(summary_csv, summary)
    write_table(table_path, summary, comparison, int(run_manifest["statement_timeout_ms"]))

    manifest = {
        "passed": all(row["query_no_end"] == card_manifest["trace_slots"] for row in summary),
        "scope": "Prefix-preserving complete SSCard modified-PostgreSQL hook replay",
        "run_manifest": str(run_manifest_path),
        "card_manifest": str(args.card_manifest),
        "detail_csv": str(detail_path),
        "summary_csv": str(summary_csv),
        "table": str(table_path),
        "query_count": run_manifest["query_count"],
        "statement_timeout_ms": run_manifest["statement_timeout_ms"],
        "trace_slots": card_manifest["trace_slots"],
        "shipped_prefix_rows": card_manifest["outputs"][0]["shipped_prefix_rows"],
        "native_fallback_rows": card_manifest["outputs"][0]["native_fallback_rows"],
        "summary": summary,
        "comparison": comparison,
        "interpretation": "complete hook coverage with prefix-preserving native fallback; not an official pure-SSCard 700-slot artifact",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
