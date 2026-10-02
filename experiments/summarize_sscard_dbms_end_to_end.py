#!/usr/bin/env python
"""Summarize SSCard modified-PostgreSQL end-to-end reproduction results.

The upstream card files shipped with SSCard contain 416 single-table planner
slots. The patched PostgreSQL hook consumes one slot per base-relation estimate
through query_no. On our full 79-query JOB-like run, query_no reaches 700, so
the full run is a hook/runtime audit while only the prefix before slot overflow
is a valid card-file injection comparison.
"""

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
DEFAULT_RUN_DIR = RESULT_DIR / "sscard_dbms_e2e"
CARD_ROOT = ROOT / "baselines" / "SSCard" / "end_to_end" / "cards" / "like_queries_single"
DEFAULT_FULLPREFIX_MANIFEST = RESULT_DIR / "sscard_fullprefix_replay_manifest.json"


def tex_escape(value: object) -> str:
    text = str(value)
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
    )


def fmt_float(value: float | None, digits: int = 2) -> str:
    if value is None or math.isnan(value):
        return "--"
    return f"{value:.{digits}f}"


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def method_label(method: str) -> str:
    return {
        "sscard": "SSCard file",
        "pg": "PostgreSQL file",
    }.get(method, method)


def count_card_rows(card_file: str) -> int:
    path = CARD_ROOT / card_file
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return sum(1 for line in handle if line.strip())


def is_int(value: str) -> bool:
    try:
        int(value)
        return True
    except ValueError:
        return False


def summarize(rows: list[dict]) -> tuple[list[dict], dict]:
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["card_file"], row["repetition"])].append(row)

    summary_rows: list[dict] = []
    for (method, card_file, repetition), items in sorted(groups.items()):
        card_rows = count_card_rows(card_file)
        slot_covered = [
            row
            for row in items
            if is_int(row["query_no_after"]) and int(row["query_no_after"]) <= card_rows
        ]
        slot_crossing = [
            row
            for row in items
            if is_int(row["query_no_before"])
            and is_int(row["query_no_after"])
            and int(row["query_no_before"]) < card_rows < int(row["query_no_after"])
        ]
        slot_overflow = [
            row
            for row in items
            if is_int(row["query_no_before"]) and int(row["query_no_before"]) >= card_rows
        ]
        completed = [row for row in slot_covered if row["status"] == "completed"]
        timeouts = [row for row in slot_covered if row["status"] == "timeout"]
        errors = [row for row in items if row["status"] == "error"]
        exec_ms = [float(row["execution_ms"]) for row in completed if row["execution_ms"]]
        wall_ms = [float(row["wall_ms"]) for row in slot_covered if row["wall_ms"]]
        q_before = [int(row["query_no_before"]) for row in items if row["query_no_before"].isdigit()]
        q_after = [int(row["query_no_after"]) for row in items if row["query_no_after"].isdigit()]
        summary_rows.append(
            {
                "method": method,
                "label": method_label(method),
                "card_file": card_file,
                "repetition": repetition,
                "attempted_queries": len(items),
                "card_rows": card_rows,
                "slots_consumed": max(q_after) if q_after else None,
                "slot_covered_queries": len(slot_covered),
                "slot_crossing_queries": len(slot_crossing),
                "slot_overflow_queries": len(slot_overflow),
                "completed": len(completed),
                "timeouts": len(timeouts),
                "errors": len(errors),
                "completed_pct": 100.0 * len(completed) / len(slot_covered) if slot_covered else 0.0,
                "median_exec_ms_completed": statistics.median(exec_ms) if exec_ms else None,
                "p90_exec_ms_completed": percentile(exec_ms, 0.90) if exec_ms else None,
                "sum_exec_s_completed": sum(exec_ms) / 1000.0,
                "sum_wall_s_all": sum(wall_ms) / 1000.0,
                "query_no_start": min(q_before) if q_before else None,
                "query_no_end": max(q_after) if q_after else None,
            }
        )

    by_method = {row["method"]: row for row in summary_rows}
    common_completed = []
    methods = sorted(by_method)
    if "sscard" in methods and "pg" in methods:
        rows_by_query = defaultdict(dict)
        for row in rows:
            card_rows = count_card_rows(row["card_file"])
            if (
                row["status"] == "completed"
                and is_int(row["query_no_after"])
                and int(row["query_no_after"]) <= card_rows
            ):
                rows_by_query[row["query_file"]][row["method"]] = row
        for query, item in rows_by_query.items():
            if "sscard" in item and "pg" in item:
                s_exec = float(item["sscard"]["execution_ms"])
                p_exec = float(item["pg"]["execution_ms"])
                common_completed.append(
                    {
                        "query_file": query,
                        "sscard_exec_ms": s_exec,
                        "pg_exec_ms": p_exec,
                        "ratio": s_exec / p_exec if p_exec > 0 else None,
                    }
                )
    ratios = [row["ratio"] for row in common_completed if row["ratio"] is not None]
    comparison = {
        "common_completed_queries": len(common_completed),
        "median_sscard_vs_pg_completed_ratio": statistics.median(ratios) if ratios else None,
        "p90_sscard_vs_pg_completed_ratio": percentile(ratios, 0.90) if ratios else None,
        "sscard_faster_common_completed": sum(1 for row in common_completed if row["ratio"] is not None and row["ratio"] < 1.0),
    }
    return summary_rows, comparison


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - pos) + ordered[upper] * (pos - lower)


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "method",
        "label",
        "card_file",
        "repetition",
        "attempted_queries",
        "card_rows",
        "slots_consumed",
        "slot_covered_queries",
        "slot_crossing_queries",
        "slot_overflow_queries",
        "completed",
        "timeouts",
        "errors",
        "completed_pct",
        "median_exec_ms_completed",
        "p90_exec_ms_completed",
        "sum_exec_s_completed",
        "sum_wall_s_all",
        "query_no_start",
        "query_no_end",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fullprefix_rows(manifest_path: Path) -> tuple[list[dict], dict] | None:
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = []
    for row in manifest["summary"]:
        rows.append(
            {
                "label": "PG fullprefix" if row["method"] == "pg" else "SSCard fullprefix",
                "card_rows": manifest["trace_slots"],
                "slots_consumed": row["query_no_end"],
                "valid_queries": row["attempted_queries"],
                "completed": row["completed"],
                "timeouts": row["timeouts"],
                "slot_overflow_queries": 0,
                "timeout_s": int(manifest["statement_timeout_ms"]) // 1000,
            }
        )
    return rows, manifest


def write_table(
    path: Path,
    rows: list[dict],
    comparison: dict,
    timeout_ms: int,
    fullprefix: tuple[list[dict], dict] | None,
) -> None:
    table_rows = []
    for row in rows:
        table_rows.append(
            {
                "label": row["label"],
                "card_rows": row["card_rows"],
                "slots_consumed": row["slots_consumed"],
                "valid_queries": row["slot_covered_queries"],
                "completed": row["completed"],
                "timeouts": row["timeouts"],
                "slot_overflow_queries": row["slot_overflow_queries"],
                "timeout_s": timeout_ms // 1000,
            }
        )
    fullprefix_manifest = None
    if fullprefix is not None:
        fullprefix_rows_value, fullprefix_manifest = fullprefix
        table_rows.extend(fullprefix_rows_value)

    lines = [
        "\\begin{table}[!t]",
        "\\centering",
        "\\caption{SSCard modified-PostgreSQL hook coverage audit on full IMDB/JOB-like SQL. Shipped files expose the official 416-slot prefix; fullprefix files preserve those 416 values and fill slots 416--699 with traced PostgreSQL native fallback rows so the hook can replay all 79 SQL files without slot overflow.}",
        "\\label{tab:sscard-dbms-e2e}",
        "\\papertablesetup",
        "\\setlength{\\tabcolsep}{1.7pt}",
        "\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}l r r r r r r r@{}}",
        "\\toprule",
        "\\tabheader{Card source} & \\tabheader{Rows} & \\tabheader{Slots} & \\tabheader{Valid Q} & \\tabheader{Done} & \\tabheader{T/O} & \\tabheader{Ovf. Q} & \\tabheader{TO(s)} \\\\",
        "\\midrule",
    ]
    for row in table_rows:
        lines.append(
            f"{tex_escape(row['label'])} & {row['card_rows']} & {row['slots_consumed']} & "
            f"{row['valid_queries']} & {row['completed']} & {row['timeouts']} & "
            f"{row['slot_overflow_queries']} & {row['timeout_s']} \\\\"
        )
    footnote = (
        "\\multicolumn{8}{@{}p{\\columnwidth}@{}}{\\footnotesize Shipped-prefix timing uses a "
        + f"{timeout_ms // 1000}s timeout and has {comparison['common_completed_queries']} common completed covered-prefix queries, with median SSCard/PostgreSQL ratio "
        + f"{fmt_float(comparison['median_sscard_vs_pg_completed_ratio'], 2)}"
        + "$\\times$. "
    )
    if fullprefix_manifest is not None:
        footnote += (
            "Fullprefix replay uses a "
            + f"{int(fullprefix_manifest['statement_timeout_ms']) // 1000}s timeout and reaches slot {fullprefix_manifest['trace_slots']} for both files; it proves hook coverage, not a pure official SSCard latency comparison."
        )
    else:
        footnote += "No fullprefix replay manifest was provided."
    footnote += "} \\\\"
    lines.extend(
        [
            "\\midrule",
            footnote,
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
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--fullprefix-manifest", type=Path, default=DEFAULT_FULLPREFIX_MANIFEST)
    args = parser.parse_args()
    if not args.run_dir.is_absolute():
        args.run_dir = ROOT / args.run_dir
    if not args.output_dir.is_absolute():
        args.output_dir = ROOT / args.output_dir
    if not args.fullprefix_manifest.is_absolute():
        args.fullprefix_manifest = ROOT / args.fullprefix_manifest
    detail_path = args.run_dir / "sscard_dbms_e2e_detail.csv"
    run_manifest_path = args.run_dir / "sscard_dbms_e2e_run_manifest.json"
    rows = read_rows(detail_path)
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    summary_rows, comparison = summarize(rows)
    summary_csv = args.output_dir / "sscard_dbms_e2e_summary.csv"
    table_path = args.output_dir / "tables" / "sscard_dbms_e2e.tex"
    write_summary_csv(summary_csv, summary_rows)
    fullprefix = fullprefix_rows(args.fullprefix_manifest)
    write_table(table_path, summary_rows, comparison, int(run_manifest["statement_timeout_ms"]), fullprefix)
    official_slot_coverage_complete = all(
        int(row["slot_overflow_queries"]) == 0 and int(row["slots_consumed"]) <= int(row["card_rows"])
        for row in summary_rows
    )
    fullprefix_coverage_complete = bool(
        fullprefix is not None
        and all(
            int(row["query_no_end"]) == int(fullprefix[1]["trace_slots"])
            for row in fullprefix[1]["summary"]
        )
    )
    manifest = {
        "passed": bool(fullprefix_coverage_complete),
        "audit_completed": True,
        "official_slot_coverage_complete": official_slot_coverage_complete,
        "fullprefix_coverage_complete": fullprefix_coverage_complete,
        "scope": "SSCard modified-PostgreSQL DBMS hook and card-slot coverage audit",
        "reason": "The patched hook runs through the full 79-query workload. Shipped 416-slot card files do not cover the 700 consumed slots, while the prefix-preserving fullprefix replay reaches slot 700 with no overflow.",
        "run_manifest": str(run_manifest_path),
        "detail_csv": str(detail_path),
        "summary_csv": str(summary_csv),
        "table": str(table_path),
        "query_count": run_manifest["query_count"],
        "statement_timeout_ms": run_manifest["statement_timeout_ms"],
        "summary": summary_rows,
        "comparison": comparison,
        "fullprefix_manifest": str(args.fullprefix_manifest) if fullprefix is not None else "",
        "fullprefix_summary": fullprefix[1]["summary"] if fullprefix is not None else [],
    }
    manifest_path = args.output_dir / "sscard_dbms_e2e_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
