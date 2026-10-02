#!/usr/bin/env python
"""Trace base-relation estimate slots consumed by the SSCard PostgreSQL hook.

The SSCard end-to-end hook consumes one number per call to
set_baserel_size_estimates().  The shipped card files contain 416 rows, but the
local PostgreSQL planner can consume a different number of slots.  This script
uses the hook's print_single_tbl_queries GUC to record the actual slot sequence
without executing the queries.
"""

from __future__ import annotations

import os

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path

import psycopg2


ROOT = Path(__file__).resolve().parents[1]
END_TO_END = ROOT / "baselines" / "SSCard" / "end_to_end"
DATA_DIR = ROOT / "tools" / "patched_pg13_data_sscard"
RESULT_DIR = ROOT / "paper" / "workload_drift_residual_field" / "results"


FIELDS = [
    "slot",
    "query_file",
    "rel_aliases",
    "rows",
    "width",
    "baserestrictinfo",
    "has_like",
    "has_non_like_filter",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5434)
    parser.add_argument("--database", default="imdb")
    parser.add_argument("--user", default="postgres")
    parser.add_argument("--password", default=os.environ.get("PGPASSWORD"))
    parser.add_argument("--query-set", default="like_queries_single")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR / "sscard_slot_trace")
    return parser.parse_args()


def query_files(query_set: str) -> list[Path]:
    files = sorted((END_TO_END / query_set).glob("*.sql"))
    if not files:
        raise FileNotFoundError(f"no SQL files under {END_TO_END / query_set}")
    return files


def read_query_no(cur) -> int:
    cur.execute("SHOW query_no")
    return int(cur.fetchone()[0])


def parse_record(path: Path, intervals: list[dict]) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    by_slot_file = {}
    for item in intervals:
        for slot in range(int(item["query_no_before"]), int(item["query_no_after"])):
            by_slot_file[slot] = item["query_file"]

    rows = []
    for block in blocks:
        lines = [line.rstrip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        slot_match = re.match(r"query:\s+(\d+)$", lines[0])
        rel_match = re.match(r"RELOPTINFO\s+\((.*?)\):\s+rows=([0-9.]+)\s+width=(\d+)", lines[1] if len(lines) > 1 else "")
        if not slot_match or not rel_match:
            continue
        slot = int(slot_match.group(1))
        restrict = ""
        for line in lines[2:]:
            if "baserestrictinfo:" in line:
                restrict = line.split("baserestrictinfo:", 1)[1].strip()
                break
        rows.append(
            {
                "slot": slot,
                "query_file": by_slot_file.get(slot, ""),
                "rel_aliases": rel_match.group(1),
                "rows": rel_match.group(2),
                "width": rel_match.group(3),
                "baserestrictinfo": restrict,
                "has_like": "~~" in restrict,
                "has_non_like_filter": bool(restrict and "~~" not in restrict),
            }
        )
    rows.sort(key=lambda row: row["slot"])
    return rows


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not args.data_dir.is_absolute():
        args.data_dir = ROOT / args.data_dir
    if not args.output_dir.is_absolute():
        args.output_dir = ROOT / args.output_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    record_path = args.data_dir / "single_tbl_est_record.txt"
    record_path.write_text("", encoding="utf-8")

    files = query_files(args.query_set)
    intervals = []
    with psycopg2.connect(
        host=args.host,
        port=args.port,
        dbname=args.database,
        user=args.user,
        password=args.password,
        connect_timeout=8,
    ) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("SET print_single_tbl_queries=true")
            cur.execute("SET ml_cardest_enabled=false")
            cur.execute("SET query_no=0")
            for path in files:
                before = read_query_no(cur)
                sql = path.read_text(encoding="utf-8", errors="replace")
                cur.execute("EXPLAIN " + sql)
                cur.fetchall()
                after = read_query_no(cur)
                intervals.append(
                    {
                        "query_file": path.name,
                        "query_no_before": before,
                        "query_no_after": after,
                        "slots": after - before,
                    }
                )
                print(f"{path.name}: {before}->{after}", flush=True)

    rows = parse_record(record_path, intervals)
    detail_csv = args.output_dir / "sscard_base_slot_trace.csv"
    interval_csv = args.output_dir / "sscard_base_slot_intervals.csv"
    manifest_path = args.output_dir / "sscard_base_slot_trace_manifest.json"
    write_csv(detail_csv, rows, FIELDS)
    write_csv(interval_csv, intervals, ["query_file", "query_no_before", "query_no_after", "slots"])

    like_slots = sum(1 for row in rows if row["has_like"])
    non_like_filter_slots = sum(1 for row in rows if row["has_non_like_filter"])
    manifest = {
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "query_set": args.query_set,
        "query_count": len(files),
        "slot_count": len(rows),
        "query_no_end": intervals[-1]["query_no_after"] if intervals else 0,
        "like_slots": like_slots,
        "non_like_filter_slots": non_like_filter_slots,
        "unfiltered_slots": len(rows) - like_slots - non_like_filter_slots,
        "detail_csv": str(detail_csv),
        "interval_csv": str(interval_csv),
        "record_path": str(record_path),
        "passed": bool(rows and len(rows) == (intervals[-1]["query_no_after"] if intervals else -1)),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
