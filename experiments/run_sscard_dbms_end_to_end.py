#!/usr/bin/env python
"""Run SSCard's PostgreSQL end-to-end estimate-injection path with auditing.

The upstream runner executes all JOB-like queries in one PostgreSQL session and
lets the patched server consume a global 416-row card file through query_no.
This script preserves that behavior, but adds per-query statement timeouts and
machine-readable logs so a long bad plan cannot hide the rest of the workload.
"""

from __future__ import annotations

import os

import argparse
import csv
import hashlib
import json
import re
import time
from datetime import datetime
from pathlib import Path

import psycopg2
from psycopg2 import errors


ROOT = Path(__file__).resolve().parents[1]
END_TO_END = ROOT / "baselines" / "SSCard" / "end_to_end"
RESULT_DIR = ROOT / "paper" / "workload_drift_residual_field" / "results"

DETAIL_FIELDS = [
    "card_file",
    "method",
    "repetition",
    "query_set",
    "query_file",
    "query_sha256",
    "status",
    "planning_ms",
    "execution_ms",
    "wall_ms",
    "query_no_before",
    "query_no_after",
    "plan_lines",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5434)
    parser.add_argument("--database", default="imdb")
    parser.add_argument("--user", default="postgres")
    parser.add_argument("--password", default=os.environ.get("PGPASSWORD"))
    parser.add_argument("--query-set", default="like_queries_single")
    parser.add_argument(
        "--card-files",
        nargs="+",
        default=["sscard_pg_single.txt", "pg_pg_single.txt"],
        help="Card files copied into the patched PostgreSQL data directory.",
    )
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--statement-timeout-ms", type=int, default=60000)
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR / "sscard_dbms_e2e")
    return parser.parse_args()


def method_name(card_file: str) -> str:
    return card_file.split("_", 1)[0]


def tex_safe_error(exc: BaseException) -> str:
    return str(exc).replace("\r", " ").replace("\n", " ").strip()[:500]


def read_query_no(cur) -> str:
    cur.execute("SHOW query_no")
    return str(cur.fetchone()[0])


def parse_time_ms(lines: list[str], prefix: str) -> float | None:
    pattern = re.compile(rf"^\s*{re.escape(prefix)}:\s+([0-9.]+)\s+ms\s*$")
    for line in lines:
        match = pattern.match(line)
        if match:
            return float(match.group(1))
    return None


def query_files(query_set: str) -> list[Path]:
    query_dir = END_TO_END / query_set
    files = sorted(query_dir.glob("*.sql"))
    if not files:
        raise FileNotFoundError(f"no SQL files under {query_dir}")
    return files


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DETAIL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def append_plan_log(path: Path, header: dict, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", errors="replace") as handle:
        handle.write(json.dumps(header, sort_keys=True) + "\n")
        for line in lines:
            handle.write(line + "\n")
        handle.write("=" * 80 + "\n")


def run_one_card(args: argparse.Namespace, card_file: str, repetition: int, files: list[Path]) -> list[dict]:
    method = method_name(card_file)
    rows: list[dict] = []
    log_path = (
        args.output_dir
        / "logs"
        / args.query_set
        / method
        / f"log_{Path(card_file).stem}_{repetition}.txt"
    )
    conn = psycopg2.connect(
        host=args.host,
        port=args.port,
        dbname=args.database,
        user=args.user,
        password=args.password,
        connect_timeout=8,
    )
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("SET ml_cardest_enabled=true")
    cur.execute("SET query_no=0")
    cur.execute("SET ml_cardest_fname=%s", (card_file,))
    cur.execute("SET statement_timeout=%s", (args.statement_timeout_ms,))
    cur.execute("SET lock_timeout=%s", (args.statement_timeout_ms,))

    for path in files:
        sql = path.read_text(encoding="utf-8", errors="replace")
        query_no_before = read_query_no(cur)
        started = time.perf_counter()
        status = "completed"
        plan_lines: list[str] = []
        error = ""
        planning_ms = None
        execution_ms = None
        try:
            cur.execute("EXPLAIN ANALYSE " + sql)
            plan_lines = [str(item[0]) for item in cur.fetchall()]
            planning_ms = parse_time_ms(plan_lines, "Planning Time")
            execution_ms = parse_time_ms(plan_lines, "Execution Time")
        except errors.QueryCanceled as exc:
            status = "timeout"
            error = tex_safe_error(exc)
            conn.rollback()
        except BaseException as exc:  # keep the full 79-query audit moving
            status = "error"
            error = tex_safe_error(exc)
            conn.rollback()
        wall_ms = (time.perf_counter() - started) * 1000.0
        try:
            query_no_after = read_query_no(cur)
        except BaseException as exc:
            query_no_after = f"unavailable: {tex_safe_error(exc)}"
            conn.rollback()
        row = {
            "card_file": card_file,
            "method": method,
            "repetition": repetition,
            "query_set": args.query_set,
            "query_file": path.name,
            "query_sha256": hashlib.sha256(sql.encode("utf-8", errors="replace")).hexdigest(),
            "status": status,
            "planning_ms": "" if planning_ms is None else f"{planning_ms:.6f}",
            "execution_ms": "" if execution_ms is None else f"{execution_ms:.6f}",
            "wall_ms": f"{wall_ms:.6f}",
            "query_no_before": query_no_before,
            "query_no_after": query_no_after,
            "plan_lines": len(plan_lines),
            "error": error,
        }
        rows.append(row)
        append_plan_log(log_path, row, plan_lines if plan_lines else ([error] if error else []))
        print(
            f"{card_file} rep={repetition} {path.name}: {status} "
            f"wall={wall_ms / 1000.0:.2f}s query_no={query_no_before}->{query_no_after}",
            flush=True,
        )

    cur.close()
    conn.close()
    return rows


def main() -> None:
    args = parse_args()
    if not args.output_dir.is_absolute():
        args.output_dir = ROOT / args.output_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = query_files(args.query_set)
    all_rows: list[dict] = []
    started_at = datetime.now().isoformat(timespec="seconds")
    for repetition in range(args.repetitions):
        for card_file in args.card_files:
            all_rows.extend(run_one_card(args, card_file, repetition, files))
            detail_path = args.output_dir / "sscard_dbms_e2e_detail.csv"
            write_rows(detail_path, all_rows)
    detail_path = args.output_dir / "sscard_dbms_e2e_detail.csv"
    write_rows(detail_path, all_rows)
    manifest = {
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "host": args.host,
        "port": args.port,
        "database": args.database,
        "query_set": args.query_set,
        "query_count": len(files),
        "card_files": args.card_files,
        "repetitions": args.repetitions,
        "statement_timeout_ms": args.statement_timeout_ms,
        "detail_csv": str(detail_path),
        "logs": str(args.output_dir / "logs"),
    }
    manifest_path = args.output_dir / "sscard_dbms_e2e_run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
