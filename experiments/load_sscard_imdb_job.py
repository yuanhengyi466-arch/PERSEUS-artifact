#!/usr/bin/env python
"""Load the JOB/IMDB CSV release into the patched PostgreSQL instance."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV_DIR = ROOT / "external_data" / "imdb_job" / "csv"
DEFAULT_PSQL = ROOT / "tools" / "patched_pg13_install" / "bin" / "psql.exe"
DEFAULT_RESULTS = ROOT / "paper" / "workload_drift_residual_field" / "results"
TABLES = [
    "aka_name",
    "aka_title",
    "cast_info",
    "char_name",
    "company_name",
    "company_type",
    "comp_cast_type",
    "complete_cast",
    "info_type",
    "keyword",
    "kind_type",
    "link_type",
    "movie_companies",
    "movie_info",
    "movie_info_idx",
    "movie_keyword",
    "movie_link",
    "name",
    "person_info",
    "role_type",
    "title",
]


def run_psql(psql: Path, host: str, port: int, db: str, sql: str, log: Path) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    command = [
        str(psql),
        "-h",
        host,
        "-p",
        str(port),
        "-U",
        "postgres",
        "-d",
        db,
        "-v",
        "ON_ERROR_STOP=1",
        "-c",
        sql,
    ]
    started = time.time()
    proc = subprocess.run(command, text=True, capture_output=True, check=False, env=env)
    elapsed = time.time() - started
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n-- elapsed={elapsed:.3f}s db={db} sql={sql[:220]}\n")
        handle.write(proc.stdout)
        handle.write(proc.stderr)
    return proc


def sql_literal(path: Path) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-dir", type=Path, default=DEFAULT_CSV_DIR)
    parser.add_argument("--psql", type=Path, default=DEFAULT_PSQL)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5434)
    parser.add_argument("--database", default="imdb")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--skip-indexes", action="store_true")
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    log = args.results_dir / "sscard_imdb_load.log"
    manifest_path = args.results_dir / "sscard_imdb_load_manifest.json"
    log.write_text("", encoding="utf-8")

    missing = [table for table in TABLES if not (args.csv_dir / f"{table}.csv").exists()]
    schema = args.csv_dir / "schematext.sql"
    if missing or not schema.exists():
        raise SystemExit(f"missing inputs: schema={schema.exists()} missing_tables={missing}")

    manifest: dict[str, object] = {
        "host": args.host,
        "port": args.port,
        "database": args.database,
        "csv_dir": str(args.csv_dir),
        "tables": TABLES,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "steps": [],
    }

    def step(name: str, db: str, sql: str) -> None:
        started = time.time()
        proc = run_psql(args.psql, args.host, args.port, db, sql, log)
        elapsed = time.time() - started
        manifest["steps"].append({"name": name, "returncode": proc.returncode, "elapsed_s": elapsed})
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if proc.returncode != 0:
            raise SystemExit(f"{name} failed; see {log}")

    step("drop connections", "postgres", f"SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '{args.database}' AND pid <> pg_backend_pid();")
    step("drop database", "postgres", f"DROP DATABASE IF EXISTS {args.database};")
    step("create database", "postgres", f"CREATE DATABASE {args.database};")
    step("tune session", args.database, "SET maintenance_work_mem='1GB'; SET synchronous_commit=off;")
    step("schema", args.database, schema.read_text(encoding="utf-8", errors="replace"))

    for table in TABLES:
        csv_path = args.csv_dir / f"{table}.csv"
        copy_sql = rf"\copy {table} FROM '{sql_literal(csv_path)}' WITH (FORMAT csv, DELIMITER ',', QUOTE E'\"', ESCAPE E'\\')"
        step(f"copy {table}", args.database, copy_sql)

    step("analyze", args.database, "ANALYZE;")
    if not args.skip_indexes:
        index_sql = (ROOT / "baselines" / "End-to-End-CardEst-Benchmark" / "scripts" / "sql" / "imdb_index.sql").read_text(
            encoding="utf-8",
            errors="replace",
        )
        step("indexes", args.database, index_sql)
        step("analyze after indexes", args.database, "ANALYZE;")

    count_sql = " UNION ALL ".join(f"SELECT '{table}' AS table_name, count(*)::bigint AS n FROM {table}" for table in TABLES)
    step("row counts", args.database, count_sql)
    manifest["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    manifest["passed"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
