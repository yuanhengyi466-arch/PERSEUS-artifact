#!/usr/bin/env python
"""Summarize the local readiness of SSCard's PostgreSQL end-to-end path."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SSCARD_ROOT = ROOT / "baselines" / "SSCard"
UPSTREAM_E2E_ROOT = ROOT / "baselines" / "End-to-End-CardEst-Benchmark"
RESULT_DIR = ROOT / "paper" / "workload_drift_residual_field" / "results"
END_TO_END = SSCARD_ROOT / "end_to_end"


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return sum(1 for _ in handle)


def port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


def bool_text(value: bool) -> str:
    return "yes" if value else "no"


def tex_escape(value: str) -> str:
    return value.replace("\\", "\\textbackslash{}").replace("_", "\\_")


def command_ok(command: list[str], cwd: Path) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=8,
            check=False,
        )
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if proc.returncode == 0:
        return True, "ok"
    message = (proc.stderr or proc.stdout or "").strip().splitlines()
    return False, message[-1][:80] if message else f"exit {proc.returncode}"


def psql_bin() -> Path | None:
    candidates = [
        ROOT / "tools" / "patched_pg13_install" / "bin" / "psql.exe",
        Path(r"C:\Program Files\PostgreSQL\18\bin\psql.exe"),
        Path(r"C:\Program Files\PostgreSQL\17\bin\psql.exe"),
        Path(r"C:\Program Files\PostgreSQL\16\bin\psql.exe"),
        Path(r"C:\Program Files\PostgreSQL\15\bin\psql.exe"),
    ]
    return first_existing(candidates)


def first_existing(paths: list[Path]) -> Path | None:
    return next((path for path in paths if path.exists()), None)


def msbuild_probe() -> tuple[str, str]:
    source_dir = UPSTREAM_E2E_ROOT / "postgresql-13.1"
    project = source_dir / "postgres.vcxproj"
    msbuild = first_existing(
        [
            Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Current\Bin\MSBuild.exe"),
            Path(r"C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe"),
            Path(r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\MSBuild\Current\Bin\MSBuild.exe"),
        ]
    )
    installed = ROOT / "tools" / "patched_pg13_install" / "bin" / "postgres.exe"
    if installed.exists():
        return "yes", "patched postgres.exe installed"
    if not project.exists():
        return "no", "postgres.vcxproj absent"
    if msbuild is None:
        return "no", "MSBuild.exe not found"
    v142_log = RESULT_DIR / "patched_pg13_postgres_msbuild_fake.log"
    v143_log = RESULT_DIR / "patched_pg13_postgres_msbuild_v143.log"
    messages = []
    for label, log in [("v142", v142_log), ("v143 override", v143_log)]:
        if log.exists():
            text = log.read_text(encoding="utf-8", errors="replace")
            if "error MSB8020" in text:
                messages.append(f"{label}: platform toolset missing")
            elif "error " in text.lower():
                tail = [line.strip() for line in text.splitlines() if "error" in line.lower()]
                messages.append(f"{label}: {tail[-1][:70] if tail else 'build error'}")
            elif "postgres.exe" in text or (source_dir / "Release" / "postgres" / "postgres.exe").exists():
                return "yes", "postgres.exe produced by MSBuild"
    if messages:
        return "no", "; ".join(messages)
    return "partial", "MSBuild available; postgres target not built in this audit"


def windows_build_chain_status() -> tuple[str, str]:
    vs_dev = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat")
    vcvars = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat")
    msvc_root = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC")
    msys_bin = Path(r"C:\msys64\usr\bin")
    conda_patch = Path(r"H:\Anaconda\Library\usr\bin\patch.exe")

    msvc_versions = list(msvc_root.glob("*")) if msvc_root.exists() else []
    found = {
        "VS BuildTools": vs_dev.exists() or vcvars.exists() or msvc_root.exists(),
        "perl": (msys_bin / "perl.exe").exists() or shutil.which("perl") is not None,
        "flex": (msys_bin / "flex.exe").exists() or shutil.which("flex") is not None,
        "bison": (msys_bin / "bison.exe").exists() or shutil.which("bison") is not None,
        "patch": (msys_bin / "patch.exe").exists() or conda_patch.exists() or shutil.which("patch") is not None,
    }
    missing = [name for name, ok in found.items() if not ok]
    installed = ROOT / "tools" / "patched_pg13_install" / "bin" / "postgres.exe"
    server_log = RESULT_DIR / "patched_pg13_server.log"
    if installed.exists():
        return "yes", f"VS BuildTools/MSYS2/Strawberry Perl built patched PG13; server log {server_log.name}"
    if not missing:
        versions = ",".join(path.name for path in msvc_versions) or "unknown"
        return "yes", f"VS BuildTools/MSYS2 found; MSVC {versions}; patched server not running"
    status = "partial" if any(found.values()) else "no"
    evidence = "; ".join(
        f"{name} {'found' if ok else 'missing'}" for name, ok in found.items()
    )
    return status, evidence


def patched_pg_source_status() -> tuple[str, str]:
    source_dir = UPSTREAM_E2E_ROOT / "postgresql-13.1"
    patch_marker = source_dir / ".codex_patch_applied"
    solution = source_dir / "pgsql.sln"
    installed = ROOT / "tools" / "patched_pg13_install" / "bin" / "postgres.exe"
    data_dir = ROOT / "tools" / "patched_pg13_data_sscard"
    if installed.exists() and (data_dir / "PG_VERSION").exists():
        return "yes", "patch applied; full solution built; installed; data dir initialized"
    smoke_error = RESULT_DIR / "patched_pg13_msbuild_direct.err"
    if patch_marker.exists() and solution.exists():
        build_status, build_evidence = msbuild_probe()
        if build_status == "yes":
            return "partial", "patch applied; pgsql.sln generated; server built but not installed/running"
        if build_evidence:
            return "partial", f"patch applied; pgsql.sln generated; {build_evidence}"
        if smoke_error.exists():
            return "partial", "patch applied; pgsql.sln generated; MSBuild smoke did not produce hook server"
        return "partial", "patch applied; pgsql.sln generated; server not installed/running"
    if source_dir.exists():
        return "partial", "PostgreSQL-13.1 source extracted; patch/build incomplete"
    return "no", "PostgreSQL-13.1 source not prepared"


def count_static_like_predicates(query_dir: Path) -> int:
    total = 0
    if not query_dir.exists():
        return 0
    for path in query_dir.glob("*.sql"):
        text = path.read_text(encoding="utf-8", errors="replace")
        if "WHERE" not in text:
            continue
        where = text.split("WHERE", 1)[1]
        for part in re.split(r"\bAND\b", where, flags=re.I):
            if re.search(r"\bNOT\s+LIKE\b|\bLIKE\b", part, flags=re.I):
                total += len(re.findall(r"\bNOT\s+LIKE\b|\bLIKE\b", part, flags=re.I))
    return total


def postgres_probe(host: str, port: int, database: str, statements: list[str]) -> tuple[bool, str]:
    psql = psql_bin()
    if psql is not None:
        commands: list[str] = []
        for stmt in statements:
            commands.extend(["-c", stmt])
        env = dict(**os.environ)
        proc = subprocess.run(
            [
                str(psql),
                "-h",
                host,
                "-p",
                str(port),
                "-U",
                "postgres",
                "-d",
                database,
                "-v",
                "ON_ERROR_STOP=1",
                *commands,
            ],
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=8,
            check=False,
            env=env,
        )
        if proc.returncode == 0:
            return True, "ok"
        message = (proc.stderr or proc.stdout or "").strip().splitlines()
        return False, message[-1][:80] if message else f"exit {proc.returncode}"

    code = """
import sys
import psycopg
host, port, database = sys.argv[1], int(sys.argv[2]), sys.argv[3]
statements = sys.argv[4:]
try:
    conn = psycopg.connect(host=host, port=port, dbname=database, user='postgres', password=os.environ.get("PGPASSWORD"), connect_timeout=2)
    cur = conn.cursor()
    for stmt in statements:
        cur.execute(stmt)
    cur.close()
    conn.close()
    print('ok')
except Exception as exc:
    print(f'{type(exc).__name__}: {exc}')
    sys.exit(1)
"""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code, host, str(port), database, *statements],
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=8,
            check=False,
        )
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    output = (proc.stdout or proc.stderr or "").strip().splitlines()
    message = output[-1][:80] if output else f"exit {proc.returncode}"
    return proc.returncode == 0, message


def has_database(host: str, port: int, database: str) -> bool:
    psql = psql_bin()
    if psql is not None:
        env = dict(**os.environ)
        proc = subprocess.run(
            [
                str(psql),
                "-h",
                host,
                "-p",
                str(port),
                "-U",
                "postgres",
                "-d",
                "postgres",
                "-At",
                "-c",
                f"SELECT 1 FROM pg_database WHERE datname = '{database}'",
            ],
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=8,
            check=False,
            env=env,
        )
        return proc.returncode == 0 and "1" in proc.stdout.splitlines()

    code = """
import sys
import psycopg
host, port, database = sys.argv[1], int(sys.argv[2]), sys.argv[3]
try:
    conn = psycopg.connect(host=host, port=port, dbname='postgres', user='postgres', password=os.environ.get("PGPASSWORD"), connect_timeout=2)
    cur = conn.cursor()
    cur.execute('SELECT 1 FROM pg_database WHERE datname = %s', (database,))
    ok = cur.fetchone() is not None
    cur.close()
    conn.close()
    sys.exit(0 if ok else 1)
except Exception:
    sys.exit(1)
"""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code, host, str(port), database],
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=8,
            check=False,
        )
    except Exception:
        return False
    return proc.returncode == 0


def readiness_rows(host: str) -> list[dict]:
    cards = END_TO_END / "cards" / "like_queries_single"
    query_dir = END_TO_END / "like_queries_single"
    columns_dir = END_TO_END / "columns"
    pg_file = cards / "pg_pg_single.txt"
    sscard_file = cards / "sscard_pg_single.txt"
    predicates_file = cards / "predicates_single.txt"
    required_columns = [
        "aka_name.name",
        "cast_info.note",
        "char_name.name",
        "comp_cast_type.kind",
        "company_name.name",
        "keyword.keyword",
        "link_type.link",
        "movie_companies.note",
        "movie_info.note",
        "name.name",
        "title.title",
    ]
    present_columns = [
        name
        for name in required_columns
        if (columns_dir / name / f"{name}.txt").exists()
    ]
    missing_columns = [name for name in required_columns if name not in present_columns]
    sql_files = len(list(query_dir.glob("*.sql"))) if query_dir.exists() else 0
    sscard_rows = count_lines(sscard_file)
    pg_rows = count_lines(pg_file)
    static_like_predicates = count_static_like_predicates(query_dir)
    build_script = END_TO_END / "sscard" / "src" / "run.sh"
    cal_sel = END_TO_END / "sscard" / "src" / "cal_sel.py"
    upstream_patch = UPSTREAM_E2E_ROOT / "benchmark.patch"
    upstream_builder = UPSTREAM_E2E_ROOT / "benchmark_builder.sh"
    build_status, build_evidence = windows_build_chain_status()
    patched_pg_status, patched_pg_evidence = patched_pg_source_status()
    can_build_all = len(missing_columns) == 0
    can_generate_cards = can_build_all and predicates_file.exists()
    vanilla_pg = port_open(host, 5432)
    hook_port = port_open(host, 5434)
    imdb_db_vanilla = has_database(host, 5432, "imdb") if vanilla_pg else False
    imdb_db_hook = has_database(host, 5434, "imdb") if hook_port else False
    imdb_db = imdb_db_hook or imdb_db_vanilla
    vanilla_gucs = False
    if vanilla_pg:
        vanilla_gucs, _vanilla_guc_message = postgres_probe(
            host,
            5432,
            "postgres",
            ["SET ml_cardest_enabled=true"],
        )
    hook_gucs = False
    hook_message = f"{host}:5434 closed"
    if hook_port:
        hook_probe_db = "imdb" if imdb_db_hook else "postgres"
        hook_gucs, hook_message = postgres_probe(
            host,
            5434,
            hook_probe_db,
            [
                "SET ml_cardest_enabled=true",
                "SET query_no=0",
                "SET ml_cardest_fname='sscard_pg_single.txt'",
            ],
        )
    can_execute_pg = hook_port and hook_gucs and imdb_db_hook and sscard_file.exists() and pg_file.exists()
    return [
        {
            "component": "JOB-like SQL",
            "local_status": bool_text(sql_files == 79),
            "evidence": f"{sql_files}/79 SQL files",
            "blocks_dbms_e2e": "no",
        },
        {
            "component": "string-column files",
            "local_status": bool_text(can_build_all),
            "evidence": f"{len(present_columns)}/11 present; missing {', '.join(missing_columns) if missing_columns else 'none'}",
            "blocks_dbms_e2e": "yes" if missing_columns else "no",
        },
        {
            "component": "LIKE predicate file",
            "local_status": bool_text(predicates_file.exists()),
            "evidence": (
                f"416-slot planner order; SQL text has {static_like_predicates} LIKE tokens, "
                "so static parsing is insufficient"
            ),
            "blocks_dbms_e2e": "yes" if not predicates_file.exists() else "no",
        },
        {
            "component": "SSCard card file",
            "local_status": bool_text(sscard_file.exists()),
            "evidence": f"{sscard_rows}/416 estimate rows",
            "blocks_dbms_e2e": "no",
        },
        {
            "component": "PostgreSQL card file",
            "local_status": bool_text(pg_file.exists()),
            "evidence": f"{pg_rows}/416 estimate rows",
            "blocks_dbms_e2e": "no",
        },
        {
            "component": "Vanilla PostgreSQL",
            "local_status": bool_text(vanilla_pg),
            "evidence": f"{host}:5432",
            "blocks_dbms_e2e": "no",
        },
        {
            "component": "IMDB database",
            "local_status": bool_text(imdb_db),
            "evidence": (
                "present on patched 5434"
                if imdb_db_hook
                else ("present on vanilla 5432 only" if imdb_db_vanilla else "required by run_job.py")
            ),
            "blocks_dbms_e2e": "yes" if not imdb_db else "no",
        },
        {
            "component": "Modified PostgreSQL hook",
            "local_status": bool_text(hook_port and hook_gucs),
            "evidence": (
                "custom CE GUCs ok"
                if hook_gucs
                else f"{hook_message}; vanilla PG GUC {'ok' if vanilla_gucs else 'absent'}"
            ),
            "blocks_dbms_e2e": "yes" if not (hook_port and hook_gucs) else "no",
        },
        {
            "component": "Upstream PG patch",
            "local_status": "manual" if upstream_patch.exists() else "no",
            "evidence": (
                "PostgreSQL-13.1 source patch/Docker path inspected"
                if upstream_patch.exists() and upstream_builder.exists()
                else "not cloned"
            ),
            "blocks_dbms_e2e": "no",
        },
        {
            "component": "Patched PG source",
            "local_status": patched_pg_status,
            "evidence": patched_pg_evidence,
            "blocks_dbms_e2e": "yes" if patched_pg_status == "no" else "no",
        },
        {
            "component": "Windows build chain",
            "local_status": build_status,
            "evidence": build_evidence,
            "blocks_dbms_e2e": "yes" if build_status == "no" else "no",
        },
        {
            "component": "README data-dir install",
            "local_status": bool_text(can_execute_pg),
            "evidence": "cards available to patched data dir" if can_execute_pg else "copy cards to data dir",
            "blocks_dbms_e2e": "yes" if not can_execute_pg else "no",
        },
        {
            "component": "Build command",
            "local_status": bool_text(can_build_all),
            "evidence": str(build_script.relative_to(END_TO_END)),
            "blocks_dbms_e2e": "yes" if not can_build_all else "no",
        },
        {
            "component": "Card command",
            "local_status": bool_text(can_generate_cards),
            "evidence": str(cal_sel.relative_to(END_TO_END)),
            "blocks_dbms_e2e": "yes" if not can_generate_cards else "no",
        },
    ]


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
        "\\caption{Readiness audit for SSCard's PostgreSQL end-to-end path. The local Windows hook path is runnable with the shipped 416-row files; the remaining blocker is regenerating a pure official 700-slot SSCard card file because the predicate-order input and end-to-end SSCard models are not tracked in the cloned snapshot.}",
        "\\label{tab:sscard-e2e-readiness}",
        "\\scriptsize\\setlength{\\tabcolsep}{1.6pt}\\renewcommand{\\arraystretch}{1.0}",
        "\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}L{0.31\\columnwidth} c L{0.35\\columnwidth} c@{}}",
        "\\toprule",
        "\\tabheader{Component} & \\tabheader{Local} & \\tabheader{Evidence} & \\tabheader{Official regen} \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{tex_escape(row['component'])} & {tex_escape(row['local_status'])} & "
            f"{tex_escape(row['evidence'])} & {tex_escape(row['blocks_dbms_e2e'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular*}", "\\end{table}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    args = parser.parse_args()
    if not args.output_dir.is_absolute():
        args.output_dir = ROOT / args.output_dir
    rows = readiness_rows(args.host)
    csv_path = args.output_dir / "sscard_end_to_end_readiness.csv"
    table_path = args.output_dir / "tables" / "sscard_end_to_end_readiness.tex"
    manifest_path = args.output_dir / "sscard_end_to_end_readiness_manifest.json"
    write_csv(csv_path, rows)
    write_table(table_path, rows)
    manifest = {
        "passed": not any(row["blocks_dbms_e2e"] == "yes" for row in rows),
        "scope": "Readiness audit for SSCard PostgreSQL end-to-end reproduction",
        "reason": "Local hook execution is available, but pure official 700-slot regeneration remains blocked by missing predicate-order/model assets.",
        "static_like_predicates": count_static_like_predicates(END_TO_END / "like_queries_single"),
        "official_card_rows": {
            "sscard_pg_single": count_lines(END_TO_END / "cards" / "like_queries_single" / "sscard_pg_single.txt"),
            "pg_pg_single": count_lines(END_TO_END / "cards" / "like_queries_single" / "pg_pg_single.txt"),
        },
        "csv": str(csv_path),
        "table": str(table_path),
        "source_root": str(SSCARD_ROOT),
        "end_to_end_root": str(END_TO_END),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
