import os
import argparse
import csv
import hashlib
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import psycopg

import run_workload_drift_mscn_finetune as ft
import run_workload_drift_optimizer_risk_proxy as risk


ROOT = Path(__file__).resolve().parents[1]
STATS_DATA = ROOT / "external_data" / "workload_drift" / "extracted" / "dataset" / "stats"
RESULTS = ROOT / "paper" / "workload_drift_residual_field" / "results"
PG_HINT_DLL = ROOT / "tools" / "pg_hint_plan" / "pg_hint_plan.dll"


TABLES = {
    "badges": [
        ("Id", "BIGINT", None),
        ("UserId", "BIGINT", None),
        ("Date", "BIGINT", "datetime"),
    ],
    "comments": [
        ("Id", "BIGINT", None),
        ("PostId", "BIGINT", None),
        ("Score", "BIGINT", None),
        ("CreationDate", "BIGINT", "datetime"),
        ("UserId", "BIGINT", None),
    ],
    "postHistory": [
        ("Id", "BIGINT", None),
        ("PostHistoryTypeId", "BIGINT", None),
        ("PostId", "BIGINT", None),
        ("CreationDate", "BIGINT", "datetime"),
        ("UserId", "BIGINT", None),
    ],
    "postLinks": [
        ("Id", "BIGINT", None),
        ("CreationDate", "BIGINT", "datetime"),
        ("PostId", "BIGINT", None),
        ("RelatedPostId", "BIGINT", None),
        ("LinkTypeId", "BIGINT", None),
    ],
    "posts": [
        ("Id", "BIGINT", None),
        ("PostTypeId", "BIGINT", None),
        ("CreationDate", "BIGINT", "datetime"),
        ("Score", "BIGINT", None),
        ("ViewCount", "BIGINT", None),
        ("OwnerUserId", "BIGINT", None),
        ("AnswerCount", "BIGINT", None),
        ("CommentCount", "BIGINT", None),
        ("FavoriteCount", "BIGINT", None),
        ("LastEditorUserId", "BIGINT", None),
    ],
    "tags": [
        ("Id", "BIGINT", None),
        ("Count", "BIGINT", None),
        ("ExcerptPostId", "BIGINT", None),
    ],
    "users": [
        ("Id", "BIGINT", None),
        ("Reputation", "BIGINT", None),
        ("CreationDate", "BIGINT", "datetime"),
        ("Views", "BIGINT", None),
        ("UpVotes", "BIGINT", None),
        ("DownVotes", "BIGINT", None),
    ],
    "votes": [
        ("Id", "BIGINT", None),
        ("PostId", "BIGINT", None),
        ("VoteTypeId", "BIGINT", None),
        ("CreationDate", "BIGINT", "datetime"),
        ("UserId", "BIGINT", None),
        ("BountyAmount", "BIGINT", None),
    ],
}


METHODS = ["postgres-default", "raw-mscn", "residual-field", "mscn-finetune-best"]
METHOD_LABELS = {
    "postgres-default": "PostgreSQL",
    "raw-mscn": "MSCN injected",
    "residual-field": "Residual injected",
    "mscn-finetune-best": "Fine-tune injected",
}


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def sql_name(name: str) -> str:
    lowered = re.sub(r"[^a-zA-Z0-9_]", "", name).lower()
    if not lowered:
        raise ValueError(f"bad SQL identifier: {name!r}")
    return lowered


def qerror(prediction: float | None, label: float | None) -> float | None:
    if prediction is None or label is None:
        return None
    if not math.isfinite(float(prediction)) or not math.isfinite(float(label)):
        return None
    pred = max(1.0, float(prediction))
    lab = max(1.0, float(label))
    return max(pred / lab, lab / pred)


def is_timeout_error(message: str | None) -> bool:
    original = str(message or "")
    text = original.lower()
    return (
        "timeout" in text
        or "querycanceled" in text
        or "超时" in original
        or "取消查询" in original
        or "瓒呮椂" in original
    )


def pct(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return "--"
    return f"{100.0 * float(value):.1f}"


def number(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return "--"
    value = float(value)
    if value >= 1000:
        return f"{value:.0f}"
    if value >= 100:
        return f"{value:.1f}"
    return f"{value:.2f}"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_target_rows(path: Path) -> list[list[str]]:
    rows = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter="#")
        for row in reader:
            if len(row) != 4:
                raise ValueError(f"expected 4 fields in {path}, got {len(row)}")
            rows.append(row)
    return rows


def query_to_sql(row: list[str]) -> str:
    tables, joins, predicates, _label = row
    conditions = []
    if joins.strip():
        conditions.extend(item.strip() for item in joins.split(",") if item.strip())
    parts = predicates.split(",") if predicates else []
    for idx in range(0, len(parts), 3):
        if idx + 2 >= len(parts):
            continue
        column, operator, value = (item.strip() for item in parts[idx : idx + 3])
        conditions.append(f"{column} {operator} {value}")
    where_clause = " AND ".join(conditions) if conditions else "1=1"
    return f"SELECT COUNT(*) FROM {tables} WHERE {where_clause}"


def table_aliases(tables_field: str) -> list[str]:
    aliases = []
    for item in tables_field.split(","):
        tokens = item.strip().split()
        if not tokens:
            continue
        alias = tokens[-1] if len(tokens) > 1 else tokens[0]
        aliases.append(sql_name(alias))
    return aliases


def inject_rows_hint(sql: str, aliases: list[str], prediction: float) -> tuple[str, int, str]:
    injected = max(1, int(round(float(prediction))))
    hint = f"Rows({' '.join(aliases)} #{injected})"
    hinted = re.sub(r"^\s*SELECT\b", f"SELECT /*+ {hint} */", sql, count=1, flags=re.I)
    if hinted == sql:
        raise ValueError("query does not start with SELECT")
    return hinted, injected, hint


def connect(args):
    return psycopg.connect(
        host=args.pg_host,
        port=args.pg_port,
        dbname=args.pg_database,
        user=args.pg_user,
        password=args.pg_password,
        autocommit=False,
    )


def ensure_pg_hint_loaded(cur, args) -> None:
    dll = args.pg_hint_dll.resolve().as_posix()
    cur.execute(f"LOAD {sql_literal(dll)}")
    cur.execute("SET pg_hint_plan.enable_hint = on")


def final_column_expr(column: str, _dtype: str, kind: str | None) -> str:
    col = sql_name(column)
    if kind == "datetime":
        return (
            f"EXTRACT(EPOCH FROM NULLIF({col}, '')::timestamp)::BIGINT AS {col}"
        )
    return f"NULLIF({col}, '')::BIGINT AS {col}"


def import_stats_data(args) -> None:
    schema = sql_name(args.schema)
    with connect(args) as conn:
        with conn.cursor() as cur:
            if args.reload_data:
                cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
            cur.execute(
                "SELECT to_regclass(%s)",
                (f"{schema}.{sql_name('posts')}",),
            )
            if cur.fetchone()[0] is not None and not args.reload_data:
                return

            for table, columns in TABLES.items():
                table_name = sql_name(table)
                raw_name = f"raw_{table_name}"
                cur.execute(f"DROP TABLE IF EXISTS {schema}.{raw_name}")
                cur.execute(f"DROP TABLE IF EXISTS {schema}.{table_name}")
                raw_cols = ", ".join(f"{sql_name(col)} TEXT" for col, _dtype, _kind in columns)
                cur.execute(f"CREATE TABLE {schema}.{raw_name} ({raw_cols})")
                copy_cols = ", ".join(sql_name(col) for col, _dtype, _kind in columns)
                copy_sql = (
                    f"COPY {schema}.{raw_name} ({copy_cols}) "
                    "FROM STDIN WITH (FORMAT csv, HEADER true, NULL '')"
                )
                csv_path = args.stats_data / f"{table}.csv"
                with cur.copy(copy_sql) as copy:
                    with csv_path.open("rb") as handle:
                        while True:
                            chunk = handle.read(1024 * 1024)
                            if not chunk:
                                break
                            copy.write(chunk)
                select_list = ", ".join(final_column_expr(*column) for column in columns)
                cur.execute(
                    f"CREATE TABLE {schema}.{table_name} AS "
                    f"SELECT {select_list} FROM {schema}.{raw_name}"
                )
                cur.execute(f"DROP TABLE {schema}.{raw_name}")

            for table, columns in TABLES.items():
                table_name = sql_name(table)
                for column, _dtype, _kind in columns:
                    col = sql_name(column)
                    idx = f"idx_{table_name}_{col}"
                    cur.execute(
                        f"CREATE INDEX IF NOT EXISTS {idx} "
                        f"ON {schema}.{table_name} ({col})"
                    )
                cur.execute(f"ANALYZE {schema}.{table_name}")


def collect_probe_cases(path: Path, limit_per_drift: int | None) -> list[dict]:
    grouped = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["method"] not in {"raw-mscn", "residual-field", "mscn-finetune-best"}:
                continue
            key = (row["drift"], int(row["seed"]), int(row["query_index"]))
            case = grouped.setdefault(
                key,
                {
                    "drift": row["drift"],
                    "seed": int(row["seed"]),
                    "query_index": int(row["query_index"]),
                    "label": int(row["label"]),
                    "raw_label": int(row["raw_label"]),
                    "table_count": int(row["table_count"]),
                    "join_count": int(row["join_count"]),
                    "predicate_count": int(row["predicate_count"]),
                    "predictions": {},
                },
            )
            case["predictions"][row["method"]] = float(row["prediction"])
    cases = [
        case
        for case in grouped.values()
        if all(method in case["predictions"] for method in METHODS if method != "postgres-default")
    ]
    cases.sort(key=lambda item: (item["drift"], item["seed"], item["query_index"]))
    if limit_per_drift is None or limit_per_drift <= 0:
        return cases
    selected = []
    counts = defaultdict(int)
    for case in cases:
        if counts[case["drift"]] < limit_per_drift:
            selected.append(case)
            counts[case["drift"]] += 1
    return selected


def plan_doc(value):
    if isinstance(value, str):
        return json.loads(value)[0]
    if isinstance(value, list):
        return value[0]
    return value


def plan_shape(node: dict) -> str:
    label = str(node.get("Node Type", ""))
    relation = node.get("Relation Name")
    alias = node.get("Alias")
    if relation or alias:
        label += f"[{relation or ''}:{alias or ''}]"
    children = node.get("Plans") or []
    if not children:
        return label
    return label + "(" + ",".join(plan_shape(child) for child in children) + ")"


def collect_plan_stats(doc: dict) -> dict:
    root = doc["Plan"]
    estimates = []
    node_types = defaultdict(int)

    def visit(node: dict) -> None:
        node_types[str(node.get("Node Type", ""))] += 1
        if "Plan Rows" in node:
            estimates.append(float(node["Plan Rows"]))
        for child in node.get("Plans") or []:
            visit(child)

    visit(root)
    children = root.get("Plans") or []
    root_input = float(children[0]["Plan Rows"]) if children and "Plan Rows" in children[0] else None
    execution_ms = doc.get("Execution Time")
    planning_ms = doc.get("Planning Time")
    shape = plan_shape(root)
    return {
        "root_input_estimate": root_input,
        "max_estimated_cardinality": max(estimates) if estimates else None,
        "estimated_node_count": len(estimates),
        "plan_node_count": int(sum(node_types.values())),
        "join_node_count": int(
            sum(count for name, count in node_types.items() if "Join" in name)
        ),
        "execution_ms": float(execution_ms) if execution_ms is not None else None,
        "planning_ms": float(planning_ms) if planning_ms is not None else None,
        "plan_shape": shape,
        "plan_shape_sha256": sha256_text(shape),
    }


def run_explain(cur, sql: str) -> dict:
    started = time.perf_counter()
    cur.execute(
        "EXPLAIN (ANALYZE, FORMAT JSON, TIMING OFF, SUMMARY ON) " + sql
    )
    doc = plan_doc(cur.fetchone()[0])
    stats = collect_plan_stats(doc)
    stats["wall_seconds"] = float(time.perf_counter() - started)
    return stats


def run_count(cur, sql: str) -> tuple[int | None, float, str]:
    started = time.perf_counter()
    try:
        cur.execute(sql)
        return int(cur.fetchone()[0]), float(time.perf_counter() - started), ""
    except Exception as exc:
        return None, float(time.perf_counter() - started), f"{type(exc).__name__}: {exc}"


def evaluate_cases(args, cases: list[dict]) -> list[dict]:
    schema = sql_name(args.schema)
    workload_rows = {}
    for drift in sorted({case["drift"] for case in cases}):
        metrics_path = risk.DRIFTS[drift]["metrics"]
        metrics = read_json(metrics_path)
        workload_rows[drift] = load_target_rows(ft.resolve_target_path(metrics_path, metrics))

    rows = []
    with connect(args) as conn:
        with conn.cursor() as cur:
            ensure_pg_hint_loaded(cur, args)
            cur.execute(f"SET search_path TO {schema}, public")
            cur.execute(f"SET statement_timeout = {int(args.statement_timeout_ms)}")
            cur.execute("SET max_parallel_workers_per_gather = 0")
            for case in cases:
                row = workload_rows[case["drift"]][case["query_index"]]
                sql = query_to_sql(row)
                aliases = table_aliases(row[0])
                actual_count, count_seconds, count_error = run_count(cur, sql)
                default_stats = None
                default_shape = None
                method_sql = {"postgres-default": sql}
                injected_estimates = {"postgres-default": None}
                hints = {"postgres-default": ""}
                for method in METHODS[1:]:
                    hinted_sql, injected, hint = inject_rows_hint(
                        sql, aliases, case["predictions"][method]
                    )
                    method_sql[method] = hinted_sql
                    injected_estimates[method] = injected
                    hints[method] = hint

                for method in METHODS:
                    base = {
                        "drift": case["drift"],
                        "seed": case["seed"],
                        "query_index": case["query_index"],
                        "method": method,
                        "label": case["label"],
                        "actual_count": actual_count,
                        "label_matches": bool(actual_count == case["label"])
                        if actual_count is not None
                        else False,
                        "count_seconds": count_seconds,
                        "count_error": count_error,
                        "table_count": case["table_count"],
                        "join_count": case["join_count"],
                        "predicate_count": case["predicate_count"],
                        "sql_sha256": sha256_text(sql),
                        "hint": hints[method],
                        "injected_estimate": injected_estimates[method],
                        "prediction": case["predictions"].get(method),
                    }
                    try:
                        stats = run_explain(cur, method_sql[method])
                        if method == "postgres-default":
                            default_stats = stats
                            default_shape = stats["plan_shape_sha256"]
                        base.update(stats)
                        base["execution_status"] = "executed"
                        base["execution_error"] = ""
                    except Exception as exc:
                        conn.rollback()
                        ensure_pg_hint_loaded(cur, args)
                        cur.execute(f"SET search_path TO {schema}, public")
                        cur.execute(f"SET statement_timeout = {int(args.statement_timeout_ms)}")
                        cur.execute("SET max_parallel_workers_per_gather = 0")
                        base.update(
                            {
                                "root_input_estimate": None,
                                "max_estimated_cardinality": None,
                                "estimated_node_count": 0,
                                "plan_node_count": 0,
                                "join_node_count": 0,
                                "execution_ms": None,
                                "planning_ms": None,
                                "wall_seconds": None,
                                "plan_shape": "",
                                "plan_shape_sha256": "",
                                "execution_status": "explain_error",
                                "execution_error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                    plan_q = qerror(base["root_input_estimate"], case["label"])
                    pred_q = qerror(base["prediction"], case["label"])
                    inject_q = qerror(base["root_input_estimate"], base["injected_estimate"])
                    base["plan_estimate_qerror"] = plan_q
                    base["prediction_qerror"] = pred_q
                    base["injection_qerror"] = inject_q
                    base["hint_applied"] = bool(inject_q is not None and inject_q <= 1.05)
                    base["plan_changed_vs_default"] = (
                        bool(default_shape and base["plan_shape_sha256"] != default_shape)
                        if method != "postgres-default"
                        else False
                    )
                    if default_stats and default_stats.get("execution_ms") and base.get("execution_ms"):
                        base["execution_ms_ratio_vs_default"] = (
                            float(base["execution_ms"]) / float(default_stats["execution_ms"])
                        )
                    else:
                        base["execution_ms_ratio_vs_default"] = None
                    rows.append(base)
                conn.commit()
    return rows


def median(values: list[float]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.median(clean)) if clean else None


def percentile(values: list[float], q: float) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.percentile(clean, q)) if clean else None


def summarize(rows: list[dict]) -> list[dict]:
    out = []
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["drift"], row["method"])].append(row)
    for (drift, method), items in sorted(grouped.items()):
        executed = [row for row in items if row["execution_status"] == "executed"]
        ratios = [row["execution_ms_ratio_vs_default"] for row in executed]
        out.append(
            {
                "drift": drift,
                "method": method,
                "queries": len(items),
                "executed_queries": len(executed),
                "execution_success_rate": len(executed) / len(items) if items else 0.0,
                "label_match_rate": float(np.mean([row["label_matches"] for row in items]))
                if items
                else 0.0,
                "hint_applied_rate": float(np.mean([row["hint_applied"] for row in items]))
                if method != "postgres-default" and items
                else None,
                "plan_change_rate": float(np.mean([row["plan_changed_vs_default"] for row in items]))
                if method != "postgres-default" and items
                else None,
                "median_prediction_qerror": median([row["prediction_qerror"] for row in items]),
                "p95_prediction_qerror": percentile([row["prediction_qerror"] for row in items], 95),
                "median_plan_qerror": median([row["plan_estimate_qerror"] for row in items]),
                "p95_plan_qerror": percentile([row["plan_estimate_qerror"] for row in items], 95),
                "median_execution_ms": median([row["execution_ms"] for row in executed]),
                "p95_execution_ms": percentile([row["execution_ms"] for row in executed], 95),
                "median_exec_ratio_vs_default": median(ratios),
                "p95_exec_ratio_vs_default": percentile(ratios, 95),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows for {path}")
    fieldnames = [key for key in rows[0].keys() if key != "plan_shape"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_table(path: Path, rows: list[dict]) -> None:
    order = {method: idx for idx, method in enumerate(METHODS)}
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{PostgreSQL estimate-injection probe using \\texttt{pg\\_hint\\_plan} \\texttt{Rows(...)} hints on the executable raw-tail subset. Each injected estimator replaces the optimizer's final-join row estimate for the same SQL query; latency is from \\texttt{EXPLAIN ANALYZE}.}",
        "\\label{tab:postgres-estimate-injection}",
        "\\begin{tabular}{llrrrrrrr}",
        "\\toprule",
        "Drift & Method & Rows & Hint OK & Plan chg. & Med. plan Q & P95 plan Q & Med. ms & Med. ms/default \\\\",
        "\\midrule",
    ]
    for row in sorted(rows, key=lambda item: (item["drift"], order[item["method"]])):
        hint_ok = "--" if row["hint_applied_rate"] is None else pct(row["hint_applied_rate"]) + "\\%"
        plan_change = "--" if row["plan_change_rate"] is None else pct(row["plan_change_rate"]) + "\\%"
        ratio = "--" if row["median_exec_ratio_vs_default"] is None else f"{row['median_exec_ratio_vs_default']:.2f}$\\times$"
        lines.append(
            f"{row['drift'].title()} & {METHOD_LABELS[row['method']]} "
            f"& {int(row['queries'])} "
            f"& {hint_ok} "
            f"& {plan_change} "
            f"& {number(row['median_plan_qerror'])} "
            f"& {number(row['p95_plan_qerror'])} "
            f"& {number(row['median_execution_ms'])} "
            f"& {ratio} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run PostgreSQL pg_hint_plan Rows(...) estimate injection on drifted CE queries."
    )
    parser.add_argument("--pg-host", default="localhost")
    parser.add_argument("--pg-port", type=int, default=5432)
    parser.add_argument("--pg-database", default="stats_drift")
    parser.add_argument("--pg-user", default="postgres")
    parser.add_argument("--pg-password", default=os.environ.get("PGPASSWORD"))
    parser.add_argument("--schema", default="stats_inj")
    parser.add_argument("--pg-hint-dll", type=Path, default=PG_HINT_DLL)
    parser.add_argument("--stats-data", type=Path, default=STATS_DATA)
    parser.add_argument("--duckdb-probe-csv", type=Path, default=RESULTS / "duckdb_execution_probe.csv")
    parser.add_argument("--output-dir", type=Path, default=RESULTS)
    parser.add_argument("--reload-data", action="store_true")
    parser.add_argument("--limit-per-drift", type=int, default=0)
    parser.add_argument("--statement-timeout-ms", type=int, default=60000)
    args = parser.parse_args()

    if not args.stats_data.is_absolute():
        args.stats_data = ROOT / args.stats_data
    if not args.output_dir.is_absolute():
        args.output_dir = ROOT / args.output_dir
    if not args.duckdb_probe_csv.is_absolute():
        args.duckdb_probe_csv = ROOT / args.duckdb_probe_csv
    if not args.pg_hint_dll.is_absolute():
        args.pg_hint_dll = ROOT / args.pg_hint_dll

    import_stats_data(args)
    limit = args.limit_per_drift if args.limit_per_drift > 0 else None
    cases = collect_probe_cases(args.duckdb_probe_csv, limit)
    detail_rows = evaluate_cases(args, cases)
    summary_rows = summarize(detail_rows)

    detail_csv = args.output_dir / "postgres_estimate_injection.csv"
    summary_csv = args.output_dir / "postgres_estimate_injection_summary.csv"
    table_path = args.output_dir / "tables" / "postgres_estimate_injection.tex"
    manifest_path = args.output_dir / "postgres_estimate_injection_manifest.json"
    write_csv(detail_csv, detail_rows)
    write_csv(summary_csv, summary_rows)
    write_table(table_path, summary_rows)
    attempted_method_executions = len(detail_rows)
    completed_method_executions = sum(
        1 for row in detail_rows if row.get("execution_status") == "executed"
    )
    completed_queries = len(
        {
            (row["drift"], int(row["seed"]), int(row["query_index"]))
            for row in detail_rows
            if row.get("execution_status") == "executed"
        }
    )
    timeout_queries = len(
        {
            (row["drift"], int(row["seed"]), int(row["query_index"]))
            for row in detail_rows
            if is_timeout_error(row.get("execution_error", ""))
        }
    )
    all_executions_completed = completed_method_executions == attempted_method_executions
    completion_rate = (
        completed_method_executions / attempted_method_executions
        if attempted_method_executions
        else 0.0
    )
    hint_rates_ok = all(
        row["hint_applied_rate"] is None or row["hint_applied_rate"] >= 0.95
        for row in summary_rows
    )
    manifest = {
        "passed": bool(detail_rows and completion_rate >= 0.99 and hint_rates_ok),
        "scope": "PostgreSQL optimizer-in-loop estimate injection with pg_hint_plan Rows hints",
        "queries": len(cases),
        "detail_rows": len(detail_rows),
        "summary_rows": len(summary_rows),
        "schema": args.schema,
        "pg_hint_dll": str(args.pg_hint_dll),
        "duckdb_probe_csv": str(args.duckdb_probe_csv),
        "detail_csv": str(detail_csv),
        "summary_csv": str(summary_csv),
        "table": str(table_path),
        "statement_timeout_ms": args.statement_timeout_ms,
        "completion_note": (
            f"{completed_method_executions} of {attempted_method_executions} method "
            f"executions completed; {timeout_queries} query group(s) timed out at "
            f"{args.statement_timeout_ms}ms."
        ),
        "all_executions_completed": all_executions_completed,
        "completed_method_executions": completed_method_executions,
        "attempted_method_executions": attempted_method_executions,
        "timeout_queries": timeout_queries,
        "completed_queries": completed_queries,
        "pass_criterion": "usable boundary audit with >=99% completed method executions and documented timeout rows",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
