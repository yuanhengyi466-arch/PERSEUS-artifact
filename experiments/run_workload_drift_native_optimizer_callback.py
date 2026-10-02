import os
import argparse
import csv
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import run_workload_drift_postgres_estimate_injection as inj
import run_workload_drift_postgres_multisubplan_injection as multi


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "paper" / "workload_drift_residual_field" / "results"

METHODS = ["default", "native-raw", "native-final", "native-multi", "native-gated"]
METHOD_LABELS = {
    "default": "Default",
    "native-raw": "Native raw",
    "native-final": "Native final",
    "native-multi": "Native multi",
    "native-gated": "Native gated",
}

DETAIL_FIELDS = [
    "drift",
    "seed",
    "query_index",
    "method",
    "label",
    "table_count",
    "join_count",
    "predicate_count",
    "map_rows",
    "map_subplans",
    "sql_sha256",
    "root_input_estimate",
    "max_estimated_cardinality",
    "estimated_node_count",
    "plan_node_count",
    "join_node_count",
    "execution_ms",
    "planning_ms",
    "plan_shape",
    "plan_shape_sha256",
    "wall_seconds",
    "total_cost",
    "status",
    "error",
    "plan_changed_vs_default",
    "execution_ms_ratio_vs_default",
]


def read_csv(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def clean(values: list[float | None]) -> list[float]:
    out = []
    for value in values:
        if value is None or value == "":
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            out.append(number)
    return out


def median(values: list[float | None]) -> float | None:
    data = clean(values)
    return float(np.median(data)) if data else None


def percentile(values: list[float | None], q: float) -> float | None:
    data = clean(values)
    return float(np.percentile(data, q)) if data else None


def pct(values: list[bool]) -> float:
    return float(np.mean(values)) if values else 0.0


def parse_methods(value: str) -> list[str]:
    methods = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [method for method in methods if method not in METHODS]
    if unknown:
        raise ValueError(f"unknown method(s): {', '.join(unknown)}")
    if "default" not in methods:
        methods.insert(0, "default")
    return list(dict.fromkeys(methods))


def case_key(row: dict) -> tuple[str, int, int]:
    return (row["drift"], int(row["seed"]), int(row["query_index"]))


def query_id(case: dict) -> str:
    return f"{case['drift']}:s{int(case['seed'])}:q{int(case['query_index'])}"


def load_final_predictions(path: Path) -> dict[tuple[str, int, int], dict]:
    grouped: dict[tuple[str, int, int], dict] = {}
    for row in read_csv(path):
        method = row.get("method")
        if method not in {"raw-mscn", "residual-field"}:
            continue
        key = case_key(row)
        item = grouped.setdefault(
            key,
            {
                "drift": row["drift"],
                "seed": int(row["seed"]),
                "query_index": int(row["query_index"]),
                "label": int(float(row["label"])),
                "table_count": int(float(row["table_count"])),
                "join_count": int(float(row["join_count"])),
                "predicate_count": int(float(row["predicate_count"])),
                "predictions": {},
            },
        )
        item["predictions"][method] = float(row["prediction"])
    return {
        key: item
        for key, item in grouped.items()
        if "raw-mscn" in item["predictions"] and "residual-field" in item["predictions"]
    }


def load_subplans(path: Path) -> dict[tuple[str, int, int], list[dict]]:
    grouped: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    for row in read_csv(path):
        key = case_key(row)
        residual = row.get("residual_estimate")
        pg_est = row.get("pg_estimate")
        if residual in ("", None) or pg_est in ("", None):
            continue
        grouped[key].append(
            {
                "aliases": " ".join(sorted(row["aliases"].split())),
                "pg_estimate": max(1.0, float(pg_est)),
                "residual_estimate": max(1.0, float(residual)),
            }
        )
    return grouped


def load_workload_rows(cases: list[dict]) -> dict[str, list[list[str]]]:
    rows = {}
    for drift in sorted({case["drift"] for case in cases}):
        spec = multi.risk.DRIFTS[drift]
        metrics = inj.read_json(spec["metrics"])
        rows[drift] = inj.load_target_rows(multi.ft.resolve_target_path(spec["metrics"], metrics))
    return rows


def select_cases(
    final_predictions: dict[tuple[str, int, int], dict],
    subplans: dict[tuple[str, int, int], list[dict]],
    drifts: list[str],
    seeds: list[int],
    limit_per_drift: int,
    min_tables: int,
) -> list[dict]:
    selected = []
    counts = defaultdict(int)
    for key, case in sorted(final_predictions.items()):
        drift, seed, _query_index = key
        if drift not in drifts or seed not in seeds:
            continue
        if int(case["table_count"]) < min_tables:
            continue
        if key not in subplans:
            continue
        if limit_per_drift > 0 and counts[drift] >= limit_per_drift:
            continue
        item = dict(case)
        item["subplans"] = subplans[key]
        selected.append(item)
        counts[drift] += 1
    return selected


def gated_method_for_drift(drift: str) -> str:
    return "native-final" if drift in {"column", "table"} else "default"


def build_estimate_map(cases: list[dict], workload_rows: dict[str, list[list[str]]]) -> list[dict]:
    out = []
    seen = set()
    for case in cases:
        row = workload_rows[case["drift"]][case["query_index"]]
        aliases = " ".join(sorted(inj.table_aliases(row[0])))
        qid = query_id(case)
        entries = [
            (aliases, "native-raw", case["predictions"]["raw-mscn"]),
            (aliases, "native-final", case["predictions"]["residual-field"]),
            (aliases, "native-multi", case["predictions"]["residual-field"]),
        ]
        for sub in case["subplans"]:
            entries.append((sub["aliases"], "native-raw", sub["pg_estimate"]))
            entries.append((sub["aliases"], "native-final", sub["pg_estimate"]))
            entries.append((sub["aliases"], "native-multi", sub["residual_estimate"]))
        for alias_key, method, estimate in entries:
            key = (qid, alias_key, method)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "query_id": qid,
                    "aliases": alias_key,
                    "method": method,
                    "estimate": max(1, int(round(float(estimate)))),
                }
            )
    return out


def setup(cur, args) -> None:
    schema = inj.sql_name(args.schema)
    cur.execute(f"SET search_path TO {schema}, public")
    cur.execute(f"SET statement_timeout = {int(args.statement_timeout_ms)}")
    cur.execute("SET max_parallel_workers_per_gather = 0")
    cur.execute("SET jit = off")
    cur.execute("SET enable_hashjoin = on")
    cur.execute("SET enable_mergejoin = on")
    cur.execute("SET enable_nestloop = on")
    cur.execute("SET enable_indexscan = on")
    cur.execute("SET enable_indexonlyscan = on")
    cur.execute("SET enable_bitmapscan = on")
    cur.execute("SET join_collapse_limit = 8")
    cur.execute("SET from_collapse_limit = 8")


def apply_native(cur, estimate_map: Path, case: dict, method: str) -> str:
    if method == "default":
        cur.execute("SET perseus_estmap_enabled = off")
        return "default"
    active_method = method
    if method == "native-gated":
        active_method = gated_method_for_drift(case["drift"])
        if active_method == "default":
            cur.execute("SET perseus_estmap_enabled = off")
            return active_method
    cur.execute("SELECT set_config('perseus_estmap_fname', %s, false)", (estimate_map.as_posix(),))
    cur.execute("SELECT set_config('perseus_estmap_query_id', %s, false)", (query_id(case),))
    cur.execute("SELECT set_config('perseus_estmap_method', %s, false)", (active_method,))
    cur.execute("SET perseus_estmap_enabled = on")
    return active_method


def run_explain(cur, sql: str) -> dict:
    doc, stats = multi.run_explain_doc(cur, sql, analyze=True)
    return stats


def evaluate(args, cases: list[dict], workload_rows: dict[str, list[list[str]]], estimate_map: Path) -> list[dict]:
    detail = []
    with inj.connect(args) as conn:
        with conn.cursor() as cur:
            setup(cur, args)
            for case in cases:
                row = workload_rows[case["drift"]][case["query_index"]]
                sql = inj.query_to_sql(row)
                default_shape = ""
                default_ms = None
                for method in args.methods:
                    base = {
                        "drift": case["drift"],
                        "seed": int(case["seed"]),
                        "query_index": int(case["query_index"]),
                        "method": method,
                        "label": int(case["label"]),
                        "table_count": int(case["table_count"]),
                        "join_count": int(case["join_count"]),
                        "predicate_count": int(case["predicate_count"]),
                        "map_rows": None,
                        "map_subplans": len(case["subplans"]),
                        "sql_sha256": inj.sha256_text(sql),
                    }
                    try:
                        setup(cur, args)
                        active_method = apply_native(cur, estimate_map, case, method)
                        base["active_method"] = active_method
                        stats = run_explain(cur, sql)
                        base.update(stats)
                        base["status"] = "executed"
                        base["error"] = ""
                        if method == "default":
                            default_shape = stats["plan_shape_sha256"]
                            default_ms = stats["execution_ms"]
                    except Exception as exc:
                        conn.rollback()
                        base.update(
                            {
                                "active_method": "",
                                "root_input_estimate": None,
                                "max_estimated_cardinality": None,
                                "estimated_node_count": 0,
                                "plan_node_count": 0,
                                "join_node_count": 0,
                                "execution_ms": None,
                                "planning_ms": None,
                                "plan_shape": "",
                                "plan_shape_sha256": "",
                                "wall_seconds": None,
                                "total_cost": None,
                                "status": "error",
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                    base["plan_changed_vs_default"] = bool(
                        method != "default"
                        and default_shape
                        and base.get("plan_shape_sha256")
                        and base["plan_shape_sha256"] != default_shape
                    )
                    if default_ms and base.get("execution_ms"):
                        base["execution_ms_ratio_vs_default"] = float(base["execution_ms"]) / float(default_ms)
                    else:
                        base["execution_ms_ratio_vs_default"] = None
                    detail.append(base)
                conn.commit()
    return detail


def summarize(detail: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in detail:
        grouped[(row["drift"], row["method"])].append(row)
    out = []
    order = {method: idx for idx, method in enumerate(METHODS)}
    for (drift, method), rows in sorted(grouped.items(), key=lambda item: (item[0][0], order.get(item[0][1], 99))):
        executed = [row for row in rows if row["status"] == "executed"]
        ratios = [row["execution_ms_ratio_vs_default"] for row in executed]
        out.append(
            {
                "drift": drift,
                "method": method,
                "queries": len(rows),
                "executed_queries": len(executed),
                "execution_success_rate": len(executed) / len(rows) if rows else 0.0,
                "median_subplans": median([row["map_subplans"] for row in rows]),
                "plan_change_rate": pct([row["plan_changed_vs_default"] for row in rows if method != "default"])
                if method != "default"
                else None,
                "median_execution_ms": median([row["execution_ms"] for row in executed]),
                "p95_execution_ms": percentile([row["execution_ms"] for row in executed], 95),
                "median_planning_ms": median([row["planning_ms"] for row in executed]),
                "median_ratio_vs_default": median(ratios),
                "p95_ratio_vs_default": percentile(ratios, 95),
                "median_total_cost": median([row["total_cost"] for row in executed]),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run native PostgreSQL Perseus estimate-map callback latency experiment.")
    parser.add_argument("--pg-host", default="127.0.0.1")
    parser.add_argument("--pg-port", type=int, default=5434)
    parser.add_argument("--pg-database", default="postgres")
    parser.add_argument("--pg-user", default="postgres")
    parser.add_argument("--pg-password", default=os.environ.get("PGPASSWORD"))
    parser.add_argument("--schema", default="stats_inj")
    parser.add_argument("--reload-data", action="store_true")
    parser.add_argument("--stats-data", type=Path, default=inj.STATS_DATA)
    parser.add_argument("--duckdb-probe-csv", type=Path, default=RESULTS / "duckdb_execution_probe.csv")
    parser.add_argument("--subplan-csv", type=Path, default=RESULTS / "postgres_multisubplan_injection_subplans.csv")
    parser.add_argument("--output-dir", type=Path, default=RESULTS / "native_optimizer_callback")
    parser.add_argument("--drifts", default="column,table,value")
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--limit-per-drift", type=int, default=8)
    parser.add_argument("--min-tables", type=int, default=3)
    parser.add_argument("--methods", default="default,native-raw,native-final,native-multi,native-gated")
    parser.add_argument("--statement-timeout-ms", type=int, default=60000)
    args = parser.parse_args()

    args.drifts = [item.strip() for item in args.drifts.split(",") if item.strip()]
    args.seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    args.methods = parse_methods(args.methods)
    args.output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    args.duckdb_probe_csv = args.duckdb_probe_csv if args.duckdb_probe_csv.is_absolute() else ROOT / args.duckdb_probe_csv
    args.subplan_csv = args.subplan_csv if args.subplan_csv.is_absolute() else ROOT / args.subplan_csv
    args.stats_data = args.stats_data if args.stats_data.is_absolute() else ROOT / args.stats_data

    inj.import_stats_data(args)
    final_predictions = load_final_predictions(args.duckdb_probe_csv)
    subplans = load_subplans(args.subplan_csv)
    cases = select_cases(final_predictions, subplans, args.drifts, args.seeds, args.limit_per_drift, args.min_tables)
    workload_rows = load_workload_rows(cases)
    estimate_rows = build_estimate_map(cases, workload_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    estimate_map = args.output_dir / "native_estimate_map.csv"
    detail_csv = args.output_dir / "native_callback_detail.csv"
    summary_csv = args.output_dir / "native_callback_summary.csv"
    manifest_path = args.output_dir / "native_callback_manifest.json"
    write_csv(estimate_map, estimate_rows, ["query_id", "aliases", "method", "estimate"])
    detail = evaluate(args, cases, workload_rows, estimate_map)
    summary = summarize(detail)
    write_csv(detail_csv, detail, DETAIL_FIELDS + ["active_method"])
    write_csv(summary_csv, summary)

    attempted = len(detail)
    completed = sum(1 for row in detail if row["status"] == "executed")
    manifest = {
        "passed": bool(cases and completed == attempted),
        "scope": "Native PostgreSQL planner estimate-map callback over Perseus final and connected-subplan estimates",
        "cases": len(cases),
        "detail_rows": len(detail),
        "estimate_map_rows": len(estimate_rows),
        "drifts": args.drifts,
        "seeds": args.seeds,
        "methods": args.methods,
        "detail_csv": str(detail_csv),
        "summary_csv": str(summary_csv),
        "estimate_map_csv": str(estimate_map),
        "statement_timeout_ms": args.statement_timeout_ms,
        "completed_method_executions": completed,
        "attempted_method_executions": attempted,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
