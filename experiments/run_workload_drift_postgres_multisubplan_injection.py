import os
import argparse
import csv
import itertools
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import duckdb

import run_workload_drift_duckdb_execution_probe as duck_probe
import run_workload_drift_mscn_finetune as ft
import run_workload_drift_optimizer_risk_proxy as risk
import run_workload_drift_postgres_estimate_injection as inj


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "paper" / "workload_drift_residual_field" / "results"

METHODS = ["default", "final-residual", "multi-residual", "multi-oracle"]
METHOD_LABELS = {
    "default": "Default",
    "final-residual": "Final residual",
    "multi-residual": "Multi residual",
    "multi-oracle": "Multi oracle",
}


def parse_methods(value: str) -> list[str]:
    methods = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [method for method in methods if method not in METHODS]
    if unknown:
        raise ValueError(f"unknown method(s): {', '.join(unknown)}")
    if "default" not in methods:
        methods.insert(0, "default")
    return list(dict.fromkeys(methods))

DETAIL_FIELDS = [
    "drift",
    "seed",
    "query_index",
    "method",
    "label",
    "table_count",
    "join_count",
    "predicate_count",
    "subplan_hints",
    "subplan_errors",
    "raw_prediction",
    "residual_prediction",
    "root_scale",
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

SUBPLAN_FIELDS = [
    "drift",
    "seed",
    "query_index",
    "aliases",
    "actual",
    "pg_estimate",
    "residual_estimate",
    "pg_qerror",
    "residual_qerror",
    "sql_sha256",
]

SELECTION_FIELDS = [
    "drift",
    "seed",
    "query_index",
    "selector",
    "selected_method",
    "selected_ms",
    "selected_ratio_vs_default",
    "best_deployable_method",
    "best_deployable_ratio_vs_default",
]


def split_indices(labels: np.ndarray, adaptation_rows: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(labels.shape[0])
    rng.shuffle(indices)
    return indices[:adaptation_rows], indices[adaptation_rows:]


def load_target_bundle(drift: str, seed: int) -> dict:
    spec = risk.DRIFTS[drift]
    metrics_path = spec["metrics"]
    metrics = inj.read_json(metrics_path)
    target_path = ft.resolve_target_path(metrics_path, metrics)
    rows = inj.load_target_rows(target_path)
    labels = np.asarray([max(1, int(row[3])) for row in rows], dtype=np.float64)
    raw_predictions = np.loadtxt(metrics["artifacts"]["predictions"], dtype=np.float64)
    adapt_idx, eval_idx = split_indices(labels, spec["adaptation_rows"], seed)
    residual_eval = risk.compute_residual_predictions(rows, labels, raw_predictions, adapt_idx, eval_idx, seed)
    residual_predictions = np.full(labels.shape[0], np.nan, dtype=np.float64)
    residual_predictions[eval_idx] = residual_eval
    return {
        "rows": rows,
        "labels": labels,
        "raw_predictions": raw_predictions,
        "residual_predictions": residual_predictions,
        "eval_idx": eval_idx,
        "adaptation_rows": spec["adaptation_rows"],
    }


def collect_cases(path: Path, drifts: list[str], limit_per_drift: int) -> list[dict]:
    cases = inj.collect_probe_cases(path, None)
    selected = []
    counts = defaultdict(int)
    for case in cases:
        if case["drift"] not in drifts:
            continue
        if int(case["table_count"]) < 3:
            continue
        if limit_per_drift > 0 and counts[case["drift"]] >= limit_per_drift:
            continue
        selected.append(case)
        counts[case["drift"]] += 1
    return selected


def parse_aliases(tables_field: str) -> dict[str, str]:
    out = {}
    for item in tables_field.split(","):
        tokens = item.strip().split()
        if not tokens:
            continue
        table = tokens[0]
        alias = tokens[-1] if len(tokens) > 1 else tokens[0]
        out[inj.sql_name(alias)] = inj.sql_name(table)
    return out


def expr_alias(expr: str) -> str | None:
    match = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\.", expr)
    if not match:
        return None
    return inj.sql_name(match.group(1))


def parse_joins(joins_field: str) -> list[tuple[str, str, str]]:
    out = []
    for join in [item.strip() for item in joins_field.split(",") if item.strip()]:
        if "=" not in join:
            continue
        left, right = join.split("=", 1)
        left_alias = expr_alias(left)
        right_alias = expr_alias(right)
        if left_alias and right_alias:
            out.append((join, left_alias, right_alias))
    return out


def parse_predicates(predicates_field: str) -> list[tuple[str, str]]:
    out = []
    parts = predicates_field.split(",") if predicates_field else []
    for idx in range(0, len(parts), 3):
        if idx + 2 >= len(parts):
            continue
        column, operator, value = (item.strip() for item in parts[idx : idx + 3])
        alias = expr_alias(column)
        if alias:
            out.append((f"{column} {operator} {value}", alias))
    return out


def connected_subset(subset: tuple[str, ...], joins: list[tuple[str, str, str]]) -> bool:
    if len(subset) <= 1:
        return True
    wanted = set(subset)
    graph = {alias: set() for alias in wanted}
    for _join, left, right in joins:
        if left in wanted and right in wanted:
            graph[left].add(right)
            graph[right].add(left)
    seen = set()
    stack = [subset[0]]
    while stack:
        item = stack.pop()
        if item in seen:
            continue
        seen.add(item)
        stack.extend(graph[item] - seen)
    return seen == wanted


def subplan_subsets(row: list[str], max_subplans: int) -> list[tuple[str, ...]]:
    aliases = sorted(parse_aliases(row[0]))
    joins = parse_joins(row[1])
    subsets = []
    for size in range(2, len(aliases)):
        for subset in itertools.combinations(aliases, size):
            if connected_subset(subset, joins):
                subsets.append(subset)
    subsets.sort(key=lambda item: (len(item), item))
    if max_subplans > 0:
        subsets = subsets[:max_subplans]
    return subsets


def subset_sql(row: list[str], subset: tuple[str, ...]) -> str:
    alias_to_table = parse_aliases(row[0])
    wanted = set(subset)
    tables = ", ".join(f"{alias_to_table[alias]} {alias}" for alias in subset)
    conditions = []
    for join, left, right in parse_joins(row[1]):
        if left in wanted and right in wanted:
            conditions.append(join)
    for predicate, alias in parse_predicates(row[2]):
        if alias in wanted:
            conditions.append(predicate)
    where_clause = " AND ".join(conditions) if conditions else "1=1"
    return f"SELECT COUNT(*) FROM {tables} WHERE {where_clause}"


def plan_alias_sets(doc: dict) -> dict[tuple[str, ...], dict]:
    root = doc["Plan"]
    out = {}

    def visit(node: dict) -> set[str]:
        children = node.get("Plans") or []
        child_sets = [visit(child) for child in children]
        aliases = set()
        alias = node.get("Alias")
        if alias:
            aliases.add(inj.sql_name(str(alias)))
        for child_set in child_sets:
            aliases.update(child_set)
        if aliases and "Plan Rows" in node:
            key = tuple(sorted(aliases))
            out[key] = {
                "plan_rows": float(node["Plan Rows"]),
                "node_type": str(node.get("Node Type", "")),
            }
        return aliases

    visit(root)
    return out


def run_explain_doc(cur, sql: str, analyze: bool) -> tuple[dict, dict]:
    prefix = "EXPLAIN (ANALYZE, FORMAT JSON, TIMING OFF, SUMMARY ON) " if analyze else "EXPLAIN (FORMAT JSON) "
    started = time.perf_counter()
    cur.execute(prefix + sql)
    doc = inj.plan_doc(cur.fetchone()[0])
    stats = inj.collect_plan_stats(doc)
    stats["wall_seconds"] = float(time.perf_counter() - started)
    stats["total_cost"] = float(doc["Plan"].get("Total Cost", 0.0))
    return doc, stats


def pg_sql_to_duckdb(sql: str) -> str:
    out = sql
    for pg_name, duck_name in {
        "posthistory": "postHistory",
        "postlinks": "postLinks",
    }.items():
        out = re.sub(rf"\b{pg_name}\b", f'"{duck_name}"', out, flags=re.I)
    return out


def run_actual_count(cur, duck_con: duckdb.DuckDBPyConnection | None, sql: str, args) -> int:
    if args.actual_engine == "duckdb":
        if duck_con is None:
            raise RuntimeError("DuckDB actual-count engine was requested but not initialized")
        return int(duck_con.execute(pg_sql_to_duckdb(sql)).fetchone()[0])
    cur.execute(sql)
    return int(cur.fetchone()[0])


def multi_hint_sql(sql: str, hints: list[str]) -> str:
    if not hints:
        return sql
    hint = " ".join(hints)
    hinted = re.sub(r"^\s*SELECT\b", f"SELECT /*+ {hint} */", sql, count=1, flags=re.I)
    if hinted == sql:
        raise ValueError("query does not start with SELECT")
    return hinted


def estimate_subplans(
    cur,
    row: list[str],
    root_scale: float,
    max_subplans: int,
    args,
    subplan_cache: dict[str, dict[str, float]],
    duck_con: duckdb.DuckDBPyConnection | None,
) -> tuple[list[dict], int]:
    out = []
    errors = 0
    for subset in subplan_subsets(row, max_subplans):
        sql = subset_sql(row, subset)
        sql_hash = inj.sha256_text(sql)
        cached = subplan_cache.get(sql_hash)
        if cached is not None:
            cached_actual = cached.get("actual")
            actual = (
                int(cached_actual)
                if cached_actual is not None and math.isfinite(float(cached_actual))
                else None
            )
            pg_estimate = float(cached["pg_estimate"])
            residual_estimate = max(1.0, pg_estimate * root_scale)
            out.append(
                {
                    "subset": subset,
                    "sql_sha256": sql_hash,
                    "actual": max(1, actual) if actual is not None else None,
                    "pg_estimate": max(1.0, pg_estimate),
                    "residual_estimate": residual_estimate,
                }
            )
            continue
        try:
            sub_doc, sub_stats = run_explain_doc(cur, sql, analyze=False)
            actual = None
            if args.subplan_actual_mode == "exact":
                actual = run_actual_count(cur, duck_con, sql, args)
            pg_estimate = sub_stats["root_input_estimate"]
            if pg_estimate is None or not math.isfinite(float(pg_estimate)):
                alias_map = plan_alias_sets(sub_doc)
                pg_estimate = alias_map.get(tuple(sorted(subset)), {}).get("plan_rows")
            if pg_estimate is None or not math.isfinite(float(pg_estimate)):
                pg_estimate = max(1.0, float(actual)) if actual is not None else 1.0
            cache_item = {"pg_estimate": float(max(1.0, float(pg_estimate)))}
            if actual is not None:
                cache_item["actual"] = float(max(1, actual))
            subplan_cache[sql_hash] = cache_item
            residual_estimate = max(1.0, float(pg_estimate) * root_scale)
            out.append(
                {
                    "subset": subset,
                    "sql_sha256": sql_hash,
                    "actual": max(1, actual) if actual is not None else None,
                    "pg_estimate": max(1.0, float(pg_estimate)),
                    "residual_estimate": residual_estimate,
                }
            )
        except Exception:
            cur.connection.rollback()
            setup(cur, args)
            apply_base_planner(cur)
            errors += 1
    return out, errors


def setup(cur, args) -> None:
    inj.ensure_pg_hint_loaded(cur, args)
    schema = inj.sql_name(args.schema)
    cur.execute(f"SET search_path TO {schema}, public")
    cur.execute(f"SET statement_timeout = {int(args.statement_timeout_ms)}")
    cur.execute("SET max_parallel_workers_per_gather = 0")
    cur.execute("SET jit = off")


def apply_base_planner(cur) -> None:
    settings = {
        "enable_hashjoin": "on",
        "enable_mergejoin": "on",
        "enable_nestloop": "on",
        "enable_indexscan": "on",
        "enable_indexonlyscan": "on",
        "enable_bitmapscan": "on",
        "join_collapse_limit": "8",
        "from_collapse_limit": "8",
    }
    for key, value in settings.items():
        cur.execute(f"SET {key} = {value}")


def evaluate_case(
    cur,
    conn,
    args,
    case: dict,
    bundle: dict,
    subplan_cache: dict[str, dict[str, float]],
    duck_con: duckdb.DuckDBPyConnection | None,
) -> tuple[list[dict], list[dict], list[dict]]:
    drift = case["drift"]
    seed = int(case["seed"])
    query_index = int(case["query_index"])
    row = bundle["rows"][query_index]
    label = float(bundle["labels"][query_index])
    raw_prediction = float(bundle["raw_predictions"][query_index])
    residual_prediction = float(bundle["residual_predictions"][query_index])
    if not math.isfinite(residual_prediction):
        return [], [], []
    root_scale = residual_prediction / max(1.0, raw_prediction)
    sql = inj.query_to_sql(row)
    aliases = inj.table_aliases(row[0])

    subplans = []
    try:
        apply_base_planner(cur)
        sub_rows, sub_errors = estimate_subplans(
            cur,
            row,
            root_scale,
            args.max_subplans,
            args,
            subplan_cache,
            duck_con,
        )
    except Exception:
        conn.rollback()
        setup(cur, args)
        return [], [], []

    for sub in sub_rows:
        subplans.append(
            {
                "drift": drift,
                "seed": seed,
                "query_index": query_index,
                "aliases": " ".join(sub["subset"]),
                "actual": sub["actual"],
                "pg_estimate": sub["pg_estimate"],
                "residual_estimate": sub["residual_estimate"],
                "pg_qerror": inj.qerror(sub["pg_estimate"], sub["actual"]),
                "residual_qerror": inj.qerror(sub["residual_estimate"], sub["actual"]),
                "sql_sha256": sub["sql_sha256"],
            }
        )

    final_hint = inj.inject_rows_hint(sql, aliases, residual_prediction)[2]
    residual_hints = [final_hint]
    oracle_hints = [f"Rows({' '.join(aliases)} #{max(1, int(round(label)))})"]
    for sub in sub_rows:
        subset = " ".join(sub["subset"])
        residual_hints.append(f"Rows({subset} #{max(1, int(round(sub['residual_estimate'])))})")
        if sub.get("actual") is not None:
            oracle_hints.append(f"Rows({subset} #{max(1, int(round(sub['actual'])))})")

    sql_by_method = {
        "default": sql,
        "final-residual": multi_hint_sql(sql, [final_hint]),
        "multi-residual": multi_hint_sql(sql, residual_hints),
    }
    if "multi-oracle" in args.methods and len(oracle_hints) == len(residual_hints):
        sql_by_method["multi-oracle"] = multi_hint_sql(sql, oracle_hints)

    default_shape = ""
    case_rows = []
    for method in args.methods:
        if method not in sql_by_method:
            continue
        row_base = {
            "drift": drift,
            "seed": seed,
            "query_index": query_index,
            "method": method,
            "label": int(label),
            "table_count": int(case["table_count"]),
            "join_count": int(case["join_count"]),
            "predicate_count": int(case["predicate_count"]),
            "subplan_hints": len(sub_rows),
            "subplan_errors": sub_errors,
            "raw_prediction": raw_prediction,
            "residual_prediction": residual_prediction,
            "root_scale": root_scale,
            "sql_sha256": inj.sha256_text(sql),
        }
        try:
            apply_base_planner(cur)
            _doc, stats = run_explain_doc(cur, sql_by_method[method], analyze=True)
            row_base.update(stats)
            row_base["status"] = "executed"
            row_base["error"] = ""
            if method == "default":
                default_shape = stats["plan_shape_sha256"]
        except Exception as exc:
            conn.rollback()
            setup(cur, args)
            row_base.update(
                {
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
        case_rows.append(row_base)

    default_ms = next((row["execution_ms"] for row in case_rows if row["method"] == "default"), None)
    for row_item in case_rows:
        row_item["plan_changed_vs_default"] = bool(
            row_item["method"] != "default"
            and default_shape
            and row_item.get("plan_shape_sha256")
            and row_item["plan_shape_sha256"] != default_shape
        )
        if default_ms and row_item.get("execution_ms"):
            row_item["execution_ms_ratio_vs_default"] = float(row_item["execution_ms"]) / float(default_ms)
        else:
            row_item["execution_ms_ratio_vs_default"] = None

    selections = []
    executable = [row for row in case_rows if row["status"] == "executed" and row["execution_ms"] is not None]
    deployable = [row for row in executable if row["method"] in {"default", "final-residual", "multi-residual"}]
    oracle = [row for row in executable if row["method"] in {"default", "multi-oracle"}]
    if deployable:
        cost_pick = min(deployable, key=lambda row: float(row["total_cost"]))
        best_deployable = min(deployable, key=lambda row: float(row["execution_ms"]))
        selections.append(
            {
                "drift": drift,
                "seed": seed,
                "query_index": query_index,
                "selector": "cost",
                "selected_method": cost_pick["method"],
                "selected_ms": cost_pick["execution_ms"],
                "selected_ratio_vs_default": cost_pick["execution_ms_ratio_vs_default"],
                "best_deployable_method": best_deployable["method"],
                "best_deployable_ratio_vs_default": best_deployable["execution_ms_ratio_vs_default"],
            }
        )
    if any(row["method"] == "multi-oracle" for row in oracle):
        best_oracle = min(oracle, key=lambda row: float(row["execution_ms"]))
        selections.append(
            {
                "drift": drift,
                "seed": seed,
                "query_index": query_index,
                "selector": "oracle-best",
                "selected_method": best_oracle["method"],
                "selected_ms": best_oracle["execution_ms"],
                "selected_ratio_vs_default": best_oracle["execution_ms_ratio_vs_default"],
                "best_deployable_method": "",
                "best_deployable_ratio_vs_default": None,
            }
        )
    conn.commit()
    return case_rows, subplans, selections


def evaluate(args, cases: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    bundles = {(drift, seed): load_target_bundle(drift, seed) for drift in args.drifts for seed in args.seeds}
    subplan_cache: dict[str, dict[str, float]] = {}
    duck_con = duck_probe.connect_duckdb(args) if args.actual_engine == "duckdb" else None
    detail = []
    subplans = []
    selections = []
    with inj.connect(args) as conn:
        with conn.cursor() as cur:
            setup(cur, args)
            for case in cases:
                case_rows, sub_rows, selection_rows = evaluate_case(
                    cur,
                    conn,
                    args,
                    case,
                    bundles[(case["drift"], int(case["seed"]))],
                    subplan_cache,
                    duck_con,
                )
                detail.extend(case_rows)
                subplans.extend(sub_rows)
                selections.extend(selection_rows)
    return detail, subplans, selections


def clean(values: list[float | None]) -> list[float]:
    return [float(value) for value in values if value is not None and math.isfinite(float(value))]


def median(values: list[float | None]) -> float | None:
    data = clean(values)
    return float(np.median(data)) if data else None


def pct(values: list[bool]) -> float:
    return float(np.mean(values)) if values else 0.0


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


def summarize(detail: list[dict], selections: list[dict]) -> list[dict]:
    by_drift_method = defaultdict(list)
    for row in detail:
        by_drift_method[(row["drift"], row["method"])].append(row)
    selection_by_drift = defaultdict(list)
    for row in selections:
        selection_by_drift[(row["drift"], row["selector"])].append(row)

    out = []
    for drift in sorted({row["drift"] for row in detail}):
        for method in [method for method in METHODS if by_drift_method[(drift, method)]]:
            rows = by_drift_method[(drift, method)]
            executed = [row for row in rows if row["status"] == "executed"]
            out.append(
                {
                    "drift": drift,
                    "method": method,
                    "queries": len(rows),
                    "executed_queries": len(executed),
                    "median_subplan_hints": median([row["subplan_hints"] for row in rows]),
                    "plan_change_rate": pct([row["plan_changed_vs_default"] for row in rows if method != "default"]),
                    "median_execution_ms": median([row["execution_ms"] for row in executed]),
                    "median_ratio_vs_default": median([row["execution_ms_ratio_vs_default"] for row in executed]),
                    "median_total_cost": median([row["total_cost"] for row in executed]),
                }
            )
        for selector in ["cost", "oracle-best"]:
            rows = selection_by_drift[(drift, selector)]
            if not rows:
                continue
            out.append(
                {
                    "drift": drift,
                    "method": selector,
                    "queries": len(rows),
                    "executed_queries": len(rows),
                    "median_subplan_hints": None,
                    "plan_change_rate": pct([row["selected_method"] != "default" for row in rows]),
                    "median_execution_ms": median([row["selected_ms"] for row in rows]),
                    "median_ratio_vs_default": median([row["selected_ratio_vs_default"] for row in rows]),
                    "median_total_cost": None,
                }
            )
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def ensure_csv(path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore").writeheader()


def append_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    if not rows:
        return
    ensure_csv(path, fieldnames)
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writerows(rows)
        handle.flush()


def load_subplan_cache(path: Path) -> dict[str, dict[str, float]]:
    cache: dict[str, dict[str, float]] = {}
    for row in read_csv(path):
        sql_hash = row.get("sql_sha256")
        if not sql_hash:
            continue
        try:
            item = {"pg_estimate": float(row["pg_estimate"])}
            if row.get("actual") not in ("", None):
                item["actual"] = float(row["actual"])
            cache[sql_hash] = item
        except (KeyError, TypeError, ValueError):
            continue
    return cache


def load_attempted_cases(path: Path) -> set[tuple[str, int, int]]:
    attempted = set()
    for row in read_csv(path):
        if row.get("drift") and row.get("seed") not in ("", None) and row.get("query_index") not in ("", None):
            attempted.add((row["drift"], int(float(row["seed"])), int(float(row["query_index"]))))
    return attempted


def coerce_detail(rows: list[dict]) -> list[dict]:
    int_fields = [
        "seed",
        "query_index",
        "label",
        "table_count",
        "join_count",
        "predicate_count",
        "subplan_hints",
        "subplan_errors",
        "estimated_node_count",
        "plan_node_count",
        "join_node_count",
    ]
    float_fields = [
        "root_input_estimate",
        "max_estimated_cardinality",
        "execution_ms",
        "planning_ms",
        "wall_seconds",
        "total_cost",
        "raw_prediction",
        "residual_prediction",
        "root_scale",
        "execution_ms_ratio_vs_default",
    ]
    out = []
    for row in rows:
        item = dict(row)
        for field in int_fields:
            item[field] = int(float(item[field])) if item.get(field) not in ("", None) else None
        for field in float_fields:
            item[field] = float(item[field]) if item.get(field) not in ("", None) else None
        item["plan_changed_vs_default"] = str(item.get("plan_changed_vs_default", "")).lower() == "true"
        out.append(item)
    return out


def coerce_selection(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        item = dict(row)
        for field in ["seed", "query_index"]:
            item[field] = int(float(item[field])) if item.get(field) not in ("", None) else None
        for field in ["selected_ms", "selected_ratio_vs_default", "best_deployable_ratio_vs_default"]:
            item[field] = float(item[field]) if item.get(field) not in ("", None) else None
        out.append(item)
    return out


def number(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return "--"
    value = float(value)
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value))}"
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def ratio_number(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return "--"
    return f"{float(value):.2f}"


def pct_str(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return "--"
    return f"{100.0 * float(value):.0f}\\%"


def write_table(path: Path, rows: list[dict]) -> None:
    by_key = {(row["drift"], row["method"]): row for row in rows}
    has_oracle = any(row["method"] == "oracle-best" for row in rows)
    oracle_note = (
        "oracle-best is an upper bound using true subquery counts."
        if has_oracle
        else "oracle-best is omitted for deployable-only stress runs that skip exact subquery counts."
    )
    lines = [
        "\\begin{table}[!t]",
        "\\centering",
        "\\caption{PostgreSQL multi-subplan \\texttt{Rows(...)} injection. \\texttt{Q done} counts completed query groups; the column drift run excludes one 60s timeout group. \\texttt{Chg.} is the cost selector's non-default plan rate; "
        + oracle_note
        + "}",
        "\\label{tab:postgres-multisubplan}",
        "\\papertablesetup",
        "\\setlength{\\tabcolsep}{2.4pt}",
        "\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}l|r r|r r r@{}}",
        "\\toprule",
        "\\tabheader{Drift} & \\tabheader{Q done} & \\tabheader{Subp.} & \\tabheader{Chg.} & \\tabheader{Cost sel.} & \\tabheader{Oracle} \\\\",
        "\\midrule",
    ]
    for drift in sorted({row["drift"] for row in rows}):
        cost = by_key[(drift, "cost")]
        oracle = by_key.get((drift, "oracle-best"))
        lines.append(
            f"{drift.title()} "
            f"& {int(cost['queries'])} "
            f"& {number(by_key[(drift, 'multi-residual')]['median_subplan_hints'])} "
            f"& {pct_str(cost['plan_change_rate'])} "
            f"& {ratio_number(cost['median_ratio_vs_default'])}$\\times$ "
            f"& {ratio_number(oracle['median_ratio_vs_default'] if oracle else None)}$\\times$ \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular*}", "\\end{table}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a multi-subplan pg_hint_plan Rows(...) optimizer-in-loop probe."
    )
    parser.add_argument("--pg-host", default="localhost")
    parser.add_argument("--pg-port", type=int, default=5432)
    parser.add_argument("--pg-database", default="stats_drift")
    parser.add_argument("--pg-user", default="postgres")
    parser.add_argument("--pg-password", default=os.environ.get("PGPASSWORD"))
    parser.add_argument("--schema", default="stats_inj")
    parser.add_argument("--pg-hint-dll", type=Path, default=inj.PG_HINT_DLL)
    parser.add_argument("--stats-data", type=Path, default=inj.STATS_DATA)
    parser.add_argument("--duckdb-probe-csv", type=Path, default=RESULTS / "duckdb_execution_probe.csv")
    parser.add_argument("--output-dir", type=Path, default=RESULTS)
    parser.add_argument("--drifts", default="column,table,value")
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--limit-per-drift", type=int, default=6)
    parser.add_argument("--max-subplans", type=int, default=8)
    parser.add_argument("--statement-timeout-ms", type=int, default=30000)
    parser.add_argument("--actual-engine", choices=["postgres", "duckdb"], default="postgres")
    parser.add_argument(
        "--subplan-actual-mode",
        choices=["exact", "skip"],
        default="exact",
        help="Use exact subplan COUNT(*) labels for oracle/q-error rows, or skip them for deployable-only stress runs.",
    )
    parser.add_argument(
        "--methods",
        default="default,final-residual,multi-residual,multi-oracle",
        help="Comma-separated method subset to execute; default is all methods.",
    )
    parser.add_argument("--duckdb-threads", type=int, default=4)
    parser.add_argument("--memory-limit", default="16GB")
    parser.add_argument("--max-temp-directory-size", default="100GB")
    parser.add_argument(
        "--subplan-cache-csv",
        type=Path,
        default=None,
        help="Optional existing subplan CSV used only to seed exact-count/cache lookups.",
    )
    parser.add_argument("--resume", action="store_true", help="Append new case rows and skip cases already present in the detail CSV.")
    parser.add_argument("--checkpoint-every", type=int, default=1, help="Refresh summary/table/manifest after this many new cases.")
    args = parser.parse_args()

    if not args.output_dir.is_absolute():
        args.output_dir = ROOT / args.output_dir
    if not args.duckdb_probe_csv.is_absolute():
        args.duckdb_probe_csv = ROOT / args.duckdb_probe_csv
    if not args.pg_hint_dll.is_absolute():
        args.pg_hint_dll = ROOT / args.pg_hint_dll
    if not args.stats_data.is_absolute():
        args.stats_data = ROOT / args.stats_data
    if args.subplan_cache_csv is not None and not args.subplan_cache_csv.is_absolute():
        args.subplan_cache_csv = ROOT / args.subplan_cache_csv
    args.drifts = [item.strip() for item in args.drifts.split(",") if item.strip()]
    args.seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    args.methods = parse_methods(args.methods)

    detail_csv = args.output_dir / "postgres_multisubplan_injection_detail.csv"
    subplan_csv = args.output_dir / "postgres_multisubplan_injection_subplans.csv"
    selection_csv = args.output_dir / "postgres_multisubplan_injection_selection.csv"
    summary_csv = args.output_dir / "postgres_multisubplan_injection_summary.csv"
    table_path = args.output_dir / "tables" / "postgres_multisubplan_injection.tex"
    manifest_path = args.output_dir / "postgres_multisubplan_injection_manifest.json"
    subplan_cache = load_subplan_cache(args.subplan_cache_csv or subplan_csv)

    cases = collect_cases(args.duckdb_probe_csv, args.drifts, args.limit_per_drift)
    if args.resume:
        attempted = load_attempted_cases(detail_csv)
        cases = [
            case
            for case in cases
            if (case["drift"], int(case["seed"]), int(case["query_index"])) not in attempted
        ]
        ensure_csv(detail_csv, DETAIL_FIELDS)
        ensure_csv(subplan_csv, SUBPLAN_FIELDS)
        ensure_csv(selection_csv, SELECTION_FIELDS)
    else:
        detail, subplans, selections = evaluate(args, cases)
        write_csv(detail_csv, detail)
        write_csv(subplan_csv, subplans)
        write_csv(selection_csv, selections)

    def refresh_outputs() -> dict:
        detail_rows = coerce_detail(read_csv(detail_csv))
        selection_rows = coerce_selection(read_csv(selection_csv))
        subplan_rows_raw = read_csv(subplan_csv)
        summary_rows = summarize(detail_rows, selection_rows)
        write_csv(summary_csv, summary_rows)
        write_table(table_path, summary_rows)
        attempted_method_executions = len(detail_rows)
        completed_method_executions = sum(1 for row in detail_rows if row.get("status") == "executed")
        completed_cases = len(
            {
                (row["drift"], int(row["seed"]), int(row["query_index"]))
                for row in detail_rows
                if row.get("status") == "executed"
            }
        )
        timeout_queries = len(
            {
                (row["drift"], int(row["seed"]), int(row["query_index"]))
                for row in detail_rows
                if is_timeout_error(row.get("error", ""))
            }
        )
        all_executions_completed = completed_method_executions == attempted_method_executions
        completion_rate = (
            completed_method_executions / attempted_method_executions
            if attempted_method_executions
            else 0.0
        )
        manifest_obj = {
            "passed": bool(detail_rows and completion_rate >= 0.99),
            "scope": "Expanded PostgreSQL optimizer-in-loop multi-subplan Rows hint probe",
            "cases": len({(row["drift"], int(row["seed"]), int(row["query_index"])) for row in detail_rows}),
            "detail_rows": len(detail_rows),
            "subplan_rows": len(subplan_rows_raw),
            "unique_subplan_sql": len({row.get("sql_sha256") for row in subplan_rows_raw if row.get("sql_sha256")}),
            "selection_rows": len(selection_rows),
            "drifts": sorted({row["drift"] for row in detail_rows}) or args.drifts,
            "seeds": sorted({int(row["seed"]) for row in detail_rows}) or args.seeds,
            "limit_per_drift": args.limit_per_drift,
            "max_subplans": args.max_subplans,
            "methods": args.methods,
            "subplan_actual_mode": args.subplan_actual_mode,
            "actual_engine": args.actual_engine,
            "detail_csv": str(detail_csv),
            "subplan_csv": str(subplan_csv),
            "selection_csv": str(selection_csv),
            "summary_csv": str(summary_csv),
            "table": str(table_path),
            "completion_note": (
                f"{completed_method_executions} of {attempted_method_executions} method "
                f"executions completed; {timeout_queries} query group(s) timed out at "
                f"{args.statement_timeout_ms}ms."
            ),
            "all_executions_completed": all_executions_completed,
            "completed_method_executions": completed_method_executions,
            "attempted_method_executions": attempted_method_executions,
            "timeout_queries": timeout_queries,
            "completed_cases": completed_cases,
            "pass_criterion": "usable boundary audit with >=99% completed method executions and documented timeout rows",
        }
        manifest_path.write_text(json.dumps(manifest_obj, indent=2), encoding="utf-8")
        return manifest_obj

    if args.resume and cases:
        bundles = {(drift, seed): load_target_bundle(drift, seed) for drift in args.drifts for seed in args.seeds}
        duck_con = duck_probe.connect_duckdb(args) if args.actual_engine == "duckdb" else None
        with inj.connect(args) as conn:
            with conn.cursor() as cur:
                setup(cur, args)
                for pos, case in enumerate(cases, start=1):
                    case_rows, subplan_rows, selection_rows = evaluate_case(
                        cur,
                        conn,
                        args,
                        case,
                        bundles[(case["drift"], int(case["seed"]))],
                        subplan_cache,
                        duck_con,
                    )
                    append_csv(detail_csv, DETAIL_FIELDS, case_rows)
                    append_csv(subplan_csv, SUBPLAN_FIELDS, subplan_rows)
                    append_csv(selection_csv, SELECTION_FIELDS, selection_rows)
                    print(
                        f"[{pos}/{len(cases)}] {case['drift']} seed={case['seed']} "
                        f"query={case['query_index']} methods={len(case_rows)} "
                        f"subplans={len(subplan_rows)}",
                        flush=True,
                    )
                    if pos % max(1, args.checkpoint_every) == 0:
                        refresh_outputs()

    manifest = refresh_outputs()
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
