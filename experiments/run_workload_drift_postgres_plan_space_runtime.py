import os
import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import run_workload_drift_mscn_finetune as ft
import run_workload_drift_optimizer_risk_proxy as risk
import run_workload_drift_postgres_estimate_injection as inj


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "paper" / "workload_drift_residual_field" / "results"

BASE_SETTINGS = {
    "enable_hashjoin": "on",
    "enable_mergejoin": "on",
    "enable_nestloop": "on",
    "enable_indexscan": "on",
    "enable_indexonlyscan": "on",
    "enable_bitmapscan": "on",
    "join_collapse_limit": "8",
    "from_collapse_limit": "8",
}

VARIANT_SETTINGS = {
    "default": {},
    "sql-order": {
        "join_collapse_limit": "1",
        "from_collapse_limit": "1",
    },
    "hash-only": {
        "enable_hashjoin": "on",
        "enable_mergejoin": "off",
        "enable_nestloop": "off",
    },
    "merge-only": {
        "enable_hashjoin": "off",
        "enable_mergejoin": "on",
        "enable_nestloop": "off",
    },
    "nestloop-only": {
        "enable_hashjoin": "off",
        "enable_mergejoin": "off",
        "enable_nestloop": "on",
    },
    "seq-scan": {
        "enable_indexscan": "off",
        "enable_indexonlyscan": "off",
        "enable_bitmapscan": "off",
    },
}

VARIANT_LABELS = {
    "default": "Default",
    "sql-order": "SQL order",
    "hash-only": "Hash only",
    "merge-only": "Merge only",
    "nestloop-only": "NestLoop only",
    "seq-scan": "Seq scan",
}


def number(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return "--"
    value = float(value)
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def pct(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return "--"
    return f"{100.0 * float(value):.0f}\\%"


def setup_session(cur, args) -> None:
    inj.ensure_pg_hint_loaded(cur, args)
    schema = inj.sql_name(args.schema)
    cur.execute(f"SET search_path TO {schema}, public")
    cur.execute(f"SET statement_timeout = {int(args.statement_timeout_ms)}")
    cur.execute("SET max_parallel_workers_per_gather = 0")
    cur.execute("SET jit = off")


def apply_variant(cur, variant: str) -> None:
    settings = dict(BASE_SETTINGS)
    settings.update(VARIANT_SETTINGS[variant])
    for key, value in settings.items():
        cur.execute(f"SET {key} = {value}")


def load_workload_rows(drifts: list[str]) -> dict[str, list[list[str]]]:
    workload_rows = {}
    for drift in drifts:
        metrics_path = risk.DRIFTS[drift]["metrics"]
        metrics = inj.read_json(metrics_path)
        workload_rows[drift] = inj.load_target_rows(ft.resolve_target_path(metrics_path, metrics))
    return workload_rows


def collect_cases(args) -> list[dict]:
    cases = inj.collect_probe_cases(args.duckdb_probe_csv, args.limit_per_drift)
    allowed = set(args.drifts)
    return [case for case in cases if case["drift"] in allowed]


def evaluate_plan_space(args, cases: list[dict]) -> list[dict]:
    workload_rows = load_workload_rows(sorted({case["drift"] for case in cases}))
    detail_rows = []
    with inj.connect(args) as conn:
        with conn.cursor() as cur:
            setup_session(cur, args)
            for case in cases:
                row = workload_rows[case["drift"]][case["query_index"]]
                sql = inj.query_to_sql(row)
                sql_hash = inj.sha256_text(sql)
                for variant in args.variants:
                    base = {
                        "drift": case["drift"],
                        "seed": case["seed"],
                        "query_index": case["query_index"],
                        "variant": variant,
                        "variant_label": VARIANT_LABELS[variant],
                        "label": case["label"],
                        "table_count": case["table_count"],
                        "join_count": case["join_count"],
                        "predicate_count": case["predicate_count"],
                        "sql_sha256": sql_hash,
                    }
                    try:
                        apply_variant(cur, variant)
                        started = time.perf_counter()
                        stats = inj.run_explain(cur, sql)
                        base.update(stats)
                        base["status"] = "executed"
                        base["error"] = ""
                        base["wall_seconds_outer"] = float(time.perf_counter() - started)
                    except Exception as exc:
                        conn.rollback()
                        setup_session(cur, args)
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
                                "wall_seconds_outer": None,
                                "plan_shape_sha256": "",
                                "status": "error",
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                    detail_rows.append(base)
                conn.commit()
    return detail_rows


def summarize_cases(detail_rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in detail_rows:
        grouped[(row["drift"], row["seed"], row["query_index"])].append(row)

    case_rows = []
    for (drift, seed, query_index), rows in sorted(grouped.items()):
        executed = [row for row in rows if row["status"] == "executed" and row["execution_ms"] is not None]
        default_rows = [row for row in executed if row["variant"] == "default"]
        default_ms = float(default_rows[0]["execution_ms"]) if default_rows else None
        times = [(row["variant"], float(row["execution_ms"])) for row in executed]
        if times:
            best_variant, best_ms = min(times, key=lambda item: item[1])
            worst_variant, worst_ms = max(times, key=lambda item: item[1])
        else:
            best_variant = worst_variant = ""
            best_ms = worst_ms = None
        shapes = {row["plan_shape_sha256"] for row in executed if row["plan_shape_sha256"]}
        timeout_count = sum(1 for row in rows if row["status"] != "executed")
        case_rows.append(
            {
                "drift": drift,
                "seed": seed,
                "query_index": query_index,
                "label": rows[0]["label"],
                "table_count": rows[0]["table_count"],
                "join_count": rows[0]["join_count"],
                "predicate_count": rows[0]["predicate_count"],
                "variant_count": len(rows),
                "executed_variant_count": len(executed),
                "timeout_or_error_count": timeout_count,
                "distinct_plan_shapes": len(shapes),
                "default_ms": default_ms,
                "best_ms": best_ms,
                "worst_ms": worst_ms,
                "best_variant": best_variant,
                "worst_variant": worst_variant,
                "default_over_best": (
                    default_ms / best_ms
                    if default_ms is not None and best_ms and best_ms > 0
                    else None
                ),
                "worst_over_best": (
                    worst_ms / best_ms
                    if worst_ms is not None and best_ms and best_ms > 0
                    else None
                ),
                "best_not_default": bool(best_variant and best_variant != "default"),
                "spread_ge_2x": bool(worst_ms is not None and best_ms and worst_ms / best_ms >= 2.0),
                "spread_ge_5x": bool(worst_ms is not None and best_ms and worst_ms / best_ms >= 5.0),
            }
        )
    return case_rows


def clean(values: list[float | None]) -> list[float]:
    return [float(value) for value in values if value is not None and math.isfinite(float(value))]


def median(values: list[float | None]) -> float | None:
    data = clean(values)
    return float(np.median(data)) if data else None


def percentile(values: list[float | None], q: float) -> float | None:
    data = clean(values)
    return float(np.percentile(data, q)) if data else None


def summarize_drifts(case_rows: list[dict], detail_rows: list[dict]) -> list[dict]:
    detail_by_drift = defaultdict(list)
    for row in detail_rows:
        detail_by_drift[row["drift"]].append(row)
    by_drift = defaultdict(list)
    for row in case_rows:
        by_drift[row["drift"]].append(row)

    out = []
    for drift in sorted(by_drift):
        cases = by_drift[drift]
        details = detail_by_drift[drift]
        out.append(
            {
                "drift": drift,
                "queries": len(cases),
                "variants": len({row["variant"] for row in details}),
                "executed_variant_rate": (
                    sum(1 for row in details if row["status"] == "executed") / len(details)
                    if details
                    else 0.0
                ),
                "median_plan_shapes": median([row["distinct_plan_shapes"] for row in cases]),
                "best_not_default_rate": float(np.mean([row["best_not_default"] for row in cases]))
                if cases
                else 0.0,
                "median_default_over_best": median([row["default_over_best"] for row in cases]),
                "p95_default_over_best": percentile([row["default_over_best"] for row in cases], 95),
                "median_worst_over_best": median([row["worst_over_best"] for row in cases]),
                "p95_worst_over_best": percentile([row["worst_over_best"] for row in cases], 95),
                "spread_ge_2x_rate": float(np.mean([row["spread_ge_2x"] for row in cases])) if cases else 0.0,
                "spread_ge_5x_rate": float(np.mean([row["spread_ge_5x"] for row in cases])) if cases else 0.0,
                "timeout_or_error_variants": sum(int(row["timeout_or_error_count"]) for row in cases),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_table(path: Path, rows: list[dict]) -> None:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{PostgreSQL alternative-plan runtime space on the executable SQL subset. Variants use native planner controls: default, SQL-order planning, hash-only, merge-only, nested-loop-only, and sequential-scan-preferred execution.}",
        "\\label{tab:postgres-plan-space-runtime}",
        "\\papertablesetup",
        "\\setlength{\\tabcolsep}{2.0pt}",
        "\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}lrrrrrr@{}}",
        "\\toprule",
        "\\tabheader{Drift} & \\tabheader{Q} & \\tabheader{Shapes} & \\tabheader{Best$\\ne$def.} & \\tabheader{Def/best} & \\tabheader{Med W/best} & \\tabheader{P95 W/best} \\\\",
        "\\midrule",
    ]
    for row in sorted(rows, key=lambda item: item["drift"]):
        lines.append(
            f"{row['drift'].title()} "
            f"& {int(row['queries'])} "
            f"& {number(row['median_plan_shapes'])} "
            f"& {pct(row['best_not_default_rate'])} "
            f"& {number(row['median_default_over_best'])} "
            f"& {number(row['median_worst_over_best'])} "
            f"& {number(row['p95_worst_over_best'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular*}", "\\end{table}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure PostgreSQL alternative-plan runtime spread on executable Workload-Drift SQL."
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
    parser.add_argument("--reload-data", action="store_true")
    parser.add_argument("--drifts", default="column,table,value")
    parser.add_argument("--variants", default="default,sql-order,hash-only,merge-only,nestloop-only,seq-scan")
    parser.add_argument("--limit-per-drift", type=int, default=0)
    parser.add_argument("--statement-timeout-ms", type=int, default=30000)
    args = parser.parse_args()

    if not args.output_dir.is_absolute():
        args.output_dir = ROOT / args.output_dir
    if not args.pg_hint_dll.is_absolute():
        args.pg_hint_dll = ROOT / args.pg_hint_dll
    if not args.stats_data.is_absolute():
        args.stats_data = ROOT / args.stats_data
    if not args.duckdb_probe_csv.is_absolute():
        args.duckdb_probe_csv = ROOT / args.duckdb_probe_csv
    args.drifts = [item.strip() for item in args.drifts.split(",") if item.strip()]
    args.variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    unknown = [variant for variant in args.variants if variant not in VARIANT_SETTINGS]
    if unknown:
        raise ValueError(f"unknown variants: {unknown}")

    inj.import_stats_data(args)
    cases = collect_cases(args)
    detail_rows = evaluate_plan_space(args, cases)
    case_rows = summarize_cases(detail_rows)
    summary_rows = summarize_drifts(case_rows, detail_rows)

    detail_csv = args.output_dir / "postgres_plan_space_runtime_detail.csv"
    case_csv = args.output_dir / "postgres_plan_space_runtime_cases.csv"
    summary_csv = args.output_dir / "postgres_plan_space_runtime_summary.csv"
    table_path = args.output_dir / "tables" / "postgres_plan_space_runtime.tex"
    manifest_path = args.output_dir / "postgres_plan_space_runtime_manifest.json"

    write_csv(detail_csv, detail_rows)
    write_csv(case_csv, case_rows)
    write_csv(summary_csv, summary_rows)
    write_table(table_path, summary_rows)

    manifest = {
        "scope": "PostgreSQL alternative-plan EXPLAIN ANALYZE runtime spread for executable Workload-Drift SQL",
        "queries": len(cases),
        "detail_rows": len(detail_rows),
        "drifts": args.drifts,
        "variants": args.variants,
        "statement_timeout_ms": args.statement_timeout_ms,
        "detail_csv": str(detail_csv),
        "case_csv": str(case_csv),
        "summary_csv": str(summary_csv),
        "table": str(table_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
