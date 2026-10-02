import os
import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import run_workload_drift_optimizer_risk_proxy as risk
import run_workload_drift_postgres_estimate_injection as inj


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "paper" / "workload_drift_residual_field" / "results"

METHODS = ["postgres-default", "raw-mscn", "residual-field"]
METHOD_LABELS = {
    "postgres-default": "PostgreSQL",
    "raw-mscn": "MSCN injected",
    "residual-field": "Residual injected",
}


def load_target(path: Path) -> tuple[list[list[str]], np.ndarray]:
    rows = inj.load_target_rows(path)
    labels = np.asarray([max(1, int(row[3])) for row in rows], dtype=np.float64)
    return rows, labels


def split_indices(labels: np.ndarray, adaptation_rows: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(labels.shape[0])
    rng.shuffle(indices)
    return indices[:adaptation_rows], indices[adaptation_rows:]


def count_predicates(row: list[str]) -> int:
    return len([part for part in row[2].split(",") if part.strip()]) // 3


def run_explain_only(cur, sql: str) -> dict:
    started = time.perf_counter()
    cur.execute("EXPLAIN (FORMAT JSON) " + sql)
    doc = inj.plan_doc(cur.fetchone()[0])
    stats = inj.collect_plan_stats(doc)
    stats["wall_seconds"] = float(time.perf_counter() - started)
    return stats


def plan_changed(default_stats: dict, method_stats: dict) -> bool:
    return (
        default_stats.get("plan_shape_sha256")
        and method_stats.get("plan_shape_sha256")
        and default_stats["plan_shape_sha256"] != method_stats["plan_shape_sha256"]
    )


def load_drift_predictions(drift: str, seed: int) -> dict:
    spec = risk.DRIFTS[drift]
    metrics_path = spec["metrics"]
    metrics = inj.read_json(metrics_path)
    target_path = risk.ft.resolve_target_path(metrics_path, metrics)
    rows, labels = load_target(target_path)
    raw_predictions = np.loadtxt(metrics["artifacts"]["predictions"], dtype=np.float64)
    if raw_predictions.shape[0] != labels.shape[0]:
        raise ValueError(f"prediction/label mismatch for {drift}")
    adapt_idx, eval_idx = split_indices(labels, spec["adaptation_rows"], seed)
    residual_eval = risk.compute_residual_predictions(
        rows, labels, raw_predictions, adapt_idx, eval_idx, seed
    )
    residual_predictions = np.full(labels.shape[0], np.nan, dtype=np.float64)
    residual_predictions[eval_idx] = residual_eval
    return {
        "drift": drift,
        "seed": seed,
        "metrics_path": metrics_path,
        "target_path": target_path,
        "rows": rows,
        "labels": labels,
        "raw_predictions": raw_predictions,
        "residual_predictions": residual_predictions,
        "adapt_idx": adapt_idx,
        "eval_idx": eval_idx,
        "adaptation_rows": spec["adaptation_rows"],
    }


def qerror_array(predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
    pred = np.maximum(1.0, predictions.astype(np.float64))
    lab = np.maximum(1.0, labels.astype(np.float64))
    return np.maximum(pred / lab, lab / pred)


def candidate_order(bundle: dict, args) -> np.ndarray:
    labels = bundle["labels"]
    rows = bundle["rows"]
    eval_idx = bundle["eval_idx"]
    raw_q = qerror_array(bundle["raw_predictions"], labels)
    candidate_mask = (
        (labels[eval_idx] >= args.min_label)
        & (labels[eval_idx] <= args.max_label)
    )
    candidates = eval_idx[candidate_mask]
    candidates = np.asarray(
        [
            int(idx)
            for idx in candidates
            if len(inj.table_aliases(rows[int(idx)][0])) >= args.min_tables
            and count_predicates(rows[int(idx)]) >= args.min_predicates
            and not math.isnan(float(bundle["residual_predictions"][int(idx)]))
        ],
        dtype=np.int64,
    )
    if candidates.shape[0] == 0:
        return candidates
    if args.order == "raw-qerror":
        scores = raw_q[candidates]
    elif args.order == "residual-delta":
        residual = bundle["residual_predictions"][candidates]
        raw = bundle["raw_predictions"][candidates]
        scores = np.abs(np.log(np.maximum(1.0, residual)) - np.log(np.maximum(1.0, raw)))
    else:
        scores = labels[candidates]
    order = np.argsort(scores)[::-1]
    return candidates[order]


def setup_session(cur, args) -> None:
    inj.ensure_pg_hint_loaded(cur, args)
    schema = inj.sql_name(args.schema)
    cur.execute(f"SET search_path TO {schema}, public")
    cur.execute(f"SET statement_timeout = {int(args.statement_timeout_ms)}")
    cur.execute("SET max_parallel_workers_per_gather = 0")


def search_plan_changes(args) -> tuple[list[dict], list[dict]]:
    changes = []
    audit = []
    with inj.connect(args) as conn:
        with conn.cursor() as cur:
            setup_session(cur, args)
            for drift in args.drifts:
                for seed in args.seeds:
                    bundle = load_drift_predictions(drift, seed)
                    ordered = candidate_order(bundle, args)
                    scanned = 0
                    explain_errors = 0
                    local_changes = 0
                    start = time.perf_counter()
                    for query_index in ordered[: args.max_candidates_per_split]:
                        query_index = int(query_index)
                        row = bundle["rows"][query_index]
                        sql = inj.query_to_sql(row)
                        aliases = inj.table_aliases(row[0])
                        try:
                            default_stats = run_explain_only(cur, sql)
                        except Exception as exc:
                            conn.rollback()
                            setup_session(cur, args)
                            explain_errors += 1
                            if args.keep_errors:
                                changes.append(
                                    {
                                        "drift": drift,
                                        "seed": seed,
                                        "query_index": query_index,
                                        "method": "postgres-default",
                                        "error": f"{type(exc).__name__}: {exc}",
                                    }
                                )
                            continue
                        scanned += 1
                        predictions = {
                            "raw-mscn": float(bundle["raw_predictions"][query_index]),
                            "residual-field": float(bundle["residual_predictions"][query_index]),
                        }
                        for method, prediction in predictions.items():
                            try:
                                hinted_sql, injected, hint = inj.inject_rows_hint(sql, aliases, prediction)
                                method_stats = run_explain_only(cur, hinted_sql)
                            except Exception as exc:
                                conn.rollback()
                                setup_session(cur, args)
                                explain_errors += 1
                                if args.keep_errors:
                                    changes.append(
                                        {
                                            "drift": drift,
                                            "seed": seed,
                                            "query_index": query_index,
                                            "method": method,
                                            "error": f"{type(exc).__name__}: {exc}",
                                        }
                                    )
                                continue
                            changed = plan_changed(default_stats, method_stats)
                            if changed:
                                local_changes += 1
                                changes.append(
                                    {
                                        "drift": drift,
                                        "seed": seed,
                                        "query_index": query_index,
                                        "method": method,
                                        "label": int(bundle["labels"][query_index]),
                                        "table_count": len(aliases),
                                        "join_count": len([part for part in row[1].split(",") if part.strip()]),
                                        "predicate_count": count_predicates(row),
                                        "prediction": prediction,
                                        "prediction_qerror": inj.qerror(prediction, bundle["labels"][query_index]),
                                        "postgres_default_root_estimate": default_stats["root_input_estimate"],
                                        "injected_estimate": injected,
                                        "default_plan_shape_sha256": default_stats["plan_shape_sha256"],
                                        "method_plan_shape_sha256": method_stats["plan_shape_sha256"],
                                        "default_root_input_estimate": default_stats["root_input_estimate"],
                                        "method_root_input_estimate": method_stats["root_input_estimate"],
                                        "default_plan_node_count": default_stats["plan_node_count"],
                                        "method_plan_node_count": method_stats["plan_node_count"],
                                        "default_join_node_count": default_stats["join_node_count"],
                                        "method_join_node_count": method_stats["join_node_count"],
                                        "hint": hint,
                                        "sql_sha256": inj.sha256_text(sql),
                                    }
                                )
                                if local_changes >= args.target_changes_per_split:
                                    break
                        if local_changes >= args.target_changes_per_split:
                            break
                    audit.append(
                        {
                            "drift": drift,
                            "seed": seed,
                            "candidate_rows": int(ordered.shape[0]),
                            "scanned_rows": int(scanned),
                            "explain_errors": int(explain_errors),
                            "plan_changes": int(local_changes),
                            "seconds": float(time.perf_counter() - start),
                            "order": args.order,
                            "min_label": args.min_label,
                            "max_label": args.max_label,
                        }
                    )
                    conn.commit()
    return changes, audit


def unique_changed_cases(changes: list[dict]) -> list[dict]:
    grouped = {}
    for row in changes:
        if "error" in row:
            continue
        key = (row["drift"], row["seed"], row["query_index"])
        case = grouped.setdefault(
            key,
            {
                "drift": row["drift"],
                "seed": row["seed"],
                "query_index": int(row["query_index"]),
                "label": int(row["label"]),
                "raw_label": int(row["label"]),
                "table_count": int(row["table_count"]),
                "join_count": int(row["join_count"]),
                "predicate_count": int(row["predicate_count"]),
                "predictions": {},
            },
        )
    for drift, seed in sorted({(row["drift"], row["seed"]) for row in grouped.values()}):
        bundle = load_drift_predictions(drift, seed)
        for case in grouped.values():
            if case["drift"] == drift and case["seed"] == seed:
                idx = case["query_index"]
                case["predictions"] = {
                    "raw-mscn": float(bundle["raw_predictions"][idx]),
                    "residual-field": float(bundle["residual_predictions"][idx]),
                    "mscn-finetune-best": float(bundle["raw_predictions"][idx]),
                }
    return sorted(grouped.values(), key=lambda item: (item["drift"], item["seed"], item["query_index"]))


def evaluate_changed_cases(args, cases: list[dict]) -> list[dict]:
    if not cases:
        return []
    detail = inj.evaluate_cases(args, cases)
    return [row for row in detail if row["method"] in METHODS]


def summarize_plan_sensitive(rows: list[dict]) -> list[dict]:
    if not rows:
        return []
    summary = [row for row in inj.summarize(rows) if row["method"] in METHODS]
    for row in summary:
        row["changed_case_set"] = "plan-sensitive"
    return summary


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


def write_table(path: Path, rows: list[dict], audit: list[dict]) -> None:
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{PostgreSQL plan-sensitive search for learned estimate injection. Candidates are held-out target queries searched directly in PostgreSQL; retained rows are those where a learned \\texttt{Rows(...)} estimate changes the plan shape under \\texttt{EXPLAIN}.}",
        "\\label{tab:postgres-plan-sensitive}",
        "\\begin{tabular}{llrrrrrr}",
        "\\toprule",
        "Drift & Method & Rows & Hint OK & Plan chg. & Med. plan Q & Med. ms & Med. ms/default \\\\",
        "\\midrule",
    ]
    order = {method: idx for idx, method in enumerate(METHODS)}
    for row in sorted(rows, key=lambda item: (item["drift"], order[item["method"]])):
        hint_ok = "--" if row["hint_applied_rate"] is None else inj.pct(row["hint_applied_rate"]) + "\\%"
        plan_change = "--" if row["plan_change_rate"] is None else inj.pct(row["plan_change_rate"]) + "\\%"
        ratio = "--" if row["median_exec_ratio_vs_default"] is None else f"{row['median_exec_ratio_vs_default']:.2f}$\\times$"
        lines.append(
            f"{row['drift'].title()} & {METHOD_LABELS[row['method']]} "
            f"& {int(row['queries'])} "
            f"& {hint_ok} "
            f"& {plan_change} "
            f"& {inj.number(row['median_plan_qerror'])} "
            f"& {inj.number(row['median_execution_ms'])} "
            f"& {ratio} \\\\"
        )
    total_scanned = sum(int(row["scanned_rows"]) for row in audit)
    total_candidates = sum(int(row["candidate_rows"]) for row in audit)
    total_changes = sum(int(row["plan_changes"]) for row in audit)
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            f"\\vspace{{2pt}}\\footnotesize Searched {total_scanned} of {total_candidates} eligible PostgreSQL candidates and found {total_changes} learned-estimate plan changes.",
            "\\end{table*}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_audit_table(path: Path, rows: list[dict]) -> None:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["drift"]].append(row)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{PostgreSQL plan-sensitivity audit for learned final-join \\texttt{Rows(...)} estimates. The search scans held-out target queries directly in PostgreSQL and counts whether raw-MSCN or residual-field injection changes the plan shape under \\texttt{EXPLAIN}.}",
        "\\label{tab:postgres-plan-search-audit}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Drift & Splits & Eligible & Scanned & Plan chg. \\\\",
        "\\midrule",
    ]
    for drift in sorted(grouped):
        items = grouped[drift]
        lines.append(
            f"{drift.title()} "
            f"& {len(items)} "
            f"& {sum(int(item['candidate_rows']) for item in items)} "
            f"& {sum(int(item['scanned_rows']) for item in items)} "
            f"& {sum(int(item['plan_changes']) for item in items)} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Search PostgreSQL queries whose plans change under learned Rows hints.")
    parser.add_argument("--pg-host", default="localhost")
    parser.add_argument("--pg-port", type=int, default=5432)
    parser.add_argument("--pg-database", default="stats_drift")
    parser.add_argument("--pg-user", default="postgres")
    parser.add_argument("--pg-password", default=os.environ.get("PGPASSWORD"))
    parser.add_argument("--schema", default="stats_inj")
    parser.add_argument("--pg-hint-dll", type=Path, default=inj.PG_HINT_DLL)
    parser.add_argument("--stats-data", type=Path, default=inj.STATS_DATA)
    parser.add_argument("--output-dir", type=Path, default=RESULTS)
    parser.add_argument("--reload-data", action="store_true")
    parser.add_argument("--drifts", default="column,table,value")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--min-label", type=int, default=10)
    parser.add_argument("--max-label", type=int, default=1_000_000)
    parser.add_argument("--min-tables", type=int, default=3)
    parser.add_argument("--min-predicates", type=int, default=2)
    parser.add_argument("--max-candidates-per-split", type=int, default=800)
    parser.add_argument("--target-changes-per-split", type=int, default=3)
    parser.add_argument("--statement-timeout-ms", type=int, default=30000)
    parser.add_argument("--order", choices=["raw-qerror", "residual-delta", "label"], default="raw-qerror")
    parser.add_argument("--keep-errors", action="store_true")
    parser.add_argument("--no-analyze", action="store_true")
    args = parser.parse_args()

    if not args.output_dir.is_absolute():
        args.output_dir = ROOT / args.output_dir
    if not args.pg_hint_dll.is_absolute():
        args.pg_hint_dll = ROOT / args.pg_hint_dll
    if not args.stats_data.is_absolute():
        args.stats_data = ROOT / args.stats_data
    args.drifts = [item.strip() for item in args.drifts.split(",") if item.strip()]
    args.seeds = [int(item) for item in args.seeds.split(",") if item.strip()]

    inj.import_stats_data(args)
    changes, audit = search_plan_changes(args)
    cases = unique_changed_cases(changes)
    detail_rows = [] if args.no_analyze else evaluate_changed_cases(args, cases)
    summary_rows = summarize_plan_sensitive(detail_rows)

    changed_csv = args.output_dir / "postgres_plan_sensitive_search.csv"
    audit_csv = args.output_dir / "postgres_plan_sensitive_search_audit.csv"
    detail_csv = args.output_dir / "postgres_plan_sensitive_injection.csv"
    summary_csv = args.output_dir / "postgres_plan_sensitive_injection_summary.csv"
    table_path = args.output_dir / "tables" / "postgres_plan_sensitive_injection.tex"
    audit_table_path = args.output_dir / "tables" / "postgres_plan_search_audit.tex"
    manifest_path = args.output_dir / "postgres_plan_sensitive_manifest.json"

    write_csv(changed_csv, changes)
    write_csv(audit_csv, audit)
    write_csv(detail_csv, detail_rows)
    write_csv(summary_csv, summary_rows)
    write_audit_table(audit_table_path, audit)
    if summary_rows:
        write_table(table_path, summary_rows, audit)

    manifest = {
        "passed": bool(changes),
        "scope": "PostgreSQL search for plan-shape changes under learned Rows estimate injection",
        "drifts": args.drifts,
        "seeds": args.seeds,
        "candidate_rows": int(sum(row["candidate_rows"] for row in audit)),
        "scanned_rows": int(sum(row["scanned_rows"] for row in audit)),
        "plan_changes": int(sum(row["plan_changes"] for row in audit)),
        "unique_changed_cases": len(cases),
        "analyzed_rows": len(detail_rows),
        "changed_csv": str(changed_csv),
        "audit_csv": str(audit_csv),
        "detail_csv": str(detail_csv),
        "summary_csv": str(summary_csv),
        "table": str(table_path) if summary_rows else None,
        "audit_table": str(audit_table_path),
        "statement_timeout_ms": args.statement_timeout_ms,
        "order": args.order,
        "min_label": args.min_label,
        "max_label": args.max_label,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
