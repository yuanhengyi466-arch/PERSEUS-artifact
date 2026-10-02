import argparse
import csv
import hashlib
import json
import math
import re
import time
from pathlib import Path
from types import SimpleNamespace

import duckdb
import numpy as np

import run_workload_drift_mscn_finetune as ft
import run_workload_drift_mscn_windows as mscn_run
import run_workload_drift_optimizer_risk_proxy as risk


ROOT = Path(__file__).resolve().parents[1]
STATS_DATA = ROOT / "external_data" / "workload_drift" / "extracted" / "dataset" / "stats"


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


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sql_ident(name: str) -> str:
    escaped = name.replace('"', '""')
    return f'"{escaped}"'


def typed_expression(column: str, dtype: str, kind: str | None) -> str:
    source = sql_ident(column)
    target = sql_ident(column)
    if kind == "datetime":
        return (
            f"CAST(epoch(strptime({source}, '%Y-%m-%d %H:%M:%S')) AS {dtype}) AS {target}"
        )
    return f"CAST({source} AS {dtype}) AS {target}"


def connect_duckdb(args) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute(f"PRAGMA threads={int(args.duckdb_threads)}")
    con.execute(f"PRAGMA memory_limit='{args.memory_limit}'")
    con.execute("PRAGMA preserve_insertion_order=false")
    con.execute(f"PRAGMA max_temp_directory_size='{args.max_temp_directory_size}'")
    for table, columns in TABLES.items():
        path = (args.stats_data / f"{table}.csv").as_posix()
        select_list = ", ".join(typed_expression(*column) for column in columns)
        con.execute(
            f"""
            CREATE TABLE {sql_ident(table)} AS
            SELECT {select_list}
            FROM read_csv_auto('{path}', header=true, all_varchar=true, nullstr='')
            """
        )
    return con


def load_target_rows(path: Path):
    rows = []
    raw_labels = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter="#")
        for row in reader:
            if len(row) != 4:
                raise ValueError(f"expected 4 fields in {path}, got {len(row)}")
            rows.append(row)
            raw_labels.append(int(row[3]))
    ce_labels = np.maximum(1, np.asarray(raw_labels, dtype=np.float64))
    return rows, np.asarray(raw_labels, dtype=np.int64), ce_labels


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


def parse_estimated_cardinality(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", str(value))
    if not match:
        return None
    return float(match.group(0).replace(",", ""))


def node_estimate(node: dict) -> float | None:
    extra = node.get("extra_info") or {}
    for key in ["Estimated Cardinality", "EC"]:
        if key in extra:
            return parse_estimated_cardinality(extra[key])
    return None


def first_estimate(node: dict) -> float | None:
    estimate = node_estimate(node)
    if estimate is not None:
        return estimate
    for child in node.get("children") or []:
        estimate = first_estimate(child)
        if estimate is not None:
            return estimate
    return None


def collect_plan_stats(plan_nodes: list[dict]) -> dict:
    operator_counts = {}
    estimates = []

    def visit(node: dict) -> None:
        name = str(node.get("name", "")).strip()
        if name:
            operator_counts[name] = operator_counts.get(name, 0) + 1
        estimate = node_estimate(node)
        if estimate is not None:
            estimates.append(estimate)
        for child in node.get("children") or []:
            visit(child)

    for node in plan_nodes:
        visit(node)

    root_input_estimate = None
    if plan_nodes:
        root = plan_nodes[0]
        children = root.get("children") or []
        if children:
            root_input_estimate = first_estimate(children[0])
        else:
            root_input_estimate = first_estimate(root)
    return {
        "root_input_estimate": root_input_estimate,
        "max_estimated_cardinality": max(estimates) if estimates else None,
        "estimated_node_count": len(estimates),
        "operator_counts": operator_counts,
        "plan_node_count": int(sum(operator_counts.values())),
    }


def execute_probe_query(con: duckdb.DuckDBPyConnection, sql: str, label: int) -> dict:
    try:
        explain_text = con.execute(f"EXPLAIN (FORMAT JSON) {sql}").fetchone()[1]
        plan = json.loads(explain_text)
        plan_stats = collect_plan_stats(plan)
    except Exception as exc:
        return {
            "sql_sha256": sha256_text(sql),
            "actual_count": None,
            "label": int(label),
            "label_matches": False,
            "count_seconds": 0.0,
            "execution_status": "explain_error",
            "execution_error": f"{type(exc).__name__}: {exc}",
            "root_input_estimate": None,
            "max_estimated_cardinality": None,
            "estimated_node_count": 0,
            "operator_counts": {},
            "plan_node_count": 0,
        }
    start = time.perf_counter()
    try:
        actual = int(con.execute(sql).fetchone()[0])
        count_seconds = time.perf_counter() - start
        execution_status = "executed"
        execution_error = ""
    except Exception as exc:
        actual = None
        count_seconds = time.perf_counter() - start
        execution_status = "count_error"
        execution_error = f"{type(exc).__name__}: {exc}"
    return {
        "sql_sha256": sha256_text(sql),
        "actual_count": actual,
        "label": int(label),
        "label_matches": bool(actual == int(label)) if actual is not None else False,
        "count_seconds": float(count_seconds),
        "execution_status": execution_status,
        "execution_error": execution_error,
        **plan_stats,
    }


def qerror_value(prediction: float, label: float) -> float:
    pred = max(1.0, float(prediction))
    lab = max(1.0, float(label))
    return max(pred / lab, lab / pred)


def error_metrics(prediction: float, label: float) -> dict:
    pred = max(1.0, float(prediction))
    lab = max(1.0, float(label))
    return {
        "prediction": float(pred),
        "qerror": qerror_value(pred, lab),
        "under_factor": float(lab / pred),
        "over_factor": float(pred / lab),
        "severe_under100": bool((lab / pred) >= 100.0),
        "severe_over100": bool((pred / lab) >= 100.0),
        "danger_q100": bool(qerror_value(pred, lab) >= 100.0),
        "catastrophic_q1000": bool(qerror_value(pred, lab) >= 1000.0),
    }


def summarize_rows(rows: list[dict]) -> dict:
    q = np.asarray([row["qerror"] for row in rows], dtype=np.float64)
    executed_rows = [row for row in rows if row["execution_status"] == "executed"]
    count_seconds = np.asarray([row["count_seconds"] for row in executed_rows], dtype=np.float64)
    label_matches = np.asarray([row["label_matches"] for row in executed_rows], dtype=np.float64)
    return {
        "sample_rows": int(len(rows)),
        "unique_query_rows": int(len({(row["seed"], row["query_index"]) for row in rows})),
        "unique_sql_queries": int(len({row["sql_sha256"] for row in rows})),
        "executed_rows": int(len(executed_rows)),
        "execution_success_rate": float(len(executed_rows) / len(rows)) if rows else 0.0,
        "label_match_rate": float(np.mean(label_matches)) if executed_rows else 0.0,
        "count_seconds_median": float(np.median(count_seconds)) if executed_rows else 0.0,
        "count_seconds_p95": float(np.percentile(count_seconds, 95)) if executed_rows else 0.0,
        "median_qerror": float(np.median(q)) if rows else 0.0,
        "p90_qerror": float(np.percentile(q, 90)) if rows else 0.0,
        "p95_qerror": float(np.percentile(q, 95)) if rows else 0.0,
        "p99_qerror": float(np.percentile(q, 99)) if rows else 0.0,
        "max_qerror": float(np.max(q)) if rows else 0.0,
        "danger_q100_rate": float(np.mean([row["danger_q100"] for row in rows])) if rows else 0.0,
        "catastrophic_q1000_rate": float(np.mean([row["catastrophic_q1000"] for row in rows])) if rows else 0.0,
        "severe_under100_rate": float(np.mean([row["severe_under100"] for row in rows])) if rows else 0.0,
        "severe_over100_rate": float(np.mean([row["severe_over100"] for row in rows])) if rows else 0.0,
    }


def aggregate_method(method_rows: dict[str, list[dict]]) -> dict:
    aggregate = {}
    baseline_median = None
    baseline_p95 = None
    baseline_under = None
    for method, rows in method_rows.items():
        metrics = summarize_rows(rows)
        if method == "raw-mscn":
            baseline_median = metrics["median_qerror"]
            baseline_p95 = metrics["p95_qerror"]
            baseline_under = metrics["severe_under100_rate"]
        aggregate[method] = metrics
    for method, metrics in aggregate.items():
        metrics["median_qerror_reduction"] = (
            (baseline_median - metrics["median_qerror"]) / baseline_median
            if baseline_median
            else 0.0
        )
        metrics["p95_qerror_reduction"] = (
            (baseline_p95 - metrics["p95_qerror"]) / baseline_p95 if baseline_p95 else 0.0
        )
        metrics["severe_under100_reduction"] = (
            (baseline_under - metrics["severe_under100_rate"]) / baseline_under
            if baseline_under
            else 0.0
        )
    return aggregate


def compute_finetune_predictions(metrics: dict, encoded: dict, adapt_idx, eval_idx, seed: int, args):
    proxy_args = SimpleNamespace(
        finetune_batch_size=args.finetune_batch_size,
        finetune_lr=args.finetune_lr,
        finetune_epochs=args.finetune_epochs,
        threads=args.threads,
    )
    return risk.compute_finetune_predictions(metrics, encoded, adapt_idx, eval_idx, seed, proxy_args)


def evaluate_drift(drift: str, con: duckdb.DuckDBPyConnection, args) -> dict:
    spec = risk.DRIFTS[drift]
    metrics_path = spec["metrics"]
    metrics = read_json(metrics_path)
    target_path = ft.resolve_target_path(metrics_path, metrics)
    rows, raw_labels, labels = load_target_rows(target_path)
    raw_predictions = np.loadtxt(metrics["artifacts"]["predictions"], dtype=np.float64)
    if raw_predictions.shape[0] != labels.shape[0]:
        raise ValueError(f"prediction/label mismatch for {drift}")

    encoding_args = SimpleNamespace(
        num_materialized_samples=metrics["num_materialized_samples"],
        train_bitmap_file=metrics["provenance"]["bitmap_files"]["train_bitmap_file"],
        predict_bitmap_file=metrics["provenance"]["bitmap_files"]["predict_bitmap_file"],
    )
    encoded = mscn_run.encode_dataset(
        Path(metrics_path.parent / "data" / metrics["provenance"]["train_rel"]),
        target_path,
        Path(metrics_path.parent / "data" / "baselines" / "MSCN-modified" / "workloads" / "stats"),
        metrics["seed"],
        encoding_args,
    )

    execution_cache = {}
    details = []
    split_reports = []
    method_rows = {
        "raw-mscn": [],
        "residual-field": [],
        "mscn-finetune-best": [],
        "duckdb-plan": [],
    }
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        indices = np.arange(labels.shape[0])
        rng.shuffle(indices)
        adapt_idx = indices[: spec["adaptation_rows"]]
        eval_idx = indices[spec["adaptation_rows"] :]
        eval_labels = labels[eval_idx]
        raw_eval = raw_predictions[eval_idx]
        residual_eval = risk.compute_residual_predictions(
            rows, labels, raw_predictions, adapt_idx, eval_idx, seed
        )
        finetune_eval, finetune_seconds, finetune_losses = compute_finetune_predictions(
            metrics, encoded, adapt_idx, eval_idx, seed, args
        )
        raw_q = np.asarray([qerror_value(pred, lab) for pred, lab in zip(raw_eval, eval_labels)])
        raw_eval_labels = raw_labels[eval_idx]
        candidate_local = np.flatnonzero(
            (raw_eval_labels >= args.min_execution_label)
            & (raw_eval_labels <= args.max_execution_label)
        )
        target_all_candidates = args.top_k_per_seed <= 0
        if not target_all_candidates and candidate_local.shape[0] < args.top_k_per_seed:
            raise ValueError(
                f"{drift} seed {seed} has only {candidate_local.shape[0]} executable "
                f"candidates under max_execution_label={args.max_execution_label}"
            )
        ranked_local = candidate_local[np.argsort(raw_q[candidate_local])[::-1]]
        selected_local_values = []
        skipped_mismatch = 0
        skipped_errors = 0
        for candidate_pos in ranked_local:
            query_index = int(eval_idx[candidate_pos])
            raw_label = int(raw_labels[query_index])
            sql = query_to_sql(rows[query_index])
            if query_index not in execution_cache:
                execution_cache[query_index] = execute_probe_query(con, sql, raw_label)
            execution = execution_cache[query_index]
            if execution["execution_status"] != "executed":
                skipped_errors += 1
                continue
            if not execution["label_matches"]:
                skipped_mismatch += 1
                continue
            selected_local_values.append(int(candidate_pos))
            if not target_all_candidates and len(selected_local_values) == args.top_k_per_seed:
                break
        if not target_all_candidates and len(selected_local_values) < args.top_k_per_seed:
            raise ValueError(
                f"{drift} seed {seed} found only {len(selected_local_values)} label-matching "
                f"executable candidates after scanning {candidate_local.shape[0]} candidates"
            )
        selected_local = np.asarray(selected_local_values, dtype=np.int64)
        selected_indices = eval_idx[selected_local]
        split = {
            "seed": seed,
            "adaptation_rows": int(adapt_idx.shape[0]),
            "eval_rows": int(eval_idx.shape[0]),
            "selected_rows": int(selected_indices.shape[0]),
            "finetune_seconds": float(finetune_seconds),
            "finetune_epoch_losses": finetune_losses,
            "selected_query_indices": [int(idx) for idx in selected_indices],
            "min_execution_label": int(args.min_execution_label),
            "max_execution_label": int(args.max_execution_label),
            "candidate_rows_under_label_cap": int(candidate_local.shape[0]),
            "skipped_label_mismatch": int(skipped_mismatch),
            "skipped_execution_error": int(skipped_errors),
            "selected_min_raw_qerror": float(np.min(raw_q[selected_local])),
            "selected_max_raw_qerror": float(np.max(raw_q[selected_local])),
        }
        for local_pos, query_index in zip(selected_local, selected_indices):
            query_index = int(query_index)
            label = float(labels[query_index])
            raw_label = int(raw_labels[query_index])
            sql = query_to_sql(rows[query_index])
            if query_index not in execution_cache:
                execution_cache[query_index] = execute_probe_query(con, sql, raw_label)
            execution = execution_cache[query_index]
            predictions = {
                "raw-mscn": float(raw_predictions[query_index]),
                "residual-field": float(residual_eval[local_pos]),
                "mscn-finetune-best": float(finetune_eval[local_pos]),
                "duckdb-plan": float(execution["root_input_estimate"] or 1.0),
            }
            for method, prediction in predictions.items():
                row = {
                    "drift": drift,
                    "seed": seed,
                    "query_index": query_index,
                    "method": method,
                    "raw_label": raw_label,
                    "label": int(label),
                    "table_count": len([part for part in rows[query_index][0].split(",") if part.strip()]),
                    "join_count": len([part for part in rows[query_index][1].split(",") if part.strip()]),
                    "predicate_count": len(
                        [part for part in rows[query_index][2].split(",") if part.strip()]
                    )
                    // 3,
                    **error_metrics(prediction, label),
                    **execution,
                }
                details.append(row)
                method_rows[method].append(row)
        selected_executions = [execution_cache[int(idx)] for idx in selected_indices]
        executed = [item for item in selected_executions if item["execution_status"] == "executed"]
        split["execution_success_rate"] = float(len(executed) / len(selected_executions))
        split["label_match_rate"] = float(
            np.mean([item["label_matches"] for item in executed])
        ) if executed else 0.0
        split_reports.append(split)
    return {
        "drift": drift,
        "baseline_metrics": str(metrics_path),
        "target_workload": str(target_path),
        "adaptation_rows": int(spec["adaptation_rows"]),
        "top_k_per_seed": int(args.top_k_per_seed),
        "min_execution_label": int(args.min_execution_label),
        "max_execution_label": int(args.max_execution_label),
        "seed_count": len(args.seeds),
        "split_reports": split_reports,
        "aggregate": aggregate_method(method_rows),
        "details": details,
    }


def write_detail_csv(path: Path, reports: list[dict]) -> list[dict]:
    rows = [row for report in reports for row in report["details"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    drop_keys = {"operator_counts"}
    fieldnames = [key for key in rows[0].keys() if key not in drop_keys]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_summary_csv(path: Path, reports: list[dict]) -> list[dict]:
    rows = []
    for report in reports:
        for method, metrics in report["aggregate"].items():
            rows.append(
                {
                    "drift": report["drift"],
                    "method": method,
                    "adaptation_rows": report["adaptation_rows"],
                    "top_k_per_seed": report["top_k_per_seed"],
                    "min_execution_label": report["min_execution_label"],
                    "max_execution_label": report["max_execution_label"],
                    **metrics,
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def aggregate_selection_audit(reports: list[dict]) -> list[dict]:
    rows = []
    for report in reports:
        splits = report["split_reports"]
        rows.append(
            {
                "drift": report["drift"],
                "splits": len(splits),
                "candidate_rows_min": min(item["candidate_rows_under_label_cap"] for item in splits),
                "candidate_rows_max": max(item["candidate_rows_under_label_cap"] for item in splits),
                "selected_rows": sum(item["selected_rows"] for item in splits),
                "skipped_label_mismatch": sum(item["skipped_label_mismatch"] for item in splits),
                "skipped_execution_error": sum(item["skipped_execution_error"] for item in splits),
                "selected_min_raw_qerror": min(item["selected_min_raw_qerror"] for item in splits),
                "selected_max_raw_qerror": max(item["selected_max_raw_qerror"] for item in splits),
                "execution_success_rate": float(
                    np.mean([item["execution_success_rate"] for item in splits])
                ),
                "label_match_rate": float(np.mean([item["label_match_rate"] for item in splits])),
                "min_execution_label": report["min_execution_label"],
                "max_execution_label": report["max_execution_label"],
            }
        )
    return rows


def write_selection_audit_csv(path: Path, reports: list[dict]) -> list[dict]:
    rows = aggregate_selection_audit(reports)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}"


def number(value: float) -> str:
    if value >= 1000:
        return f"{value:.0f}"
    if value >= 100:
        return f"{value:.1f}"
    return f"{value:.2f}"


def write_table(path: Path, rows: list[dict]) -> None:
    labels = {
        "raw-mscn": "MSCN",
        "residual-field": "Residual-field",
        "mscn-finetune-best": "MSCN fine-tune",
    }
    order = {"raw-mscn": 0, "residual-field": 1, "mscn-finetune-best": 2}
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{DuckDB SQL sanity check on executable raw-MSCN tail queries. Each selected benchmark CSV row is converted to SQL, executed as \\texttt{COUNT(*)}, and checked against its label.}",
        "\\label{tab:duckdb-probe}",
        "\\begin{tabular}{llrrrrrrr}",
        "\\toprule",
        "Drift & Estimator & Rows & Exec. OK & Label OK & Median Q & P95 Q & P95 Red. & $Q\\ge100$ \\\\",
        "\\midrule",
    ]
    filtered_rows = [row for row in rows if row["method"] in order]
    for row in sorted(filtered_rows, key=lambda item: (item["drift"], order[item["method"]])):
        lines.append(
            f"{row['drift'].title()} & {labels[row['method']]} "
            f"& {int(row['sample_rows'])} "
            f"& {pct(row['execution_success_rate'])}\\% "
            f"& {pct(row['label_match_rate'])}\\% "
            f"& {number(row['median_qerror'])} "
            f"& {number(row['p95_qerror'])} "
            f"& {pct(row['p95_qerror_reduction'])}\\% "
            f"& {pct(row['danger_q100_rate'])}\\% \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_selection_audit_table(path: Path, rows: list[dict]) -> None:
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Selection audit for the DuckDB SQL sanity check. Candidates are held-out raw-MSCN tail queries with raw labels in the configured executable range.}",
        "\\label{tab:duckdb-selection}",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Drift & Cand. range & Kept & Label skip & Exec. skip & Raw-Q range \\\\",
        "\\midrule",
    ]
    for row in sorted(rows, key=lambda item: item["drift"]):
        lines.append(
            f"{row['drift'].title()} "
            f"& {int(row['candidate_rows_min'])}--{int(row['candidate_rows_max'])} "
            f"& {int(row['selected_rows'])} "
            f"& {int(row['skipped_label_mismatch'])} "
            f"& {int(row['skipped_execution_error'])} "
            f"& {number(row['selected_min_raw_qerror'])}--{number(row['selected_max_raw_qerror'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_plan_reference_table(path: Path, rows: list[dict]) -> None:
    plan_rows = [row for row in rows if row["method"] == "duckdb-plan"]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{DuckDB root-plan estimate on the same SQL sanity-check subset. This is a database-engine reference, not a learned-estimator baseline.}",
        "\\label{tab:duckdb-plan-reference}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Drift & Median Q & P95 Q & $Q\\ge100$ & P95 Red. \\\\",
        "\\midrule",
    ]
    for row in sorted(plan_rows, key=lambda item: item["drift"]):
        lines.append(
            f"{row['drift'].title()} "
            f"& {number(row['median_qerror'])} "
            f"& {number(row['p95_qerror'])} "
            f"& {pct(row['danger_q100_rate'])}\\% "
            f"& {pct(row['p95_qerror_reduction'])}\\% \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def make_json_safe(report: dict) -> dict:
    safe = dict(report)
    safe["details"] = [
        {key: value for key, value in row.items() if key != "operator_counts"}
        for row in report["details"]
    ]
    return safe


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a DuckDB SQL execution probe for drifted CE queries.")
    parser.add_argument("--output-dir", default="paper/workload_drift_residual_field/results")
    parser.add_argument("--stats-data", default=str(STATS_DATA))
    parser.add_argument("--drifts", default="column,table,value")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument(
        "--top-k-per-seed",
        type=int,
        default=12,
        help="Tail candidates per drift/seed; use 0 to keep all executable label-matching candidates.",
    )
    parser.add_argument("--min-execution-label", type=int, default=10)
    parser.add_argument("--max-execution-label", type=int, default=1_000_000)
    parser.add_argument("--finetune-epochs", type=int, default=20)
    parser.add_argument("--finetune-lr", type=float, default=1e-3)
    parser.add_argument("--finetune-batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--duckdb-threads", type=int, default=4)
    parser.add_argument("--memory-limit", default="16GB")
    parser.add_argument("--max-temp-directory-size", default="128GB")
    args = parser.parse_args()

    args.output_dir = ROOT / args.output_dir
    args.stats_data = Path(args.stats_data)
    if not args.stats_data.is_absolute():
        args.stats_data = ROOT / args.stats_data
    args.seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    drifts = [item.strip() for item in args.drifts.split(",") if item.strip()]

    con = connect_duckdb(args)
    reports = [evaluate_drift(drift, con, args) for drift in drifts]
    detail_rows = write_detail_csv(args.output_dir / "duckdb_execution_probe.csv", reports)
    summary_rows = write_summary_csv(args.output_dir / "duckdb_execution_probe_summary.csv", reports)
    selection_rows = write_selection_audit_csv(
        args.output_dir / "duckdb_execution_probe_selection_audit.csv", reports
    )
    write_table(args.output_dir / "tables" / "duckdb_execution_probe.tex", summary_rows)
    write_selection_audit_table(
        args.output_dir / "tables" / "duckdb_execution_probe_selection_audit.tex",
        selection_rows,
    )
    write_plan_reference_table(
        args.output_dir / "tables" / "duckdb_plan_reference.tex",
        summary_rows,
    )
    manifest = {
        "passed": bool(
            all(
                row["execution_success_rate"] == 1.0 and row["label_match_rate"] == 1.0
                for row in summary_rows
                if row["method"] == "raw-mscn"
            )
        ),
        "scope": "DuckDB SQL COUNT(*) execution and plan-estimate probe for raw-MSCN tail target queries",
        "drifts": drifts,
        "seeds": args.seeds,
        "top_k_per_seed": args.top_k_per_seed,
        "min_execution_label": args.min_execution_label,
        "max_execution_label": args.max_execution_label,
        "stats_data": str(args.stats_data),
        "duckdb_threads": args.duckdb_threads,
        "finetune_epochs": args.finetune_epochs,
        "finetune_lr": args.finetune_lr,
        "detail_csv": str(args.output_dir / "duckdb_execution_probe.csv"),
        "summary_csv": str(args.output_dir / "duckdb_execution_probe_summary.csv"),
        "selection_audit_csv": str(args.output_dir / "duckdb_execution_probe_selection_audit.csv"),
        "table": str(args.output_dir / "tables" / "duckdb_execution_probe.tex"),
        "selection_audit_table": str(
            args.output_dir / "tables" / "duckdb_execution_probe_selection_audit.tex"
        ),
        "plan_reference_table": str(args.output_dir / "tables" / "duckdb_plan_reference.tex"),
        "json": str(args.output_dir / "duckdb_execution_probe.json"),
        "detail_rows": len(detail_rows),
        "selection_rows": len(selection_rows),
    }
    write_json(
        args.output_dir / "duckdb_execution_probe.json",
        {"manifest": manifest, "reports": [make_json_safe(report) for report in reports]},
    )
    write_json(args.output_dir / "duckdb_execution_probe_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
