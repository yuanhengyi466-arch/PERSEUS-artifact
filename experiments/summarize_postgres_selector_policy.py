#!/usr/bin/env python
"""Validation/test selector policy over existing PostgreSQL multi-subplan traces."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "paper" / "workload_drift_residual_field"
RESULT_DIR = PAPER_DIR / "results"
METHODS = ["default", "final-residual", "multi-residual"]
RESIDUAL_METHODS = ["final-residual", "multi-residual"]
SAFE_THRESHOLDS = [1.00, 1.05, 1.10, 1.25]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def ffloat(value: str | float | int | None) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def percentile(values: list[float], pct_value: float) -> float | None:
    if not values:
        return None
    data = sorted(values)
    pos = (len(data) - 1) * pct_value / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return data[lo]
    return data[lo] * (hi - pos) + data[hi] * (pos - lo)


def ratio_text(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "--"
    return f"{value:.2f}x"


def pct_text(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "--"
    return f"{100.0 * value:.0f}\\%"


def method_label(method: str) -> str:
    return {
        "default": "default",
        "final-residual": "final",
        "multi-residual": "multi",
    }.get(method, method)


def summarize_ratios(methods_list: list[dict[str, dict]], method: str) -> dict[str, float | None]:
    ratios = [
        float(methods[method]["execution_ms"]) / float(methods["default"]["execution_ms"])
        for methods in methods_list
        if float(methods["default"]["execution_ms"]) > 0
    ]
    if not ratios:
        return {
            "median_ratio": None,
            "p95_ratio": None,
            "win_rate": None,
            "loss_rate": None,
        }
    return {
        "median_ratio": median(ratios),
        "p95_ratio": percentile(ratios, 95),
        "win_rate": sum(value < 0.99 for value in ratios) / len(ratios),
        "loss_rate": sum(value > 1.01 for value in ratios) / len(ratios),
    }


def choose_safe_method(validation_ratio_medians: dict[str, float | None], validation_ratio_p95: dict[str, float | None], threshold: float) -> str:
    candidates = [
        method
        for method in RESIDUAL_METHODS
        if validation_ratio_medians[method] is not None
        and validation_ratio_p95[method] is not None
        and validation_ratio_medians[method] < 0.995
        and validation_ratio_p95[method] <= threshold
    ]
    return min(candidates, key=lambda method: validation_ratio_medians[method]) if candidates else "default"


def stable_bucket(scope: str, drift: str, seed: int, query_index: int, split_seed: int) -> str:
    key = f"{scope}|{drift}|{seed}|{query_index}|{split_seed}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / float(2**64)
    return "validation" if value < 0.30 else "test"


def load_groups(path: Path) -> dict[tuple[str, int, int], dict[str, dict[str, float | str]]]:
    groups: dict[tuple[str, int, int], dict[str, dict[str, float | str]]] = defaultdict(dict)
    for row in read_csv(path):
        if row.get("method") not in METHODS or row.get("status") != "executed":
            continue
        execution_ms = ffloat(row.get("execution_ms"))
        if execution_ms is None:
            continue
        key = (row["drift"], int(float(row["seed"])), int(float(row["query_index"])))
        groups[key][row["method"]] = {
            "execution_ms": execution_ms,
            "total_cost": ffloat(row.get("total_cost")),
            "ratio_vs_default": ffloat(row.get("execution_ms_ratio_vs_default")),
        }
    return {key: methods for key, methods in groups.items() if all(method in methods for method in METHODS)}


def summarize_scope(scope: str, groups: dict[tuple[str, int, int], dict[str, dict]], split_seed: int) -> list[dict]:
    by_drift: dict[str, list[tuple[tuple[str, int, int], dict[str, dict]]]] = defaultdict(list)
    for key, methods in groups.items():
        by_drift[key[0]].append((key, methods))

    rows = []
    for drift, items in sorted(by_drift.items()):
        validation = [
            methods
            for key, methods in items
            if stable_bucket(scope, key[0], key[1], key[2], split_seed) == "validation"
        ]
        test = [
            methods
            for key, methods in items
            if stable_bucket(scope, key[0], key[1], key[2], split_seed) == "test"
        ]
        if not validation or not test:
            continue
        validation_medians = {
            method: median([float(methods[method]["execution_ms"]) for methods in validation])
            for method in METHODS
        }
        validation_ratios = {
            method: [
                float(methods[method]["execution_ms"]) / float(methods["default"]["execution_ms"])
                for methods in validation
                if float(methods["default"]["execution_ms"]) > 0
            ]
            for method in METHODS
        }
        validation_ratio_medians = {method: median(values) for method, values in validation_ratios.items()}
        validation_ratio_p95 = {method: percentile(values, 95) for method, values in validation_ratios.items()}
        median_selected = min(METHODS, key=lambda method: validation_medians[method])
        safe_selected = choose_safe_method(validation_ratio_medians, validation_ratio_p95, 1.05)

        default_ms = [float(methods["default"]["execution_ms"]) for methods in test]
        median_selected_ms = [float(methods[median_selected]["execution_ms"]) for methods in test]
        median_selected_summary = summarize_ratios(test, median_selected)
        safe_selected_ms = [float(methods[safe_selected]["execution_ms"]) for methods in test]
        safe_selected_summary = summarize_ratios(test, safe_selected)
        best_ratios = [
            min(float(methods[method]["execution_ms"]) for method in METHODS)
            / float(methods["default"]["execution_ms"])
            for methods in test
            if float(methods["default"]["execution_ms"]) > 0
        ]
        cost_picks = []
        cost_ratios = []
        for methods in test:
            cost_candidates = [
                (method, methods[method]["total_cost"])
                for method in METHODS
                if methods[method]["total_cost"] is not None
            ]
            if not cost_candidates:
                continue
            cost_pick = min(cost_candidates, key=lambda item: float(item[1]))[0]
            cost_picks.append(cost_pick)
            if float(methods["default"]["execution_ms"]) > 0:
                cost_ratios.append(float(methods[cost_pick]["execution_ms"]) / float(methods["default"]["execution_ms"]))

        rows.append(
            {
                "scope": scope,
                "drift": drift,
                "validation_groups": len(validation),
                "test_groups": len(test),
                "selected_method": median_selected,
                "median_selected_method": median_selected,
                "safe_selected_method": safe_selected,
                "validation_default_ms": validation_medians["default"],
                "validation_final_ms": validation_medians["final-residual"],
                "validation_multi_ms": validation_medians["multi-residual"],
                "validation_default_median_ratio": validation_ratio_medians["default"],
                "validation_default_p95_ratio": validation_ratio_p95["default"],
                "validation_final_median_ratio": validation_ratio_medians["final-residual"],
                "validation_final_p95_ratio": validation_ratio_p95["final-residual"],
                "validation_multi_median_ratio": validation_ratio_medians["multi-residual"],
                "validation_multi_p95_ratio": validation_ratio_p95["multi-residual"],
                "test_default_ms": median(default_ms),
                "test_selected_ms": median(median_selected_ms),
                "test_selected_median_ratio": median_selected_summary["median_ratio"],
                "test_selected_p95_ratio": median_selected_summary["p95_ratio"],
                "test_selected_win_rate": median_selected_summary["win_rate"],
                "test_selected_loss_rate": median_selected_summary["loss_rate"],
                "test_safe_ms": median(safe_selected_ms),
                "test_safe_median_ratio": safe_selected_summary["median_ratio"],
                "test_safe_p95_ratio": safe_selected_summary["p95_ratio"],
                "test_safe_win_rate": safe_selected_summary["win_rate"],
                "test_safe_loss_rate": safe_selected_summary["loss_rate"],
                "test_best_median_ratio": median(best_ratios),
                "test_best_win_rate": sum(value < 0.99 for value in best_ratios) / len(best_ratios),
                "cost_median_ratio": median(cost_ratios),
                "cost_pick_nondefault_rate": sum(method != "default" for method in cost_picks) / len(cost_picks)
                if cost_picks
                else None,
            }
        )
    return rows


def build_sensitivity_rows(rows: list[dict], balanced_groups: dict[tuple[str, int, int], dict], column_groups: dict[tuple[str, int, int], dict], split_seed: int) -> list[dict]:
    source_groups = {
        "balanced-k1450": balanced_groups,
        "column-alltail": column_groups,
    }
    out = []
    for base in rows:
        scope = base["scope"]
        drift = base["drift"]
        groups = source_groups[scope]
        test = [
            methods
            for key, methods in groups.items()
            if key[0] == drift and stable_bucket(scope, key[0], key[1], key[2], split_seed) == "test"
        ]
        validation_ratio_medians = {
            "default": base["validation_default_median_ratio"],
            "final-residual": base["validation_final_median_ratio"],
            "multi-residual": base["validation_multi_median_ratio"],
        }
        validation_ratio_p95 = {
            "default": base["validation_default_p95_ratio"],
            "final-residual": base["validation_final_p95_ratio"],
            "multi-residual": base["validation_multi_p95_ratio"],
        }
        for threshold in SAFE_THRESHOLDS:
            chosen = choose_safe_method(validation_ratio_medians, validation_ratio_p95, threshold)
            summary = summarize_ratios(test, chosen)
            out.append(
                {
                    "scope": scope,
                    "drift": drift,
                    "threshold": threshold,
                    "selected_method": chosen,
                    "test_median_ratio": summary["median_ratio"],
                    "test_p95_ratio": summary["p95_ratio"],
                    "test_win_rate": summary["win_rate"],
                    "test_loss_rate": summary["loss_rate"],
                }
            )
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_sensitivity_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_table(path: Path, rows: list[dict]) -> None:
    show_rows = [
        row
        for row in rows
        if row["scope"] == "balanced-k1450" or row["scope"] == "column-alltail"
    ]
    order = {"balanced-k1450": 0, "column-alltail": 1}
    show_rows.sort(key=lambda row: (order.get(row["scope"], 99), row["drift"]))
    lines = [
        "\\begin{table}[!t]",
        "\\centering",
        "\\caption{Held-out PostgreSQL selector policies over deployable multi-subplan traces. A deterministic 30/70 validation/test split chooses one fixed method per scope/drift. Median selects by validation median runtime; safe selects a residual method only when validation median improves and validation P95 stays within 1.05$\\times$ default, otherwise it falls back to default. Best is an unattainable per-query upper bound within the same deployable candidate set.}",
        "\\label{tab:postgres-selector-policy}",
        "\\papertablesetup",
        "\\setlength{\\tabcolsep}{1.8pt}",
        "\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}l l r r l r r r@{}}",
        "\\toprule",
        "\\tabheader{Scope} & \\tabheader{Drift} & \\tabheader{Val/Test} & \\tabheader{Med.} & \\tabheader{Med. ratio} & \\tabheader{Safe} & \\tabheader{Safe ratio} & \\tabheader{Best} \\\\",
        "\\midrule",
    ]
    labels = {
        "balanced-k1450": "k1450",
        "column-alltail": "stress",
    }
    for row in show_rows:
        lines.append(
            f"{labels.get(row['scope'], row['scope'])} & {str(row['drift']).title()} "
            f"& {int(row['validation_groups'])}/{int(row['test_groups'])} "
            f"& {method_label(row['median_selected_method'])} "
            f"& {ratio_text(row['test_selected_median_ratio'])}/{ratio_text(row['test_selected_p95_ratio'])} "
            f"& {method_label(row['safe_selected_method'])} "
            f"& {ratio_text(row['test_safe_median_ratio'])}/{ratio_text(row['test_safe_p95_ratio'])} "
            f"& {ratio_text(row['test_best_median_ratio'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular*}",
            "\\end{table}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--balanced-detail", type=Path, default=PAPER_DIR / "results_pg_deploy_k1450" / "postgres_multisubplan_injection_detail.csv")
    parser.add_argument("--column-detail", type=Path, default=PAPER_DIR / "results_pg_deploy_alltail_column" / "postgres_multisubplan_injection_detail.csv")
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--split-seed", type=int, default=20260616)
    args = parser.parse_args()

    for attr in ["balanced_detail", "column_detail", "output_dir"]:
        value = getattr(args, attr)
        if not value.is_absolute():
            setattr(args, attr, ROOT / value)

    balanced_groups = load_groups(args.balanced_detail)
    column_groups = load_groups(args.column_detail)
    rows = []
    rows.extend(summarize_scope("balanced-k1450", balanced_groups, args.split_seed))
    rows.extend(summarize_scope("column-alltail", column_groups, args.split_seed))
    sensitivity_rows = build_sensitivity_rows(rows, balanced_groups, column_groups, args.split_seed)

    csv_path = args.output_dir / "postgres_selector_policy.csv"
    sensitivity_csv_path = args.output_dir / "postgres_selector_threshold_sensitivity.csv"
    table_path = args.output_dir / "tables" / "postgres_selector_policy.tex"
    manifest_path = args.output_dir / "postgres_selector_policy_manifest.json"
    write_csv(csv_path, rows)
    write_sensitivity_csv(sensitivity_csv_path, sensitivity_rows)
    write_table(table_path, rows)
    manifest = {
        "passed": bool(rows),
        "scope": "Held-out selector policy over PostgreSQL deployable multi-subplan traces",
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "split_seed": args.split_seed,
        "balanced_detail": str(args.balanced_detail),
        "column_detail": str(args.column_detail),
        "csv": str(csv_path),
        "sensitivity_csv": str(sensitivity_csv_path),
        "table": str(table_path),
        "rows": len(rows),
        "sensitivity_rows": len(sensitivity_rows),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
