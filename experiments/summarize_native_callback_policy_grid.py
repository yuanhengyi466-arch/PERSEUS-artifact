#!/usr/bin/env python
"""Search held-out native PostgreSQL callback policies over deployment-safe features."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "paper" / "workload_drift_residual_field"
RESULT_DIR = PAPER_DIR / "results"
NATIVE_DETAIL = RESULT_DIR / "native_optimizer_callback_hash_full_seeds123" / "native_callback_detail.csv"

METHODS = ["default", "native-final", "native-multi", "native-gated"]
DEPLOY_METHODS = ["native-final", "native-multi", "native-gated"]
SPLIT_SEED = 20260618

CARDINALITY_THRESHOLDS = [
    50,
    100,
    150,
    200,
    300,
    500,
    750,
    1000,
    1500,
    2000,
    3000,
    5000,
    7500,
    10000,
    15000,
    20000,
    30000,
    50000,
    75000,
    100000,
]
SUBPLAN_THRESHOLDS = [2, 3, 4, 5, 8, 12]
JOIN_THRESHOLDS = [1, 2, 3, 4, 5, 8, 12]
PREDICATE_THRESHOLDS = [2, 3, 4, 5, 6, 8, 10]
DRIFT_SETS = [
    ("column",),
    ("table",),
    ("value",),
    ("column", "table"),
    ("column", "value"),
    ("table", "value"),
    ("column", "table", "value"),
]

POLICY_CLASSES = [
    {
        "class": "strict",
        "validation_subset_p95_cap": 1.05,
        "validation_subset_loss_cap": 0.25,
        "validation_all_p95_cap": 1.01,
        "min_validation_deploy": 20,
        "objective": "maximize validation deploy under subset P95 <= 1.05",
    },
    {
        "class": "balanced",
        "validation_subset_p95_cap": 1.10,
        "validation_subset_loss_cap": 0.25,
        "validation_all_p95_cap": 1.01,
        "min_validation_deploy": 20,
        "objective": "maximize validation deploy under subset P95 <= 1.10",
    },
    {
        "class": "coverage",
        "validation_subset_p95_cap": 1.12,
        "validation_subset_loss_cap": 0.25,
        "validation_all_p95_cap": 1.01,
        "min_validation_deploy": 20,
        "objective": "maximize validation deploy under subset P95 <= 1.12",
    },
    {
        "class": "frontier",
        "validation_subset_p95_cap": 1.25,
        "validation_subset_loss_cap": 0.35,
        "validation_all_p95_cap": 1.02,
        "min_validation_deploy": 20,
        "objective": "maximize validation deploy under subset P95 <= 1.25",
    },
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def ffloat(value: str | float | int | None) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def stable_split(drift: str, seed: int, query_index: int, split_seed: int) -> str:
    key = f"native-callback|{drift}|{seed}|{query_index}|{split_seed}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / float(2**64)
    return "validation" if value < 0.30 else "test"


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


def summarize(values: list[float]) -> dict[str, float | int | None]:
    values = [value for value in values if math.isfinite(value)]
    if not values:
        return {
            "n": 0,
            "median_ratio": None,
            "p90_ratio": None,
            "p95_ratio": None,
            "p99_ratio": None,
            "win_rate": None,
            "loss_rate": None,
        }
    return {
        "n": len(values),
        "median_ratio": statistics.median(values),
        "p90_ratio": percentile(values, 90),
        "p95_ratio": percentile(values, 95),
        "p99_ratio": percentile(values, 99),
        "win_rate": sum(value < 0.99 for value in values) / len(values),
        "loss_rate": sum(value > 1.01 for value in values) / len(values),
    }


def ratio_text(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "--"
    return f"{value:.2f}$\\times$"


def pct_text(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "--"
    return f"{100.0 * value:.0f}\\%"


def int_text(value: float | int | None) -> str:
    if value is None:
        return "--"
    return str(int(round(float(value))))


def load_groups(path: Path, split_seed: int) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int, int], dict[str, dict[str, float | int | bool | str]]] = {}
    for row in read_csv(path):
        method = row.get("method", "")
        if method not in METHODS or row.get("status") != "executed":
            continue
        execution_ms = ffloat(row.get("execution_ms"))
        max_card = ffloat(row.get("max_estimated_cardinality"))
        total_cost = ffloat(row.get("total_cost"))
        if execution_ms is None or max_card is None or total_cost is None:
            continue
        key = (
            row["drift"],
            int(float(row["seed"])),
            int(float(row["query_index"])),
        )
        grouped.setdefault(key, {})[method] = {
            "execution_ms": execution_ms,
            "total_cost": total_cost,
            "max_estimated_cardinality": max_card,
            "table_count": int(float(row["table_count"])),
            "join_count": int(float(row["join_count"])),
            "predicate_count": int(float(row["predicate_count"])),
            "subplan_count": int(float(row.get("map_subplans") or 0)),
            "plan_changed": str(row.get("plan_changed_vs_default", "")).lower() == "true",
        }
    rows = []
    for key, methods in grouped.items():
        if not all(method in methods for method in METHODS):
            continue
        drift, seed, query_index = key
        if not methods["default"]["execution_ms"]:
            continue
        rows.append(
            {
                "key": key,
                "drift": drift,
                "seed": seed,
                "query_index": query_index,
                "split": stable_split(drift, seed, query_index, split_seed),
                "methods": methods,
            }
        )
    return rows


def candidate_rules() -> list[dict[str, object]]:
    rules: list[dict[str, object]] = []
    for drift_set in DRIFT_SETS:
        for method in DEPLOY_METHODS:
            for card in CARDINALITY_THRESHOLDS:
                base = {
                    "drifts": tuple(drift_set),
                    "method": method,
                    "max_cardinality": card,
                    "max_subplans": None,
                    "max_joins": None,
                    "max_predicates": None,
                }
                rules.append(base.copy())
                for max_subplans in SUBPLAN_THRESHOLDS:
                    rule = base.copy()
                    rule["max_subplans"] = max_subplans
                    rules.append(rule)
                for max_joins in JOIN_THRESHOLDS:
                    rule = base.copy()
                    rule["max_joins"] = max_joins
                    rules.append(rule)
                for max_predicates in PREDICATE_THRESHOLDS:
                    rule = base.copy()
                    rule["max_predicates"] = max_predicates
                    rules.append(rule)
    return rules


def rule_name(rule: dict[str, object]) -> str:
    parts = [
        "+".join(rule["drifts"]),
        str(rule["method"]).replace("native-", ""),
        f"card<={int(rule['max_cardinality'])}",
    ]
    if rule.get("max_subplans") is not None:
        parts.append(f"subp<={int(rule['max_subplans'])}")
    if rule.get("max_joins") is not None:
        parts.append(f"join<={int(rule['max_joins'])}")
    if rule.get("max_predicates") is not None:
        parts.append(f"pred<={int(rule['max_predicates'])}")
    return " ".join(parts)


def short_rule_name(rule_name_value: str) -> str:
    replacements = {
        "column gated card<=2000 subp<=5": "gated c$\\le$2k, s$\\le$5",
        "column gated card<=7500 subp<=5": "gated c$\\le$7.5k, s$\\le$5",
        "column final card<=30000 subp<=5": "final c$\\le$30k, s$\\le$5",
        "column+table final card<=30000 pred<=6": "final c$\\le$30k, p$\\le$6",
    }
    return replacements.get(rule_name_value, rule_name_value.replace("<=", "$\\le$"))


def rule_applies(rule: dict[str, object], group: dict[str, object]) -> bool:
    if group["drift"] not in set(rule["drifts"]):
        return False
    method = str(rule["method"])
    values = group["methods"][method]  # type: ignore[index]
    if float(values["max_estimated_cardinality"]) > float(rule["max_cardinality"]):
        return False
    if rule.get("max_subplans") is not None and int(values["subplan_count"]) > int(rule["max_subplans"]):
        return False
    if rule.get("max_joins") is not None and int(values["join_count"]) > int(rule["max_joins"]):
        return False
    if rule.get("max_predicates") is not None and int(values["predicate_count"]) > int(rule["max_predicates"]):
        return False
    return True


def evaluate_rule(
    groups: list[dict[str, object]],
    rule: dict[str, object],
    split_name: str,
) -> dict[str, object]:
    subset_ratios: list[float] = []
    policy_ratios: list[float] = []
    counts: dict[str, int] = {}
    changed = 0
    method = str(rule["method"])
    for group in groups:
        if group["split"] != split_name:
            continue
        methods = group["methods"]  # type: ignore[assignment]
        default_ms = float(methods["default"]["execution_ms"])  # type: ignore[index]
        if rule_applies(rule, group):
            deployed = methods[method]  # type: ignore[index]
            ratio = float(deployed["execution_ms"]) / default_ms
            subset_ratios.append(ratio)
            policy_ratios.append(ratio)
            drift = str(group["drift"])
            counts[drift] = counts.get(drift, 0) + 1
            changed += int(bool(deployed["plan_changed"]))
        else:
            policy_ratios.append(1.0)
    subset = summarize(subset_ratios)
    policy = summarize(policy_ratios)
    deployed = int(subset["n"] or 0)
    return {
        "groups": len([group for group in groups if group["split"] == split_name]),
        "deployed": deployed,
        "coverage": deployed / len([group for group in groups if group["split"] == split_name]),
        "drift_counts": counts,
        "plan_changed": changed,
        "plan_changed_rate": changed / deployed if deployed else None,
        **{f"subset_{key}": value for key, value in subset.items()},
        **{f"policy_{key}": value for key, value in policy.items()},
    }


def passes_class(metrics: dict[str, object], policy_class: dict[str, object]) -> bool:
    return (
        int(metrics["deployed"]) >= int(policy_class["min_validation_deploy"])
        and metrics["subset_median_ratio"] is not None
        and float(metrics["subset_median_ratio"]) <= 0.98
        and float(metrics["subset_p95_ratio"]) <= float(policy_class["validation_subset_p95_cap"])
        and float(metrics["subset_loss_rate"]) <= float(policy_class["validation_subset_loss_cap"])
        and float(metrics["policy_p95_ratio"]) <= float(policy_class["validation_all_p95_cap"])
    )


def select_policies(groups: list[dict[str, object]]) -> list[dict[str, object]]:
    selected = []
    rules = candidate_rules()
    for policy_class in POLICY_CLASSES:
        eligible = []
        for rule in rules:
            val = evaluate_rule(groups, rule, "validation")
            if passes_class(val, policy_class):
                eligible.append((rule, val))
        if not eligible:
            continue
        rule, val = sorted(
            eligible,
            key=lambda item: (
                int(item[1]["deployed"]),
                -float(item[1]["subset_median_ratio"]),
                -float(item[1]["subset_p95_ratio"]),
            ),
            reverse=True,
        )[0]
        test = evaluate_rule(groups, rule, "test")
        selected.append(
            {
                "policy_class": policy_class["class"],
                "objective": policy_class["objective"],
                "validation_subset_p95_cap": policy_class["validation_subset_p95_cap"],
                "rule_name": rule_name(rule),
                "rule": rule,
                "validation": val,
                "test": test,
            }
        )
    return selected


def flatten(selected: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for policy in selected:
        for split_name in ["validation", "test"]:
            metrics = policy[split_name]  # type: ignore[index]
            row = {
                "policy_class": policy["policy_class"],
                "split": split_name,
                "rule_name": policy["rule_name"],
                "validation_subset_p95_cap": policy["validation_subset_p95_cap"],
                "groups": metrics["groups"],
                "deployed": metrics["deployed"],
                "coverage": metrics["coverage"],
                "drift_counts": json.dumps(metrics["drift_counts"], sort_keys=True),
                "plan_changed": metrics["plan_changed"],
                "plan_changed_rate": metrics["plan_changed_rate"],
            }
            for prefix in ["subset", "policy"]:
                for key in ["median_ratio", "p90_ratio", "p95_ratio", "p99_ratio", "win_rate", "loss_rate"]:
                    row[f"{prefix}_{key}"] = metrics[f"{prefix}_{key}"]
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_tex_rows(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "\\multicolumn{8}{@{}l}{\\tabheader{Held-out native-callback deployment policies}} \\\\",
        "\\midrule",
        "\\tabheader{Policy} & \\tabheader{Rule} & \\tabheader{Split} & \\tabheader{Deploy} & \\tabheader{Sub med/P95} & \\tabheader{Sub win/loss} & \\tabheader{All P95/P99} & \\tabheader{Chg.} \\\\",
        "\\midrule",
    ]
    for row in rows:
        if row["split"] != "test":
            continue
        lines.append(
            f"{str(row['policy_class']).title()} "
            f"& {short_rule_name(str(row['rule_name']))} "
            f"& Test "
            f"& {int_text(row['deployed'])}/{int_text(row['groups'])} ({pct_text(row['coverage'])}) "
            f"& {ratio_text(row['subset_median_ratio'])}/{ratio_text(row['subset_p95_ratio'])} "
            f"& {pct_text(row['subset_win_rate'])}/{pct_text(row['subset_loss_rate'])} "
            f"& {ratio_text(row['policy_p95_ratio'])}/{ratio_text(row['policy_p99_ratio'])} "
            f"& {pct_text(row['plan_changed_rate'])} \\\\"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detail", type=Path, default=NATIVE_DETAIL)
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    args = parser.parse_args()

    if not args.detail.is_absolute():
        args.detail = ROOT / args.detail
    if not args.output_dir.is_absolute():
        args.output_dir = ROOT / args.output_dir

    groups = load_groups(args.detail, args.split_seed)
    selected = select_policies(groups)
    rows = flatten(selected)
    csv_path = args.output_dir / "native_callback_policy_grid.csv"
    tex_path = args.output_dir / "tables" / "native_callback_policy_grid_rows.tex"
    manifest_path = args.output_dir / "native_callback_policy_grid_manifest.json"
    write_csv(csv_path, rows)
    write_tex_rows(tex_path, rows)
    manifest = {
        "passed": bool(rows),
        "scope": "Held-out native PostgreSQL callback policy grid over deployment-safe features",
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "detail": str(args.detail),
        "csv": str(csv_path),
        "tex_rows": str(tex_path),
        "split_seed": args.split_seed,
        "groups": len(groups),
        "candidate_rules": len(candidate_rules()),
        "policy_classes": POLICY_CLASSES,
        "selected": selected,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ["passed", "scope", "groups", "candidate_rules", "csv", "tex_rows"]}, indent=2))


if __name__ == "__main__":
    main()
