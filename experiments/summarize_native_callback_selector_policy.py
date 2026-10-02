#!/usr/bin/env python
"""Held-out deployable selector policy over native PostgreSQL callback traces."""

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
NATIVE_DETAIL = (
    RESULT_DIR
    / "native_optimizer_callback_hash_full_seeds123"
    / "native_callback_detail.csv"
)

METHODS = ["default", "native-final", "native-multi", "native-gated"]
SPLIT_SEED = 20260618


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
            "win_rate": None,
            "loss_rate": None,
        }
    return {
        "n": len(values),
        "median_ratio": statistics.median(values),
        "p90_ratio": percentile(values, 90),
        "p95_ratio": percentile(values, 95),
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


def load_groups(path: Path) -> dict[tuple[str, int, int], dict[str, dict[str, float | str]]]:
    groups: dict[tuple[str, int, int], dict[str, dict[str, float | str]]] = {}
    for row in read_csv(path):
        method = row.get("method", "")
        if method not in METHODS or row.get("status") != "executed":
            continue
        execution_ms = ffloat(row.get("execution_ms"))
        total_cost = ffloat(row.get("total_cost"))
        max_card = ffloat(row.get("max_estimated_cardinality"))
        if execution_ms is None or total_cost is None or max_card is None:
            continue
        key = (
            row["drift"],
            int(float(row["seed"])),
            int(float(row["query_index"])),
        )
        groups.setdefault(key, {})[method] = {
            "execution_ms": execution_ms,
            "total_cost": total_cost,
            "max_estimated_cardinality": max_card,
            "plan_changed": str(row.get("plan_changed_vs_default", "")).lower() == "true",
        }
    return {key: methods for key, methods in groups.items() if all(method in methods for method in METHODS)}


def evaluate_policy(
    groups: dict[tuple[str, int, int], dict[str, dict[str, float | str]]],
    max_cardinality: float,
    split_seed: int,
) -> list[dict[str, float | int | str | None]]:
    rows = []
    for drift in sorted({key[0] for key in groups}):
        for split_name in ["validation", "test"]:
            split_groups = [
                (key, methods)
                for key, methods in groups.items()
                if key[0] == drift and stable_split(key[0], key[1], key[2], split_seed) == split_name
            ]
            subset_ratios: list[float] = []
            policy_ratios: list[float] = []
            changed = 0
            deployed = 0
            for _key, methods in split_groups:
                default_ms = float(methods["default"]["execution_ms"])
                final = methods["native-final"]
                use_final = drift == "column" and float(final["max_estimated_cardinality"]) <= max_cardinality
                if use_final and default_ms > 0:
                    ratio = float(final["execution_ms"]) / default_ms
                    subset_ratios.append(ratio)
                    policy_ratios.append(ratio)
                    deployed += 1
                    changed += int(bool(final["plan_changed"]))
                else:
                    policy_ratios.append(1.0)

            subset = summarize(subset_ratios)
            policy = summarize(policy_ratios)
            rows.append(
                {
                    "drift": drift,
                    "split": split_name,
                    "groups": len(split_groups),
                    "deployed": deployed,
                    "coverage": deployed / len(split_groups) if split_groups else None,
                    "plan_changed": changed,
                    "plan_changed_rate": changed / deployed if deployed else None,
                    "subset_n": subset["n"],
                    "subset_median_ratio": subset["median_ratio"],
                    "subset_p90_ratio": subset["p90_ratio"],
                    "subset_p95_ratio": subset["p95_ratio"],
                    "subset_win_rate": subset["win_rate"],
                    "subset_loss_rate": subset["loss_rate"],
                    "policy_median_ratio": policy["median_ratio"],
                    "policy_p90_ratio": policy["p90_ratio"],
                    "policy_p95_ratio": policy["p95_ratio"],
                    "policy_win_rate": policy["win_rate"],
                    "policy_loss_rate": policy["loss_rate"],
                    "rule": f"column native-final if max_estimated_cardinality <= {int(max_cardinality)}; otherwise default",
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, float | int | str | None]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_tex_rows(path: Path, rows: list[dict[str, float | int | str | None]]) -> None:
    show_rows = [
        row
        for row in rows
        if row["drift"] == "column" and row["split"] in {"validation", "test"}
    ]
    lines = [
        "\\multicolumn{8}{@{}l}{\\tabheader{Held-out native-callback deployment policy}} \\\\",
        "\\midrule",
        "\\tabheader{Split} & \\tabheader{Rule} & \\tabheader{Groups} & \\tabheader{Deploy} & \\tabheader{Sub med/P95} & \\tabheader{Sub win/loss} & \\tabheader{All med/P95} & \\tabheader{Chg.} \\\\",
        "\\midrule",
    ]
    for row in show_rows:
        lines.append(
            f"{str(row['split']).title()} "
            f"& Final if card$\\le$200 "
            f"& {int_text(row['groups'])} "
            f"& {int_text(row['deployed'])} ({pct_text(row['coverage'])}) "
            f"& {ratio_text(row['subset_median_ratio'])}/{ratio_text(row['subset_p95_ratio'])} "
            f"& {pct_text(row['subset_win_rate'])}/{pct_text(row['subset_loss_rate'])} "
            f"& {ratio_text(row['policy_median_ratio'])}/{ratio_text(row['policy_p95_ratio'])} "
            f"& {pct_text(row['plan_changed_rate'])} \\\\"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detail", type=Path, default=NATIVE_DETAIL)
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--max-cardinality", type=float, default=200.0)
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    args = parser.parse_args()

    if not args.detail.is_absolute():
        args.detail = ROOT / args.detail
    if not args.output_dir.is_absolute():
        args.output_dir = ROOT / args.output_dir

    groups = load_groups(args.detail)
    rows = evaluate_policy(groups, args.max_cardinality, args.split_seed)
    csv_path = args.output_dir / "native_callback_selector_policy.csv"
    tex_path = args.output_dir / "tables" / "native_callback_selector_policy_rows.tex"
    manifest_path = args.output_dir / "native_callback_selector_policy_manifest.json"
    write_csv(csv_path, rows)
    write_tex_rows(tex_path, rows)
    manifest = {
        "passed": bool(rows),
        "scope": "Held-out deployable native PostgreSQL callback selector policy",
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "detail": str(args.detail),
        "csv": str(csv_path),
        "tex_rows": str(tex_path),
        "split_seed": args.split_seed,
        "rule": f"column native-final if max_estimated_cardinality <= {int(args.max_cardinality)}; otherwise default",
        "rows": len(rows),
        "methods_required": METHODS,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
