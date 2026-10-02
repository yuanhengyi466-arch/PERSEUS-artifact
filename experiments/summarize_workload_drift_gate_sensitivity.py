#!/usr/bin/env python
"""Summarize validation-gate threshold sensitivity for the paper."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "paper" / "workload_drift_residual_field" / "results"
GATE_DIR = RESULT_DIR / "gate_sweep"

DEFAULT_REPORTS = {
    "column": GATE_DIR / "column_residual-field_2000.json",
    "table": GATE_DIR / "table_residual-field_2000.json",
    "value": GATE_DIR / "value_residual-field_500.json",
}

PROFILES = [
    {
        "name": "Lenient",
        "tail": "20/30/50",
        "vs_global": "15/25",
        "min_p90": 0.20,
        "min_p95": 0.30,
        "min_p99": 0.50,
        "min_vg95": 0.15,
        "min_vg99": 0.25,
        "max_median_ratio": 1.25,
    },
    {
        "name": "Default",
        "tail": "30/40/60",
        "vs_global": "25/40",
        "min_p90": 0.30,
        "min_p95": 0.40,
        "min_p99": 0.60,
        "min_vg95": 0.25,
        "min_vg99": 0.40,
        "max_median_ratio": 1.20,
    },
    {
        "name": "Strict",
        "tail": "50/70/90",
        "vs_global": "50/70",
        "min_p90": 0.50,
        "min_p95": 0.70,
        "min_p99": 0.90,
        "min_vg95": 0.50,
        "min_vg99": 0.70,
        "max_median_ratio": 1.10,
    },
]


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}\\%"


def passes(split: dict, profile: dict) -> bool:
    rel = split["relative_reductions"]
    relg = split["relative_reductions_vs_global_calibration"]
    median_ratio = split["method_qerror"]["median"] / split["baseline_qerror"]["median"]
    return (
        rel["p90"] >= profile["min_p90"]
        and rel["p95"] >= profile["min_p95"]
        and rel["p99"] >= profile["min_p99"]
        and relg["p95"] >= profile["min_vg95"]
        and relg["p99"] >= profile["min_vg99"]
        and median_ratio <= profile["max_median_ratio"]
    )


def load_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize() -> list[dict]:
    rows = []
    for drift, path in DEFAULT_REPORTS.items():
        report = load_report(path)
        splits = report["split_reports"]
        min_tail = {
            q: min(float(split["relative_reductions"][q]) for split in splits)
            for q in ["p90", "p95", "p99"]
        }
        min_vg = {
            q: min(float(split["relative_reductions_vs_global_calibration"][q]) for split in splits)
            for q in ["p95", "p99"]
        }
        profile_counts = {
            profile["name"].lower(): sum(1 for split in splits if passes(split, profile))
            for profile in PROFILES
        }
        rows.append(
            {
                "drift": drift,
                "labels": report["requires_labeled_target_adaptation_rows"],
                "min_tail_p90": min_tail["p90"],
                "min_tail_p95": min_tail["p95"],
                "min_tail_p99": min_tail["p99"],
                "min_vs_global_p95": min_vg["p95"],
                "min_vs_global_p99": min_vg["p99"],
                "lenient_pass": f"{profile_counts['lenient']}/{len(splits)}",
                "default_pass": f"{profile_counts['default']}/{len(splits)}",
                "strict_pass": f"{profile_counts['strict']}/{len(splits)}",
            }
        )
    return rows


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
        "\\caption{Validation-gate threshold sensitivity for residual-field transport. Min reductions are the weakest split among three adaptation seeds at the default label budget.}",
        "\\label{tab:gate-sensitivity}",
        "\\papertablesetup",
        "\\setlength{\\tabcolsep}{1.9pt}",
        "\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}l|r r|c c c@{}}",
        "\\toprule",
        "\\tabheader{Drift} & \\tabheader{Min Tail Red.} & \\tabheader{Min vs Glob.} & \\tabheader{Len.} & \\tabheader{Def.} & \\tabheader{Str.} \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['drift'].title()} "
            f"& {pct(row['min_tail_p90'])}/{pct(row['min_tail_p95'])}/{pct(row['min_tail_p99'])} "
            f"& {pct(row['min_vs_global_p95'])}/{pct(row['min_vs_global_p99'])} "
            f"& {row['lenient_pass']} & {row['default_pass']} & {row['strict_pass']} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular*}",
            "\\vspace{1pt}",
            "{\\footnotesize Len./Def./Str. require P90/P95/P99 reductions of 20/30/50, 30/40/60, and 50/70/90\\%, respectively.}",
            "\\end{table}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = summarize()
    write_csv(RESULT_DIR / "gate_threshold_sensitivity.csv", rows)
    write_table(RESULT_DIR / "tables" / "gate_threshold_sensitivity.tex", rows)
    print(json.dumps({"rows": rows, "profiles": PROFILES}, indent=2))


if __name__ == "__main__":
    main()
