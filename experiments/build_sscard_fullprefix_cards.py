#!/usr/bin/env python
"""Build complete SSCard DBMS card files by preserving the shipped prefix.

The SSCard end-to-end artifact ships 416 values for the modified PostgreSQL
single-table hook, while the local patched PostgreSQL 13.1 planner consumes
700 base-relation slots for the full 79-query JOB-like workload.  This script
does not invent SSCard estimates for the missing suffix.  It preserves the
shipped 416-value prefix exactly and fills the remaining slots with the traced
PostgreSQL native base-relation row estimates, producing complete files that
can exercise the full hook without reading past the artifact.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "paper" / "workload_drift_residual_field" / "results"
CARD_DIR = ROOT / "baselines" / "SSCard" / "end_to_end" / "cards" / "like_queries_single"
DATA_DIR = ROOT / "tools" / "patched_pg13_data_sscard"
TRACE_CSV = RESULT_DIR / "sscard_slot_trace" / "sscard_base_slot_trace.csv"


def read_values(path: Path) -> list[float]:
    return [float(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_trace_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row["slot"]))
    expected = list(range(len(rows)))
    actual = [int(row["slot"]) for row in rows]
    if actual != expected:
        raise ValueError("trace slots are not contiguous from zero")
    return rows


def write_values(path: Path, values: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{value:.12g}\n" for value in values), encoding="utf-8")


def build_fullprefix(prefix_path: Path, trace_rows: list[dict]) -> tuple[list[float], dict]:
    prefix = read_values(prefix_path)
    native_suffix = [float(row["rows"]) for row in trace_rows[len(prefix) :]]
    values = prefix + native_suffix
    if len(values) != len(trace_rows):
        raise ValueError(
            f"complete file length {len(values)} does not match trace length {len(trace_rows)}"
        )
    return values, {
        "source_file": str(prefix_path),
        "shipped_prefix_rows": len(prefix),
        "native_fallback_rows": len(native_suffix),
        "complete_rows": len(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-csv", type=Path, default=TRACE_CSV)
    parser.add_argument("--card-dir", type=Path, default=CARD_DIR)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR / "sscard_fullprefix_cards")
    parser.add_argument("--copy-to-data-dir", action="store_true")
    args = parser.parse_args()

    for attr in ["trace_csv", "card_dir", "data_dir", "output_dir"]:
        value = getattr(args, attr)
        if not value.is_absolute():
            setattr(args, attr, ROOT / value)

    trace_rows = read_trace_rows(args.trace_csv)
    outputs = []
    for source_name, output_name in [
        ("sscard_pg_single.txt", "sscard_pg_single_fullprefix.txt"),
        ("pg_pg_single.txt", "pg_pg_single_fullprefix.txt"),
    ]:
        values, item = build_fullprefix(args.card_dir / source_name, trace_rows)
        output_path = args.card_dir / output_name
        write_values(output_path, values)
        copied_to = ""
        if args.copy_to_data_dir:
            args.data_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(output_path, args.data_dir / output_name)
            copied_to = str(args.data_dir / output_name)
        item.update(
            {
                "output_file": str(output_path),
                "copied_to": copied_to,
                "first_fallback_slot": item["shipped_prefix_rows"],
            }
        )
        outputs.append(item)

    manifest = {
        "passed": all(item["complete_rows"] == len(trace_rows) for item in outputs),
        "scope": "Prefix-preserving complete SSCard card files for modified-PostgreSQL hook replay",
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "trace_csv": str(args.trace_csv),
        "trace_slots": len(trace_rows),
        "like_slots": sum(1 for row in trace_rows if row["has_like"] == "True"),
        "non_like_filter_slots": sum(1 for row in trace_rows if row["has_non_like_filter"] == "True"),
        "filterless_slots": sum(
            1
            for row in trace_rows
            if row["has_like"] != "True" and row["has_non_like_filter"] != "True"
        ),
        "semantics": "first 416 values are the shipped card file; slots 416..699 use traced PostgreSQL native base-relation row estimates",
        "outputs": outputs,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "sscard_fullprefix_cards_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
