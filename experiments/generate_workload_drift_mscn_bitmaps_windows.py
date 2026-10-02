import argparse
import csv
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

ALIAS_TO_TABLE = {
    "b": "badges",
    "c": "comments",
    "ph": "postHistory",
    "pl": "postLinks",
    "p": "posts",
    "t": "tags",
    "u": "users",
    "v": "votes",
    "badges": "badges",
    "comments": "comments",
    "postHistory": "postHistory",
    "postLinks": "postLinks",
    "posts": "posts",
    "tags": "tags",
    "users": "users",
    "votes": "votes",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_tables(field: str) -> list[tuple[str, str]]:
    tables = []
    for item in field.split(","):
        parts = item.strip().split()
        if not parts:
            continue
        table = parts[0]
        alias = parts[1] if len(parts) > 1 else table
        tables.append((table, alias))
    return tables


def parse_predicates(field: str) -> list[tuple[str, str, str, str]]:
    if not field:
        return []
    parts = field.split(",")
    predicates = []
    for i in range(0, len(parts), 3):
        if i + 2 >= len(parts):
            continue
        table_col, op, value = parts[i], parts[i + 1], parts[i + 2]
        if "." not in table_col:
            continue
        alias, column = table_col.split(".", 1)
        predicates.append((ALIAS_TO_TABLE.get(alias, alias), column, op, value))
    return predicates


def normalize_column(name: str) -> str:
    return name.strip().lower()


def parse_number(value: str):
    text = value.strip()
    if text == "":
        return np.nan
    try:
        return float(text)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue
    return np.nan


def compare(values: np.ndarray, op: str, raw_value: str) -> np.ndarray:
    threshold = parse_number(raw_value.strip("'"))
    if op == "=":
        return values == threshold
    if op == ">":
        return values > threshold
    if op == ">=":
        return values >= threshold
    if op == "<":
        return values < threshold
    if op == "<=":
        return values <= threshold
    raise ValueError(f"unsupported operator: {op}")


def load_table_sample(dataset_dir: Path, table: str, sample_size: int, rng: random.Random) -> dict:
    path = dataset_dir / f"{table}.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing STATS table CSV: {path}")

    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"missing header in {path}")
        reservoir = []
        for idx, row in enumerate(reader):
            if len(reservoir) < sample_size:
                reservoir.append(row)
            else:
                j = rng.randint(0, idx)
                if j < sample_size:
                    reservoir[j] = row

    if len(reservoir) < sample_size:
        raise ValueError(f"{path} has only {len(reservoir)} rows, need {sample_size}")

    columns: dict[str, list[float]] = {}
    for row in reservoir:
        for key, value in row.items():
            columns.setdefault(normalize_column(key), []).append(parse_number(value))

    arrays = {name: np.asarray(vals, dtype=np.float64) for name, vals in columns.items()}
    return {
        "csv": str(path),
        "rows_sampled": len(reservoir),
        "sha256": sha256(path),
        "columns": arrays,
    }


def load_samples(dataset_dir: Path, tables: list[str], sample_size: int, seed: int) -> dict:
    samples = {}
    for table in sorted(set(tables)):
        table_seed = int(hashlib.sha256(f"{seed}:{table}".encode("utf-8")).hexdigest()[:8], 16)
        samples[table] = load_table_sample(dataset_dir, table, sample_size, random.Random(table_seed))
    return samples


def query_table_bitmap(
    table: str, predicates: list[tuple[str, str, str, str]], table_sample: dict, sample_size: int
) -> np.ndarray:
    mask = np.ones(sample_size, dtype=bool)
    columns = table_sample["columns"]
    for pred_table, column, op, value in predicates:
        if pred_table != table:
            continue
        col_key = normalize_column(column)
        if col_key not in columns:
            raise KeyError(f"missing sampled column {table}.{column}")
        mask &= compare(columns[col_key], op, value)
    return mask.astype(np.uint8)


def pack_original_mscn_bits(bits: np.ndarray) -> bytes:
    out = bytearray()
    for start in range(0, len(bits), 8):
        byte = 0
        for offset, flag in enumerate(bits[start : start + 8]):
            byte += int(flag) * (2**offset)
        out.append(byte)
    return bytes(out)


def generate_bitmap_file(
    query_file: Path,
    output_file: Path,
    samples: dict,
    sample_size: int,
) -> dict:
    rows = 0
    bitmaps = 0
    matched_bits = 0
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with query_file.open("r", encoding="utf-8", errors="replace", newline="") as src, output_file.open("wb") as out:
        reader = csv.reader(src, delimiter="#")
        for row in reader:
            if len(row) != 4:
                raise ValueError(f"expected 4 # fields in {query_file}, got {len(row)}")
            query_tables = parse_tables(row[0])
            predicates = parse_predicates(row[2])
            out.write(len(query_tables).to_bytes(4, "little"))
            rows += 1
            for table, _alias in query_tables:
                bitmap = query_table_bitmap(table, predicates, samples[table], sample_size)
                matched_bits += int(bitmap.sum())
                bitmaps += 1
                out.write(pack_original_mscn_bits(bitmap))
    return {
        "query_file": str(query_file),
        "output_file": str(output_file),
        "query_rows": rows,
        "bitmaps": bitmaps,
        "bitmap_sha256": sha256(output_file),
        "mean_selectivity": float(matched_bits / max(1, bitmaps * sample_size)),
    }


def collect_tables(query_files: list[Path]) -> list[str]:
    tables = set()
    for query_file in query_files:
        with query_file.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.reader(handle, delimiter="#")
            for row in reader:
                if len(row) != 4:
                    continue
                for table, _alias in parse_tables(row[0]):
                    tables.add(table)
    return sorted(tables)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate original MSCN-format materialized-sample bitmaps for STATS drift on Windows."
    )
    parser.add_argument("--dataset-dir", required=True, help="Path containing stats_simplified CSV files.")
    parser.add_argument("--query-file", action="append", required=True, help="Official Workload-Drift query CSV.")
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--report-json", required=True)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = ROOT / dataset_dir
    query_files = [Path(path) if Path(path).is_absolute() else ROOT / path for path in args.query_file]
    report_path = Path(args.report_json)
    if not report_path.is_absolute():
        report_path = ROOT / report_path

    tables = collect_tables(query_files)
    samples = load_samples(dataset_dir, tables, args.sample_size, args.seed)
    outputs = []
    for query_file in query_files:
        output_file = query_file.with_name(query_file.stem + "_bitmap.csv")
        outputs.append(generate_bitmap_file(query_file, output_file, samples, args.sample_size))

    sample_report = {
        table: {
            "csv": meta["csv"],
            "rows_sampled": meta["rows_sampled"],
            "sha256": meta["sha256"],
            "columns": sorted(meta["columns"].keys()),
        }
        for table, meta in samples.items()
    }
    report = {
        "passed": True,
        "generator": "experiments/generate_workload_drift_mscn_bitmaps_windows.py",
        "scope": "Windows-native reconstruction of MSCN materialized-sample bitmaps from official STATS table CSVs",
        "dataset_dir": str(dataset_dir),
        "sample_size": args.sample_size,
        "seed": args.seed,
        "tables": tables,
        "samples": sample_report,
        "outputs": outputs,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
