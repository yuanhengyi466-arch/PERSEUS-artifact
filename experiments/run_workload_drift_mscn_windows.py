import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WDB_ROOT = ROOT / "baselines" / "workload-drift-benchmark"
MSCN_ROOT = WDB_ROOT / "baselines" / "MSCN-modified"


def require_torch():
    try:
        import torch
        from torch.utils.data import DataLoader
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: torch. Install CPU PyTorch before running the real MSCN baseline."
        ) from exc
    return torch, DataLoader


def add_mscn_to_path() -> None:
    path = str(MSCN_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_show(rel_path: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["git", "show", f"HEAD:{rel_path}"],
        cwd=str(WDB_ROOT),
        text=False,
        capture_output=True,
        timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"failed to export {rel_path} from workload-drift-benchmark git object: "
            f"{proc.stderr.decode(errors='replace')}"
        )
    output.write_bytes(proc.stdout)


def copy_or_export(rel_path: str, output: Path) -> str:
    source = WDB_ROOT / rel_path
    output.parent.mkdir(parents=True, exist_ok=True)
    if source.exists():
        shutil.copy2(source, output)
        return "working_tree"
    git_show(rel_path, output)
    return "git_object_HEAD"


def prepare_stats_drift(
    work_dir: Path, train_rel: str, predict_rel: str, limit_train: int, limit_predict: int
) -> dict:
    data_dir = work_dir / "data"
    exported = {}
    for rel in [
        "baselines/MSCN-modified/workloads/stats/tables.csv",
        "baselines/MSCN-modified/workloads/stats/joins.csv",
        "baselines/MSCN-modified/workloads/stats/column_min_max_vals.csv",
        train_rel,
        predict_rel,
    ]:
        out = data_dir / rel
        exported[rel] = copy_or_export(rel, out)

    train_path = data_dir / train_rel
    predict_path = data_dir / predict_rel
    if limit_train:
        train_path = truncate_query_file(train_path, work_dir / "data" / "train_limited.csv", limit_train)
    if limit_predict:
        predict_path = truncate_query_file(
            predict_path, work_dir / "data" / "predict_limited.csv", limit_predict
        )

    stats_dir = data_dir / "baselines" / "MSCN-modified" / "workloads" / "stats"
    return {
        "stats_dir": str(stats_dir),
        "train_path": str(train_path),
        "predict_path": str(predict_path),
        "exported": exported,
        "train_sha256": sha256(train_path),
        "predict_sha256": sha256(predict_path),
        "train_rows": count_rows(train_path),
        "predict_rows": count_rows(predict_path),
    }


def truncate_query_file(source: Path, output: Path, limit: int) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8", errors="replace", newline="") as src:
        lines = [line for _, line in zip(range(limit), src)]
    output.write_text("".join(lines), encoding="utf-8")
    return output


def count_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return sum(1 for line in handle if line.strip())


def load_query_file(path: Path):
    tables, joins, predicates, labels = [], [], [], []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter="#")
        for row in reader:
            if len(row) != 4:
                raise ValueError(f"expected 4 # separated fields in {path}, got {len(row)}: {row[:3]}")
            tables.append(row[0].split(","))
            joins.append(row[1].split(",") if row[1] else [""])
            predicates.append(row[2].split(",") if row[2] else [""])
            labels.append(str(max(1, int(row[3]))))
    predicates = [list(chunks(item, 3)) for item in predicates]
    return joins, predicates, tables, labels


def chunks(values, n):
    for i in range(0, len(values), n):
        yield values[i : i + n]


def load_table_names(stats_dir: Path):
    names = []
    with (stats_dir / "tables.csv").open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        for idx, row in enumerate(reader):
            if idx == 0 or not row:
                continue
            names.append(row[0])
    return names


def load_join_names(stats_dir: Path):
    joins = {""}
    with (stats_dir / "joins.csv").open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        for idx, row in enumerate(reader):
            if idx == 0 or not row or "=" not in row[0]:
                continue
            left, right = row[0].split("=", 1)
            pair = sorted([left, right])
            joins.add(f"{pair[0]}={pair[1]}")
    return joins


def load_column_min_max(stats_dir: Path):
    values = {}
    with (stats_dir / "column_min_max_vals.csv").open(
        "r", encoding="utf-8", errors="replace", newline=""
    ) as handle:
        reader = csv.reader(handle)
        for idx, row in enumerate(reader):
            if idx == 0 or len(row) < 3:
                continue
            values[row[0]] = [float(row[1]), float(row[2])]
    return values


def load_bitmap_file(path: Path, query_count: int, num_materialized_samples: int) -> list[np.ndarray]:
    num_bytes_per_bitmap = int((num_materialized_samples + 7) >> 3)
    samples = []
    with path.open("rb") as handle:
        for _ in range(query_count):
            four_bytes = handle.read(4)
            if not four_bytes:
                raise ValueError(f"unexpected end of bitmap file while reading {path}")
            num_bitmaps = int.from_bytes(four_bytes, byteorder="little")
            bitmaps = np.empty((num_bitmaps, num_bytes_per_bitmap * 8), dtype=np.uint8)
            for j in range(num_bitmaps):
                bitmap_bytes = handle.read(num_bytes_per_bitmap)
                if len(bitmap_bytes) != num_bytes_per_bitmap:
                    raise ValueError(f"truncated bitmap in {path}")
                bitmaps[j] = np.unpackbits(np.frombuffer(bitmap_bytes, dtype=np.uint8))
            samples.append(bitmaps[:, :num_materialized_samples])
        extra = handle.read(1)
        if extra:
            raise ValueError(f"bitmap file has trailing bytes after {query_count} queries: {path}")
    return samples


def encode_dataset(train_path: Path, predict_path: Path, stats_dir: Path, seed: int, args: argparse.Namespace):
    add_mscn_to_path()
    from mscn.data import make_dataset
    from mscn.util import (
        encode_data,
        encode_samples,
        get_all_column_names,
        get_all_operators,
        get_set_encoding,
        normalize_labels,
    )

    joins, predicates, tables, labels = load_query_file(train_path)
    pred_joins, pred_predicates, pred_tables, pred_labels = load_query_file(predict_path)

    table2vec, _ = get_set_encoding(load_table_names(stats_dir))
    column2vec, _ = get_set_encoding(get_all_column_names(predicates))
    op2vec, _ = get_set_encoding(get_all_operators(predicates))
    join2vec, _ = get_set_encoding(load_join_names(stats_dir))
    column_min_max_vals = load_column_min_max(stats_dir)

    if args.num_materialized_samples > 0:
        if not args.train_bitmap_file or not args.predict_bitmap_file:
            raise ValueError("bitmap mode requires --train-bitmap-file and --predict-bitmap-file")
        samples = load_bitmap_file(Path(args.train_bitmap_file), len(tables), args.num_materialized_samples)
        pred_samples = load_bitmap_file(
            Path(args.predict_bitmap_file), len(pred_tables), args.num_materialized_samples
        )
    else:
        samples = [[]] * len(tables)
        pred_samples = [[]] * len(pred_tables)
    samples_enc = encode_samples(tables, samples, table2vec)
    predicates_enc, joins_enc = encode_data(
        predicates, joins, column_min_max_vals, column2vec, op2vec, join2vec
    )
    label_norm, min_val, max_val = normalize_labels(labels)

    rng = np.random.default_rng(seed)
    indices = np.arange(len(labels), dtype=int)
    rng.shuffle(indices)
    train_cutoff = max(1, int(len(indices) * 0.9))
    train_indices = indices[:train_cutoff]
    valid_indices = indices[train_cutoff:] if train_cutoff < len(indices) else indices[-1:]

    train_parts = select_parts(samples_enc, predicates_enc, joins_enc, label_norm, train_indices)
    valid_parts = select_parts(samples_enc, predicates_enc, joins_enc, label_norm, valid_indices)
    max_num_joins = max(
        max(len(x) for x in train_parts["joins"]),
        max(len(x) for x in valid_parts["joins"]),
    )
    max_num_predicates = max(
        max(len(x) for x in train_parts["predicates"]),
        max(len(x) for x in valid_parts["predicates"]),
    )

    train_dataset = make_dataset(
        train_parts["samples"],
        train_parts["predicates"],
        train_parts["joins"],
        train_parts["labels"],
        max_num_joins=max_num_joins,
        max_num_predicates=max_num_predicates,
    )
    valid_dataset = make_dataset(
        valid_parts["samples"],
        valid_parts["predicates"],
        valid_parts["joins"],
        valid_parts["labels"],
        max_num_joins=max_num_joins,
        max_num_predicates=max_num_predicates,
    )

    pred_samples_enc = encode_samples(pred_tables, pred_samples, table2vec)
    pred_predicates_enc, pred_joins_enc = encode_data(
        pred_predicates, pred_joins, column_min_max_vals, column2vec, op2vec, join2vec
    )
    pred_label_norm, _, _ = normalize_labels(pred_labels, min_val, max_val)
    pred_max_num_joins = max(max_num_joins, max(len(x) for x in pred_joins_enc))
    pred_max_num_predicates = max(max_num_predicates, max(len(x) for x in pred_predicates_enc))
    predict_dataset = make_dataset(
        pred_samples_enc,
        pred_predicates_enc,
        pred_joins_enc,
        pred_label_norm,
        max_num_joins=pred_max_num_joins,
        max_num_predicates=pred_max_num_predicates,
    )

    return {
        "train_dataset": train_dataset,
        "valid_dataset": valid_dataset,
        "predict_dataset": predict_dataset,
        "labels": {
            "train": [labels[i] for i in train_indices],
            "valid": [labels[i] for i in valid_indices],
            "predict": pred_labels,
        },
        "min_val": min_val,
        "max_val": max_val,
        "feature_dims": {
            "sample_feats": len(table2vec) + args.num_materialized_samples,
            "predicate_feats": len(column2vec) + len(op2vec) + 1,
            "join_feats": len(join2vec),
        },
        "encoding_sizes": {
            "tables": len(table2vec),
            "columns": len(column2vec),
            "operators": len(op2vec),
            "joins": len(join2vec),
        },
    }


def select_parts(samples, predicates, joins, labels, indices):
    return {
        "samples": [samples[i] for i in indices],
        "predicates": [predicates[i] for i in indices],
        "joins": [joins[i] for i in indices],
        "labels": [labels[i] for i in indices],
    }


def unnormalize_tensor(torch, values, min_val: float, max_val: float) -> np.ndarray:
    if isinstance(values, list):
        values = torch.stack([v.detach().cpu().reshape(()) for v in values])
    values = values.detach().cpu().numpy().reshape(-1)
    labels = (values * (max_val - min_val)) + min_val
    return np.maximum(1, np.round(np.exp(labels))).astype(np.int64)


def qerror_array(preds: np.ndarray, labels) -> np.ndarray:
    labels_arr = np.maximum(1, np.asarray([float(x) for x in labels], dtype=np.float64))
    preds_arr = np.maximum(1, preds.astype(np.float64))
    return np.maximum(preds_arr / labels_arr, labels_arr / preds_arr)


def summarize_qerror(qerror: np.ndarray) -> dict:
    return {
        "median": float(np.median(qerror)),
        "p90": float(np.percentile(qerror, 90)),
        "p95": float(np.percentile(qerror, 95)),
        "p99": float(np.percentile(qerror, 99)),
        "max": float(np.max(qerror)),
        "mean": float(np.mean(qerror)),
    }


def qerror_loss(torch, outputs, targets, min_val, max_val):
    preds = torch.exp((outputs.view(-1) * (max_val - min_val)) + min_val)
    actual = torch.exp((targets.float().view(-1) * (max_val - min_val)) + min_val)
    return torch.maximum(torch.log(preds / actual), torch.log(actual / preds)).mean()


def run_model(args: argparse.Namespace, prepared: dict, output_dir: Path) -> dict:
    torch, DataLoader = require_torch()
    add_mscn_to_path()
    from mscn.model import SetConv

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(args.threads)

    encoded = encode_dataset(
        Path(prepared["train_path"]),
        Path(prepared["predict_path"]),
        Path(prepared["stats_dir"]),
        seed,
        args,
    )
    dims = encoded["feature_dims"]
    model = SetConv(
        dims["sample_feats"],
        dims["predicate_feats"],
        dims["join_feats"],
        args.hidden_units,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_loader = DataLoader(encoded["train_dataset"], batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(encoded["valid_dataset"], batch_size=args.batch_size)
    predict_loader = DataLoader(encoded["predict_dataset"], batch_size=args.batch_size)

    model.train()
    train_start = time.perf_counter()
    epoch_losses = []
    for epoch in range(args.epochs):
        losses = []
        for data_batch in train_loader:
            samples, predicates, joins, targets, sample_masks, predicate_masks, join_masks = data_batch
            optimizer.zero_grad()
            outputs = model(samples, predicates, joins, sample_masks, predicate_masks, join_masks)
            loss = qerror_loss(torch, outputs, targets, encoded["min_val"], encoded["max_val"])
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        epoch_losses.append(float(np.mean(losses)))
        print(f"Epoch {epoch}: loss={epoch_losses[-1]:.6f}")
    train_seconds = time.perf_counter() - train_start

    valid_preds = predict(torch, model, valid_loader)
    pred_preds = predict(torch, model, predict_loader)
    valid_unnorm = unnormalize_tensor(torch, valid_preds, encoded["min_val"], encoded["max_val"])
    pred_unnorm = unnormalize_tensor(torch, pred_preds, encoded["min_val"], encoded["max_val"])
    valid_qerror = qerror_array(valid_unnorm, encoded["labels"]["valid"])
    pred_qerror = qerror_array(pred_unnorm, encoded["labels"]["predict"])

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "predict_qerror.npy", pred_qerror)
    np.save(output_dir / "valid_qerror.npy", valid_qerror)
    np.savetxt(output_dir / "predict_predictions.txt", pred_unnorm, fmt="%d")
    torch.save(model.state_dict(), output_dir / "model.pt")

    uses_materialized_samples = args.num_materialized_samples > 0
    metrics = {
        "passed": True,
        "baseline": "workload-drift-benchmark MSCN SetConv",
        "scope": (
            "Windows CPU reproduction on official stats-drift CSV with reconstructed MSCN "
            "materialized-sample bitmaps"
            if uses_materialized_samples
            else "Windows CPU sub-reproduction on official stats-drift CSV without materialized bitmap samples"
        ),
        "uses_materialized_samples": uses_materialized_samples,
        "num_materialized_samples": args.num_materialized_samples,
        "counts_as_full_paper_reproduction": bool(uses_materialized_samples),
        "counts_as_real_data_baseline_subreproduction": True,
        "counts_as_real_data_baseline_reproduction": bool(uses_materialized_samples),
        "train_seconds": train_seconds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden_units": args.hidden_units,
        "seed": seed,
        "threads": args.threads,
        "feature_dims": encoded["feature_dims"],
        "encoding_sizes": encoded["encoding_sizes"],
        "train_rows": prepared["train_rows"],
        "predict_rows": prepared["predict_rows"],
        "valid_rows": len(encoded["labels"]["valid"]),
        "epoch_losses": epoch_losses,
        "valid_qerror": summarize_qerror(valid_qerror),
        "predict_qerror": summarize_qerror(pred_qerror),
        "artifacts": {
            "predict_qerror": str(output_dir / "predict_qerror.npy"),
            "valid_qerror": str(output_dir / "valid_qerror.npy"),
            "predictions": str(output_dir / "predict_predictions.txt"),
            "model": str(output_dir / "model.pt"),
        },
    }
    return metrics


def predict(torch, model, loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for data_batch in loader:
            samples, predicates, joins, _targets, sample_masks, predicate_masks, join_masks = data_batch
            outputs = model(samples, predicates, joins, sample_masks, predicate_masks, join_masks)
            preds.append(outputs.reshape(-1).detach().cpu())
    return torch.cat(preds)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a Windows/CPU MSCN baseline on official Workload-Drift stats-drift CSV."
    )
    parser.add_argument(
        "--train-rel",
        default="workloads/stats-drift/varying_column_queries/queries_4_columns.csv",
    )
    parser.add_argument(
        "--predict-rel",
        default="workloads/stats-drift/varying_column_queries/queries_5_columns.csv",
    )
    parser.add_argument("--output-dir", default="research_plan/workload_drift_mscn_windows")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-units", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-predict", type=int, default=0)
    parser.add_argument("--num-materialized-samples", type=int, default=0)
    parser.add_argument("--train-bitmap-file", default="")
    parser.add_argument("--predict-bitmap-file", default="")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    run_dir = output_dir / f"seed_{args.seed}"
    prepared = prepare_stats_drift(
        run_dir,
        args.train_rel,
        args.predict_rel,
        args.limit_train,
        args.limit_predict,
    )
    metrics = run_model(args, prepared, run_dir)
    metrics["provenance"] = {
        "workload_drift_root": str(WDB_ROOT),
        "mscn_root": str(MSCN_ROOT),
        "train_rel": args.train_rel,
        "predict_rel": args.predict_rel,
        "train_sha256": prepared["train_sha256"],
        "predict_sha256": prepared["predict_sha256"],
        "exported_sources": prepared["exported"],
        "bitmap_files": {
            "train_bitmap_file": args.train_bitmap_file,
            "predict_bitmap_file": args.predict_bitmap_file,
            "train_bitmap_sha256": sha256(Path(args.train_bitmap_file))
            if args.train_bitmap_file and Path(args.train_bitmap_file).exists()
            else "",
            "predict_bitmap_sha256": sha256(Path(args.predict_bitmap_file))
            if args.predict_bitmap_file and Path(args.predict_bitmap_file).exists()
            else "",
        },
        "command": sys.argv,
        "cwd": os.getcwd(),
        "python": sys.version,
    }
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
