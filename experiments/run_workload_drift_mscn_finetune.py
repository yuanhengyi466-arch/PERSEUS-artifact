import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import run_workload_drift_mscn_windows as mscn_run


ROOT = Path(__file__).resolve().parents[1]


def sha256_array(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(values, dtype=np.int64).tobytes())
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_target_rows(path: Path):
    labels = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter="#")
        for row in reader:
            if len(row) != 4:
                raise ValueError(f"expected 4 fields in {path}, got {len(row)}")
            labels.append(max(1, int(row[3])))
    return np.asarray(labels, dtype=np.float64)


def qerror(predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
    pred = np.maximum(1.0, predictions.astype(np.float64))
    lab = np.maximum(1.0, labels.astype(np.float64))
    return np.maximum(pred / lab, lab / pred)


def summarize(values: np.ndarray) -> dict:
    return {
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def relative_reduction(before: float, after: float) -> float:
    return float((before - after) / before) if before else 0.0


def select_dataset(torch, dataset, indices: np.ndarray):
    tensors = tuple(tensor[indices] for tensor in dataset.tensors)
    return torch.utils.data.TensorDataset(*tensors)


def train_epoch(torch, model, loader, optimizer, min_val: float, max_val: float) -> float:
    model.train()
    losses = []
    for data_batch in loader:
        samples, predicates, joins, targets, sample_masks, predicate_masks, join_masks = data_batch
        optimizer.zero_grad()
        outputs = model(samples, predicates, joins, sample_masks, predicate_masks, join_masks)
        loss = mscn_run.qerror_loss(torch, outputs, targets, min_val, max_val)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else 0.0


def build_encoding_args(metrics: dict, args) -> SimpleNamespace:
    bitmap_files = metrics["provenance"]["bitmap_files"]
    return SimpleNamespace(
        num_materialized_samples=metrics["num_materialized_samples"],
        train_bitmap_file=bitmap_files["train_bitmap_file"],
        predict_bitmap_file=bitmap_files["predict_bitmap_file"],
    )


def resolve_target_path(metrics_path: Path, metrics: dict) -> Path:
    target_path = metrics_path.parent / "data" / metrics["provenance"]["predict_rel"].replace("/", "\\")
    if not target_path.exists():
        target_path = metrics_path.parent / "data" / metrics["provenance"]["predict_rel"]
    return target_path


def evaluate_split(args, seed: int, metrics_path: Path, metrics: dict, encoded: dict, labels: np.ndarray):
    torch, DataLoader = mscn_run.require_torch()
    mscn_run.add_mscn_to_path()
    from mscn.model import SetConv

    rng = np.random.default_rng(seed)
    indices = np.arange(labels.shape[0])
    rng.shuffle(indices)
    if args.adaptation_rows >= labels.shape[0]:
        raise ValueError("adaptation_rows must be smaller than target row count")
    adapt_idx = indices[: args.adaptation_rows]
    eval_idx = indices[args.adaptation_rows :]

    dims = encoded["feature_dims"]
    checkpoint = Path(metrics["artifacts"]["model"])
    model = SetConv(
        dims["sample_feats"],
        dims["predicate_feats"],
        dims["join_feats"],
        metrics["hidden_units"],
    )
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))

    predict_dataset = encoded["predict_dataset"]
    adapt_dataset = select_dataset(torch, predict_dataset, adapt_idx)
    eval_dataset = select_dataset(torch, predict_dataset, eval_idx)
    adapt_loader = DataLoader(adapt_dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size)

    raw_predictions_all = np.loadtxt(metrics["artifacts"]["predictions"], dtype=np.float64)
    raw_q = summarize(qerror(raw_predictions_all[eval_idx], labels[eval_idx]))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(args.threads)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start = time.perf_counter()
    epoch_losses = []
    for _epoch in range(args.epochs):
        epoch_losses.append(
            train_epoch(torch, model, adapt_loader, optimizer, encoded["min_val"], encoded["max_val"])
        )
    train_seconds = time.perf_counter() - start

    ft_outputs = mscn_run.predict(torch, model, eval_loader)
    ft_predictions = mscn_run.unnormalize_tensor(
        torch, ft_outputs, encoded["min_val"], encoded["max_val"]
    ).astype(np.float64)
    ft_q = summarize(qerror(ft_predictions, labels[eval_idx]))
    reductions = {
        key: relative_reduction(raw_q[key], ft_q[key])
        for key in ["median", "p90", "p95", "p99", "mean"]
    }
    pass_gates = {
        "p90_reduction": reductions["p90"] >= args.min_p90_reduction,
        "p95_reduction": reductions["p95"] >= args.min_p95_reduction,
        "p99_reduction": reductions["p99"] >= args.min_p99_reduction,
        "median_not_much_worse": ft_q["median"] <= raw_q["median"] * args.max_median_regression,
        "eval_rows": int(eval_idx.shape[0]) >= args.min_eval_rows,
    }
    return {
        "seed": seed,
        "passed": all(pass_gates.values()),
        "pass_gates": pass_gates,
        "adaptation_rows": int(adapt_idx.shape[0]),
        "eval_rows": int(eval_idx.shape[0]),
        "baseline_qerror": raw_q,
        "method_qerror": ft_q,
        "relative_reductions": reductions,
        "train_seconds": train_seconds,
        "epoch_losses": epoch_losses,
        "adaptation_indices_sha256": sha256_array(adapt_idx),
        "eval_indices_sha256": sha256_array(eval_idx),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune reproduced MSCN on target adaptation rows.")
    parser.add_argument("--baseline-metrics", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--adaptation-rows", type=int, default=2000)
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--min-eval-rows", type=int, default=5000)
    parser.add_argument("--min-p90-reduction", type=float, default=0.30)
    parser.add_argument("--min-p95-reduction", type=float, default=0.40)
    parser.add_argument("--min-p99-reduction", type=float, default=0.60)
    parser.add_argument("--max-median-regression", type=float, default=1.20)
    args = parser.parse_args()

    metrics_path = Path(args.baseline_metrics)
    if not metrics_path.is_absolute():
        metrics_path = ROOT / metrics_path
    metrics = read_json(metrics_path)
    target_path = resolve_target_path(metrics_path, metrics)
    labels = load_target_rows(target_path)
    encoding_args = build_encoding_args(metrics, args)
    encoded = mscn_run.encode_dataset(
        Path(metrics_path.parent / "data" / metrics["provenance"]["train_rel"]),
        target_path,
        Path(metrics_path.parent / "data" / "baselines" / "MSCN-modified" / "workloads" / "stats"),
        metrics["seed"],
        encoding_args,
    )
    if labels.shape[0] != len(encoded["labels"]["predict"]):
        raise ValueError("label and encoded prediction length mismatch")

    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    split_reports = [
        evaluate_split(args, seed, metrics_path, metrics, encoded, labels)
        for seed in seeds
    ]
    aggregate_reductions = {
        key: float(np.median([item["relative_reductions"][key] for item in split_reports]))
        for key in ["median", "p90", "p95", "p99", "mean"]
    }
    passed_seed_count = sum(1 for item in split_reports if item["passed"])
    report = {
        "passed": passed_seed_count == len(split_reports),
        "method": "mscn-finetune",
        "baseline": "workload-drift-benchmark MSCN SetConv",
        "uses_api": False,
        "windows_native": True,
        "requires_labeled_target_adaptation_rows": args.adaptation_rows,
        "seed_count": len(split_reports),
        "passed_seed_count": passed_seed_count,
        "aggregate_reductions": aggregate_reductions,
        "gate_thresholds": {
            "min_p90_reduction": args.min_p90_reduction,
            "min_p95_reduction": args.min_p95_reduction,
            "min_p99_reduction": args.min_p99_reduction,
            "max_median_regression": args.max_median_regression,
            "min_eval_rows": args.min_eval_rows,
        },
        "fine_tune": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "threads": args.threads,
        },
        "split_reports": split_reports,
        "baseline_metrics": str(metrics_path),
        "target_workload": str(target_path),
        "command": sys.argv,
    }
    output = Path(args.output_json)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
