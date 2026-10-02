import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

from run_workload_drift_mscn_windows import (
    ROOT,
    WDB_ROOT,
    MSCN_ROOT,
    add_mscn_to_path,
    load_bitmap_file,
    load_column_min_max,
    load_join_names,
    load_query_file,
    load_table_names,
    prepare_stats_drift,
    predict,
    qerror_array,
    qerror_loss,
    require_torch,
    select_parts,
    sha256,
    summarize_qerror,
    unnormalize_tensor,
)


def encode_robust_dataset(
    train_path: Path,
    predict_path: Path,
    stats_dir: Path,
    seed: int,
    args: argparse.Namespace,
):
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
        robust=True,
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


def run_model(args: argparse.Namespace, prepared: dict, output_dir: Path) -> dict:
    torch, DataLoader = require_torch()
    add_mscn_to_path()
    from mscn.model import RobustSetConv

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(args.threads)

    encoded = encode_robust_dataset(
        Path(prepared["train_path"]),
        Path(prepared["predict_path"]),
        Path(prepared["stats_dir"]),
        seed,
        args,
    )
    dims = encoded["feature_dims"]
    model = RobustSetConv(
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
    return {
        "passed": True,
        "baseline": "workload-drift-benchmark Robust-MSCN RobustSetConv",
        "scope": (
            "Windows CPU reproduction of Robust-MSCN on official stats-drift CSV with "
            "reconstructed MSCN materialized-sample bitmaps"
            if uses_materialized_samples
            else "Windows CPU sub-reproduction of Robust-MSCN on official stats-drift CSV without materialized bitmaps"
        ),
        "uses_materialized_samples": uses_materialized_samples,
        "num_materialized_samples": args.num_materialized_samples,
        "uses_official_make_dataset_robust_flag": True,
        "uses_robust_model_dropout": True,
        "robust_training_note": (
            "Follows the upstream Robust-MSCN path: RobustSetConv dropout plus the official "
            "make_dataset(..., robust=True) training-data path."
        ),
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a Windows/CPU Robust-MSCN baseline on official Workload-Drift stats-drift CSV."
    )
    parser.add_argument(
        "--train-rel",
        default="workloads/stats-drift/varying_column_queries/queries_4_columns.csv",
    )
    parser.add_argument(
        "--predict-rel",
        default="workloads/stats-drift/varying_column_queries/queries_5_columns.csv",
    )
    parser.add_argument("--output-dir", default="research_plan/workload_drift_robust_mscn_windows")
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
