import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import run_workload_drift_mscn_finetune as ft
import run_workload_drift_mscn_windows as mscn_run
import run_workload_drift_tail_residual_transport as rt


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research_plan"


DRIFTS = {
    "column": {
        "metrics": RESEARCH / "workload_drift_mscn_windows_bitmap" / "seed_1" / "metrics.json",
        "adaptation_rows": 2000,
        "min_eval_rows": 5000,
    },
    "table": {
        "metrics": RESEARCH / "workload_drift_mscn_windows_table_bitmap" / "seed_1" / "metrics.json",
        "adaptation_rows": 2000,
        "min_eval_rows": 5000,
    },
    "value": {
        "metrics": RESEARCH / "workload_drift_mscn_windows_value_bitmap" / "seed_1" / "metrics.json",
        "adaptation_rows": 500,
        "min_eval_rows": 1000,
    },
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_target_rows(path: Path):
    rows = []
    labels = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter="#")
        for row in reader:
            if len(row) != 4:
                raise ValueError(f"expected 4 fields in {path}, got {len(row)}")
            rows.append(row)
            labels.append(max(1, int(row[3])))
    return rows, np.asarray(labels, dtype=np.float64)


def qerror(predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
    pred = np.maximum(1.0, predictions.astype(np.float64))
    lab = np.maximum(1.0, labels.astype(np.float64))
    return np.maximum(pred / lab, lab / pred)


def risk_metrics(predictions: np.ndarray, labels: np.ndarray) -> dict:
    pred = np.maximum(1.0, predictions.astype(np.float64))
    lab = np.maximum(1.0, labels.astype(np.float64))
    log_under = np.maximum(np.log(lab) - np.log(pred), 0.0)
    log_over = np.maximum(np.log(pred) - np.log(lab), 0.0)
    abs_log = np.maximum(log_under, log_over)
    q = np.exp(abs_log)
    # Underestimation receives double weight because it is the classic optimizer failure
    # mode for missed joins, undersized memory, and nested-loop misselection.
    optimizer_risk = (2.0 * np.square(log_under)) + np.square(log_over)
    return {
        "eval_rows": int(lab.shape[0]),
        "danger_q100_rate": float(np.mean(q >= 100.0)),
        "catastrophic_q1000_rate": float(np.mean(q >= 1000.0)),
        "severe_under100_rate": float(np.mean((lab / pred) >= 100.0)),
        "severe_over100_rate": float(np.mean((pred / lab) >= 100.0)),
        "under_risk_mass": float(np.mean(np.square(log_under))),
        "over_risk_mass": float(np.mean(np.square(log_over))),
        "optimizer_risk_score": float(np.mean(optimizer_risk)),
        "p95_log_error": float(np.percentile(abs_log, 95)),
        "p99_log_error": float(np.percentile(abs_log, 99)),
        "p95_qerror": float(np.percentile(q, 95)),
        "p99_qerror": float(np.percentile(q, 99)),
    }


def relative_reduction(before: float, after: float) -> float:
    return float((before - after) / before) if before else 0.0


def top_tail_repair(raw_predictions: np.ndarray, method_predictions: np.ndarray, labels: np.ndarray) -> dict:
    base_q = qerror(raw_predictions, labels)
    method_q = qerror(method_predictions, labels)
    top_k = max(10, int(math.ceil(base_q.shape[0] * 0.01)))
    top_idx = np.argsort(base_q)[-top_k:]
    base_tail = base_q[top_idx]
    method_tail = method_q[top_idx]
    return {
        "top1pct_rows": int(top_idx.shape[0]),
        "top1pct_median_qerror_before": float(np.median(base_tail)),
        "top1pct_median_qerror_after": float(np.median(method_tail)),
        "top1pct_median_qerror_reduction": relative_reduction(
            float(np.median(base_tail)), float(np.median(method_tail))
        ),
        "top1pct_repaired_below_100": float(np.mean(method_tail < 100.0)),
        "top1pct_repaired_below_1000": float(np.mean(method_tail < 1000.0)),
    }


def aggregate_metric(rows: list[dict], key: str) -> float:
    return float(np.median([row[key] for row in rows]))


def aggregate_method(method_rows: list[dict], baseline_rows: list[dict] | None = None) -> dict:
    keys = [
        "danger_q100_rate",
        "catastrophic_q1000_rate",
        "severe_under100_rate",
        "severe_over100_rate",
        "under_risk_mass",
        "over_risk_mass",
        "optimizer_risk_score",
        "p95_log_error",
        "p99_log_error",
        "p95_qerror",
        "p99_qerror",
    ]
    out = {key: aggregate_metric(method_rows, key) for key in keys}
    if "top1pct_median_qerror_reduction" in method_rows[0]:
        for key in [
            "top1pct_median_qerror_reduction",
            "top1pct_repaired_below_100",
            "top1pct_repaired_below_1000",
        ]:
            out[key] = aggregate_metric(method_rows, key)
    else:
        out["top1pct_median_qerror_reduction"] = 0.0
        out["top1pct_repaired_below_100"] = 0.0
        out["top1pct_repaired_below_1000"] = 0.0
    if baseline_rows:
        baseline_risk = aggregate_metric(baseline_rows, "optimizer_risk_score")
        baseline_under = aggregate_metric(baseline_rows, "severe_under100_rate")
        baseline_danger = aggregate_metric(baseline_rows, "danger_q100_rate")
        out["optimizer_risk_reduction"] = relative_reduction(
            baseline_risk, out["optimizer_risk_score"]
        )
        out["severe_under100_reduction"] = relative_reduction(
            baseline_under, out["severe_under100_rate"]
        )
        out["danger_q100_reduction"] = relative_reduction(
            baseline_danger, out["danger_q100_rate"]
        )
    else:
        out["optimizer_risk_reduction"] = 0.0
        out["severe_under100_reduction"] = 0.0
        out["danger_q100_reduction"] = 0.0
    return out


def compute_residual_predictions(rows, labels, raw_predictions, adapt_idx, eval_idx, seed: int):
    residuals = np.log(labels) - np.log(np.maximum(1.0, raw_predictions))
    model = rt.learn_residual_field(
        rows,
        raw_predictions,
        residuals,
        adapt_idx,
        seed,
        lower_clip=1.0,
        upper_clip=99.0,
        feature_preset="full",
    )
    shifts = rt.apply_residual_field(model, eval_idx)
    return raw_predictions[eval_idx] * np.exp(shifts)


def compute_finetune_predictions(metrics: dict, encoded: dict, adapt_idx, eval_idx, seed: int, args):
    torch, DataLoader = mscn_run.require_torch()
    mscn_run.add_mscn_to_path()
    from mscn.model import SetConv

    dims = encoded["feature_dims"]
    model = SetConv(
        dims["sample_feats"],
        dims["predicate_feats"],
        dims["join_feats"],
        metrics["hidden_units"],
    )
    model.load_state_dict(torch.load(Path(metrics["artifacts"]["model"]), map_location="cpu"))

    predict_dataset = encoded["predict_dataset"]
    adapt_dataset = ft.select_dataset(torch, predict_dataset, adapt_idx)
    eval_dataset = ft.select_dataset(torch, predict_dataset, eval_idx)
    adapt_loader = DataLoader(adapt_dataset, batch_size=args.finetune_batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.finetune_batch_size)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(args.threads)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.finetune_lr)
    start = time.perf_counter()
    losses = []
    for _epoch in range(args.finetune_epochs):
        losses.append(ft.train_epoch(torch, model, adapt_loader, optimizer, encoded["min_val"], encoded["max_val"]))
    outputs = mscn_run.predict(torch, model, eval_loader)
    predictions = mscn_run.unnormalize_tensor(torch, outputs, encoded["min_val"], encoded["max_val"])
    return predictions.astype(np.float64), float(time.perf_counter() - start), losses


def evaluate_drift(drift: str, args) -> dict:
    spec = DRIFTS[drift]
    metrics_path = spec["metrics"]
    metrics = read_json(metrics_path)
    target_path = ft.resolve_target_path(metrics_path, metrics)
    rows, labels = load_target_rows(target_path)
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

    split_reports = []
    method_rows = {"raw-mscn": [], "residual-field": [], "mscn-finetune-best": []}
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        indices = np.arange(labels.shape[0])
        rng.shuffle(indices)
        adapt_idx = indices[: spec["adaptation_rows"]]
        eval_idx = indices[spec["adaptation_rows"] :]
        eval_labels = labels[eval_idx]
        raw_eval = raw_predictions[eval_idx]
        residual_eval = compute_residual_predictions(rows, labels, raw_predictions, adapt_idx, eval_idx, seed)
        finetune_eval, finetune_seconds, finetune_losses = compute_finetune_predictions(
            metrics, encoded, adapt_idx, eval_idx, seed, args
        )

        split = {
            "seed": seed,
            "adaptation_rows": int(adapt_idx.shape[0]),
            "eval_rows": int(eval_idx.shape[0]),
            "finetune_seconds": finetune_seconds,
            "finetune_epoch_losses": finetune_losses,
            "adaptation_indices_sha256": ft.sha256_array(adapt_idx),
            "eval_indices_sha256": ft.sha256_array(eval_idx),
            "methods": {},
        }
        baseline_metrics = risk_metrics(raw_eval, eval_labels)
        split["methods"]["raw-mscn"] = baseline_metrics
        method_rows["raw-mscn"].append(baseline_metrics)

        for name, predictions in [
            ("residual-field", residual_eval),
            ("mscn-finetune-best", finetune_eval),
        ]:
            metrics_row = risk_metrics(predictions, eval_labels)
            metrics_row.update(top_tail_repair(raw_eval, predictions, eval_labels))
            split["methods"][name] = metrics_row
            method_rows[name].append(metrics_row)
        split_reports.append(split)

    aggregate = {
        "raw-mscn": aggregate_method(method_rows["raw-mscn"]),
        "residual-field": aggregate_method(method_rows["residual-field"], method_rows["raw-mscn"]),
        "mscn-finetune-best": aggregate_method(method_rows["mscn-finetune-best"], method_rows["raw-mscn"]),
    }
    return {
        "drift": drift,
        "baseline_metrics": str(metrics_path),
        "target_workload": str(target_path),
        "adaptation_rows": spec["adaptation_rows"],
        "seed_count": len(args.seeds),
        "split_reports": split_reports,
        "aggregate": aggregate,
    }


def write_csv(path: Path, reports: list[dict]) -> list[dict]:
    rows = []
    for report in reports:
        for method, metrics in report["aggregate"].items():
            row = {
                "drift": report["drift"],
                "method": method,
                "adaptation_rows": report["adaptation_rows"],
                **metrics,
            }
            rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}"


def best_cell(text: str, is_best: bool) -> str:
    return f"\\bestcell{{{text}}}" if is_best else text


def reduction_cell(text: str, value: float, best_value=None) -> str:
    if value <= 0:
        return text
    if best_value is not None and best_value > 0 and value == best_value:
        return best_cell(text, True)
    return f"\\passcell{{{text}}}"


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
        "\\caption{Optimizer-facing risk proxy on held-out target queries. Risk uses double weight for severe underestimation.}",
        "\\label{tab:optimizer-risk}",
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "Drift & Method & $Q\\ge100$ & Under $\\ge100$ & Risk Red. & Top1\\% $<1000$ & Top1\\% Red. \\\\",
        "\\midrule",
    ]
    best_by_drift = {}
    for drift in {row["drift"] for row in rows}:
        drift_rows = [row for row in rows if row["drift"] == drift]
        best_by_drift[drift] = {
            "danger_q100_rate": min(row["danger_q100_rate"] for row in drift_rows),
            "severe_under100_rate": min(row["severe_under100_rate"] for row in drift_rows),
            "optimizer_risk_reduction": max(row["optimizer_risk_reduction"] for row in drift_rows),
            "top1pct_repaired_below_1000": max(row["top1pct_repaired_below_1000"] for row in drift_rows),
            "top1pct_median_qerror_reduction": max(
                row["top1pct_median_qerror_reduction"] for row in drift_rows
            ),
        }
    for row in sorted(rows, key=lambda item: (item["drift"], order[item["method"]])):
        best = best_by_drift[row["drift"]]
        danger = f"{pct(row['danger_q100_rate'])}\\%"
        under = f"{pct(row['severe_under100_rate'])}\\%"
        risk_reduction = f"{pct(row['optimizer_risk_reduction'])}\\%"
        top1_repaired = f"{pct(row['top1pct_repaired_below_1000'])}\\%"
        top1_reduction = f"{pct(row['top1pct_median_qerror_reduction'])}\\%"
        lines.append(
            f"{row['drift'].title()} & {labels[row['method']]} "
            f"& {best_cell(danger, row['danger_q100_rate'] == best['danger_q100_rate'])} "
            f"& {best_cell(under, row['severe_under100_rate'] == best['severe_under100_rate'])} "
            f"& {reduction_cell(risk_reduction, row['optimizer_risk_reduction'], best['optimizer_risk_reduction'])} "
            f"& {reduction_cell(top1_repaired, row['top1pct_repaired_below_1000'], best['top1pct_repaired_below_1000'])} "
            f"& {reduction_cell(top1_reduction, row['top1pct_median_qerror_reduction'], best['top1pct_median_qerror_reduction'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute optimizer-facing risk proxies for CE adaptation.")
    parser.add_argument("--output-dir", default="paper/workload_drift_residual_field/results")
    parser.add_argument("--drifts", default="column,table,value")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--finetune-epochs", type=int, default=20)
    parser.add_argument("--finetune-lr", type=float, default=1e-3)
    parser.add_argument("--finetune-batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    args.seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    drifts = [item.strip() for item in args.drifts.split(",") if item.strip()]
    output_dir = ROOT / args.output_dir

    reports = [evaluate_drift(drift, args) for drift in drifts]
    rows = write_csv(output_dir / "optimizer_risk_proxy.csv", reports)
    write_table(output_dir / "tables" / "optimizer_risk_proxy.tex", rows)
    manifest = {
        "passed": True,
        "scope": "Optimizer-facing risk proxy for raw MSCN, residual-field, and same-budget MSCN fine-tuning",
        "drifts": drifts,
        "seeds": args.seeds,
        "finetune_epochs": args.finetune_epochs,
        "finetune_lr": args.finetune_lr,
        "csv": str(output_dir / "optimizer_risk_proxy.csv"),
        "table": str(output_dir / "tables" / "optimizer_risk_proxy.tex"),
        "json": str(output_dir / "optimizer_risk_proxy.json"),
    }
    output = {"manifest": manifest, "reports": reports}
    write_json(output_dir / "optimizer_risk_proxy.json", output)
    write_json(output_dir / "optimizer_risk_proxy_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
