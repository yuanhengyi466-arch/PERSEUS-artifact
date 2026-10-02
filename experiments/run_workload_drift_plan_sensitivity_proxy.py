import argparse
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import run_workload_drift_mscn_windows as mscn_run
import run_workload_drift_mscn_finetune as ft
import run_workload_drift_optimizer_risk_proxy as risk


ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def qerror(predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
    pred = np.maximum(1.0, predictions.astype(np.float64))
    lab = np.maximum(1.0, labels.astype(np.float64))
    return np.maximum(pred / lab, lab / pred)


def sample_pairs(labels: np.ndarray, seed: int, max_pairs: int, min_true_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng((seed * 1009) + 17)
    n = int(labels.shape[0])
    if n < 2:
        raise ValueError("need at least two evaluation rows for pair sampling")

    left_chunks = []
    right_chunks = []
    accepted = 0
    attempts = 0
    max_attempts = max(10 * max_pairs, max_pairs + 10000)
    batch = min(max(10000, max_pairs // 2), 250000)

    while accepted < max_pairs and attempts < max_attempts:
        draw = min(batch, max_attempts - attempts)
        left = rng.integers(0, n, size=draw, endpoint=False)
        right = rng.integers(0, n, size=draw, endpoint=False)
        attempts += draw

        not_same = left != right
        low = np.minimum(labels[left], labels[right])
        high = np.maximum(labels[left], labels[right])
        has_margin = (high / np.maximum(low, 1.0)) >= min_true_ratio
        keep = not_same & has_margin
        if not np.any(keep):
            continue

        keep_left = left[keep]
        keep_right = right[keep]
        remaining = max_pairs - accepted
        if keep_left.shape[0] > remaining:
            keep_left = keep_left[:remaining]
            keep_right = keep_right[:remaining]
        left_chunks.append(keep_left)
        right_chunks.append(keep_right)
        accepted += int(keep_left.shape[0])

    if not left_chunks:
        raise ValueError("pair sampler found no pairs with the requested true-cardinality margin")
    return np.concatenate(left_chunks), np.concatenate(right_chunks)


def pair_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    tail_mask: np.ndarray,
) -> dict:
    y_left = labels[left]
    y_right = labels[right]
    p_left = np.maximum(1.0, predictions[left])
    p_right = np.maximum(1.0, predictions[right])

    true_left = y_left <= y_right
    pred_left = p_left <= p_right
    flip = true_left != pred_left
    best_true = np.minimum(y_left, y_right)
    chosen_true = np.where(pred_left, y_left, y_right)
    regret = chosen_true / np.maximum(best_true, 1.0)
    regret = np.maximum(1.0, regret)

    tail_pair = tail_mask[left] | tail_mask[right]
    if np.any(tail_pair):
        tail_flip = flip[tail_pair]
        tail_regret = regret[tail_pair]
        tail_flip_rate = float(np.mean(tail_flip))
        tail_regret10_rate = float(np.mean(tail_regret >= 10.0))
        tail_p95_regret = float(np.percentile(tail_regret, 95))
        tail_pair_count = int(tail_regret.shape[0])
    else:
        tail_flip_rate = 0.0
        tail_regret10_rate = 0.0
        tail_p95_regret = 1.0
        tail_pair_count = 0

    return {
        "pair_count": int(left.shape[0]),
        "tail_pair_count": tail_pair_count,
        "flip_rate": float(np.mean(flip)),
        "regret10_rate": float(np.mean(regret >= 10.0)),
        "regret100_rate": float(np.mean(regret >= 100.0)),
        "mean_log_regret": float(np.mean(np.log(regret))),
        "p95_regret": float(np.percentile(regret, 95)),
        "p99_regret": float(np.percentile(regret, 99)),
        "tail_flip_rate": tail_flip_rate,
        "tail_regret10_rate": tail_regret10_rate,
        "tail_p95_regret": tail_p95_regret,
    }


def median_metric(rows: list[dict], key: str) -> float:
    return float(np.median([float(row[key]) for row in rows]))


def aggregate_method(method_rows: list[dict], baseline_rows: list[dict] | None = None) -> dict:
    keys = [
        "pair_count",
        "tail_pair_count",
        "flip_rate",
        "regret10_rate",
        "regret100_rate",
        "mean_log_regret",
        "p95_regret",
        "p99_regret",
        "tail_flip_rate",
        "tail_regret10_rate",
        "tail_p95_regret",
    ]
    out = {key: median_metric(method_rows, key) for key in keys}
    if baseline_rows:
        base_flip = median_metric(baseline_rows, "flip_rate")
        base_regret10 = median_metric(baseline_rows, "regret10_rate")
        base_tail_flip = median_metric(baseline_rows, "tail_flip_rate")
        out["flip_reduction"] = risk.relative_reduction(base_flip, out["flip_rate"])
        out["regret10_reduction"] = risk.relative_reduction(base_regret10, out["regret10_rate"])
        out["tail_flip_reduction"] = risk.relative_reduction(base_tail_flip, out["tail_flip_rate"])
    else:
        out["flip_reduction"] = 0.0
        out["regret10_reduction"] = 0.0
        out["tail_flip_reduction"] = 0.0
    return out


def load_encoded_dataset(metrics_path: Path, metrics: dict, target_path: Path) -> dict:
    encoding_args = SimpleNamespace(
        num_materialized_samples=metrics["num_materialized_samples"],
        train_bitmap_file=metrics["provenance"]["bitmap_files"]["train_bitmap_file"],
        predict_bitmap_file=metrics["provenance"]["bitmap_files"]["predict_bitmap_file"],
    )
    return mscn_run.encode_dataset(
        Path(metrics_path.parent / "data" / metrics["provenance"]["train_rel"]),
        target_path,
        Path(metrics_path.parent / "data" / "baselines" / "MSCN-modified" / "workloads" / "stats"),
        metrics["seed"],
        encoding_args,
    )


def evaluate_drift(drift: str, args) -> dict:
    spec = risk.DRIFTS[drift]
    metrics_path = spec["metrics"]
    metrics = risk.read_json(metrics_path)
    target_path = ft.resolve_target_path(metrics_path, metrics)
    rows, labels = risk.load_target_rows(target_path)
    raw_predictions = np.loadtxt(metrics["artifacts"]["predictions"], dtype=np.float64)
    if raw_predictions.shape[0] != labels.shape[0]:
        raise ValueError(f"prediction/label mismatch for {drift}")

    encoded = load_encoded_dataset(metrics_path, metrics, target_path)
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
        residual_eval = risk.compute_residual_predictions(rows, labels, raw_predictions, adapt_idx, eval_idx, seed)
        finetune_eval, finetune_seconds, finetune_losses = risk.compute_finetune_predictions(
            metrics, encoded, adapt_idx, eval_idx, seed, args
        )

        left, right = sample_pairs(eval_labels, seed, args.max_pairs, args.min_true_ratio)
        raw_q = qerror(raw_eval, eval_labels)
        top_k = max(10, int(math.ceil(raw_q.shape[0] * args.tail_fraction)))
        tail_threshold = float(np.partition(raw_q, -top_k)[-top_k])
        tail_mask = raw_q >= tail_threshold

        split = {
            "seed": seed,
            "adaptation_rows": int(adapt_idx.shape[0]),
            "eval_rows": int(eval_idx.shape[0]),
            "pair_count": int(left.shape[0]),
            "min_true_ratio": args.min_true_ratio,
            "tail_fraction": args.tail_fraction,
            "tail_threshold_qerror": tail_threshold,
            "finetune_seconds": finetune_seconds,
            "finetune_epoch_losses": finetune_losses,
            "adaptation_indices_sha256": ft.sha256_array(adapt_idx),
            "eval_indices_sha256": ft.sha256_array(eval_idx),
            "methods": {},
        }
        for name, predictions in [
            ("raw-mscn", raw_eval),
            ("residual-field", residual_eval),
            ("mscn-finetune-best", finetune_eval),
        ]:
            metrics_row = pair_metrics(eval_labels, predictions, left, right, tail_mask)
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
            rows.append(
                {
                    "drift": report["drift"],
                    "method": method,
                    "adaptation_rows": report["adaptation_rows"],
                    **metrics,
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}"


def number(value: float) -> str:
    if value >= 1000:
        return f"{value:.0f}"
    if value >= 100:
        return f"{value:.1f}"
    return f"{value:.2f}"


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
        "\\caption{Cardinality-ordering sensitivity proxy. Held-out query pairs, not alternative plans for the same SQL query, have true cardinalities differing by at least $10\\times$; a flip means the estimator ranks the higher-cardinality candidate as cheaper under a lower-cardinality-is-cheaper rule.}",
        "\\label{tab:plan-sensitivity}",
        "\\begin{tabular}{llrrrrrr}",
        "\\toprule",
        "Drift & Method & Pairs & Flip & Flip Red. & Regret $\\ge10$ & Tail Flip & Tail Red. \\\\",
        "\\midrule",
    ]
    best_by_drift = {}
    for drift in {row["drift"] for row in rows}:
        drift_rows = [row for row in rows if row["drift"] == drift]
        best_by_drift[drift] = {
            "flip_rate": min(float(row["flip_rate"]) for row in drift_rows),
            "flip_reduction": max(float(row["flip_reduction"]) for row in drift_rows),
            "regret10_rate": min(float(row["regret10_rate"]) for row in drift_rows),
            "tail_flip_rate": min(float(row["tail_flip_rate"]) for row in drift_rows),
            "tail_flip_reduction": max(float(row["tail_flip_reduction"]) for row in drift_rows),
        }
    for row in sorted(rows, key=lambda item: (item["drift"], order[item["method"]])):
        best = best_by_drift[row["drift"]]
        flip_rate = float(row["flip_rate"])
        flip_reduction = float(row["flip_reduction"])
        regret10_rate = float(row["regret10_rate"])
        tail_flip_rate = float(row["tail_flip_rate"])
        tail_flip_reduction = float(row["tail_flip_reduction"])
        lines.append(
            f"{row['drift'].title()} & {labels[row['method']]} "
            f"& {int(round(float(row['pair_count'])))} "
            f"& {best_cell(f'{pct(flip_rate)}\\%', flip_rate == best['flip_rate'])} "
            f"& {reduction_cell(f'{pct(flip_reduction)}\\%', flip_reduction, best['flip_reduction'])} "
            f"& {best_cell(f'{pct(regret10_rate)}\\%', regret10_rate == best['regret10_rate'])} "
            f"& {best_cell(f'{pct(tail_flip_rate)}\\%', tail_flip_rate == best['tail_flip_rate'])} "
            f"& {reduction_cell(f'{pct(tail_flip_reduction)}\\%', tail_flip_reduction, best['tail_flip_reduction'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute a cardinality-ordering sensitivity proxy.")
    parser.add_argument("--output-dir", default="paper/workload_drift_residual_field/results")
    parser.add_argument("--drifts", default="column,table,value")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--max-pairs", type=int, default=200000)
    parser.add_argument("--min-true-ratio", type=float, default=10.0)
    parser.add_argument("--tail-fraction", type=float, default=0.01)
    parser.add_argument("--finetune-epochs", type=int, default=20)
    parser.add_argument("--finetune-lr", type=float, default=1e-3)
    parser.add_argument("--finetune-batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    args.seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    drifts = [item.strip() for item in args.drifts.split(",") if item.strip()]
    output_dir = ROOT / args.output_dir

    reports = [evaluate_drift(drift, args) for drift in drifts]
    rows = write_csv(output_dir / "plan_sensitivity_proxy.csv", reports)
    write_table(output_dir / "tables" / "plan_sensitivity_proxy.tex", rows)
    manifest = {
        "passed": True,
        "scope": "Cardinality-ordering sensitivity proxy for held-out target query pairs",
        "drifts": drifts,
        "seeds": args.seeds,
        "max_pairs": args.max_pairs,
        "min_true_ratio": args.min_true_ratio,
        "tail_fraction": args.tail_fraction,
        "finetune_epochs": args.finetune_epochs,
        "finetune_lr": args.finetune_lr,
        "csv": str(output_dir / "plan_sensitivity_proxy.csv"),
        "table": str(output_dir / "tables" / "plan_sensitivity_proxy.tex"),
        "json": str(output_dir / "plan_sensitivity_proxy.json"),
    }
    write_json(output_dir / "plan_sensitivity_proxy.json", {"manifest": manifest, "reports": reports})
    write_json(output_dir / "plan_sensitivity_proxy_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
