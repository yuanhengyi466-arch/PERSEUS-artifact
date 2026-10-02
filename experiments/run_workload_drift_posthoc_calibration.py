import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import run_workload_drift_mscn_finetune as ft


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_array(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(values, dtype=np.int64).tobytes())
    return digest.hexdigest()


def chunks(values, n):
    for i in range(0, len(values), n):
        yield values[i : i + n]


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


def predicate_bin(value: str) -> int:
    try:
        numeric = float(value)
    except ValueError:
        numeric = 0.0
    return int(np.clip(np.floor(np.log1p(abs(numeric)) / 2.0), 0, 9))


def parse_predicates(row: list[str]) -> list[tuple[str, str, float]]:
    parsed = []
    parts = row[2].split(",") if row[2] else []
    for pred in chunks(parts, 3):
        if len(pred) != 3:
            continue
        column, operator, value = pred
        try:
            numeric = float(value)
        except ValueError:
            numeric = 0.0
        parsed.append((column.strip(), operator.strip(), numeric))
    return parsed


def feature_vocab(rows: list[list[str]]) -> dict:
    tables = sorted({item.strip().split()[0] for row in rows for item in row[0].split(",") if item.strip()})
    columns = set()
    operators = set()
    for row in rows:
        for column, operator, _value in parse_predicates(row):
            if "." in column:
                columns.add(column)
                operators.add(operator)
    return {"tables": tables, "columns": sorted(columns), "operators": sorted(operators)}


def row_features(rows: list[list[str]], predictions: np.ndarray, vocab: dict, feature_preset: str) -> np.ndarray:
    features = []
    for row, prediction in zip(rows, predictions):
        table_names = [item.strip().split()[0] for item in row[0].split(",") if item.strip()]
        joins = [item.strip() for item in row[1].split(",") if item.strip()]
        columns = []
        operators = []
        values = []
        bins = []
        ranges = {}
        for column, operator, numeric in parse_predicates(row):
            columns.append(column)
            operators.append(operator)
            values.append(numeric)
            bins.append(predicate_bin(str(numeric)))
            slot = ranges.setdefault(column, {"lower": None, "upper": None})
            if operator == ">":
                slot["lower"] = numeric if slot["lower"] is None else max(slot["lower"], numeric)
            elif operator == "<":
                slot["upper"] = numeric if slot["upper"] is None else min(slot["upper"], numeric)

        table_set = set(table_names)
        column_set = set(columns)
        vector = [
            math.log(max(1.0, float(prediction))),
            float(len(table_names)),
            float(len(joins)),
            float(len(columns)),
        ]
        vector.extend(1.0 if table in table_set else 0.0 for table in vocab["tables"])
        vector.extend(1.0 if column in column_set else 0.0 for column in vocab["columns"])
        vector.extend(float(operators.count(operator)) for operator in vocab["operators"])
        vector.extend(float(bins.count(bin_id)) for bin_id in range(10))
        if values:
            log_values = np.log1p(np.abs(np.asarray(values, dtype=np.float64)))
            vector.extend(
                [
                    float(np.min(log_values)),
                    float(np.median(log_values)),
                    float(np.max(log_values)),
                    float(np.mean(log_values)),
                    float(np.std(log_values)),
                ]
            )
        else:
            vector.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        if feature_preset == "value":
            for column in vocab["columns"]:
                bounds = ranges.get(column, {})
                lower = bounds.get("lower")
                upper = bounds.get("upper")
                has_lower = lower is not None
                has_upper = upper is not None
                if has_lower and has_upper:
                    center = 0.5 * (lower + upper)
                    width = max(0.0, upper - lower)
                elif has_lower:
                    center = lower
                    width = 0.0
                elif has_upper:
                    center = upper
                    width = 0.0
                else:
                    center = 0.0
                    width = 0.0
                vector.extend(
                    [
                        1.0 if has_lower else 0.0,
                        1.0 if has_upper else 0.0,
                        math.log1p(abs(center)),
                        math.log1p(width),
                        math.log1p(abs(lower)) if has_lower else 0.0,
                        math.log1p(abs(upper)) if has_upper else 0.0,
                    ]
                )
        features.append(vector)
    return np.asarray(features, dtype=np.float64)


def fit_predict(method: str, rows, labels, predictions, adapt_idx, eval_idx, seed: int, args):
    raw = np.maximum(1.0, predictions.astype(np.float64))
    labels = np.maximum(1.0, labels.astype(np.float64))
    log_raw = np.log(raw)
    log_labels = np.log(labels)
    residuals = log_labels - log_raw

    global_shift = float(np.median(residuals[adapt_idx]))
    if method == "global":
        return raw[eval_idx] * math.exp(global_shift), {
            "model_family": "median log-residual global calibration",
            "global_shift": global_shift,
        }
    if method == "shrink-global":
        if not 0.0 <= args.shrink_factor <= 1.5:
            raise ValueError("shrink factor must be between 0 and 1.5")
        shift = args.shrink_factor * global_shift
        return raw[eval_idx] * math.exp(shift), {
            "model_family": "shrink median log-residual global calibration",
            "global_shift": global_shift,
            "shrink_factor": float(args.shrink_factor),
        }
    if method in {"isotonic", "shrink-isotonic"}:
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(log_raw[adapt_idx], log_labels[adapt_idx])
        calibrated_log = model.predict(log_raw[eval_idx])
        if method == "shrink-isotonic":
            if not 0.0 <= args.shrink_factor <= 1.0:
                raise ValueError("shrink factor must be between 0 and 1")
            calibrated_log = log_raw[eval_idx] + args.shrink_factor * (calibrated_log - log_raw[eval_idx])
            return np.exp(calibrated_log), {
                "model_family": "shrink IsotonicRegression",
                "shrink_factor": float(args.shrink_factor),
            }
        return np.exp(calibrated_log), {"model_family": "IsotonicRegression"}

    vocab = feature_vocab(rows)
    features = row_features(rows, raw, vocab, args.feature_preset)
    clip_lo, clip_hi = np.percentile(residuals[adapt_idx], [1.0, 99.0])
    targets = np.clip(residuals[adapt_idx], clip_lo, clip_hi)
    if method in {"ridge", "shrink-ridge"}:
        model = make_pipeline(StandardScaler(), Ridge(alpha=args.ridge_alpha))
        model.fit(features[adapt_idx], targets)
        raw_shift = np.clip(model.predict(features[eval_idx]), clip_lo, clip_hi)
        if method == "shrink-ridge":
            if not 0.0 <= args.shrink_factor <= 1.5:
                raise ValueError("shrink factor must be between 0 and 1.5")
            if args.shrink_anchor == "global":
                shift = global_shift + args.shrink_factor * (raw_shift - global_shift)
            else:
                shift = args.shrink_factor * raw_shift
        else:
            shift = raw_shift
        return raw[eval_idx] * np.exp(shift), {
            "model_family": "Ridge residual calibrator",
            "feature_preset": args.feature_preset,
            "feature_dim": int(features.shape[1]),
            "residual_clip_range": [float(clip_lo), float(clip_hi)],
            "ridge_alpha": float(args.ridge_alpha),
            "shrink_factor": float(args.shrink_factor) if method == "shrink-ridge" else 1.0,
            "shrink_anchor": args.shrink_anchor if method == "shrink-ridge" else "none",
        }
    if method == "hist-gradient":
        model = HistGradientBoostingRegressor(
            loss=args.hist_loss,
            max_iter=args.hist_max_iter,
            learning_rate=args.hist_learning_rate,
            l2_regularization=args.hist_l2,
            max_leaf_nodes=args.hist_max_leaf_nodes,
            min_samples_leaf=args.hist_min_samples_leaf,
            random_state=seed + 314159,
        )
        model.fit(features[adapt_idx], targets)
        shift = np.clip(model.predict(features[eval_idx]), clip_lo, clip_hi)
        return raw[eval_idx] * np.exp(shift), {
            "model_family": "HistGradientBoosting residual calibrator",
            "feature_preset": args.feature_preset,
            "feature_dim": int(features.shape[1]),
            "residual_clip_range": [float(clip_lo), float(clip_hi)],
            "hist_loss": args.hist_loss,
            "hist_max_iter": int(args.hist_max_iter),
            "hist_learning_rate": float(args.hist_learning_rate),
            "hist_l2": float(args.hist_l2),
            "hist_max_leaf_nodes": int(args.hist_max_leaf_nodes),
            "hist_min_samples_leaf": int(args.hist_min_samples_leaf),
        }
    if method == "tail-risk-ridge":
        residual_model = make_pipeline(StandardScaler(), Ridge(alpha=args.ridge_alpha))
        risk_model = HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            l2_regularization=0.01,
            max_leaf_nodes=15,
            random_state=seed + 104729,
        )
        residual_model.fit(features[adapt_idx], targets)
        risk_model.fit(features[adapt_idx], np.abs(residuals[adapt_idx]))
        raw_shift = np.clip(residual_model.predict(features[eval_idx]), clip_lo, clip_hi)
        if args.shrink_anchor == "global":
            shift = global_shift + args.shrink_factor * (raw_shift - global_shift)
        else:
            shift = args.shrink_factor * raw_shift
        risk_eval = risk_model.predict(features[eval_idx])
        threshold = float(np.quantile(risk_model.predict(features[adapt_idx]), args.tail_gate_quantile))
        tail_mask = risk_eval >= threshold
        final_shift = np.where(tail_mask, shift, global_shift)
        return raw[eval_idx] * np.exp(final_shift), {
            "model_family": "tail-risk-gated Ridge residual calibrator",
            "feature_preset": args.feature_preset,
            "feature_dim": int(features.shape[1]),
            "residual_clip_range": [float(clip_lo), float(clip_hi)],
            "ridge_alpha": float(args.ridge_alpha),
            "shrink_factor": float(args.shrink_factor),
            "shrink_anchor": args.shrink_anchor,
            "tail_gate_quantile": float(args.tail_gate_quantile),
            "tail_gate_threshold": threshold,
            "tail_eval_fraction": float(np.mean(tail_mask)),
        }
    raise ValueError(f"unknown method: {method}")


def evaluate_split(args, rows, labels, predictions, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    indices = np.arange(labels.shape[0])
    rng.shuffle(indices)
    if args.adaptation_rows >= labels.shape[0]:
        raise ValueError("adaptation rows must be smaller than target row count")
    adapt_idx = indices[: args.adaptation_rows]
    eval_idx = indices[args.adaptation_rows :]
    eval_labels = labels[eval_idx]
    raw_eval = predictions[eval_idx]

    baseline_q = summarize(qerror(raw_eval, eval_labels))
    global_pred, global_info = fit_predict("global", rows, labels, predictions, adapt_idx, eval_idx, seed, args)
    global_q = summarize(qerror(global_pred, eval_labels))
    method_pred, method_info = fit_predict(args.method, rows, labels, predictions, adapt_idx, eval_idx, seed, args)
    method_q = summarize(qerror(method_pred, eval_labels))
    reductions = {key: relative_reduction(baseline_q[key], method_q[key]) for key in ["median", "p90", "p95", "p99", "mean"]}
    reductions_vs_global = {
        key: relative_reduction(global_q[key], method_q[key]) for key in ["median", "p90", "p95", "p99", "mean"]
    }
    pass_gates = {
        "p90_reduction": reductions["p90"] >= args.min_p90_reduction,
        "p95_reduction": reductions["p95"] >= args.min_p95_reduction,
        "p99_reduction": reductions["p99"] >= args.min_p99_reduction,
        "beats_global_p95": reductions_vs_global["p95"] >= args.min_vs_global_p95_reduction,
        "beats_global_p99": reductions_vs_global["p99"] >= args.min_vs_global_p99_reduction,
        "median_not_much_worse": method_q["median"] <= baseline_q["median"] * args.max_median_regression,
        "eval_rows": int(eval_idx.shape[0]) >= args.min_eval_rows,
    }
    return {
        "seed": seed,
        "passed": all(pass_gates.values()),
        "pass_gates": pass_gates,
        "adaptation_rows": int(adapt_idx.shape[0]),
        "eval_rows": int(eval_idx.shape[0]),
        "baseline_qerror": baseline_q,
        "global_calibration_qerror": global_q,
        "method_qerror": method_q,
        "relative_reductions": reductions,
        "relative_reductions_vs_global_calibration": reductions_vs_global,
        "global_model": global_info,
        "method_model": method_info,
        "adaptation_indices_sha256": sha256_array(adapt_idx),
        "eval_indices_sha256": sha256_array(eval_idx),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc calibration for workload-drift prediction artifacts.")
    parser.add_argument("--baseline-metrics", required=True)
    parser.add_argument(
        "--method",
        choices=[
            "global",
            "shrink-global",
            "isotonic",
            "shrink-isotonic",
            "ridge",
            "shrink-ridge",
            "hist-gradient",
            "tail-risk-ridge",
        ],
        default="tail-risk-ridge",
    )
    parser.add_argument("--adaptation-rows", type=int, default=500)
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--min-eval-rows", type=int, default=1000)
    parser.add_argument("--min-p90-reduction", type=float, default=0.30)
    parser.add_argument("--min-p95-reduction", type=float, default=0.40)
    parser.add_argument("--min-p99-reduction", type=float, default=0.60)
    parser.add_argument("--min-vs-global-p95-reduction", type=float, default=0.25)
    parser.add_argument("--min-vs-global-p99-reduction", type=float, default=0.40)
    parser.add_argument("--max-median-regression", type=float, default=1.20)
    parser.add_argument("--shrink-factor", type=float, default=0.50)
    parser.add_argument("--shrink-anchor", choices=["zero", "global"], default="global")
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--feature-preset", choices=["basic", "value"], default="basic")
    parser.add_argument("--hist-max-iter", type=int, default=300)
    parser.add_argument("--hist-loss", choices=["squared_error", "absolute_error"], default="squared_error")
    parser.add_argument("--hist-learning-rate", type=float, default=0.05)
    parser.add_argument("--hist-l2", type=float, default=0.01)
    parser.add_argument("--hist-max-leaf-nodes", type=int, default=15)
    parser.add_argument("--hist-min-samples-leaf", type=int, default=20)
    parser.add_argument("--tail-gate-quantile", type=float, default=0.70)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    metrics_path = Path(args.baseline_metrics)
    if not metrics_path.is_absolute():
        metrics_path = ROOT / metrics_path
    metrics = read_json(metrics_path)
    target_path = ft.resolve_target_path(metrics_path, metrics)
    rows, labels = load_target_rows(target_path)
    predictions = np.loadtxt(metrics["artifacts"]["predictions"], dtype=np.float64)
    if predictions.shape[0] != labels.shape[0]:
        raise ValueError(f"prediction/label mismatch: {predictions.shape[0]} vs {labels.shape[0]}")
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    split_reports = [evaluate_split(args, rows, labels, predictions, seed) for seed in seeds]
    passed_seed_count = sum(1 for split in split_reports if split["passed"])
    aggregate_reductions = {
        key: float(np.median([split["relative_reductions"][key] for split in split_reports]))
        for key in ["median", "p90", "p95", "p99", "mean"]
    }
    aggregate_reductions_vs_global = {
        key: float(
            np.median([split["relative_reductions_vs_global_calibration"][key] for split in split_reports])
        )
        for key in ["median", "p90", "p95", "p99", "mean"]
    }
    report = {
        "passed": passed_seed_count == len(split_reports),
        "baseline": metrics["baseline"],
        "baseline_metrics": str(metrics_path),
        "target_workload": str(target_path),
        "method": args.method,
        "uses_api": False,
        "windows_native": True,
        "requires_labeled_target_adaptation_rows": args.adaptation_rows,
        "row_count": int(labels.shape[0]),
        "seed_count": len(split_reports),
        "passed_seed_count": passed_seed_count,
        "aggregate_reductions": aggregate_reductions,
        "aggregate_reductions_vs_global_calibration": aggregate_reductions_vs_global,
        "gate_thresholds": {
            "min_p90_reduction": args.min_p90_reduction,
            "min_p95_reduction": args.min_p95_reduction,
            "min_p99_reduction": args.min_p99_reduction,
            "min_vs_global_p95_reduction": args.min_vs_global_p95_reduction,
            "min_vs_global_p99_reduction": args.min_vs_global_p99_reduction,
            "max_median_regression": args.max_median_regression,
            "min_eval_rows": args.min_eval_rows,
        },
        "split_reports": split_reports,
        "command": sys.argv,
    }
    output = Path(args.output_json)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {output}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
