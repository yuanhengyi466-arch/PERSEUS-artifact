import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SSCARD_ROOT = ROOT / "baselines" / "SSCard"


def normalize_columns(frame: pd.DataFrame) -> dict[str, str]:
    return {column.strip().lower(): column for column in frame.columns}


def pick_column(frame: pd.DataFrame, names: list[str]) -> str:
    columns = normalize_columns(frame)
    for name in names:
        if name in columns:
            return columns[name]
    raise KeyError(f"missing any of {names} in {list(frame.columns)}")


def load_result_csv_by_numeric_tail(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        header = handle.readline().strip()
        columns = [column.strip().lower() for column in header.split(",")]
        rows = []
        for line in handle:
            line = line.rstrip("\r\n")
            if not line:
                continue
            if "true_count" in columns and "est_count" in columns:
                parts = line.rsplit(",", 4)
                if len(parts) != 5:
                    continue
                query, _, truth, estimate, _ = parts
            elif "padding_latency" in columns:
                parts = line.rsplit(",", 4)
                if len(parts) != 5:
                    continue
                query, truth, estimate, _, _ = parts
            else:
                parts = line.rsplit(",", 3)
                if len(parts) != 4:
                    continue
                query, truth, estimate, _ = parts
            rows.append((query.strip().strip('"'), truth, estimate))
    data = pd.DataFrame(rows, columns=["query", "truth", "estimate"])
    data["truth"] = pd.to_numeric(data["truth"], errors="coerce")
    data["estimate"] = pd.to_numeric(data["estimate"], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    data["truth"] = np.maximum(1.0, data["truth"].to_numpy(dtype=np.float64))
    data["estimate"] = np.maximum(1.0, data["estimate"].to_numpy(dtype=np.float64))
    return data


def load_result_csv(path: Path) -> pd.DataFrame:
    robust = load_result_csv_by_numeric_tail(path)
    try:
        frame = pd.read_csv(path)
    except pd.errors.ParserError:
        frame = pd.read_csv(path, engine="python", on_bad_lines="skip")
    query_col = pick_column(frame, ["s_q", "string", "pattern"])
    truth_col = pick_column(frame, ["ground_truth", "true_count", "true"])
    estimate_col = pick_column(frame, ["estimate", "est_count", "estimate_card", "est"])
    data = pd.DataFrame(
        {
            "query": frame[query_col].astype(str),
            "truth": pd.to_numeric(frame[truth_col], errors="coerce"),
            "estimate": pd.to_numeric(frame[estimate_col], errors="coerce"),
        }
    )
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    data["truth"] = np.maximum(1.0, data["truth"].to_numpy(dtype=np.float64))
    data["estimate"] = np.maximum(1.0, data["estimate"].to_numpy(dtype=np.float64))
    if robust.shape[0] > data.shape[0]:
        return robust
    return data


def qerror(estimate: np.ndarray, truth: np.ndarray) -> np.ndarray:
    est = np.maximum(1.0, np.asarray(estimate, dtype=np.float64))
    tru = np.maximum(1.0, np.asarray(truth, dtype=np.float64))
    return np.maximum(est / tru, tru / est)


def summarize(values: np.ndarray) -> dict:
    return {
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def reduction(before: float, after: float) -> float:
    return float((before - after) / before) if before else 0.0


def query_features(frame: pd.DataFrame) -> np.ndarray:
    queries = frame["query"].astype(str).tolist()
    estimates = np.maximum(1.0, frame["estimate"].to_numpy(dtype=np.float64))
    rows = []
    for query, estimate in zip(queries, estimates):
        chars = list(query)
        length = len(chars)
        counts = {
            "digit": sum(ch.isdigit() for ch in chars),
            "upper": sum(ch.isupper() for ch in chars),
            "lower": sum(ch.islower() for ch in chars),
            "space": sum(ch.isspace() for ch in chars),
        }
        unique = len(set(chars))
        first = ord(chars[0]) if chars else 0
        last = ord(chars[-1]) if chars else 0
        rows.append(
            [
                math.log(estimate),
                math.log1p(estimate),
                float(length),
                math.log1p(length),
                float(unique),
                float(unique / max(1, length)),
                float(counts["digit"]),
                float(counts["upper"]),
                float(counts["lower"]),
                float(counts["space"]),
                float(first % 128) / 128.0,
                float(last % 128) / 128.0,
            ]
        )
    return np.asarray(rows, dtype=np.float64)


def make_residual_estimator(method: str, seed: int):
    if method in {"ridge", "shrink-ridge"}:
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0)), "Ridge residual calibrator"
    if method == "random-forest":
        return (
            RandomForestRegressor(
                n_estimators=250,
                max_depth=14,
                min_samples_leaf=3,
                random_state=seed,
                n_jobs=1,
            ),
            "RandomForest residual calibrator",
        )
    if method in {"hist-gradient", "tail-gated"}:
        return (
            HistGradientBoostingRegressor(
                max_iter=300,
                learning_rate=0.05,
                l2_regularization=0.01,
                max_leaf_nodes=31,
                random_state=seed,
            ),
            "HistGradientBoosting residual calibrator",
        )
    raise ValueError(f"unknown residual estimator: {method}")


def fit_predict(method: str, frame: pd.DataFrame, adapt_idx: np.ndarray, eval_idx: np.ndarray, seed: int, args):
    estimates = np.maximum(1.0, frame["estimate"].to_numpy(dtype=np.float64))
    truth = np.maximum(1.0, frame["truth"].to_numpy(dtype=np.float64))
    log_estimate = np.log(estimates)
    log_truth = np.log(truth)
    residual = log_truth - log_estimate

    if method == "global":
        shift = float(np.median(residual[adapt_idx]))
        return estimates[eval_idx] * math.exp(shift), {"model_family": "median log-residual global calibration"}
    if method in {"isotonic", "shrink-isotonic", "tail-shrink-isotonic"}:
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(log_estimate[adapt_idx], log_truth[adapt_idx])
        calibrated_log_truth = model.predict(log_estimate[eval_idx])
        if method in {"shrink-isotonic", "tail-shrink-isotonic"}:
            if not 0.0 <= args.shrink_factor <= 1.0:
                raise ValueError("shrink factor must be between 0 and 1")
            blended_log_truth = log_estimate[eval_idx] + args.shrink_factor * (
                calibrated_log_truth - log_estimate[eval_idx]
            )
            if method == "tail-shrink-isotonic":
                if not 0.0 <= args.isotonic_tail_quantile <= 1.0:
                    raise ValueError("isotonic tail quantile must be between 0 and 1")
                tail_threshold = float(np.quantile(log_estimate[adapt_idx], args.isotonic_tail_quantile))
                tail_mask = log_estimate[eval_idx] >= tail_threshold
                final_log_truth = np.where(tail_mask, blended_log_truth, log_estimate[eval_idx])
                return np.exp(final_log_truth), {
                    "model_family": "tail shrink IsotonicRegression",
                    "shrink_factor": float(args.shrink_factor),
                    "isotonic_tail_quantile": float(args.isotonic_tail_quantile),
                    "isotonic_tail_threshold": tail_threshold,
                    "tail_eval_fraction": float(np.mean(tail_mask)),
                }
            return np.exp(blended_log_truth), {
                "model_family": "shrink IsotonicRegression",
                "shrink_factor": float(args.shrink_factor),
            }
        return np.exp(calibrated_log_truth), {"model_family": "IsotonicRegression"}

    features = query_features(frame)
    clip_lo, clip_hi = np.percentile(residual[adapt_idx], [1.0, 99.0])
    targets = np.clip(residual[adapt_idx], clip_lo, clip_hi)
    if method == "tail-gated":
        residual_model, residual_family = make_residual_estimator("tail-gated", seed)
        risk_model = HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            l2_regularization=0.01,
            max_leaf_nodes=15,
            random_state=seed + 7919,
        )
        residual_model.fit(features[adapt_idx], targets)
        risk_model.fit(features[adapt_idx], np.abs(residual[adapt_idx]))
        raw_shifts = np.clip(residual_model.predict(features[eval_idx]), clip_lo, clip_hi)
        risk_eval = risk_model.predict(features[eval_idx])
        gate_threshold = float(np.quantile(risk_model.predict(features[adapt_idx]), args.tail_gate_quantile))
        gated = risk_eval >= gate_threshold
        shifts = np.where(gated, raw_shifts, 0.0)
        return estimates[eval_idx] * np.exp(shifts), {
            "model_family": f"tail-gated {residual_family}",
            "feature_dim": int(features.shape[1]),
            "residual_clip_range": [float(clip_lo), float(clip_hi)],
            "tail_gate_quantile": float(args.tail_gate_quantile),
            "tail_gate_threshold": gate_threshold,
            "gated_eval_fraction": float(np.mean(gated)),
        }

    if method == "shrink-ridge":
        if not 0.0 <= args.shrink_factor <= 1.5:
            raise ValueError("shrink factor must be between 0 and 1.5")
        if args.shrink_anchor not in {"zero", "global"}:
            raise ValueError("shrink anchor must be zero or global")
        model, family = make_residual_estimator("shrink-ridge", seed)
        model.fit(features[adapt_idx], targets)
        raw_shifts = np.clip(model.predict(features[eval_idx]), clip_lo, clip_hi)
        if args.shrink_anchor == "global":
            global_shift = float(np.median(residual[adapt_idx]))
            shifts = global_shift + args.shrink_factor * (raw_shifts - global_shift)
        else:
            shifts = args.shrink_factor * raw_shifts
        return estimates[eval_idx] * np.exp(shifts), {
            "model_family": f"shrink {family}",
            "feature_dim": int(features.shape[1]),
            "residual_clip_range": [float(clip_lo), float(clip_hi)],
            "shrink_factor": float(args.shrink_factor),
            "shrink_anchor": args.shrink_anchor,
        }

    model, family = make_residual_estimator(method, seed)
    model.fit(features[adapt_idx], targets)
    shifts = np.clip(model.predict(features[eval_idx]), clip_lo, clip_hi)
    return estimates[eval_idx] * np.exp(shifts), {
        "model_family": family,
        "feature_dim": int(features.shape[1]),
        "residual_clip_range": [float(clip_lo), float(clip_hi)],
    }


def evaluate_split(args, frame: pd.DataFrame, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    indices = np.arange(frame.shape[0], dtype=int)
    rng.shuffle(indices)
    if args.max_rows and args.max_rows < indices.shape[0]:
        indices = indices[: args.max_rows]
    if args.adaptation_rows >= indices.shape[0]:
        raise ValueError("adaptation rows must be smaller than available rows")

    adapt_idx = indices[: args.adaptation_rows]
    eval_idx = indices[args.adaptation_rows :]
    estimates = frame["estimate"].to_numpy(dtype=np.float64)
    truth = frame["truth"].to_numpy(dtype=np.float64)
    baseline_q = summarize(qerror(estimates[eval_idx], truth[eval_idx]))
    global_pred, global_info = fit_predict("global", frame, adapt_idx, eval_idx, seed, args)
    global_q = summarize(qerror(global_pred, truth[eval_idx]))
    method_pred, method_info = fit_predict(args.method, frame, adapt_idx, eval_idx, seed, args)
    method_q = summarize(qerror(method_pred, truth[eval_idx]))

    reductions = {key: reduction(baseline_q[key], method_q[key]) for key in ["median", "p90", "p95", "p99", "mean"]}
    reductions_vs_global = {
        key: reduction(global_q[key], method_q[key]) for key in ["median", "p90", "p95", "p99", "mean"]
    }
    gates = {
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
        "passed": all(gates.values()),
        "pass_gates": gates,
        "adaptation_rows": int(adapt_idx.shape[0]),
        "eval_rows": int(eval_idx.shape[0]),
        "baseline_qerror": baseline_q,
        "global_calibration_qerror": global_q,
        "method_qerror": method_q,
        "relative_reductions": reductions,
        "relative_reductions_vs_global_calibration": reductions_vs_global,
        "global_model": global_info,
        "method_model": method_info,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc calibration on SSCard result CSVs.")
    parser.add_argument("--dataset", default="DBLP_AN")
    parser.add_argument("--baseline-method", default="SSCard")
    parser.add_argument(
        "--method",
        choices=[
            "isotonic",
            "shrink-isotonic",
            "tail-shrink-isotonic",
            "ridge",
            "shrink-ridge",
            "random-forest",
            "hist-gradient",
            "tail-gated",
        ],
        default="random-forest",
    )
    parser.add_argument("--adaptation-rows", type=int, default=2000)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--min-eval-rows", type=int, default=10000)
    parser.add_argument("--min-p90-reduction", type=float, default=0.05)
    parser.add_argument("--min-p95-reduction", type=float, default=0.05)
    parser.add_argument("--min-p99-reduction", type=float, default=0.05)
    parser.add_argument("--min-vs-global-p95-reduction", type=float, default=0.01)
    parser.add_argument("--min-vs-global-p99-reduction", type=float, default=0.01)
    parser.add_argument("--max-median-regression", type=float, default=1.05)
    parser.add_argument("--tail-gate-quantile", type=float, default=0.80)
    parser.add_argument("--isotonic-tail-quantile", type=float, default=0.60)
    parser.add_argument("--shrink-factor", type=float, default=0.55)
    parser.add_argument("--shrink-anchor", choices=["zero", "global"], default="zero")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    result_csv = SSCARD_ROOT / "results" / args.dataset / f"{args.baseline_method}.csv"
    frame = load_result_csv(result_csv)
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    split_reports = [evaluate_split(args, frame, seed) for seed in seeds]
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
        "baseline": f"SSCard/{args.baseline_method}",
        "dataset": args.dataset,
        "result_csv": str(result_csv),
        "method": args.method,
        "uses_api": False,
        "windows_native": True,
        "requires_labeled_target_adaptation_rows": args.adaptation_rows,
        "row_count": int(frame.shape[0]),
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
    output = Path(args.output_json) if args.output_json else (
        ROOT
        / "research_plan"
        / f"sscard_posthoc_{args.dataset}_{args.baseline_method}_{args.method}_{args.adaptation_rows}.json"
    )
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
