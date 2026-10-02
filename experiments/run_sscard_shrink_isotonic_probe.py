import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

import run_sscard_posthoc_calibration as cal


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research_plan"


DEFAULT_CASES = [
    "DBLP_AN:LBS",
    "DBLP_AN:SSCard",
    "DBLP_AN:LPLM",
    "DBLP_AN:Astrid",
    "WIKI:Astrid",
    "WIKI:DREAM",
    "WIKI:LPLM",
    "WIKI:SSCard",
    "WIKI:LBS",
    "IMDB_AN:LPLM",
    "IMDB_MT:LPLM",
]


def parse_list(text: str, cast):
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def split_case(text: str) -> tuple[str, str]:
    if ":" not in text:
        raise ValueError(f"case must be DATASET:BASELINE, got {text}")
    dataset, baseline = text.split(":", 1)
    return dataset, baseline


def evaluate(frame, adaptation_rows: int, shrink_factor: float, seed: int, args) -> dict:
    rng = np.random.default_rng(seed)
    indices = np.arange(frame.shape[0], dtype=int)
    rng.shuffle(indices)
    if args.max_rows and args.max_rows < indices.shape[0]:
        indices = indices[: args.max_rows]
    if adaptation_rows >= indices.shape[0]:
        raise ValueError("adaptation rows must be smaller than available rows")

    adapt_idx = indices[:adaptation_rows]
    eval_idx = indices[adaptation_rows:]
    estimates = np.maximum(1.0, frame["estimate"].to_numpy(dtype=np.float64))
    truth = np.maximum(1.0, frame["truth"].to_numpy(dtype=np.float64))
    log_estimate = np.log(estimates)
    log_truth = np.log(truth)

    baseline_q = cal.summarize(cal.qerror(estimates[eval_idx], truth[eval_idx]))
    global_pred, _ = cal.fit_predict("global", frame, adapt_idx, eval_idx, seed, args)
    global_q = cal.summarize(cal.qerror(global_pred, truth[eval_idx]))

    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(log_estimate[adapt_idx], log_truth[adapt_idx])
    calibrated_log_truth = model.predict(log_estimate[eval_idx])
    method_pred = np.exp(log_estimate[eval_idx] + shrink_factor * (calibrated_log_truth - log_estimate[eval_idx]))
    method_q = cal.summarize(cal.qerror(method_pred, truth[eval_idx]))

    reductions = {key: cal.reduction(baseline_q[key], method_q[key]) for key in ["median", "p90", "p95", "p99", "mean"]}
    reductions_vs_global = {
        key: cal.reduction(global_q[key], method_q[key]) for key in ["median", "p90", "p95", "p99", "mean"]
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
        "baseline_qerror": baseline_q,
        "global_calibration_qerror": global_q,
        "method_qerror": method_q,
        "relative_reductions": reductions,
        "relative_reductions_vs_global_calibration": reductions_vs_global,
    }


def summarize_splits(splits: list[dict], key: str) -> dict:
    return {
        metric: float(np.median([split[key][metric] for split in splits]))
        for metric in ["median", "p90", "p95", "p99", "mean"]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe shrink-isotonic SSCard operating points.")
    parser.add_argument("--cases", default=",".join(DEFAULT_CASES))
    parser.add_argument("--adaptation-rows", default="1,2,3,5,10,25,50,100")
    parser.add_argument("--shrink-factors", default="0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--min-eval-rows", type=int, default=10000)
    parser.add_argument("--min-p90-reduction", type=float, default=0.05)
    parser.add_argument("--min-p95-reduction", type=float, default=0.05)
    parser.add_argument("--min-p99-reduction", type=float, default=0.05)
    parser.add_argument("--min-vs-global-p95-reduction", type=float, default=0.01)
    parser.add_argument("--min-vs-global-p99-reduction", type=float, default=0.01)
    parser.add_argument("--max-median-regression", type=float, default=1.05)
    parser.add_argument("--tail-gate-quantile", type=float, default=0.80)
    parser.add_argument("--output-csv", default="research_plan/sscard_shrink_isotonic_probe_summary.csv")
    parser.add_argument("--output-json", default="research_plan/sscard_shrink_isotonic_probe_summary.json")
    args = parser.parse_args()

    cases = [split_case(item) for item in parse_list(args.cases, str)]
    adaptation_rows = parse_list(args.adaptation_rows, int)
    shrink_factors = parse_list(args.shrink_factors, float)
    seeds = parse_list(args.seeds, int)

    rows = []
    detail = []
    for dataset, baseline_method in cases:
        result_csv = cal.SSCARD_ROOT / "results" / dataset / f"{baseline_method}.csv"
        frame = cal.load_result_csv(result_csv)
        for labels in adaptation_rows:
            if labels >= frame.shape[0] or frame.shape[0] - labels < args.min_eval_rows:
                continue
            for shrink_factor in shrink_factors:
                splits = [evaluate(frame, labels, shrink_factor, seed, args) for seed in seeds]
                passed_seed_count = sum(1 for split in splits if split["passed"])
                failures = sorted(
                    {
                        gate
                        for split in splits
                        for gate, passed in split["pass_gates"].items()
                        if not passed
                    }
                )
                aggregate = summarize_splits(splits, "relative_reductions")
                aggregate_vs_global = summarize_splits(splits, "relative_reductions_vs_global_calibration")
                row = {
                    "dataset": dataset,
                    "baseline": f"SSCard/{baseline_method}",
                    "method": f"shrink-isotonic s{int(round(100.0 * shrink_factor))}",
                    "adaptation_rows": labels,
                    "shrink_factor": shrink_factor,
                    "row_count": int(frame.shape[0]),
                    "seed_count": len(seeds),
                    "passed_seed_count": passed_seed_count,
                    "passed": passed_seed_count == len(seeds),
                    "p90_reduction": aggregate["p90"],
                    "p95_reduction": aggregate["p95"],
                    "p99_reduction": aggregate["p99"],
                    "p95_vs_global": aggregate_vs_global["p95"],
                    "p99_vs_global": aggregate_vs_global["p99"],
                    "fail_reasons": "+".join(failures),
                }
                rows.append(row)
                detail.append({**row, "split_reports": splits})

    rows.sort(key=lambda row: (row["dataset"], row["baseline"], row["adaptation_rows"], row["shrink_factor"]))
    out_csv = Path(args.output_csv)
    out_json = Path(args.output_json)
    if not out_csv.is_absolute():
        out_csv = ROOT / out_csv
    if not out_json.is_absolute():
        out_json = ROOT / out_json
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    out_json.write_text(json.dumps({"rows": rows, "details": detail}, indent=2), encoding="utf-8")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
