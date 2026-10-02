import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def check_metrics(metrics_path: Path, args: argparse.Namespace) -> dict:
    errors = []
    if not metrics_path.exists():
        return {
            "passed": False,
            "metrics_path": str(metrics_path),
            "errors": [f"missing metrics file: {metrics_path}"],
        }

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not metrics.get("passed"):
        errors.append("metrics.passed is not true")
    if not metrics.get("counts_as_real_data_baseline_subreproduction"):
        errors.append("run is not marked as a real-data baseline sub-reproduction")
    if args.require_materialized_samples:
        if not metrics.get("uses_materialized_samples"):
            errors.append("run does not use materialized bitmap samples")
        if metrics.get("num_materialized_samples", 0) < args.min_materialized_samples:
            errors.append(
                "num_materialized_samples below threshold: "
                f"{metrics.get('num_materialized_samples')} < {args.min_materialized_samples}"
            )
        if not metrics.get("counts_as_full_paper_reproduction"):
            errors.append("materialized-sample run is not marked as full paper reproduction")
    elif metrics.get("counts_as_full_paper_reproduction"):
        errors.append("run incorrectly claims full paper reproduction")
    if metrics.get("train_rows", 0) < args.min_train_rows:
        errors.append(f"train_rows below threshold: {metrics.get('train_rows')} < {args.min_train_rows}")
    if metrics.get("predict_rows", 0) < args.min_predict_rows:
        errors.append(
            f"predict_rows below threshold: {metrics.get('predict_rows')} < {args.min_predict_rows}"
        )
    if metrics.get("epochs", 0) < args.min_epochs:
        errors.append(f"epochs below threshold: {metrics.get('epochs')} < {args.min_epochs}")

    q = metrics.get("predict_qerror", {})
    for key in ["median", "p90", "p95", "p99", "max", "mean"]:
        value = q.get(key)
        if not isinstance(value, (int, float)) or value < 1:
            errors.append(f"invalid predict_qerror.{key}: {value}")

    provenance = metrics.get("provenance", {})
    if "workloads/stats-drift/" not in provenance.get("train_rel", ""):
        errors.append("train_rel does not point to official stats-drift workload")
    if "workloads/stats-drift/" not in provenance.get("predict_rel", ""):
        errors.append("predict_rel does not point to official stats-drift workload")
    if len(provenance.get("train_sha256", "")) != 64:
        errors.append("missing train sha256 provenance")
    if len(provenance.get("predict_sha256", "")) != 64:
        errors.append("missing predict sha256 provenance")

    artifact_errors = []
    artifact_shapes = {}
    for name, path_text in metrics.get("artifacts", {}).items():
        path = Path(path_text)
        if not path.exists():
            artifact_errors.append(f"missing artifact {name}: {path}")
            continue
        if name.endswith("qerror"):
            arr = np.load(path)
            artifact_shapes[name] = list(arr.shape)
            if arr.ndim != 1:
                artifact_errors.append(f"qerror artifact is not 1D: {name} shape={arr.shape}")
            if name == "predict_qerror" and arr.shape[0] != metrics.get("predict_rows"):
                artifact_errors.append(
                    f"predict_qerror length mismatch: {arr.shape[0]} != {metrics.get('predict_rows')}"
                )
            if np.any(~np.isfinite(arr)) or np.any(arr < 1):
                artifact_errors.append(f"invalid qerror values in {name}")
    errors.extend(artifact_errors)

    return {
        "passed": not errors,
        "metrics_path": str(metrics_path),
        "errors": errors,
        "summary": {
            "baseline": metrics.get("baseline"),
            "scope": metrics.get("scope"),
            "train_rows": metrics.get("train_rows"),
            "predict_rows": metrics.get("predict_rows"),
            "epochs": metrics.get("epochs"),
            "predict_qerror": metrics.get("predict_qerror"),
            "counts_as_full_paper_reproduction": metrics.get("counts_as_full_paper_reproduction"),
            "uses_materialized_samples": metrics.get("uses_materialized_samples"),
            "num_materialized_samples": metrics.get("num_materialized_samples"),
            "counts_as_real_data_baseline_subreproduction": metrics.get(
                "counts_as_real_data_baseline_subreproduction"
            ),
            "artifact_shapes": artifact_shapes,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Workload-Drift MSCN Windows reproduction evidence.")
    parser.add_argument(
        "--metrics-json",
        default="research_plan/workload_drift_mscn_windows/seed_1/metrics.json",
    )
    parser.add_argument("--report-json", default="research_plan/workload_drift_mscn_repro_check.json")
    parser.add_argument("--min-train-rows", type=int, default=2000)
    parser.add_argument("--min-predict-rows", type=int, default=2000)
    parser.add_argument("--min-epochs", type=int, default=5)
    parser.add_argument("--require-materialized-samples", action="store_true")
    parser.add_argument("--min-materialized-samples", type=int, default=1000)
    args = parser.parse_args()

    metrics_path = Path(args.metrics_json)
    if not metrics_path.is_absolute():
        metrics_path = ROOT / metrics_path
    report = check_metrics(metrics_path, args)
    report_path = Path(args.report_json)
    if not report_path.is_absolute():
        report_path = ROOT / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
