import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research_plan"


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def compact_number(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def percent_param(value: float) -> str:
    scaled = 100.0 * float(value)
    if abs(scaled - round(scaled)) < 1e-9:
        return str(int(round(scaled)))
    return compact_number(scaled)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def drift_name(report: dict, path: Path) -> str:
    text = " ".join([path.name, report.get("target_workload", ""), report.get("baseline_metrics", "")])
    if "varying_column_queries" in text or "_column_" in text:
        return "column"
    if "varying_table_queries" in text or "_table_" in text:
        return "table"
    if "varying_value_queries" in text or "_value_" in text:
        return "value"
    return "unknown"


def display_method(report: dict) -> str:
    method = report.get("method", "")
    first = report.get("split_reports", [{}])[0]
    model = first.get("method_model", {})
    preset = model.get("feature_preset", "basic")
    suffix = " value" if preset == "value" else ""
    alpha_suffix = ""
    if "ridge_alpha" in model and abs(float(model["ridge_alpha"]) - 1.0) > 1e-12:
        alpha_text = compact_number(float(model["ridge_alpha"]))
        alpha_suffix = f" a{alpha_text}"
    if method == "shrink-ridge" and "shrink_factor" in model:
        anchor = model.get("shrink_anchor", "zero")
        return f"shrink-ridge {anchor} s{percent_param(model['shrink_factor'])}{alpha_suffix}{suffix}"
    if method == "shrink-global" and "shrink_factor" in model:
        return f"shrink-global s{percent_param(model['shrink_factor'])}"
    if method == "shrink-isotonic" and "shrink_factor" in model:
        return f"shrink-isotonic s{percent_param(model['shrink_factor'])}"
    if method == "ridge":
        return f"{method}{alpha_suffix}{suffix}"
    if method == "hist-gradient":
        hist_suffixes = []
        if model.get("hist_loss", "squared_error") == "absolute_error":
            hist_suffixes.append("abs")
        if int(model.get("hist_min_samples_leaf", 20)) != 20:
            hist_suffixes.append(f"leaf{int(model['hist_min_samples_leaf'])}")
        if int(model.get("hist_max_leaf_nodes", 15)) != 15:
            hist_suffixes.append(f"nodes{int(model['hist_max_leaf_nodes'])}")
        if int(model.get("hist_max_iter", 300)) != 300:
            hist_suffixes.append(f"iter{int(model['hist_max_iter'])}")
        if abs(float(model.get("hist_learning_rate", 0.05)) - 0.05) > 1e-12:
            lr_text = compact_number(float(model["hist_learning_rate"]))
            hist_suffixes.append(f"lr{lr_text}")
        if abs(float(model.get("hist_l2", 0.01)) - 0.01) > 1e-12:
            l2_text = compact_number(float(model["hist_l2"]))
            hist_suffixes.append(f"l2_{l2_text}")
        tuned = " " + " ".join(hist_suffixes) if hist_suffixes else ""
        return f"{method}{suffix}{tuned}"
    if method == "tail-risk-ridge" and "shrink_factor" in model:
        anchor = model.get("shrink_anchor", "zero")
        gate = model.get("tail_gate_quantile", 0.0)
        return f"tail-risk-ridge {anchor} s{percent_param(model['shrink_factor'])}{alpha_suffix} q{percent_param(gate)}{suffix}"
    return method


def baseline_family(report: dict) -> str:
    baseline = report.get("baseline", "")
    if "Robust-MSCN" in baseline:
        return "robust_mscn"
    if "MSCN" in baseline:
        return "mscn"
    return "unknown"


def read_report(path: Path) -> dict | None:
    report = read_json(path)
    if "aggregate_reductions" not in report:
        return None
    failures = sorted(
        {
            gate
            for split in report.get("split_reports", [])
            for gate, passed in split.get("pass_gates", {}).items()
            if not passed
        }
    )
    return {
        "family": baseline_family(report),
        "drift": drift_name(report, path),
        "baseline": report.get("baseline", ""),
        "method": display_method(report),
        "labels": report.get("requires_labeled_target_adaptation_rows", 0),
        "passed": report.get("passed", False),
        "passed_seed_count": report.get("passed_seed_count", 0),
        "seed_count": report.get("seed_count", 0),
        "median_reduction": report["aggregate_reductions"]["median"],
        "p90_reduction": report["aggregate_reductions"]["p90"],
        "p95_reduction": report["aggregate_reductions"]["p95"],
        "p99_reduction": report["aggregate_reductions"]["p99"],
        "p95_vs_global": report["aggregate_reductions_vs_global_calibration"]["p95"],
        "p99_vs_global": report["aggregate_reductions_vs_global_calibration"]["p99"],
        "fail_reasons": "+".join(failures),
        "report_path": str(path.relative_to(ROOT)),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def find_row(rows: list[dict], drift: str, method: str, labels: int) -> dict:
    for row in rows:
        if row["drift"] == drift and row["method"] == method and row["labels"] == labels:
            return row
    raise KeyError((drift, method, labels))


def table_row(row: dict) -> str:
    return (
        f"| {row['drift']} | {row['method']} | {row['labels']} | "
        f"{row['passed_seed_count']}/{row['seed_count']} | {pct(row['p90_reduction'])} | "
        f"{pct(row['p95_reduction'])} | {pct(row['p99_reduction'])} | "
        f"{pct(row['p95_vs_global'])} | {pct(row['p99_vs_global'])} |"
    )


def boundary_table_row(row: dict) -> str:
    return (
        f"| {row['drift']} | {row['method']} | {row['labels']} | "
        f"{row['passed_seed_count']}/{row['seed_count']} | {pct(row['p90_reduction'])} | "
        f"{pct(row['p95_reduction'])} | {pct(row['p99_reduction'])} | "
        f"{pct(row['p95_vs_global'])} | {pct(row['p99_vs_global'])} | {row['fail_reasons']} |"
    )


def reproduction_command(row: dict) -> str:
    report = read_json(ROOT / row["report_path"])
    command = report.get("command")
    if isinstance(command, list):
        parts = [str(part) for part in command]
        if parts and parts[0].endswith(".py"):
            parts = ["python", *parts]
        return " ".join(parts)
    if isinstance(command, str):
        return command
    return f"# Missing reproduction command for {row['report_path']}"


def append_reproduction_commands(lines: list[str], rows: list[dict]) -> None:
    lines.extend(["", "## Reproduction Commands", "", "```powershell"])
    seen = set()
    for row in rows:
        command = reproduction_command(row)
        if command in seen:
            continue
        seen.add(command)
        lines.append(command)
    lines.append("python experiments\\summarize_workload_drift_posthoc_calibration.py")
    lines.append("```")


def row_by_path(rows: list[dict], relative_path: str) -> dict:
    normalized = relative_path.replace("/", "\\")
    for row in rows:
        if row["report_path"] == normalized:
            return row
    raise KeyError(relative_path)


def is_posthoc_baseline_report(path: Path) -> bool:
    normalized = str(path).lower().replace("\\", "/")
    excluded_fragments = (
        "residual_field",
        "residual_transport",
        "tail_residual",
        "drift_signature",
        "residual-field",
        "_smoke",
        "/windows",
    )
    if any(fragment in normalized for fragment in excluded_fragments):
        return False
    report = read_json(path)
    command = report.get("command", "")
    if isinstance(command, list):
        command_text = " ".join(str(part) for part in command)
    else:
        command_text = str(command)
    return "run_workload_drift_posthoc_calibration.py" in command_text.replace("\\", "/")


def write_robust_note(rows: list[dict]) -> None:
    csv_path = RESEARCH / "workload_drift_robust_mscn_posthoc_summary.csv"
    write_csv(csv_path, rows)

    selected_passes = [
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_column_shrink_ridge_global_s425m_a0p02_145.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_column_shrink_ridge_global_s425m_a0p02_146.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_column_shrink_ridge_zero_s50_150.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_column_shrink_ridge_zero_s52p5_161_fine.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_column_shrink_ridge_zero_s55_164_fine.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_column_shrink_ridge_zero_s55_167_fine.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_column_shrink_ridge_zero_s55_170_gap.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_column_shrink_ridge_zero_s57p5_174_fine.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_column_shrink_ridge_zero_s60_179_fine.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_column_shrink_ridge_zero_s60_180_gap.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_column_shrink_ridge_zero_s62p5_184_fine.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_column_shrink_ridge_zero_s62p5_187_fine.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_column_shrink_ridge_zero_s65_190_gap.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_column_shrink_ridge_zero_s65_200.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_column_shrink_ridge_zero_s50_225.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_column_shrink_ridge_zero_s65_225.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_column_shrink_ridge_zero_s75_500.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_column_shrink_ridge_zero_s99_1000.json"),
        find_row(rows, "column", "ridge", 1500),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_table_hist_gradient_abs_value_39_leaf10_lr0p045_l2_0p01_probe.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_table_hist_gradient_abs_value_40_leaf9_lr0p045_l2_0p01_probe.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_table_hist_gradient_abs_value_41_leaf8_lr0p05_l2_0p01_probe.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_table_hist_gradient_abs_value_44_leaf9_lr0p045_l2_0p01_guard.json"),
        find_row(rows, "table", "shrink-ridge zero s95 a50 value", 49),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_table_shrink_ridge_value_zero_s1p035_a60_49.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_table_shrink_ridge_value_zero_s1p039_a60p05_49.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_table_shrink_ridge_value_zero_s1p04_a60p25_49.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_table_shrink_ridge_value_zero_s100_a30_52.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_table_shrink_ridge_value_zero_s100_a30_54.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_table_shrink_ridge_value_zero_s100_a30_55.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_table_shrink_ridge_value_zero_s80_a30_67.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_table_shrink_ridge_value_zero_s65_a10_71.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_table_shrink_ridge_value_zero_s70_a10_74.json"),
        find_row(rows, "table", "shrink-ridge zero s55", 77),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_table_shrink_ridge_zero_s65_90.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_table_shrink_ridge_zero_s75_91.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_table_shrink_ridge_zero_s85_95.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_table_shrink_ridge_zero_s85_98.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_table_shrink_ridge_zero_s80_102_gap.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_table_shrink_ridge_zero_s75_110.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_table_shrink_ridge_zero_s90_110_gap.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_table_shrink_ridge_zero_s90_112_gap.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_table_shrink_ridge_zero_s85_114.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_table_shrink_ridge_zero_s85_115.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_table_shrink_ridge_zero_s85_118.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_table_shrink_ridge_zero_s75_120.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_table_shrink_ridge_zero_s90_120.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_table_shrink_ridge_zero_s93_120_gap.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_table_shrink_ridge_zero_s93_123.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_table_shrink_ridge_zero_s95_123.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_table_shrink_ridge_zero_s97_126.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_table_shrink_ridge_zero_s99p5_126.json"),
        find_row(rows, "table", "ridge", 127),
        find_row(rows, "table", "ridge", 128),
        find_row(rows, "table", "ridge", 130),
        find_row(rows, "table", "ridge", 200),
        find_row(rows, "table", "ridge", 250),
        find_row(rows, "table", "ridge", 500),
        find_row(rows, "table", "ridge", 2000),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_value_shrink_ridge_value_zero_s630m_a3p5_52.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_value_shrink_ridge_value_zero_s640m_a3p5_52.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_value_shrink_ridge_value_zero_s640m_a4_52.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_value_shrink_ridge_value_zero_s625m_a2_53.json"),
        find_row(rows, "value", "shrink-ridge zero s57 value", 54),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_value_shrink_ridge_value_zero_s58_54.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_value_shrink_ridge_value_zero_s60_59.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_value_shrink_ridge_value_zero_s60_60.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_value_shrink_ridge_value_zero_s70_65.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_value_shrink_ridge_value_zero_s80_68.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_value_shrink_ridge_value_zero_s80_69.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_value_shrink_ridge_value_zero_s80_70.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_value_shrink_ridge_value_zero_s80_76.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_value_shrink_ridge_value_zero_s65_80.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_value_shrink_ridge_value_zero_s80_85.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_value_shrink_ridge_value_zero_s75_90.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_value_shrink_ridge_value_zero_s70_95.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_value_shrink_ridge_value_zero_s68_96.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_value_shrink_ridge_value_zero_s70_100.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_value_shrink_ridge_value_zero_s68_107.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_value_shrink_ridge_value_zero_s70_108.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_value_shrink_ridge_value_zero_s68_109.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_value_shrink_ridge_value_zero_s68_110.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_value_shrink_ridge_value_zero_s70_111.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_value_shrink_ridge_value_zero_s68_a0p35_120.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_value_shrink_ridge_value_zero_s75p5_a1_140.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_value_shrink_ridge_value_zero_s73p25_a1p3_200.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_value_shrink_ridge_value_zero_s73p5_a0p75_250.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_value_shrink_ridge_value_zero_s80_a1p25_250.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_value_hist_gradient_value_122_leaf11_l2_0p3_lr0p06.json"),
        find_row(rows, "value", "hist-gradient value leaf11", 124),
        find_row(rows, "value", "hist-gradient value leaf10", 136),
        find_row(rows, "value", "hist-gradient value leaf7", 137),
        find_row(rows, "value", "hist-gradient value leaf10", 174),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_value_hist_gradient_value_300.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_value_hist_gradient_value_400.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_value_hist_gradient_value_450.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_value_hist_gradient_value_500.json"),
    ]
    selected_boundaries = [
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_column_shrink_ridge_global_s425m_a0p02_144.json"),
        find_row(rows, "column", "shrink-ridge zero s45", 148),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_column_shrink_ridge_zero_s55_160_gap.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_column_shrink_ridge_zero_s60_160_gap.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_column_shrink_ridge_zero_s55_161_fine.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_column_shrink_ridge_zero_s57p5_169_fine.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_column_shrink_ridge_zero_s60_177_fine.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_column_shrink_ridge_zero_s65_187_fine.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_column_shrink_ridge_zero_s65_170_gap.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_column_shrink_ridge_zero_s65_180_gap.json"),
        find_row(rows, "column", "ridge", 1000),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_column_shrink_ridge_zero_s99p5_1000.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_table_hist_gradient_abs_value_38_leaf10_lr0p04_l2_0p01_probe.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_table_hist_gradient_abs_value_38_leaf10_lr0p045_l2_0p01_probe.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_table_hist_gradient_abs_value_38_leaf10_lr0p05_l2_0p01_probe.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_table_hist_gradient_abs_value_38_leaf11_lr0p045_l2_0p01_probe.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_table_hist_gradient_abs_value_39_leaf9_lr0p045_l2_0p01_probe.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_table_hist_gradient_abs_value_39_leaf10_lr0p05_l2_0p01_probe.json"),
        find_row(rows, "table", "shrink-ridge zero s95 a50 value", 48),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_table_shrink_ridge_value_zero_s1p039_a60_49.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_table_shrink_ridge_value_zero_s1p0388_a60_49.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_table_shrink_ridge_value_zero_s1p04_a60_49.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_table_shrink_ridge_value_zero_s1p0405_a60p05_49.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_table_shrink_ridge_value_zero_s1p05_a60p5_49.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_table_shrink_ridge_value_zero_s105_a30_51.json"),
        find_row(rows, "table", "shrink-ridge zero s55", 75),
        find_row(rows, "table", "shrink-ridge zero s55", 76),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_table_shrink_ridge_zero_s75_90.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_table_shrink_ridge_zero_s85_90.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_table_shrink_ridge_zero_s90_116_gap.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_table_shrink_ridge_zero_s93_117_gap.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_table_shrink_ridge_zero_s95_117_gap.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_table_shrink_ridge_zero_s97_123.json"),
        find_row(rows, "table", "ridge", 120),
        find_row(rows, "table", "ridge", 125),
        find_row(rows, "table", "ridge", 126),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_value_shrink_ridge_value_zero_s650m_a3p75_51.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_value_shrink_ridge_value_zero_s64_a4_51_boundary.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_value_shrink_ridge_value_zero_s65_a4p5_51_boundary.json"),
        find_row(rows, "value", "shrink-ridge zero s57 value", 53),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_value_shrink_ridge_value_zero_s60_a3_53.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_value_shrink_ridge_value_zero_s65_a3_53.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_value_shrink_ridge_value_zero_s60_58.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_value_shrink_ridge_value_zero_s80_67.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_value_shrink_ridge_value_zero_s85_71.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_value_shrink_ridge_value_zero_s85_73.json"),
        find_row(rows, "value", "shrink-ridge zero s57 value", 50),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_value_hist_gradient_value_121_leaf10_nodes15_lr0p045_l2_0p19_guard.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_value_hist_gradient_value_121_leaf10_nodes15_lr0p055_l2_0p17_guard.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_value_hist_gradient_value_121_leaf10_nodes9_l2_0p19_lr0p08.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_value_hist_gradient_value_122_leaf11_l2_0p1.json"),
        find_row(rows, "value", "hist-gradient value leaf11", 123),
        find_row(rows, "value", "hist-gradient value leaf8", 138),
        find_row(rows, "value", "hist-gradient value", 150),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_value_shrink_ridge_zero_s75_a0p1_500.json"),
        row_by_path(rows, r"research_plan\workload_drift_robust_mscn_value_shrink_ridge_global_s75_a1_500.json"),
    ]

    md_path = RESEARCH / "workload_drift_robust_mscn_posthoc_calibration.md"
    lines = [
        "# Robust-MSCN Post-hoc Calibration Expansion",
        "",
        "This note records simple post-hoc calibration baselines on reproduced Robust-MSCN prediction/label artifacts. It is isolated from the Drift-Signature Residual Field Transport implementation and paper artifacts.",
        "",
        "## Strict 3-Seed Passes",
        "",
        "| drift | method | labels | gate | p90 reduction | p95 reduction | p99 reduction | p95 vs global | p99 vs global |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected_passes:
        lines.append(table_row(row))
    lines.extend(
        [
            "",
            "## Boundary And Negative Results",
            "",
            "| drift | method | labels | gate | p90 reduction | p95 reduction | p99 reduction | p95 vs global | p99 vs global | fail reasons |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in selected_boundaries:
        lines.append(boundary_table_row(row))
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `column / shrink-ridge global s42p5 a0p02` gives a new low-label Robust-MSCN post-hoc calibration point: 145 labeled target queries, 3/3 strict-gate pass, and 50.5%/48.3%/83.5% p90/p95/p99 reductions, lowering the previous 149-label zero-s45 point. At 146 labels the same global-anchor setting is a p99-oriented 3/3 tradeoff at 50.6%/48.3%/83.8%. At 144 labels it is only 2/3 because seed 3 misses the p99-vs-global guard; nearby zero-anchor 148-label probes remain 2/3 boundaries.",
            "- Mined pass-table points add stronger Robust-MSCN column operating points without changing the low-budget selection: 150-label zero-s50 reaches 56.5%/51.7%/82.9% as a p95-oriented early tradeoff. A focused 160-190 label gap sweep first added 170/180/190-label passes, and a finer follow-up tightened that front: zero-s52.5 at 161 labels reaches 59.9%/56.2%/86.6%, zero-s55 at 164 labels raises p95 to 58.5% with 86.8% p99, zero-s55 at 167 labels gives a p99-oriented 59.1%/56.4%/87.2%, zero-s57.5 at 174 labels reaches 60.6%/56.5%/87.5%, zero-s60 at 179 labels gives 60.9%/58.3%/87.0%, and zero-s62.5 at 184/187 labels reaches 60.8%/58.6%/87.3% and 60.0%/58.6%/87.4%. These points improve the p99-vs-budget curve before the older 200/225-label points; more aggressive same-budget variants such as s55@161, s57.5@169, s60@177, and s65@187 remain 2/3 because the median guard is still too tight. Existing 200-label zero-s65 remains a p90-oriented point at 62.7%/56.2%/83.4%, 225-label zero-s50 reaches 60.0%/52.8%/86.0%, and same-budget zero-s65 gives a p90/p95 tradeoff at 65.2%/57.0%/84.5%. Higher-budget points then include 500-label zero-s75 at 66.7%/68.3%/89.9% and a newly mined 1,000-label zero-s99 point at 66.2%/76.4%/93.8%, with all seed-level p99 guards above 92.9%.",
            "- The 1,000-label zero-s99 point is now the selected Robust-MSCN column high-tail simple calibration point; ridge/s100 at the same budget and zero-s99.5 are 2/3 because seed 1 narrowly violates the median guard. The 1,500-label ridge point remains a nearby high-budget p95 tradeoff at 66.4%/77.2%/93.2%.",
            "- `table / shrink-ridge zero s55` now lowers the simple post-hoc calibration budget to 77 labels, with 88.7%/92.5%/70.8% reductions. At 75 and 76 labels it is 2/3 because seed 3 misses p99 and the median guard.",
            "- `table / hist-gradient value abs leaf10 lr0p045` now lowers the Robust-MSCN value-aware table-drift low-budget point to 39 labels, with 79.1%/76.9%/68.8% p90/p95/p99 reductions and 91.0%/78.0% p95/p99 gains over global calibration. The 40-label abs leaf9/lr0.045 point is a p90/p95-oriented tradeoff at 84.9%/86.9%/65.0%, 41-label abs leaf8/lr0.05 gives the strongest low-budget balanced tradeoff at 87.7%/92.5%/72.5%, and 44-label abs leaf9/lr0.045 gives the stronger p99 tradeoff at 82.3%/84.7%/76.2%. Focused 38-label leaf10/lr guards remain 2/3 and leaf11 is 1/3, all collapsing on seed-3 p99, so 39 labels is the current honest strict boundary. These absolute-error hist-gradient points still have occasional seed-level mean outliers, so they are quantile-tail baselines rather than mean-risk fixes.",
            "- `table / shrink-ridge zero s95 a50 value` remains the stronger 49-label Robust-MSCN table-drift p99 representative, with 84.1%/86.8%/76.1% reductions. Same-budget stronger-tail settings are also 3/3: shrink factor 1.035 with alpha 60 improves the 49-label tail point to 83.7%/87.1%/78.4%, and the tighter shrink factor 1.039 with alpha 60.05 improves it again to 83.8%/87.1%/78.6% p90/p95/p99 reductions; s1.04/a60.25 is a nearby pass with nearly identical tail. At 48 labels, nearby s95/a50 probes remain 2/3. A narrow s1.0385-s1.0405/a60.05-a60.5 sweep found no stronger passing point: the slightly higher-p99 candidates such as s1.0405/a60.05 reach 78.6% p99 but fail only because seed 2 narrowly violates the median guard, so the existing s1.039/a60.05 remains the selected 49-label 3/3 shrink-ridge representative.",
            "- Additional mined table-drift passes strengthen the Robust-MSCN tail curve above the 49-label budget. Value-aware zero-s100/a30 gives 52/54/55-label points at 89.1%/90.9%/80.2%, 90.2%/92.6%/82.5%, and 89.5%/92.6%/83.1%; seed 3 is the weakest tail split for 54/55 labels but still clears the p99 gate at 61.6%-61.8%. Value-aware p95 tradeoffs then add 67-label zero-s80/a30 at 89.5%/92.8%/77.9%, 71-label zero-s65/a10 at 89.7%/93.8%/77.2%, and 74-label zero-s70/a10 at 90.5%/94.5%/81.6%, all strict 3/3. A nearby zero-s105/a30 51-label probe is 2/3 because seed 2 still violates the median guard, so it does not lower that stronger-tail curve. Basic shrink-ridge then gives 90-label zero-s65 at 90.1%/94.3%/84.4%, and zero-s75 passes from 91 labels with 91.0%/95.4%/85.0% p90/p95/p99 reductions; the 90-label zero-s75 probe is 2/3 because seed 3 misses p99. Higher-budget zero-s75 references at 110 and 120 labels reach 92.7%/95.4%/85.0% and 92.5%/95.3%/86.4%. The 91-label point is a quantile-tail operating point: it clears median and tail gates, but one split still has an extreme mean outlier.",
            "- Newly mined ridge-like table points improve the Robust-MSCN stronger-tail simple curve without changing the 90/91-label low-budget selection: zero-s85 gives p95-front tradeoffs at 95 and 98 labels with 91.1%/95.6%/81.9% and 92.0%/96.1%/83.0%, and a new zero-s80 102-label point gives an early p99 tradeoff at 91.9%/95.7%/83.6%. A focused 101-120 label gap sweep then adds cleaner mid-budget simple points: zero-s90 at 110 labels reaches 92.7%/96.3%/86.1%, zero-s90 at 112 labels reaches 92.4%/96.5%/88.4%, and zero-s93 at 120 labels reaches 92.8%/96.5%/88.4%. These points move the previous 118-label zero-s85 tail strength to 112 labels while retaining a 3/3 gate. The same sweep found 2/3 median boundaries at zero-s90/116, zero-s93/117, and zero-s95/117. Older 114/115-label zero-s85 passes remain useful p90-oriented intermediates at 92.7%/95.9%/86.9% and 93.0%/96.0%/87.1%; zero-s85 at 118 labels still gives a slightly higher p90/p99 tradeoff at 93.2%/95.8%/88.6%. A previously buried 123-label zero-s93 point reaches 93.3%/96.5%/88.7%, becoming the selected stronger-tail simple point; zero-s95 at the same budget is a near-identical p95 tradeoff at 93.1%/96.6%/88.7%. At 126 labels, zero-s97 remains a balanced pass at 93.2%/96.7%/88.1%, while zero-s99.5 gives the p95-oriented tradeoff at 93.0%/96.9%/88.0%. The 90-label zero-s85 probe is 2/3 because seed 3 misses p99 and the median guard, and zero-s97 at 123/124/125 labels remains 2/3 on the seed-3 median guard. Higher-budget ridge points add a clean p99 frontier: 200 labels reaches 95.1%/97.9%/91.9%, 250 labels reaches 95.5%/98.1%/93.9%, and 500 labels reaches 96.4%/98.6%/96.9%, so the 500-label ridge point now supersedes the 2,000-label ridge point on p99 while using one quarter of the labels.",
            "- Basic value-drift shrink-ridge remains a useful negative boundary: at 500 labels it improves p95/p99 by up to 44.9%/53.5% or 46.4%/50.9%, but stays 0/3 because p99 reduction and vs-global gains are below the strict gate.",
            "- Value-aware post-hoc features turn Robust-MSCN value drift into passing baseline-improvement points. `value / shrink-ridge zero s63 a3p5 value` first lowered the budget to 52 labels with 52.8%/54.8%/60.9% p90/p95/p99 reductions, improving the previous 53-label s62p5/a2 and 54-label s57 points. At the same 52-label budget, `s64/a4` gives a stronger p90/p99 operating point at 53.1%/54.3%/61.5%, while `s64/a3.5` gives the stronger p95/p99 tradeoff at 52.5%/55.1%/61.3%. Focused 51-label shrink-ridge guards remain below strict 3/3: s64/a4 and s65/a4.5 are 2/3, both failing seed 1 on p99 and p99-vs-global despite aggregate p99 at 61.2%-61.9%; s63/a3.5 and s62/a4 additionally miss seed 3's p99-vs-global guard. The older 53-label s57 setting is only 1/3, while nearby s60/a3 and s65/a3 probes are useful 2/3 boundaries: s60/a3 misses only seed 3's p99-vs-global guard, and s65/a3 trades that for a seed 1 p99/p99-vs-global miss. A stronger-tail shrink-ridge curve now gives strict 3/3 points at 76 labels (`s80`, 59.4%/66.9%/73.2%), 80 labels (`s65`, 58.2%/62.4%/72.1%), 85 labels (`s80`, 60.4%/66.0%/74.5%), 90 labels (`s75`, 59.6%/65.5%/74.7%), 95 labels (`s70`, 61.1%/66.4%/75.1%), 96 labels (`s68`, 61.6%/66.3%/75.3%), 100 labels (`s70`, 61.3%/66.4%/73.5%), 108 labels (`s70`, 61.5%/67.4%/74.9%), 109 labels (`s68`, 62.7%/66.8%/75.7%), 110 labels (`s68`, 62.1%/65.6%/76.1%), 111 labels (`s70`, 62.8%/66.5%/76.7%), 120 labels (`s68/a0.35`, 62.4%/66.2%/77.7%), 140 labels (`s75.5/a1`, 63.5%/68.5%/77.9%), and 200 labels (`s73.25/a1.30`, 64.0%/67.9%/78.7%). The 120-label `s68/a0.35` point improves same-budget p99 over the previous `s70/a1` representative (76.9%) while giving up some p95; nearby 116/118/119-label `s68/a0.35` probes pass but have weaker 76.1%-76.6% p99, and 121 labels is a close pass at 77.6% p99. The 140-label `s75.5/a1` refinement improves the old `s75/a1` point from 63.3%/68.3%/77.8% to 63.5%/68.5%/77.9%, with all seed-level p99 guards above 74%. The 200-label `s73.25/a1.30` micro-refinement improves the old selected `s73.5/a1.25` point from 64.0%/68.1%/78.6% to 64.0%/67.9%/78.7%, trading a small p95 loss for a same-budget p99 gain. A 250-label sweep did not beat that 200-label p99, but it adds two useful 3/3 tradeoff points: `s73.5/a0.75` is the same-budget p99-best 250-label setting at 66.2%/70.5%/74.9%, while `s80/a1.25` is the p90/p95-best 250-label setting at 68.5%/72.1%/74.4%. The hist-gradient value track lowers its passing budget to 122 labels with leaf11/lr0p06/l2_0p3 and 56.8%/59.3%/64.4% reductions. Previously buried but strict 3/3 stronger-tail hist-gradient operating points are now selected as separate evidence: 124-label leaf11 reaches 57.7%/60.7%/68.8%, 136-label leaf10 reaches 58.4%/62.1%/70.5%, 137-label leaf7 reaches 60.3%/62.5%/71.8%, 300-label default reaches 73.2%/76.1%/76.7%, 400-label default reaches 75.8%/77.9%/80.0%, 450-label default gives a p95-oriented 76.4%/78.8%/78.3% tradeoff, and 500-label default raises p95 to 79.7% while keeping p99 at 77.9%. A focused 121-label leaf10 guard sweep did not lower the 122-label hist-gradient budget: the best guard-oriented point `nodes15/lr0.045/l2_0p19` is still 2/3, with seed 2 at only 59.3% p99 and 38.7% p99-vs-global, while a higher aggregate-p99 `nodes15/lr0.055/l2_0p17` point reaches 64.6% aggregate p99 but seed 2 falls to 55.2%. The older 121-label leaf10/nodes9/lr0p08/l2_0p19 probe also has strong aggregate tails but stays 2/3 because seed 2 misses the p99 guards, and 138-label leaf8 has even stronger aggregate tails but also fails seed 2's p99 guards.",
            "- Mined Robust-MSCN value-drift passes add intermediate and high-budget operating points: 54-label zero-s58 slightly improves the 54-label zero-s57 p99 point to 63.4%, zero-s60 now passes from 59 labels at 64.4% p99 and reaches 66.4% p99 at 60 labels, and a newly surfaced 65-label zero-s70 point reaches 56.0%/61.1%/69.1%, improving the previously selected 68-label zero-s80 p99 point at a lower budget. Zero-s80 then passes from 68 labels at 68.4% p99, and 69/70 labels strengthen that curve to 69.8%/71.0% p99. A 12-point 71-73 label sweep over shrink factors 0.70-0.85 found no new selected front: zero-s80 passes at 71/72/73 labels with 70.8%/70.7%/70.2% p99, all dominated by the 70-label zero-s80 p99 point, while zero-s85 at 71 and 73 labels is 2/3 because seed 1 misses p99-vs-global. A 20-point 80-95 label sweep over shrink factors 0.65-0.85 passed 20/20; selected representatives first extended the mid-budget p99 curve to 72.1%/74.5%/74.7%/75.1% at 80/85/90/95 labels. A follow-up 60-point sweep over 76-79 and 96-106 labels also passed 60/60; selected p99-front representatives add 76-label zero-s80 at 59.4%/66.9%/73.2% and 96-label zero-s68 at 61.6%/66.3%/75.3%, while 104-label zero-s75 is a p95-oriented tradeoff at 62.4%/68.3%/74.6%. The 67-label zero-s80 probe is only 2/3 because seed 1 misses the p99-vs-global guard, and the 58-label zero-s60 probe remains 2/3 because seed 1 misses p99 and p99-vs-global. The high-budget default hist-gradient curve adds 300/400/450/500-label strict 3/3 points at p95 76.1%/77.9%/78.8%/79.7% and p99 76.7%/80.0%/78.3%/77.9%; 400 labels remains p99-best, while 450/500 labels are useful p95 tradeoffs. These do not lower the 52-label budget, but they give a cleaner tail curve for baseline comparison.",
            "- These calibrators are useful baseline-expansion evidence. They do not replace or modify the existing residual-field track.",
            "",
            "## Evidence Files",
            "",
            f"- Summary CSV: `{csv_path.relative_to(ROOT)}`",
            "- Runner: `experiments\\run_workload_drift_posthoc_calibration.py`",
        ]
    )
    for row in selected_passes + selected_boundaries:
        lines.append(f"- `{row['report_path']}`")
    lines.extend(
        [
            "",
            "## Reproduction Commands",
            "",
            "```powershell",
            "python experiments\\run_workload_drift_posthoc_calibration.py --baseline-metrics research_plan\\workload_drift_robust_mscn_windows_bitmap\\seed_1\\metrics.json --method shrink-ridge --adaptation-rows 145 --seeds 1,2,3 --min-eval-rows 5000 --shrink-factor 0.425 --shrink-anchor global --ridge-alpha 0.02 --output-json research_plan\\workload_drift_robust_mscn_column_shrink_ridge_global_s425m_a0p02_145.json",
            "python experiments\\run_workload_drift_posthoc_calibration.py --baseline-metrics research_plan\\workload_drift_robust_mscn_windows_bitmap\\seed_1\\metrics.json --method ridge --adaptation-rows 1500 --seeds 1,2,3 --min-eval-rows 5000 --output-json research_plan\\workload_drift_robust_mscn_column_ridge_1500.json",
            "python experiments\\run_workload_drift_posthoc_calibration.py --baseline-metrics research_plan\\workload_drift_robust_mscn_windows_table_bitmap\\seed_1\\metrics.json --method shrink-ridge --adaptation-rows 77 --seeds 1,2,3 --min-eval-rows 5000 --shrink-factor 0.55 --shrink-anchor zero --output-json research_plan\\workload_drift_robust_mscn_table_shrink_ridge_zero_s55_77.json",
            "python experiments\\run_workload_drift_posthoc_calibration.py --baseline-metrics research_plan\\workload_drift_robust_mscn_windows_table_bitmap\\seed_1\\metrics.json --method shrink-ridge --feature-preset value --adaptation-rows 49 --seeds 1,2,3 --min-eval-rows 5000 --shrink-factor 0.95 --shrink-anchor zero --ridge-alpha 50 --output-json research_plan\\workload_drift_robust_mscn_table_shrink_ridge_value_zero_s95_a50_49.json",
            "python experiments\\run_workload_drift_posthoc_calibration.py --baseline-metrics research_plan\\workload_drift_robust_mscn_windows_table_bitmap\\seed_1\\metrics.json --method shrink-ridge --feature-preset value --adaptation-rows 49 --seeds 1,2,3 --min-eval-rows 5000 --shrink-factor 1.035 --shrink-anchor zero --ridge-alpha 60 --output-json research_plan\\workload_drift_robust_mscn_table_shrink_ridge_value_zero_s1p035_a60_49.json",
            "python experiments\\run_workload_drift_posthoc_calibration.py --baseline-metrics research_plan\\workload_drift_robust_mscn_windows_table_bitmap\\seed_1\\metrics.json --method shrink-ridge --feature-preset value --adaptation-rows 49 --seeds 1,2,3 --min-eval-rows 5000 --shrink-factor 1.039 --shrink-anchor zero --ridge-alpha 60.05 --output-json research_plan\\workload_drift_robust_mscn_table_shrink_ridge_value_zero_s1p039_a60p05_49.json",
            "python experiments\\run_workload_drift_posthoc_calibration.py --baseline-metrics research_plan\\workload_drift_robust_mscn_windows_table_bitmap\\seed_1\\metrics.json --method shrink-ridge --feature-preset value --adaptation-rows 49 --seeds 1,2,3 --min-eval-rows 5000 --shrink-factor 1.04 --shrink-anchor zero --ridge-alpha 60.25 --output-json research_plan\\workload_drift_robust_mscn_table_shrink_ridge_value_zero_s1p04_a60p25_49.json",
            "python experiments\\run_workload_drift_posthoc_calibration.py --baseline-metrics research_plan\\workload_drift_robust_mscn_windows_table_bitmap\\seed_1\\metrics.json --method ridge --adaptation-rows 127 --seeds 1,2,3 --min-eval-rows 5000 --output-json research_plan\\workload_drift_robust_mscn_table_ridge_127.json",
            "python experiments\\run_workload_drift_posthoc_calibration.py --baseline-metrics research_plan\\workload_drift_robust_mscn_windows_table_bitmap\\seed_1\\metrics.json --method ridge --adaptation-rows 130 --seeds 1,2,3 --min-eval-rows 5000 --output-json research_plan\\workload_drift_robust_mscn_table_ridge_130.json",
            "python experiments\\run_workload_drift_posthoc_calibration.py --baseline-metrics research_plan\\workload_drift_robust_mscn_windows_value_bitmap\\seed_1\\metrics.json --method shrink-ridge --feature-preset value --adaptation-rows 52 --seeds 1,2,3 --min-eval-rows 1000 --shrink-factor 0.63 --shrink-anchor zero --ridge-alpha 3.5 --output-json research_plan\\workload_drift_robust_mscn_value_shrink_ridge_value_zero_s630m_a3p5_52.json",
            "python experiments\\run_workload_drift_posthoc_calibration.py --baseline-metrics research_plan\\workload_drift_robust_mscn_windows_value_bitmap\\seed_1\\metrics.json --method shrink-ridge --feature-preset value --adaptation-rows 52 --seeds 1,2,3 --min-eval-rows 1000 --shrink-factor 0.64 --shrink-anchor zero --ridge-alpha 3.5 --output-json research_plan\\workload_drift_robust_mscn_value_shrink_ridge_value_zero_s640m_a3p5_52.json",
            "python experiments\\run_workload_drift_posthoc_calibration.py --baseline-metrics research_plan\\workload_drift_robust_mscn_windows_value_bitmap\\seed_1\\metrics.json --method shrink-ridge --feature-preset value --adaptation-rows 52 --seeds 1,2,3 --min-eval-rows 1000 --shrink-factor 0.64 --shrink-anchor zero --ridge-alpha 4 --output-json research_plan\\workload_drift_robust_mscn_value_shrink_ridge_value_zero_s640m_a4_52.json",
            "python experiments\\run_workload_drift_posthoc_calibration.py --baseline-metrics research_plan\\workload_drift_robust_mscn_windows_value_bitmap\\seed_1\\metrics.json --method shrink-ridge --feature-preset value --adaptation-rows 53 --seeds 1,2,3 --min-eval-rows 1000 --shrink-factor 0.625 --shrink-anchor zero --ridge-alpha 2 --output-json research_plan\\workload_drift_robust_mscn_value_shrink_ridge_value_zero_s625m_a2_53.json",
            "python experiments\\run_workload_drift_posthoc_calibration.py --baseline-metrics research_plan\\workload_drift_robust_mscn_windows_value_bitmap\\seed_1\\metrics.json --method shrink-ridge --feature-preset value --adaptation-rows 54 --seeds 1,2,3 --min-eval-rows 1000 --shrink-factor 0.57 --shrink-anchor zero --output-json research_plan\\workload_drift_robust_mscn_value_shrink_ridge_value_zero_s57_54.json",
            "python experiments\\run_workload_drift_posthoc_calibration.py --baseline-metrics research_plan\\workload_drift_robust_mscn_windows_value_bitmap\\seed_1\\metrics.json --method shrink-ridge --feature-preset value --adaptation-rows 111 --seeds 1,2,3 --min-eval-rows 1000 --shrink-factor 0.70 --shrink-anchor zero --output-json research_plan\\workload_drift_robust_mscn_value_shrink_ridge_value_zero_s70_111.json",
            "python experiments\\run_workload_drift_posthoc_calibration.py --baseline-metrics research_plan\\workload_drift_robust_mscn_windows_value_bitmap\\seed_1\\metrics.json --method shrink-ridge --feature-preset value --adaptation-rows 120 --seeds 1,2,3 --min-eval-rows 1000 --shrink-factor 0.68 --shrink-anchor zero --ridge-alpha 0.35 --output-json research_plan\\workload_drift_robust_mscn_value_shrink_ridge_value_zero_s68_a0p35_120.json",
            "python experiments\\run_workload_drift_posthoc_calibration.py --baseline-metrics research_plan\\workload_drift_robust_mscn_windows_value_bitmap\\seed_1\\metrics.json --method shrink-ridge --feature-preset value --adaptation-rows 140 --seeds 1,2,3 --min-eval-rows 1000 --shrink-factor 0.755 --shrink-anchor zero --ridge-alpha 1.0 --output-json research_plan\\workload_drift_robust_mscn_value_shrink_ridge_value_zero_s75p5_a1_140.json",
            "python experiments\\run_workload_drift_posthoc_calibration.py --baseline-metrics research_plan\\workload_drift_robust_mscn_windows_value_bitmap\\seed_1\\metrics.json --method shrink-ridge --feature-preset value --adaptation-rows 200 --seeds 1,2,3 --min-eval-rows 1000 --shrink-factor 0.7325 --shrink-anchor zero --ridge-alpha 1.30 --output-json research_plan\\workload_drift_robust_mscn_value_shrink_ridge_value_zero_s73p25_a1p3_200.json",
            "python experiments\\run_workload_drift_posthoc_calibration.py --baseline-metrics research_plan\\workload_drift_robust_mscn_windows_value_bitmap\\seed_1\\metrics.json --method shrink-ridge --feature-preset value --adaptation-rows 250 --seeds 1,2,3 --min-eval-rows 1000 --shrink-factor 0.735 --shrink-anchor zero --ridge-alpha 0.75 --output-json research_plan\\workload_drift_robust_mscn_value_shrink_ridge_value_zero_s73p5_a0p75_250.json",
            "python experiments\\run_workload_drift_posthoc_calibration.py --baseline-metrics research_plan\\workload_drift_robust_mscn_windows_value_bitmap\\seed_1\\metrics.json --method shrink-ridge --feature-preset value --adaptation-rows 250 --seeds 1,2,3 --min-eval-rows 1000 --shrink-factor 0.80 --shrink-anchor zero --ridge-alpha 1.25 --output-json research_plan\\workload_drift_robust_mscn_value_shrink_ridge_value_zero_s80_a1p25_250.json",
            "python experiments\\run_workload_drift_posthoc_calibration.py --baseline-metrics research_plan\\workload_drift_robust_mscn_windows_value_bitmap\\seed_1\\metrics.json --method hist-gradient --feature-preset value --adaptation-rows 122 --seeds 1,2,3 --min-eval-rows 1000 --hist-min-samples-leaf 11 --hist-learning-rate 0.06 --hist-l2 0.3 --output-json research_plan\\workload_drift_robust_mscn_value_hist_gradient_value_122_leaf11_l2_0p3_lr0p06.json",
            "python experiments\\run_workload_drift_posthoc_calibration.py --baseline-metrics research_plan\\workload_drift_robust_mscn_windows_value_bitmap\\seed_1\\metrics.json --method hist-gradient --feature-preset value --adaptation-rows 121 --seeds 1,2,3 --min-eval-rows 1000 --hist-min-samples-leaf 10 --hist-max-leaf-nodes 15 --hist-learning-rate 0.045 --hist-l2 0.19 --output-json research_plan\\workload_drift_robust_mscn_value_hist_gradient_value_121_leaf10_nodes15_lr0p045_l2_0p19_guard.json",
            "python experiments\\run_workload_drift_posthoc_calibration.py --baseline-metrics research_plan\\workload_drift_robust_mscn_windows_value_bitmap\\seed_1\\metrics.json --method hist-gradient --feature-preset value --adaptation-rows 124 --seeds 1,2,3 --min-eval-rows 1000 --hist-min-samples-leaf 11 --output-json research_plan\\workload_drift_robust_mscn_value_hist_gradient_value_124_leaf11.json",
            "python experiments\\run_workload_drift_posthoc_calibration.py --baseline-metrics research_plan\\workload_drift_robust_mscn_windows_value_bitmap\\seed_1\\metrics.json --method hist-gradient --feature-preset value --adaptation-rows 136 --seeds 1,2,3 --min-eval-rows 1000 --hist-min-samples-leaf 10 --output-json research_plan\\workload_drift_robust_mscn_value_hist_gradient_value_136_leaf10.json",
            "python experiments\\run_workload_drift_posthoc_calibration.py --baseline-metrics research_plan\\workload_drift_robust_mscn_windows_value_bitmap\\seed_1\\metrics.json --method hist-gradient --feature-preset value --adaptation-rows 137 --seeds 1,2,3 --min-eval-rows 1000 --hist-min-samples-leaf 7 --output-json research_plan\\workload_drift_robust_mscn_value_hist_gradient_value_137_leaf7.json",
            "python experiments\\run_workload_drift_posthoc_calibration.py --baseline-metrics research_plan\\workload_drift_robust_mscn_windows_value_bitmap\\seed_1\\metrics.json --method hist-gradient --feature-preset value --adaptation-rows 174 --seeds 1,2,3 --min-eval-rows 1000 --hist-min-samples-leaf 10 --output-json research_plan\\workload_drift_robust_mscn_value_hist_gradient_value_174_leaf10.json",
            "python experiments\\summarize_workload_drift_posthoc_calibration.py",
            "```",
        ]
    )
    reproduction_header = lines.index("## Reproduction Commands")
    del lines[reproduction_header - 1 :]
    append_reproduction_commands(lines, selected_passes + selected_boundaries)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


def write_mscn_note(rows: list[dict]) -> None:
    csv_path = RESEARCH / "workload_drift_mscn_posthoc_summary.csv"
    write_csv(csv_path, rows)
    selected_passes = [
        find_row(rows, "column", "shrink-ridge zero s45", 290),
        row_by_path(rows, r"research_plan\workload_drift_mscn_column_shrink_ridge_value_zero_s48_a0p5_273.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_column_shrink_ridge_value_zero_s48_a0p5_274.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_column_shrink_ridge_value_zero_s49_276.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_column_shrink_ridge_value_zero_s49_277.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_column_shrink_ridge_value_zero_s50_280.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_column_shrink_ridge_zero_s50_350.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_column_shrink_ridge_zero_s55_400.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_column_shrink_ridge_zero_s58p5_475.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_column_shrink_ridge_zero_s60_499.json"),
        find_row(rows, "column", "shrink-ridge zero s75", 1500),
        find_row(rows, "table", "shrink-ridge zero s55", 77),
        row_by_path(rows, r"research_plan\workload_drift_mscn_table_hist_gradient_abs_value_26_leaf7_pass_26_leaf7_lr0p09_l2_0p2.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_table_hist_gradient_abs_value_29_leaf7_pass_29_leaf7_lr0p035_l2_0p2.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_table_hist_gradient_abs_value_30_leaf8_pass_30_leaf8_default.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_table_hist_gradient_abs_value_35_leaf4_pass_35_leaf4_default.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_table_hist_gradient_abs_value_38_leaf3_pass_38_leaf3_default.json"),
        find_row(rows, "table", "hist-gradient value leaf7", 51),
        row_by_path(rows, r"research_plan\workload_drift_mscn_table_hist_gradient_value_53_leaf8.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_table_hist_gradient_value_54_leaf8.json"),
        find_row(rows, "table", "hist-gradient value leaf6", 55),
        row_by_path(rows, r"research_plan\workload_drift_mscn_table_hist_gradient_value_56_leaf6.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_table_hist_gradient_value_58_leaf6.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_table_hist_gradient_value_59_leaf6.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_table_shrink_ridge_zero_s85_102.json"),
        find_row(rows, "table", "ridge", 103),
        find_row(rows, "table", "ridge", 105),
        find_row(rows, "table", "ridge", 109),
        row_by_path(rows, r"research_plan\workload_drift_mscn_table_shrink_ridge_zero_s73_91.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_table_shrink_ridge_zero_s75_92.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_table_shrink_ridge_zero_s75_95.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_table_ridge_120.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_table_ridge_130.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_table_ridge_200.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_table_ridge_250.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_table_ridge_500.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_table_ridge_1000.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_table_ridge_1500.json"),
        find_row(rows, "table", "ridge", 2000),
        row_by_path(rows, r"research_plan\workload_drift_mscn_table_shrink_ridge_global_s125_2000.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_value_hist_gradient_value_463_leaf6_nodes8_iter425_lr0p03_l2_0p78.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_value_hist_gradient_value_464_leaf6_nodes8_iter400_lr0p0301_l2_0p765.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_value_hist_gradient_value_465_leaf6_nodes8_iter425_lr0p0299_l2_0p765.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_value_hist_gradient_value_466_leaf6_nodes8_iter425_lr0p0299_l2_0p75.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_value_hist_gradient_value_470_leaf6_nodes8_iter400_lr0p0295_l2_0p675.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_value_hist_gradient_value_470_leaf6_nodes8_iter425_lr0p0295_l2_0p675.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_value_hist_gradient_value_471_leaf6_nodes9_iter375_lr0p03_l2_0p625.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_value_hist_gradient_value_471_leaf6_nodes9_iter400_lr0p03_l2_0p625.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_value_hist_gradient_value_472_leaf6_nodes10_iter375_lr0p03_l2_0p65.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_value_hist_gradient_value_473_leaf6_nodes10_iter350_lr0p0305_l2_0p625.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_value_hist_gradient_value_474_leaf6_nodes7_iter350_lr0p0335_l2_0p55.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_value_hist_gradient_value_474_leaf6_nodes8_lr0p0335_l2_0p525.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_value_hist_gradient_value_475_leaf6_nodes9_lr0p035_l2_0p65.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_value_hist_gradient_value_476_leaf7_lr0p035_l2_0p02.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_value_hist_gradient_value_477_leaf7_lr0p03_l2_0p05.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_value_hist_gradient_value_477_leaf7_lr0p03_l2_1.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_value_hist_gradient_value_478_leaf6_lr003.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_value_hist_gradient_value_500_nodes7.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_value_hist_gradient_value_516_leaf7.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_value_hist_gradient_value_521_nodes7.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_value_hist_gradient_value_529_nodes9.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_value_hist_gradient_value_539_nodes7.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_value_hist_gradient_value_560.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_value_hist_gradient_value_600.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_value_hist_gradient_value_750.json"),
    ]
    selected_boundaries = [
        find_row(rows, "column", "shrink-ridge zero s45", 275),
        find_row(rows, "column", "shrink-ridge zero s50", 250),
        find_row(rows, "column", "shrink-ridge zero s48 value", 270),
        find_row(rows, "column", "shrink-ridge zero s48 value", 275),
        row_by_path(rows, r"research_plan\workload_drift_mscn_column_shrink_ridge_value_zero_s48_a0p5_272.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_column_shrink_ridge_value_zero_s51_a0p3_272_guard.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_column_shrink_ridge_value_zero_s518m_a0p5_272_guard.json"),
        find_row(rows, "column", "shrink-ridge zero s47 value", 278),
        row_by_path(rows, r"research_plan\workload_drift_mscn_column_shrink_ridge_zero_s58p6_475.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_column_shrink_ridge_zero_s59_475.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_table_hist_gradient_abs_value_25_leaf8_boundary_25_leaf8_default.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_table_hist_gradient_abs_value_25_leaf7_boundary_25_leaf7_default.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_table_hist_gradient_abs_value_25_leaf7_nodes7_iter150_lr0p09_l2_0p2_guard.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_table_hist_gradient_abs_value_25_leaf8_lr0p08_l2_0p3_guard.json"),
        find_row(rows, "table", "hist-gradient value leaf7", 50),
        row_by_path(rows, r"research_plan\workload_drift_mscn_table_hist_gradient_value_52_leaf8.json"),
        find_row(rows, "table", "hist-gradient value leaf6", 54),
        find_row(rows, "table", "hist-gradient value leaf10", 60),
        find_row(rows, "table", "hist-gradient value", 68),
        row_by_path(rows, r"research_plan\workload_drift_mscn_table_tail_risk_ridge_value_global_s75_q50_a10_50.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_table_tail_risk_ridge_value_zero_s75_q50_a10_50.json"),
        find_row(rows, "table", "shrink-ridge zero s45", 76),
        row_by_path(rows, r"research_plan\workload_drift_mscn_table_shrink_ridge_zero_s72_91.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_table_shrink_ridge_zero_s74_91.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_table_shrink_ridge_zero_s75_91.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_table_shrink_ridge_zero_s90_102.json"),
        find_row(rows, "table", "ridge", 101),
        find_row(rows, "table", "ridge", 102),
        row_by_path(rows, r"research_plan\workload_drift_mscn_value_hist_gradient_value_462_leaf6_nodes8_iter425_lr0p0301_l2_0p795.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_value_hist_gradient_value_462_leaf6_nodes8_iter450_lr0p03_l2_0p81.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_value_hist_gradient_value_463_leaf6_nodes8_iter375_lr0p03_l2_0p765.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_value_hist_gradient_value_464_leaf6_nodes8_iter425_lr0p0301_l2_0p765.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_value_hist_gradient_value_470_leaf6_nodes8_iter425_lr0p0295_l2_0p625.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_value_hist_gradient_value_472_leaf6_nodes10_iter350_lr0p03_l2_0p65.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_value_hist_gradient_value_473_leaf6_nodes11_lr0p03_l2_0p65.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_value_hist_gradient_value_473_leaf6_nodes7_lr0p0345_l2_0p55.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_value_hist_gradient_value_474_leaf6_nodes7_lr0p035_l2_0p55.json"),
        row_by_path(rows, r"research_plan\workload_drift_mscn_value_hist_gradient_value_475_leaf5_lr0p035_l2_0p02.json"),
        find_row(rows, "value", "hist-gradient value leaf6 lr0p03", 476),
        row_by_path(rows, r"research_plan\workload_drift_mscn_value_hist_gradient_value_477_leaf6_lr003_l2_0p05.json"),
        find_row(rows, "value", "hist-gradient value nodes9", 477),
        find_row(rows, "value", "hist-gradient value", 538),
        find_row(rows, "value", "hist-gradient value", 545),
        find_row(rows, "value", "shrink-ridge global s75", 500),
        find_row(rows, "value", "shrink-ridge zero s75", 500),
    ]
    md_path = RESEARCH / "workload_drift_mscn_posthoc_calibration.md"
    lines = [
        "# MSCN Post-hoc Calibration Expansion",
        "",
        "This note records simple post-hoc calibration baselines on reproduced MSCN prediction/label artifacts. It is isolated from the Drift-Signature Residual Field Transport implementation and paper artifacts.",
        "",
        "## Strict 3-Seed Passes",
        "",
        "| drift | method | labels | gate | p90 reduction | p95 reduction | p99 reduction | p95 vs global | p99 vs global |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected_passes:
        lines.append(table_row(row))
    lines.extend(
        [
            "",
            "## Boundary And Negative Results",
            "",
            "| drift | method | labels | gate | p90 reduction | p95 reduction | p99 reduction | p95 vs global | p99 vs global | fail reasons |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in selected_boundaries:
        lines.append(boundary_table_row(row))
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `column / shrink-ridge zero s45` gives a low-label MSCN simple post-hoc calibration point: 290 labeled target queries, 3/3 strict-gate pass, and 35.6%/42.3%/66.2% p90/p95/p99 reductions. At 275 labels it is 2/3.",
            "- `column / shrink-ridge zero s48 a0p5 value` improves that low-label MSCN column point to 273 labeled target queries, 3/3 strict-gate pass, and 31.5%/45.3%/65.6% p90/p95/p99 reductions, lowering the previous 276-label s48 value point. Focused 272-label guards did not lower this boundary: s48/a0.5 is 2/3 because seed 3 misses p90, s51/a0.3 is 2/3 because seed 2 misses the median guard, and s51.8/a0.5 is also 2/3 on seed-2 median. Nearby 270-label s48, 275-label unregularized s48, and 278-label s47 settings remain additional 2/3 boundaries.",
            "- Mined column-drift passes add stronger MSCN tail tradeoffs above the 273-label budget: value-aware zero-s48/a0.5 at 274 labels gives a nearby p99 tradeoff at 31.4%/45.1%/65.7%, zero-s49 at 276/277 labels reaches 31.7%/47.3%/65.3% and 33.2%/47.5%/66.7%, and zero-s50 at 280 labels gives a p95/p99 tradeoff at 33.8%/47.6%/66.4%. Higher-budget zero-s50 at 350 labels gives a p95-oriented intermediate at 36.9%/52.9%/68.3%, zero-s55 at 400 labels reaches 34.1%/50.8%/71.2%, zero-s58.5 at 475 labels reaches 36.5%/53.2%/71.3%, and zero-s60 at 499 labels reaches 36.3%/51.8%/72.1%. The fractional 475-label sweep shows s58.5 is the last stable setting; s58.6 and s59+ are 2/3 median boundaries on seed 2. `column / shrink-ridge zero s75` remains the strongest selected simple MSCN column point at 1,500 labels, with 41.6%/62.4%/77.1% reductions.",
            "- `table / shrink-ridge zero s55` lowers the MSCN simple post-hoc budget to 77 labels, with 86.7%/91.6%/66.0% reductions. At 76 labels the nearby s45 setting is 2/3 due to the p99-reduction guard.",
            "- `table / hist-gradient value leaf7` was the older squared-error value-aware MSCN table-drift low-budget point at 51 labels, with 87.2%/92.5%/61.7% p90/p95/p99 reductions; its 50-label squared-error neighbor is only 2/3. Switching the same simple hist-gradient residual calibrator to `absolute_error` loss gives a new lower-budget track: 26-label abs leaf7/lr0.09/l2=0.2 is a strict 3/3 pass at 80.7%/87.4%/62.3%. A focused 25-label guard sweep did not lower the boundary: default leaf8 is 2/3 because seed 2 violates the median guard, default leaf7 is 1/3 because seed 2 violates median and seed 3 misses p99, capacity-limited leaf7/nodes7/iter150 is still 1/3 with seed-2 median and seed-3 p99 failures, and leaf8/lr0.08/l2=0.3 is 2/3 on seed-2 median. Above the low-budget point, abs leaf7 at 29 labels is the stronger p99 tradeoff at 82.7%/86.9%/79.2%, abs leaf8 at 30 labels gives a balanced 81.2%/90.9%/76.2%, abs leaf4 at 35 labels gives a p95-oriented 89.0%/93.1%/75.5%, and abs leaf3 at 38 labels reaches 88.4%/94.1%/79.0%. The older squared-error 53-label leaf8, 54-label leaf8, 55/56/58/59-label leaf6 points remain useful high-quantile references, but they are no longer the lowest-label hist-gradient table-drift evidence. Tail-risk-ridge variants at 50 labels remain negative: selected global/zero anchor probes are 0/3 because p99 is unstable across seeds.",
            "- `table / shrink-ridge zero s85` now lowers the strict MSCN simple table ridge-like budget to 102 labels, with 90.6%/96.3%/81.5% reductions; s90/s95/s98/s99 at 102 labels and unshrunk ridge at 102 labels remain 2/3 because seed 3 misses the p99-reduction guard. The 103-label ridge point remains a ridge-only reference at 88.5%/95.5%/77.4%. Stronger ridge-like points improve the low-budget tail curve: zero-s73 now passes at 91 labels with 90.1%/94.9%/83.1%, while s72/s74/s75 at 91 labels are 2/3 p99 boundaries; zero-s75 still gives the stronger 92-label point at 90.6%/95.0%/84.8%, and 95 labels reaches 90.5%/94.7%/82.4%. A previously unselected high-budget ridge frontier adds 200/250/500/1,000/1,500-label points with p99 -90.5%/-92.7%/-96.3%/-96.4%/-96.9%; the 500-label ridge point nearly matches the old 2,000-label ridge p99 while using one quarter of the labels, and 1,000 labels gives the small intermediate p95/p99 bump to 98.5%/96.4%. At 2,000 labels, shrink-ridge global s125 is the strongest selected simple table tail point at 96.3%/99.0%/98.2%. Like other aggressive table tail points, the 91/92-label shrink-ridge points clear the median and quantile gates but still have extreme seed-3 mean outliers.",
            "- A value-aware column-drift shrink-ridge sweep around 250-275 labels did not lower the current 276-label point; nearby settings remain 1/3-2/3, usually on the p90 or p99 guards.",
            "- Basic value-drift shrink-ridge remains a negative target on reproduced MSCN artifacts: the best 500-label variants improve p95 by only 22.5%-24.4% and p99 by 14.3%-15.0%, so all stay 0/3.",
            "- Value-aware features give new MSCN value-drift passing points. `value / hist-gradient value leaf6 nodes8 iter425 lr0p03 l2_0p78` lowers the passing budget to 463 labels with 62.1%/66.2%/60.3% p90/p95/p99 reductions, improving on the previous 470-label, 471-label, 472-label, 473-label, 474-label, 475-label, 476-label, 477-label, and 478-label value-aware points. This 463-label point is a strict 3/3 pass but sits near the p99 boundary: the seed-level p99 reductions are 60.1%, 60.7%, and 60.3%. A focused 462-label guard-protection sweep around the 463-label point did not find a strict 3/3 pass: the best aggregate-tail neighbors remain 2/3, with `iter425/lr0.0301/l2_0p795` reaching 62.1% aggregate p99 but seed 1 falling to 58.2%, and `iter450/lr0.03/l2_0p81` reaching 61.4% aggregate p99 but seed 2 falling to 57.3%. A second conservative 462-label guard sweep over lower leaf/nodes counts and stronger L2 also failed 36/36 settings, so 463 labels remains the honest low-budget boundary for this value-drift hist-gradient track. The 464-label `leaf6/nodes8/iter400/lr0.0301/l2_0p765` point is a stronger-tail intermediate with 62.2%/67.4%/63.0% reductions, while 465-label `iter425/lr0.0299/l2_0p765` and 466-label `iter425/lr0.0299/l2_0p75` give useful higher-budget operating points at 62.8%/68.3%/61.7% and 61.6%/66.4%/62.0%. The older 470-label `iter400/lr0.0295/l2_0p675` and `iter425/lr0.0295/l2_0p675` settings remain valid intermediate points at 62.0%/66.7%/60.6% and 61.9%/66.6%/61.7%, but are now superseded on label budget. Nearby candidates with larger aggregate p99 can still be 2/3 because one seed misses the strict p99 guard; for example the 463-label `iter375/lr0.03/l2_0p765` neighbor reaches 60.6% aggregate p99 but seed 3 reaches only 58.6%, and the 464-label `iter425/lr0.0301/l2_0p765` neighbor reaches 63.1% aggregate p99 but seed 2 reaches only 59.6%.",
            "- Mined high-budget MSCN value-drift hist-gradient passes show how much tail gain is available when labels are less constrained: 477-label leaf7/lr0.03/l2_1 is a p95 tradeoff at 62.4%/69.0%/60.5%, 478-label leaf6/lr0.03 reaches 63.1% p99, 500-label nodes7 reaches 64.3%, 516-label leaf7 gives a p95 tradeoff at 64.4%/69.1%/60.9%, 521 labels reaches 65.6% p99, 529-label nodes9 gives 64.8%/69.6%/64.0%, 539/560-label operating points reach 66.8%/68.9% p99, 600 labels reaches 69.4%, and 750 labels reaches 70.7% p99 with weakest-seed p99 still 65.6%. These are stronger operating points, not replacements for the 463-label budget result.",
            "- These results are baseline-expansion evidence, not edits to the residual-field method.",
            "",
            "## Evidence Files",
            "",
            f"- Summary CSV: `{csv_path.relative_to(ROOT)}`",
            "- Runner: `experiments\\run_workload_drift_posthoc_calibration.py`",
        ]
    )
    for row in selected_passes + selected_boundaries:
        lines.append(f"- `{row['report_path']}`")
    lines.extend(
        [
            "",
            "## Reproduction Commands",
            "",
            "```powershell",
            "python experiments\\run_workload_drift_posthoc_calibration.py --baseline-metrics research_plan\\workload_drift_mscn_windows_bitmap\\seed_1\\metrics.json --method shrink-ridge --adaptation-rows 290 --seeds 1,2,3 --min-eval-rows 5000 --shrink-factor 0.45 --shrink-anchor zero --output-json research_plan\\workload_drift_mscn_column_shrink_ridge_zero_s45_290.json",
            "python experiments\\run_workload_drift_posthoc_calibration.py --baseline-metrics research_plan\\workload_drift_mscn_windows_bitmap\\seed_1\\metrics.json --method shrink-ridge --feature-preset value --adaptation-rows 273 --seeds 1,2,3 --min-eval-rows 5000 --shrink-factor 0.48 --shrink-anchor zero --ridge-alpha 0.5 --output-json research_plan\\workload_drift_mscn_column_shrink_ridge_value_zero_s48_a0p5_273.json",
            "python experiments\\run_workload_drift_posthoc_calibration.py --baseline-metrics research_plan\\workload_drift_mscn_windows_table_bitmap\\seed_1\\metrics.json --method shrink-ridge --adaptation-rows 77 --seeds 1,2,3 --min-eval-rows 5000 --shrink-factor 0.55 --shrink-anchor zero --output-json research_plan\\workload_drift_mscn_table_shrink_ridge_zero_s55_77.json",
            "python experiments\\run_workload_drift_posthoc_calibration.py --baseline-metrics research_plan\\workload_drift_mscn_windows_table_bitmap\\seed_1\\metrics.json --method hist-gradient --feature-preset value --adaptation-rows 51 --seeds 1,2,3 --min-eval-rows 5000 --hist-min-samples-leaf 7 --output-json research_plan\\workload_drift_mscn_table_hist_gradient_value_51_leaf7.json",
            "python experiments\\run_workload_drift_posthoc_calibration.py --baseline-metrics research_plan\\workload_drift_mscn_windows_table_bitmap\\seed_1\\metrics.json --method ridge --adaptation-rows 103 --seeds 1,2,3 --min-eval-rows 5000 --output-json research_plan\\workload_drift_mscn_table_ridge_103.json",
            "python experiments\\run_workload_drift_posthoc_calibration.py --baseline-metrics research_plan\\workload_drift_mscn_windows_table_bitmap\\seed_1\\metrics.json --method ridge --adaptation-rows 105 --seeds 1,2,3 --min-eval-rows 5000 --output-json research_plan\\workload_drift_mscn_table_ridge_105.json",
            "python experiments\\run_workload_drift_posthoc_calibration.py --baseline-metrics research_plan\\workload_drift_mscn_windows_value_bitmap\\seed_1\\metrics.json --method hist-gradient --feature-preset value --adaptation-rows 462 --seeds 1,2,3 --min-eval-rows 1000 --hist-min-samples-leaf 6 --hist-max-leaf-nodes 8 --hist-max-iter 425 --hist-learning-rate 0.0301 --hist-l2 0.795 --output-json research_plan\\workload_drift_mscn_value_hist_gradient_value_462_leaf6_nodes8_iter425_lr0p0301_l2_0p795.json",
            "python experiments\\run_workload_drift_posthoc_calibration.py --baseline-metrics research_plan\\workload_drift_mscn_windows_value_bitmap\\seed_1\\metrics.json --method hist-gradient --feature-preset value --adaptation-rows 463 --seeds 1,2,3 --min-eval-rows 1000 --hist-min-samples-leaf 6 --hist-max-leaf-nodes 8 --hist-max-iter 425 --hist-learning-rate 0.03 --hist-l2 0.78 --output-json research_plan\\workload_drift_mscn_value_hist_gradient_value_463_leaf6_nodes8_iter425_lr0p03_l2_0p78.json",
            "python experiments\\run_workload_drift_posthoc_calibration.py --baseline-metrics research_plan\\workload_drift_mscn_windows_value_bitmap\\seed_1\\metrics.json --method hist-gradient --feature-preset value --adaptation-rows 464 --seeds 1,2,3 --min-eval-rows 1000 --hist-min-samples-leaf 6 --hist-max-leaf-nodes 8 --hist-max-iter 400 --hist-learning-rate 0.0301 --hist-l2 0.765 --output-json research_plan\\workload_drift_mscn_value_hist_gradient_value_464_leaf6_nodes8_iter400_lr0p0301_l2_0p765.json",
            "python experiments\\run_workload_drift_posthoc_calibration.py --baseline-metrics research_plan\\workload_drift_mscn_windows_value_bitmap\\seed_1\\metrics.json --method hist-gradient --feature-preset value --adaptation-rows 465 --seeds 1,2,3 --min-eval-rows 1000 --hist-min-samples-leaf 6 --hist-max-leaf-nodes 8 --hist-max-iter 425 --hist-learning-rate 0.0299 --hist-l2 0.765 --output-json research_plan\\workload_drift_mscn_value_hist_gradient_value_465_leaf6_nodes8_iter425_lr0p0299_l2_0p765.json",
            "python experiments\\run_workload_drift_posthoc_calibration.py --baseline-metrics research_plan\\workload_drift_mscn_windows_value_bitmap\\seed_1\\metrics.json --method hist-gradient --feature-preset value --adaptation-rows 466 --seeds 1,2,3 --min-eval-rows 1000 --hist-min-samples-leaf 6 --hist-max-leaf-nodes 8 --hist-max-iter 425 --hist-learning-rate 0.0299 --hist-l2 0.75 --output-json research_plan\\workload_drift_mscn_value_hist_gradient_value_466_leaf6_nodes8_iter425_lr0p0299_l2_0p75.json",
            "python experiments\\run_workload_drift_posthoc_calibration.py --baseline-metrics research_plan\\workload_drift_mscn_windows_value_bitmap\\seed_1\\metrics.json --method hist-gradient --feature-preset value --adaptation-rows 470 --seeds 1,2,3 --min-eval-rows 1000 --hist-min-samples-leaf 6 --hist-max-leaf-nodes 8 --hist-max-iter 400 --hist-learning-rate 0.0295 --hist-l2 0.675 --output-json research_plan\\workload_drift_mscn_value_hist_gradient_value_470_leaf6_nodes8_iter400_lr0p0295_l2_0p675.json",
            "python experiments\\run_workload_drift_posthoc_calibration.py --baseline-metrics research_plan\\workload_drift_mscn_windows_value_bitmap\\seed_1\\metrics.json --method hist-gradient --feature-preset value --adaptation-rows 470 --seeds 1,2,3 --min-eval-rows 1000 --hist-min-samples-leaf 6 --hist-max-leaf-nodes 8 --hist-max-iter 425 --hist-learning-rate 0.0295 --hist-l2 0.675 --output-json research_plan\\workload_drift_mscn_value_hist_gradient_value_470_leaf6_nodes8_iter425_lr0p0295_l2_0p675.json",
            "python experiments\\run_workload_drift_posthoc_calibration.py --baseline-metrics research_plan\\workload_drift_mscn_windows_value_bitmap\\seed_1\\metrics.json --method hist-gradient --feature-preset value --adaptation-rows 471 --seeds 1,2,3 --min-eval-rows 1000 --hist-min-samples-leaf 6 --hist-max-leaf-nodes 9 --hist-max-iter 375 --hist-learning-rate 0.03 --hist-l2 0.625 --output-json research_plan\\workload_drift_mscn_value_hist_gradient_value_471_leaf6_nodes9_iter375_lr0p03_l2_0p625.json",
            "python experiments\\run_workload_drift_posthoc_calibration.py --baseline-metrics research_plan\\workload_drift_mscn_windows_value_bitmap\\seed_1\\metrics.json --method hist-gradient --feature-preset value --adaptation-rows 471 --seeds 1,2,3 --min-eval-rows 1000 --hist-min-samples-leaf 6 --hist-max-leaf-nodes 9 --hist-max-iter 400 --hist-learning-rate 0.03 --hist-l2 0.625 --output-json research_plan\\workload_drift_mscn_value_hist_gradient_value_471_leaf6_nodes9_iter400_lr0p03_l2_0p625.json",
            "python experiments\\run_workload_drift_posthoc_calibration.py --baseline-metrics research_plan\\workload_drift_mscn_windows_value_bitmap\\seed_1\\metrics.json --method hist-gradient --feature-preset value --adaptation-rows 472 --seeds 1,2,3 --min-eval-rows 1000 --hist-min-samples-leaf 6 --hist-max-leaf-nodes 10 --hist-max-iter 375 --hist-learning-rate 0.03 --hist-l2 0.65 --output-json research_plan\\workload_drift_mscn_value_hist_gradient_value_472_leaf6_nodes10_iter375_lr0p03_l2_0p65.json",
            "python experiments\\run_workload_drift_posthoc_calibration.py --baseline-metrics research_plan\\workload_drift_mscn_windows_value_bitmap\\seed_1\\metrics.json --method hist-gradient --feature-preset value --adaptation-rows 473 --seeds 1,2,3 --min-eval-rows 1000 --hist-min-samples-leaf 6 --hist-max-leaf-nodes 10 --hist-max-iter 350 --hist-learning-rate 0.0305 --hist-l2 0.625 --output-json research_plan\\workload_drift_mscn_value_hist_gradient_value_473_leaf6_nodes10_iter350_lr0p0305_l2_0p625.json",
            "python experiments\\run_workload_drift_posthoc_calibration.py --baseline-metrics research_plan\\workload_drift_mscn_windows_value_bitmap\\seed_1\\metrics.json --method hist-gradient --feature-preset value --adaptation-rows 474 --seeds 1,2,3 --min-eval-rows 1000 --hist-min-samples-leaf 6 --hist-max-leaf-nodes 7 --hist-max-iter 350 --hist-learning-rate 0.0335 --hist-l2 0.55 --output-json research_plan\\workload_drift_mscn_value_hist_gradient_value_474_leaf6_nodes7_iter350_lr0p0335_l2_0p55.json",
            "python experiments\\run_workload_drift_posthoc_calibration.py --baseline-metrics research_plan\\workload_drift_mscn_windows_value_bitmap\\seed_1\\metrics.json --method hist-gradient --feature-preset value --adaptation-rows 474 --seeds 1,2,3 --min-eval-rows 1000 --hist-min-samples-leaf 6 --hist-max-leaf-nodes 8 --hist-learning-rate 0.0335 --hist-l2 0.525 --output-json research_plan\\workload_drift_mscn_value_hist_gradient_value_474_leaf6_nodes8_lr0p0335_l2_0p525.json",
            "python experiments\\summarize_workload_drift_posthoc_calibration.py",
            "```",
        ]
    )
    reproduction_header = lines.index("## Reproduction Commands")
    del lines[reproduction_header - 1 :]
    append_reproduction_commands(lines, selected_passes + selected_boundaries)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


def main() -> None:
    paths = sorted(
        list(RESEARCH.glob("workload_drift_robust_mscn_*.json"))
        + list(RESEARCH.glob("workload_drift_mscn_*.json"))
    )
    rows = []
    for path in paths:
        if not is_posthoc_baseline_report(path):
            continue
        row = read_report(path)
        if row is not None:
            rows.append(row)
    if not rows:
        raise SystemExit("no workload-drift post-hoc calibration reports found")
    rows.sort(key=lambda item: (item["family"], item["drift"], item["method"], item["labels"], item["report_path"]))
    robust_rows = [row for row in rows if row["family"] == "robust_mscn"]
    mscn_rows = [row for row in rows if row["family"] == "mscn"]
    if robust_rows:
        write_robust_note(robust_rows)
    if mscn_rows:
        write_mscn_note(mscn_rows)


if __name__ == "__main__":
    main()
