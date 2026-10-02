import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def find_row(rows: list[dict], **conditions) -> dict:
    for row in rows:
        if all(row.get(key) == value for key, value in conditions.items()):
            return row
    raise KeyError(f"missing row matching {conditions}")


def bool_value(value: str) -> bool:
    return value.strip().lower() == "true"


def float_value(row: dict, key: str) -> float:
    return float(row[key])


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


def pass_cell(text: str, enabled: bool = True) -> str:
    return f"\\passcell{{{text}}}" if enabled else text


def reduction_cell(text: str, value: float, best_value=None) -> str:
    if value <= 0:
        return text
    if best_value is not None and best_value > 0 and value == best_value:
        return best_cell(text, True)
    return pass_cell(text)


def build_policy_rows(result_dir: Path) -> list[dict]:
    expanded = read_csv(result_dir / "expanded_method_comparison.csv")
    finetune = read_csv(result_dir / "finetune_comparison.csv")
    risk = read_csv(result_dir / "optimizer_risk_proxy.csv")
    rows = []
    for drift in ["column", "table", "value"]:
        residual = find_row(expanded, drift=drift, method="residual-field", variant="full")
        gated = find_row(expanded, drift=drift, method="gated-residual-field", variant="full")
        ft = find_row(finetune, drift=drift, method="mscn-finetune")
        raw_risk = find_row(risk, drift=drift, method="raw-mscn")
        residual_risk = find_row(risk, drift=drift, method="residual-field")
        ft_risk = find_row(risk, drift=drift, method="mscn-finetune-best")

        residual_passed = bool_value(residual["passed"])
        gated_passed = bool_value(gated["passed"])
        ft_passed = bool_value(ft["passed"])
        raw_score = float_value(raw_risk, "optimizer_risk_score")
        residual_score = float_value(residual_risk, "optimizer_risk_score")
        ft_score = float_value(ft_risk, "optimizer_risk_score")

        if residual_passed:
            action = "Use residual-field"
            selected = residual
            selected_risk = residual_risk
            rationale = "strict gate passes"
        elif ft_score < raw_score and ft_score <= residual_score:
            action = "Use MSCN fine-tune"
            selected = ft
            selected_risk = ft_risk
            rationale = "no gate pass; lowest risk proxy"
        elif gated_passed:
            action = "Use gated fallback"
            selected = gated
            selected_risk = residual_risk
            rationale = "residual field rejected on validation gate"
        else:
            action = "Abstain / keep MSCN"
            selected = None
            selected_risk = raw_risk
            rationale = "no candidate passes validation"

        if selected is None:
            selected_p95 = float_value(residual, "base_p95")
            selected_p99 = float_value(residual, "base_p99")
            selected_p95_reduction = 0.0
            selected_p99_reduction = 0.0
            selected_passed_seed_count = 0
        else:
            selected_p95 = float_value(selected, "method_p95")
            selected_p99 = float_value(selected, "method_p99")
            selected_p95_reduction = float_value(selected, "p95_reduction")
            selected_p99_reduction = float_value(selected, "p99_reduction")
            selected_passed_seed_count = int(selected["passed_seed_count"])

        rows.append(
            {
                "drift": drift,
                "action": action,
                "rationale": rationale,
                "residual_gate_passed": residual_passed,
                "fine_tune_gate_passed": ft_passed,
                "selected_passed_seed_count": selected_passed_seed_count,
                "seed_count": int(residual["seed_count"]),
                "raw_p95": float_value(residual, "base_p95"),
                "selected_p95": selected_p95,
                "selected_p95_reduction": selected_p95_reduction,
                "raw_p99": float_value(residual, "base_p99"),
                "selected_p99": selected_p99,
                "selected_p99_reduction": selected_p99_reduction,
                "raw_optimizer_risk_score": raw_score,
                "selected_optimizer_risk_score": float_value(selected_risk, "optimizer_risk_score"),
                "selected_optimizer_risk_reduction": (
                    (raw_score - float_value(selected_risk, "optimizer_risk_score")) / raw_score
                    if raw_score
                    else 0.0
                ),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_table(path: Path, rows: list[dict]) -> None:
    action_labels = {
        "Use residual-field": "Residual-field",
        "Use MSCN fine-tune": "MSCN fine-tune",
        "Use gated fallback": "Gated fallback",
        "Abstain / keep MSCN": "Keep MSCN",
    }
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Validation-gated repair policy derived from held-out gates and optimizer-risk proxy. The policy is conservative: it enables residual-field transport only when the strict gate passes.}",
        "\\label{tab:selector-policy}",
        "\\begin{tabular}{lllrrrrr}",
        "\\toprule",
        "Drift & Action & Trigger & P95 Q & P95 Red. & P99 Q & P99 Red. & Risk Red. \\\\",
        "\\midrule",
    ]
    best = {
        "selected_p95_reduction": max(row["selected_p95_reduction"] for row in rows),
        "selected_p99_reduction": max(row["selected_p99_reduction"] for row in rows),
        "selected_optimizer_risk_reduction": max(
            row["selected_optimizer_risk_reduction"] for row in rows
        ),
    }
    for row in rows:
        trigger = (
            f"gate {row['selected_passed_seed_count']}/{row['seed_count']}"
            if row["action"] == "Use residual-field"
            else "risk fallback"
        )
        p95_reduction = f"{pct(row['selected_p95_reduction'])}\\%"
        p99_reduction = f"{pct(row['selected_p99_reduction'])}\\%"
        risk_reduction = f"{pct(row['selected_optimizer_risk_reduction'])}\\%"
        lines.append(
            f"{row['drift'].title()} & {pass_cell(action_labels[row['action']])} "
            f"& {pass_cell(trigger)} "
            f"& {number(row['selected_p95'])} "
            f"& {reduction_cell(p95_reduction, row['selected_p95_reduction'], best['selected_p95_reduction'])} "
            f"& {number(row['selected_p99'])} "
            f"& {reduction_cell(p99_reduction, row['selected_p99_reduction'], best['selected_p99_reduction'])} "
            f"& {reduction_cell(risk_reduction, row['selected_optimizer_risk_reduction'], best['selected_optimizer_risk_reduction'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a validation-gated repair policy table.")
    parser.add_argument("--result-dir", default="paper/workload_drift_residual_field/results")
    args = parser.parse_args()
    result_dir = ROOT / args.result_dir
    rows = build_policy_rows(result_dir)
    csv_path = result_dir / "selector_policy.csv"
    table_path = result_dir / "tables" / "selector_policy.tex"
    manifest_path = result_dir / "selector_policy_manifest.json"
    write_csv(csv_path, rows)
    write_table(table_path, rows)
    manifest = {
        "passed": True,
        "scope": "Conservative validation-gated repair policy from held-out gates and optimizer-risk proxy",
        "csv": str(csv_path),
        "table": str(table_path),
        "rows": len(rows),
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
