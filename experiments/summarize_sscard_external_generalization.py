import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_CSV = ROOT / "research_plan" / "sscard_posthoc_summary.csv"
STABILITY10_CSV = ROOT / "research_plan" / "sscard_stability10_summary.csv"
RESULT_DIR = ROOT / "paper" / "workload_drift_residual_field" / "results"

def pct(value: float) -> str:
    return f"{100.0 * value:.1f}\\%"


def number(value: float) -> str:
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def read_rows() -> list[dict]:
    with SUMMARY_CSV.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_stability10_rows() -> list[dict]:
    if not STABILITY10_CSV.exists():
        return []
    with STABILITY10_CSV.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_table(path: Path, rows: list[dict]) -> None:
    lines = [
        "\\begin{table}[!t]",
        "\\centering",
        "\\caption{External protocol check using shipped SSCard result CSVs. Cases report 3-seed validation-gate passes; the last columns summarize selected 10-seed stability reruns, where Min Labels is the smallest stable selected rerun budget.}",
        "\\label{tab:sscard-external}",
        "\\papertablesetup",
        "\\setlength{\\tabcolsep}{1.9pt}",
        "\\begin{tabular*}{\\columnwidth}{@{\\extracolsep{\\fill}}l|c r|r r r@{}}",
        "\\toprule",
        "\\tabheader{Dataset} & \\tabheader{Cases} & \\tabheader{P99 Red.} & \\tabheader{10-seed Fam.} & \\tabheader{Min Labels} & \\tabheader{Best P99} \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['dataset'].replace('_', '\\_')} "
            f"& \\passcell{{{row['passing_cases']}}} "
            f"& \\passcell{{{row['p99_range']}}} "
            f"& {row['stable_estimator_count']} "
            f"& {row['stable_min_labels']} "
            f"& \\passcell{{{row['stable_best_p99']}}} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular*}", "\\end{table}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = read_rows()
    stability_rows = read_stability10_rows()
    passed = [row for row in rows if row["passed"] == "True" and row["dataset"] in {"DBLP_AN", "WIKI"}]
    out_rows = []
    for dataset in ["DBLP_AN", "WIKI"]:
        group = [row for row in passed if row["dataset"] == dataset]
        tested = [row for row in rows if row["dataset"] == dataset]
        stable_group = [
            row
            for row in stability_rows
            if row.get("passed") == "True" and row.get("dataset") == dataset
        ]
        p95 = [float(row["p95_reduction"]) for row in group]
        p99 = [float(row["p99_reduction"]) for row in group]
        estimators = sorted(row["baseline"].replace("SSCard/", "") for row in group)
        stable_estimators = sorted({row["baseline"].replace("SSCard/", "") for row in stable_group})
        stable_min_labels = min((int(row["adaptation_rows"]) for row in stable_group), default=0)
        stable_best_p99 = max((float(row["p99_reduction"]) for row in stable_group), default=float("nan"))
        out_rows.append(
            {
                "dataset": dataset,
                "passing_cases": f"{len(group)}/{len(tested)}",
                "estimators": ",".join(estimators),
                "p95_min": min(p95),
                "p95_max": max(p95),
                "p99_min": min(p99),
                "p99_max": max(p99),
                "p95_range": f"{pct(min(p95))}--{pct(max(p95))}",
                "p99_range": f"{pct(min(p99))}--{pct(max(p99))}",
                "stable_estimator_count": len(stable_estimators),
                "stable_estimators": ",".join(stable_estimators),
                "stable_min_labels": stable_min_labels,
                "stable_best_p99": pct(stable_best_p99) if stable_group else "--",
                "source_repo": "https://github.com/marlcplhra/SSCard",
                "uses_api": "False",
                "windows_native_summary": "True",
            }
        )

    out_csv = RESULT_DIR / "sscard_external_generalization.csv"
    out_table = RESULT_DIR / "tables" / "sscard_external_generalization.tex"
    manifest = RESULT_DIR / "sscard_external_generalization_manifest.json"
    write_csv(out_csv, out_rows)
    write_table(out_table, out_rows)
    manifest.write_text(
        json.dumps(
            {
                "passed": True,
                "scope": "External estimator/data check over shipped SSCard result CSVs",
                "selected_rows": len(out_rows),
                "datasets": sorted({row["dataset"] for row in out_rows}),
                "estimators": sorted({name for row in out_rows for name in row["estimators"].split(",")}),
                "summary_csv": str(out_csv),
                "table": str(out_table),
                "source_summary": str(SUMMARY_CSV),
                "source_repo": "https://github.com/marlcplhra/SSCard",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(json.loads(manifest.read_text(encoding="utf-8")), indent=2))


if __name__ == "__main__":
    main()
