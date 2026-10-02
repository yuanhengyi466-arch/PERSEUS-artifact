import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research_plan"


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def display_method(report: dict) -> str:
    method = report["method"]
    first = report["split_reports"][0]
    method_model = first.get("method_model", {})
    if method == "tail-gated" and "tail_gate_quantile" in method_model:
        return f"{method} q{int(round(100.0 * method_model['tail_gate_quantile']))}"
    return method


def read_report(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    failures = sorted(
        {
            gate
            for split in report["split_reports"]
            for gate, passed in split.get("pass_gates", {}).items()
            if not passed
        }
    )
    return {
        "dataset": report["dataset"],
        "baseline": report["baseline"],
        "method": display_method(report),
        "adaptation_rows": report["requires_labeled_target_adaptation_rows"],
        "rows": report["row_count"],
        "passed": report["passed"],
        "passed_seed_count": report["passed_seed_count"],
        "seed_count": report["seed_count"],
        "median_reduction": report["aggregate_reductions"]["median"],
        "p90_reduction": report["aggregate_reductions"]["p90"],
        "p95_reduction": report["aggregate_reductions"]["p95"],
        "p99_reduction": report["aggregate_reductions"]["p99"],
        "p95_vs_global": report["aggregate_reductions_vs_global_calibration"]["p95"],
        "p99_vs_global": report["aggregate_reductions_vs_global_calibration"]["p99"],
        "fail_reasons": "+".join(failures),
        "report_path": str(path),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    paths = sorted(RESEARCH.glob("sscard_budget_*.json"))
    if not paths:
        raise SystemExit("no SSCard budget sweep JSON files found")
    rows = [read_report(path) for path in paths]
    rows.sort(key=lambda row: (row["dataset"], row["baseline"], row["method"], row["adaptation_rows"]))

    out_csv = RESEARCH / "sscard_budget_sweep_summary.csv"
    write_csv(out_csv, rows)

    passing = [row for row in rows if row["passed"]]
    min_pass = {}
    for row in passing:
        key = (row["dataset"], row["baseline"], row["method"])
        if key not in min_pass or row["adaptation_rows"] < min_pass[key]["adaptation_rows"]:
            min_pass[key] = row

    out_md = RESEARCH / "sscard_budget_sweep.md"
    lines = [
        "# SSCard Label-Budget Sweep",
        "",
        "This sweep lowers the labeled target adaptation budget for already promising SSCard artifacts. It is isolated from the existing Drift-Signature Residual Field Transport implementation and paper artifacts.",
        "",
        "## Minimum Passing Budgets",
        "",
        "| dataset | baseline | method | min labels | gate | p90 reduction | p95 reduction | p99 reduction | p95 vs global | p99 vs global |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(min_pass.values(), key=lambda item: (item["adaptation_rows"], item["dataset"], item["baseline"])):
        lines.append(
            "| {dataset} | {baseline} | {method} | {labels} | {gate} | {p90} | {p95} | {p99} | {p95g} | {p99g} |".format(
                dataset=row["dataset"],
                baseline=row["baseline"],
                method=row["method"],
                labels=row["adaptation_rows"],
                gate=f"{row['passed_seed_count']}/{row['seed_count']}",
                p90=pct(row["p90_reduction"]),
                p95=pct(row["p95_reduction"]),
                p99=pct(row["p99_reduction"]),
                p95g=pct(row["p95_vs_global"]),
                p99g=pct(row["p99_vs_global"]),
            )
        )
    lines.extend(
        [
            "",
            "## All Swept Budgets",
            "",
            "| dataset | baseline | method | labels | gate | p90 reduction | p95 reduction | p99 reduction | fail reasons |",
            "|---|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        gate = (
            f"{row['passed_seed_count']}/{row['seed_count']}"
            if row["passed"]
            else f"fail ({row['passed_seed_count']}/{row['seed_count']})"
        )
        lines.append(
            "| {dataset} | {baseline} | {method} | {labels} | {gate} | {p90} | {p95} | {p99} | {fails} |".format(
                dataset=row["dataset"],
                baseline=row["baseline"],
                method=row["method"],
                labels=row["adaptation_rows"],
                gate=gate,
                p90=pct(row["p90_reduction"]),
                p95=pct(row["p95_reduction"]),
                p99=pct(row["p99_reduction"]),
                fails=row["fail_reasons"],
            )
        )
    lines.extend(
        [
            "",
            "## Takeaway",
            "",
            "- `WIKI / MO / isotonic` is the strongest ultra-low-label point found: it still passes with only 10 labeled target queries and improves p95/p99 by 61.2%/77.3%. `WIKI / MO / random-forest` also passes at 10 labels, with smaller p95/p99 gains.",
            "- `DBLP_AN / LBS / ridge` and `DBLP_AN / SSCard / random-forest` still pass with only 25 labeled target queries.",
            "- `DBLP_AN / LPLM / ridge`, `DBLP_AN / SSCard / ridge`, and `DBLP_AN / SSCard / isotonic` still pass with 50 labeled target queries.",
            "- `DBLP_AN / DREAM / ridge` still passes at 250 labels but not 100 labels.",
            "- `WIKI / LPLM / random-forest` and `WIKI / DREAM / ridge` need 1,000 labels for 3/3 strict-gate stability in this sweep.",
            "- `WIKI / Astrid / tail-gated q60` needs the full 2,000-label setting from the main SSCard report; lower budgets become seed-sensitive.",
            "",
            f"- Summary CSV: `{out_csv.relative_to(ROOT)}`",
        ]
    )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
