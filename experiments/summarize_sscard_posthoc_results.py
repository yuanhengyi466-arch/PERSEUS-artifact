import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


REPORTS = [
    ROOT / "research_plan" / "sscard_posthoc_DBLP_AN_SSCard_random_forest_2000.json",
    ROOT / "research_plan" / "sscard_posthoc_DBLP_AN_LPLM_ridge_2000.json",
    ROOT / "research_plan" / "sscard_posthoc_DBLP_AN_LBS_ridge_2000.json",
    ROOT / "research_plan" / "sscard_posthoc_DBLP_AN_DREAM_ridge_2000.json",
    ROOT / "research_plan" / "sscard_posthoc_WIKI_LPLM_random_forest_2000.json",
    ROOT / "research_plan" / "sscard_posthoc_WIKI_MO_random_forest_2000.json",
    ROOT / "research_plan" / "sscard_posthoc_WIKI_Astrid_tail_gated_q60_2000.json",
    ROOT / "research_plan" / "sscard_posthoc_WIKI_DREAM_ridge_2000.json",
    ROOT / "research_plan" / "sscard_posthoc_IMDB_AN_SSCard_random_forest_2000.json",
    ROOT / "research_plan" / "sscard_posthoc_IMDB_MT_SSCard_random_forest_2000.json",
    ROOT / "research_plan" / "sscard_posthoc_IMDB_AN_SSCard_tail_gated_2000.json",
    ROOT / "research_plan" / "sscard_posthoc_IMDB_MT_SSCard_tail_gated_2000.json",
    ROOT / "research_plan" / "sscard_posthoc_DBLP_AN_MO_random_forest_2000.json",
    ROOT / "research_plan" / "sscard_posthoc_DBLP_AN_LBS_random_forest_2000.json",
    ROOT / "research_plan" / "sscard_posthoc_DBLP_AN_LBS_isotonic_2000_probe.json",
    ROOT / "research_plan" / "sscard_posthoc_DBLP_AN_LBS_hist_gradient_2000_probe.json",
    ROOT / "research_plan" / "sscard_posthoc_DBLP_AN_DREAM_random_forest_2000.json",
    ROOT / "research_plan" / "sscard_posthoc_DBLP_AN_DREAM_isotonic_2000_probe.json",
    ROOT / "research_plan" / "sscard_posthoc_WIKI_SSCard_random_forest_2000_max50000_probe.json",
    ROOT / "research_plan" / "sscard_posthoc_TPCH_PN_SSCard_random_forest_100_probe.json",
    ROOT / "research_plan" / "sscard_posthoc_WIKI_DREAM_random_forest_2000_max50000_probe.json",
    ROOT / "research_plan" / "sscard_posthoc_DBLP_AN_Astrid_random_forest_2000_probe.json",
    ROOT / "research_plan" / "sscard_posthoc_DBLP_AN_LPLM_random_forest_2000_probe.json",
    ROOT / "research_plan" / "sscard_posthoc_DBLP_AN_LPLM_isotonic_2000_probe.json",
    ROOT / "research_plan" / "sscard_posthoc_DBLP_AN_LPLM_hist_gradient_2000_probe.json",
    ROOT / "research_plan" / "sscard_posthoc_DBLP_AN_LPLM_tail_gated_q70_2000_probe.json",
    ROOT / "research_plan" / "sscard_posthoc_DBLP_AN_LPLM_tail_gated_q80_2000_probe.json",
    ROOT / "research_plan" / "sscard_posthoc_DBLP_AN_LPLM_tail_gated_q90_2000_probe.json",
    ROOT / "research_plan" / "sscard_posthoc_WIKI_Astrid_random_forest_2000_probe.json",
    ROOT / "research_plan" / "sscard_posthoc_WIKI_Astrid_ridge_2000_probe.json",
    ROOT / "research_plan" / "sscard_posthoc_WIKI_Astrid_hist_gradient_2000_probe.json",
    ROOT / "research_plan" / "sscard_posthoc_WIKI_Astrid_isotonic_2000_probe.json",
    ROOT / "research_plan" / "sscard_posthoc_WIKI_Astrid_q50_2000_probe.json",
    ROOT / "research_plan" / "sscard_posthoc_WIKI_Astrid_q55_2000_probe.json",
    ROOT / "research_plan" / "sscard_posthoc_WIKI_Astrid_q75_2000_probe.json",
    ROOT / "research_plan" / "sscard_posthoc_WIKI_Astrid_tail_gated_q70_2000_probe.json",
    ROOT / "research_plan" / "sscard_posthoc_WIKI_Astrid_tail_gated_q80_2000_probe.json",
    ROOT / "research_plan" / "sscard_posthoc_WIKI_Astrid_tail_gated_q90_2000_probe.json",
    ROOT / "research_plan" / "sscard_posthoc_WIKI_LBS_random_forest_2000_probe.json",
    ROOT / "research_plan" / "sscard_posthoc_WIKI_DREAM_isotonic_2000_probe.json",
    ROOT / "research_plan" / "sscard_posthoc_IMDB_AN_LPLM_random_forest_2000_probe.json",
    ROOT / "research_plan" / "sscard_posthoc_IMDB_MT_LPLM_random_forest_2000_probe.json",
    ROOT / "research_plan" / "sscard_posthoc_IMDB_AN_Astrid_random_forest_2000_probe.json",
    ROOT / "research_plan" / "sscard_posthoc_IMDB_MT_Astrid_random_forest_2000_probe.json",
    ROOT / "research_plan" / "sscard_posthoc_IMDB_AN_DREAM_random_forest_2000_probe.json",
    ROOT / "research_plan" / "sscard_posthoc_IMDB_MT_DREAM_random_forest_2000_probe.json",
    ROOT / "research_plan" / "sscard_posthoc_IMDB_AN_LBS_random_forest_2000_probe.json",
    ROOT / "research_plan" / "sscard_posthoc_IMDB_MT_LBS_random_forest_2000_probe.json",
    ROOT / "research_plan" / "sscard_posthoc_IMDB_AN_MO_random_forest_2000_probe.json",
    ROOT / "research_plan" / "sscard_posthoc_IMDB_MT_MO_random_forest_2000_probe.json",
]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def display_method(report: dict, first_split: dict) -> str:
    method = report["method"]
    method_model = first_split.get("method_model", {})
    if method == "tail-gated" and "tail_gate_quantile" in method_model:
        return f"{method} q{int(round(100.0 * method_model['tail_gate_quantile']))}"
    return method


def main() -> None:
    rows = []
    for path in REPORTS:
        report = read_json(path)
        first = report["split_reports"][0]
        baseline_q = first["baseline_qerror"]
        rows.append(
            {
                "dataset": report["dataset"],
                "baseline": report["baseline"],
                "method": display_method(report, first),
                "rows": report["row_count"],
                "adaptation_rows": report["requires_labeled_target_adaptation_rows"],
                "passed": report["passed"],
                "passed_seed_count": report["passed_seed_count"],
                "seed_count": report["seed_count"],
                "baseline_median": baseline_q["median"],
                "baseline_p90": baseline_q["p90"],
                "baseline_p95": baseline_q["p95"],
                "baseline_p99": baseline_q["p99"],
                "median_reduction": report["aggregate_reductions"]["median"],
                "p90_reduction": report["aggregate_reductions"]["p90"],
                "p95_reduction": report["aggregate_reductions"]["p95"],
                "p99_reduction": report["aggregate_reductions"]["p99"],
                "mean_reduction": report["aggregate_reductions"]["mean"],
                "p95_vs_global": report["aggregate_reductions_vs_global_calibration"]["p95"],
                "p99_vs_global": report["aggregate_reductions_vs_global_calibration"]["p99"],
                "report_path": str(path),
            }
        )

    out_csv = ROOT / "research_plan" / "sscard_posthoc_summary.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    out_md = ROOT / "research_plan" / "sscard_posthoc_candidate.md"
    lines = [
        "# SSCard Post-hoc Calibration Candidate",
        "",
        "This is a separate baseline-expansion track using the cloned SSCard repository and its shipped result CSVs. It does not modify the existing Drift-Signature Residual Field Transport implementation or paper artifacts.",
        "",
        "## Summary",
        "",
        "| dataset | baseline | method | gate | p90 reduction | p95 reduction | p99 reduction | p95 vs global |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        gate = f"{row['passed_seed_count']}/{row['seed_count']}" if row["passed"] else f"fail ({row['passed_seed_count']}/{row['seed_count']})"
        lines.append(
            "| {dataset} | {baseline} | {method} | {gate} | {p90} | {p95} | {p99} | {p95g} |".format(
                dataset=row["dataset"],
                baseline=row["baseline"],
                method=row["method"],
                gate=gate,
                p90=pct(row["p90_reduction"]),
                p95=pct(row["p95_reduction"]),
                p99=pct(row["p99_reduction"]),
                p95g=pct(row["p95_vs_global"]),
            )
        )
    lines.extend(
        [
            "",
            "## Main Takeaway",
            "",
            "- `DBLP_AN / SSCard / random-forest` is the clean pass: 3/3 splits, 28.4% p90 reduction, 35.6% p95 reduction, and 39.2% p99 reduction with 2,000 labeled adaptation queries.",
            "- `DBLP_AN / LPLM / ridge` is a new clean pass: 3/3 splits, 48.9% p90 reduction, 49.8% p95 reduction, and 50.8% p99 reduction with 2,000 labeled adaptation queries; it also beats median global calibration by 15.3% at p95 and 15.9% at p99.",
            "- `DBLP_AN / LBS / ridge` is another strong pass: 3/3 splits, 54.7% p90 reduction, 53.1% p95 reduction, and 51.7% p99 reduction.",
            "- `DBLP_AN / DREAM / ridge` is a smaller but stable pass: 3/3 splits, 12.4% p90 reduction, 13.4% p95 reduction, and 11.8% p99 reduction.",
            "- `WIKI / LPLM / random-forest` is a strong full-row pass after robust WIKI CSV parsing: 3/3 splits, 92.4% p90 reduction, 93.0% p95 reduction, and 90.0% p99 reduction with 2,000 labeled adaptation queries.",
            "- `WIKI / MO / random-forest` is another full-row pass: 3/3 splits, 54.6% p90 reduction, 71.3% p95 reduction, and 94.0% p99 reduction with 2,000 labeled adaptation queries.",
            "- `WIKI / Astrid / tail-gated q60` converts the large-tail-gain Astrid case into a median-safe pass: 3/3 splits, 40.4% p90 reduction, 57.1% p95 reduction, and 64.3% p99 reduction.",
            "- `WIKI / DREAM / ridge` is also a stable pass: 3/3 splits, 19.1% p90 reduction, 16.6% p95 reduction, and 12.3% p99 reduction.",
            "- `DBLP_AN / LPLM / random-forest` and isotonic are near misses; both improve baseline tail strongly, but one seed does not clear the p99-vs-global guard. Ridge is the stable variant on this artifact.",
            "- `WIKI / Astrid` non-gated calibrators still fail the median guard, and q50/q55/q65/q75 show the expected threshold tradeoff; q60 is the strongest strict-gate operating point found.",
            "- IMDB and WIKI LBS/MO-style baselines often improve tails but fail because their baseline median is already 1.0 or near 1.0 and calibration worsens the median.",
            "- WIKI SSCard, WIKI DREAM, DBLP Astrid, TPCH_PN, and some competitor baselines improve partly or have little headroom, but fail the strict gate because median worsens, p99 does not move enough, or global calibration is already competitive.",
            "",
            "## Evidence",
            "",
            f"- Summary CSV: `{out_csv.relative_to(ROOT)}`",
            "- Gate reports:",
        ]
    )
    for row in rows:
        lines.append(f"  - `{Path(row['report_path']).relative_to(ROOT)}`")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
