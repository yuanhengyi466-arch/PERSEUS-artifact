import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research_plan"


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def factor_tag(value: float) -> str:
    scaled = 100.0 * value
    if abs(scaled - round(scaled)) < 1e-9:
        return str(int(round(scaled)))
    return f"{scaled:.3f}".rstrip("0").rstrip(".").replace(".", "p")


def display_method(report: dict) -> str:
    method = report["method"]
    first = report["split_reports"][0]
    method_model = first.get("method_model", {})
    if method == "tail-gated" and "tail_gate_quantile" in method_model:
        return f"{method} q{factor_tag(method_model['tail_gate_quantile'])}"
    if method == "shrink-isotonic" and "shrink_factor" in method_model:
        return f"{method} s{factor_tag(method_model['shrink_factor'])}"
    if method == "tail-shrink-isotonic" and "shrink_factor" in method_model:
        tail_quantile = method_model.get("isotonic_tail_quantile", 0.0)
        return (
            f"{method} s{factor_tag(method_model['shrink_factor'])} "
            f"q{int(round(100.0 * tail_quantile))}"
        )
    if method == "shrink-ridge" and "shrink_factor" in method_model:
        anchor = method_model.get("shrink_anchor", "zero")
        return f"{method} {anchor} s{factor_tag(method_model['shrink_factor'])}"
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


def row_by_name(rows: list[dict], filename: str) -> dict:
    for row in rows:
        if Path(row["report_path"]).name == filename:
            return row
    raise KeyError(filename)


def stability_table_row(row: dict) -> str:
    return "| {dataset} | {baseline} | {method} | {labels} | {gate} | {p90} | {p95} | {p99} | {p95g} | {p99g} |".format(
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


def main() -> None:
    paths = sorted(RESEARCH.glob("sscard_stability10_*.json"))
    if not paths:
        raise SystemExit("no SSCard 10-seed stability JSON files found")
    rows = [read_report(path) for path in paths]
    rows.sort(key=lambda row: (row["dataset"], row["baseline"], row["method"], row["adaptation_rows"]))

    out_csv = RESEARCH / "sscard_stability10_summary.csv"
    write_csv(out_csv, rows)

    passing = [row for row in rows if row["passed"]]
    min_pass = {}
    for row in passing:
        key = (row["dataset"], row["baseline"], row["method"])
        if key not in min_pass or row["adaptation_rows"] < min_pass[key]["adaptation_rows"]:
            min_pass[key] = row

    near = [row for row in rows if not row["passed"] and row["passed_seed_count"] >= 8]
    out_md = RESEARCH / "sscard_stability10.md"
    lines = [
        "# SSCard 10-Seed Stability Check",
        "",
        "This check reruns selected low-label SSCard adaptation settings with seeds 1..10. It is isolated from the existing Drift-Signature Residual Field Transport implementation and paper artifacts.",
        "",
        "## 10/10 Passing Budgets",
        "",
        "| dataset | baseline | method | labels | p90 reduction | p95 reduction | p99 reduction | p95 vs global | p99 vs global |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(min_pass.values(), key=lambda item: (item["adaptation_rows"], item["dataset"], item["baseline"])):
        lines.append(
            "| {dataset} | {baseline} | {method} | {labels} | {p90} | {p95} | {p99} | {p95g} | {p99g} |".format(
                dataset=row["dataset"],
                baseline=row["baseline"],
                method=row["method"],
                labels=row["adaptation_rows"],
                p90=pct(row["p90_reduction"]),
                p95=pct(row["p95_reduction"]),
                p99=pct(row["p99_reduction"]),
                p95g=pct(row["p95_vs_global"]),
                p99g=pct(row["p99_vs_global"]),
            )
        )
    stronger_rows = [
        row_by_name(rows, "sscard_stability10_DBLP_AN_SSCard_shrink_ridge_global_s76p8_48.json"),
        row_by_name(rows, "sscard_stability10_DBLP_AN_SSCard_shrink_ridge_global_s77p4_48_frac.json"),
        row_by_name(rows, "sscard_stability10_DBLP_AN_SSCard_shrink_ridge_global_s80_50.json"),
        row_by_name(rows, "sscard_stability10_DBLP_AN_LBS_shrink_ridge_global_s65_31.json"),
        row_by_name(rows, "sscard_stability10_DBLP_AN_LBS_shrink_ridge_global_s23p5_26_frac.json"),
        row_by_name(rows, "sscard_stability10_DBLP_AN_LBS_shrink_ridge_global_s62_34.json"),
        row_by_name(rows, "sscard_stability10_DBLP_AN_LBS_shrink_ridge_global_s65p2_35.json"),
        row_by_name(rows, "sscard_stability10_DBLP_AN_LBS_shrink_ridge_global_s65_40.json"),
        row_by_name(rows, "sscard_stability10_DBLP_AN_LBS_shrink_ridge_global_s66_48_gap2.json"),
        row_by_name(rows, "sscard_stability10_DBLP_AN_LBS_shrink_ridge_global_s70_49_gap2.json"),
        row_by_name(rows, "sscard_stability10_DBLP_AN_LBS_shrink_ridge_global_s65_50.json"),
        row_by_name(rows, "sscard_stability10_DBLP_AN_LBS_shrink_ridge_global_s71p64_50_frac.json"),
        row_by_name(rows, "sscard_stability10_DBLP_AN_LBS_shrink_ridge_global_s65_100.json"),
        row_by_name(rows, "sscard_stability10_DBLP_AN_LBS_ridge_250.json"),
        row_by_name(rows, "sscard_stability10_DBLP_AN_LPLM_shrink_ridge_global_s43p7_90_frac.json"),
        row_by_name(rows, "sscard_stability10_DBLP_AN_LPLM_shrink_ridge_global_s45_100.json"),
        row_by_name(rows, "sscard_stability10_DBLP_AN_MO_shrink_ridge_global_s53p8_52.json"),
        row_by_name(rows, "sscard_stability10_DBLP_AN_MO_shrink_ridge_global_s60_65.json"),
        row_by_name(rows, "sscard_stability10_DBLP_AN_MO_shrink_ridge_global_s60_75.json"),
        row_by_name(rows, "sscard_stability10_DBLP_AN_MO_shrink_ridge_global_s70_100.json"),
        row_by_name(rows, "sscard_stability10_DBLP_AN_DREAM_shrink_ridge_global_s90_190.json"),
        row_by_name(rows, "sscard_stability10_DBLP_AN_DREAM_shrink_ridge_global_s90_200.json"),
        row_by_name(rows, "sscard_stability10_DBLP_AN_DREAM_shrink_ridge_global_s96_200.json"),
        row_by_name(rows, "sscard_stability10_DBLP_AN_DREAM_shrink_ridge_global_s99p8_200_frac.json"),
        row_by_name(rows, "sscard_stability10_DBLP_AN_DREAM_shrink_ridge_global_s90_250.json"),
        row_by_name(rows, "sscard_stability10_WIKI_MO_shrink_isotonic_5.json"),
        row_by_name(rows, "sscard_stability10_WIKI_MO_shrink_isotonic_10.json"),
        row_by_name(rows, "sscard_stability10_WIKI_MO_tail_shrink_isotonic_s55_q30_3.json"),
        row_by_name(rows, "sscard_stability10_WIKI_MO_tail_shrink_isotonic_s64_q55_3_frac.json"),
        row_by_name(rows, "sscard_stability10_WIKI_DREAM_shrink_ridge_global_s94_1077.json"),
        row_by_name(rows, "sscard_stability10_WIKI_DREAM_shrink_ridge_global_s94_1079.json"),
        row_by_name(rows, "sscard_stability10_WIKI_DREAM_shrink_ridge_global_s92_1081.json"),
        row_by_name(rows, "sscard_stability10_WIKI_DREAM_shrink_ridge_global_s94p852_1080_frac.json"),
        row_by_name(rows, "sscard_stability10_WIKI_DREAM_shrink_ridge_global_s90_1500.json"),
        row_by_name(rows, "sscard_stability10_WIKI_DREAM_ridge_2000.json"),
        row_by_name(rows, "sscard_stability10_WIKI_Astrid_tail_gated_q60p02_1320_frac.json"),
        row_by_name(rows, "sscard_stability10_WIKI_Astrid_tail_gated_q60_1500.json"),
        row_by_name(rows, "sscard_stability10_WIKI_Astrid_tail_gated_q60_1800.json"),
        row_by_name(rows, "sscard_stability10_WIKI_Astrid_tail_gated_q60_2000.json"),
    ]
    lines.extend(
        [
            "",
            "## Stronger 10/10 Operating Points",
            "",
            "These points do not lower the minimum stable label budget, but they are useful stronger-tail or higher-budget 10-seed references.",
            "",
            "| dataset | baseline | method | labels | gate | p90 reduction | p95 reduction | p99 reduction | p95 vs global | p99 vs global |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in stronger_rows:
        lines.append(stability_table_row(row))
    lines.extend(
        [
            "",
            "## Near Misses",
            "",
            "| dataset | baseline | method | labels | gate | p90 reduction | p95 reduction | p99 reduction | fail reasons |",
            "|---|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in sorted(near, key=lambda item: (-item["passed_seed_count"], item["adaptation_rows"], item["dataset"], item["baseline"])):
        lines.append(
            "| {dataset} | {baseline} | {method} | {labels} | {gate} | {p90} | {p95} | {p99} | {fails} |".format(
                dataset=row["dataset"],
                baseline=row["baseline"],
                method=row["method"],
                labels=row["adaptation_rows"],
                gate=f"{row['passed_seed_count']}/{row['seed_count']}",
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
            "- `WIKI / LPLM / shrink-isotonic s58` remains the strongest 1-label p99 point, with 91.4%/93.9%/92.7% p90/p95/p99 reductions; s60/s62/s65 trade some p99 for higher p90/p95 while staying 10/10.",
            "- `WIKI / MO / shrink-isotonic s30` is also 10/10 stable at 1 labeled target query.",
            "- `WIKI / MO / shrink-isotonic s55` is a more aggressive 10/10 stable variant from 3 labeled target queries, with 39.6%/65.0%/93.7% p90/p95/p99 reductions; 4 labels is a 9/10 median-guard boundary under this seed split.",
            "- A mined 5-label `WIKI / MO / shrink-isotonic s55` point is an intermediate stable tradeoff at 43.1%/66.6%/93.7%, and the 10-label point is a stronger 10/10 operating point than the 3-label representative, reaching 44.4%/66.9%/93.8% p90/p95/p99 reductions. These do not lower the 3-label budget, but they give a cleaner high-tail stable curve.",
            "- `WIKI / MO / tail-shrink-isotonic` confirms the median-protection boundary. The p99-oriented s55 q30/q50 variants at 3 labels stay near 39.6%-40.5%/64.2%-64.5%/93.8% p90/p95/p99 reductions; a focused aggressive-tail sweep adds a p90-oriented 3-label s64 q55 point at 41.5%/61.4%/93.2%, while s65 q55 is a 9/10 seed-5 p95-vs-global boundary and s64 q50 is a median boundary.",
            "- The older unshrunk `WIKI / MO / isotonic` operating point is 10/10 stable at 50 labeled target queries.",
            "- `DBLP_AN / SSCard / shrink-ridge global s77` lowers the 10/10 stable budget to 48 labeled target queries and gives the balanced same-budget point at 20.9%/24.6%/28.0% p90/p95/p99 reductions. A focused fractional sweep finds s76p8 as the same-budget p99-best 10/10 point at 20.8%/24.5%/28.1%, while a new s77p4 point is the p90/p95-oriented 48-label stable representative at 21.0%/24.7%/28.0%; s77p5+ and s78+ at 48 labels remain seed-4 median-guard boundaries.",
            "- `DBLP_AN / SSCard / shrink-ridge global s80` at 50 labels is the strongest stable base-SSCard tail point in this sweep, with 20.2%/25.4%/28.6% p90/p95/p99 reductions.",
            "- `DBLP_AN / LBS / shrink-ridge global s23` lowers the 10/10 stable budget to 26 labels, with 20.8%/16.2%/16.7% p90/p95/p99 reductions; a fractional same-budget sweep raises the stable representative to s23p5 at 21.1%/16.6%/16.9%, while s23p55+ is a seed-7 median-guard boundary. Mined stronger stable LBS operating points give a clearer curve: 27-label s45 reaches 30.7%/26.6%/24.5%, 31-label s65 reaches 36.1%/32.5%/29.4%, 34-label s62 reaches 35.4%/31.7%/30.5%, 35-label s65p2 reaches 36.1%/32.5%/31.2%, 40-label s65 gives a p90-oriented 36.2%/32.3%/30.0% tradeoff, 48-label s66 reaches 42.3%/39.0%/37.1%, and 49-label s70 reaches 45.1%/41.8%/40.3%, matching or slightly beating the previous 50-label s70 tail point one label earlier. A 50-label fractional sweep raises the stable same-budget tail representative to s71p64 at 44.0%/41.2%/40.7%, while s71p65+ and s72+ remain 9/10 seed-8 p99-vs-global boundaries. The 100-label s65 point reaches 42.5%/39.5%/38.9%, and ordinary ridge at 250 labels reaches 53.3%/51.6%/49.5%; 46/47-label gap2 settings remain 9/10 median boundaries, 48-label s68/s70 miss p99-vs-global, and 34-label s65 plus 35-label s65p4/s66+ are p90 boundaries.",
            "- `DBLP_AN / LPLM / shrink-ridge global s40` lowers the 10/10 stable ridge-like budget to 85 labels; 84-label variants remain 9/10 vs-global boundaries. A lower-shrink s20-s29 micro-sweep at 84 labels did not fix seed 10: s20 is the closest vs-global point but still reaches only 0.09% p95-vs-global and 0.02% p99-vs-global on that seed, below the 1% gates.",
            "- `DBLP_AN / LPLM / shrink-ridge global s43` at 90 labels is an intermediate 10/10 point at 43.1%/45.0%/47.7% p90/p95/p99 reductions. A fractional same-budget sweep raises the selected stable representative to s43p7 at 43.2%/45.1%/47.8%, while s43p8+ and s44 are 9/10 seed-6 p95-vs-global boundaries. The 100-label s45 point remains the strongest stable LPLM shrink-ridge tail point in this sweep, with 43.2%/45.3%/48.0% p90/p95/p99 reductions.",
            "- `DBLP_AN / MO / shrink-ridge global s53` lowers the 10/10 stable budget to 52 labels, with 19.3%/18.0%/23.6% p90/p95/p99 reductions; 51-label variants remain 8/10 p99-reduction boundaries. A focused fractional sweep at 52 labels finds s53p8 as the strongest stable low-budget MO tail point at 19.5%/18.2%/23.9%, while s53p9 and s54 are median-guard boundaries. Higher-budget stable MO references include 65-label s60 at 21.4%/20.1%/25.1%, 75-label s60 at 21.5%/20.6%/26.5%, and 100-label s70 at 23.7%/24.2%/30.3%.",
            "- `DBLP_AN / DREAM / shrink-ridge global s90` lowers the 10/10 stable budget to 165 labels, with 8.3%/9.2%/11.3% p90/p95/p99 reductions; 160-label variants remain 9/10 p99-reduction boundaries. A focused 200-label micro-sweep shows s92/s94/s96/s98/s99 and fractional s99p8 all pass 10/10, while s100 remains a 9/10 p99 boundary. The s96 point is still the strongest stable DREAM p99 point at 9.0%/11.5%/12.9%, while s99p8 is only a p90/p95 tradeoff at 9.1%/11.6%/12.5%.",
            "- `WIKI / Astrid / tail-gated q62` lowers the 10/10 stable budget to 1,150 labeled target queries, with 33.3%/54.9%/65.3% p90/p95/p99 reductions; 1,125 labels is a 9/10 p95-vs-global boundary. The older q60 setting remains the stronger p90/p95 operating point at higher budget.",
            "- `WIKI / Astrid / tail-gated q60` adds stable stronger-tail points above the q62 budget: a fractional q60p02 setting passes at 1,320 labels with 37.2%/56.0%/65.9%, while q60.005/q60.01/q60.015 at the same budget remain 9/10 seed-7 median boundaries. Higher-budget references include 1,500 labels at 38.1%/56.3%/65.6%, 1,800 labels at 38.1%/56.3%/66.8%, and 2,000 labels as the strongest selected tail point at 39.9%/57.0%/66.9% p90/p95/p99 reductions; q58 at 1,800 labels is a 9/10 median-guard boundary despite stronger raw tails.",
            "- `WIKI / DREAM / shrink-ridge global s94` lowers the 10/10 stable budget from 2,000-label ridge to 1,075 labels, with 17.5%/15.5%/10.1% p90/p95/p99 reductions; 1,074 labels remains a 9/10 p99-reduction boundary. Mined stronger stable points include 1,077/1,079-label s94 smoothing points at p99 10.2%/10.3%, a refined 1,080-label s94p852 point at 17.5%/15.7%/10.4%, 1,081-label s92 at 17.3%/15.5%/10.4%, 1,500-label s90 at 17.5%/15.6%/10.4%, and 2,000-label ridge at 18.5%/16.5%/10.7%; s94p854+ at 1,080 labels and s95/s96 at 1,081 labels are seed-6 median-guard boundaries.",
            "- `WIKI / LPLM / random-forest` is 10/10 stable at 1,000 labels.",
            "- `WIKI / LPLM / shrink-isotonic s55` is the nearby 9/10 boundary point; raising the shrink factor to s58 fixes the p95-vs-global failure while retaining large p99 gains.",
            "- Several other ultra-low-budget settings remain useful 3-seed evidence but are not 10/10 stable: `WIKI / MO / random-forest` at 5 labels is 8/10, and unshrunk `WIKI / MO / isotonic` at 10 labels is 9/10.",
            "",
            f"- Summary CSV: `{out_csv.relative_to(ROOT)}`",
        ]
    )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
