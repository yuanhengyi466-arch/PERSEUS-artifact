"""Paired, SQL-cluster-aware summaries of the frozen protocol ablation."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent


def read_json(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def read_csv(name):
    with (OUT / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(name, rows):
    with (OUT / name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def interval(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return [None, None]
    return np.percentile(values, [2.5, 97.5]).tolist()


def statistics(times, base, admitted, failures):
    ratios = times / base
    sub = ratios[admitted]
    worst = int(np.argmax(ratios))
    return {
        "cases": len(times), "admitted": int(admitted.sum()), "coverage": float(admitted.mean()),
        "base_total_ms": float(base.sum()), "policy_total_ms": float(times.sum()),
        "total_ratio": float(times.sum()/base.sum()), "saved_ms": float(base.sum()-times.sum()),
        "ratio_median": float(np.median(ratios)), "ratio_p95": float(np.percentile(ratios, 95)),
        "ratio_p99": float(np.percentile(ratios, 99)),
        "latency_p50_ms": float(np.percentile(times, 50)), "latency_p95_ms": float(np.percentile(times, 95)),
        "latency_p99_ms": float(np.percentile(times, 99)),
        "ratio_of_latency_p50": float(np.percentile(times, 50)/np.percentile(base, 50)),
        "ratio_of_latency_p95": float(np.percentile(times, 95)/np.percentile(base, 95)),
        "ratio_of_latency_p99": float(np.percentile(times, 99)/np.percentile(base, 99)),
        "subset_median": float(np.median(sub)) if len(sub) else None,
        "subset_p95": float(np.percentile(sub, 95)) if len(sub) else None,
        "subset_max": float(np.max(sub)) if len(sub) else None,
        "max_ratio": float(ratios[worst]), "worst_ratio_policy_ms": float(times[worst]),
        "worst_ratio_base_ms": float(base[worst]), "max_absolute_increase_ms": float(np.max(times-base)),
        "failed_repetitions": int(failures.sum()),
    }


def weighted_quantile_samples(values, weights, quantile):
    if not len(values):
        return np.full(weights.shape[0], np.nan)
    order = np.argsort(values)
    vals, w = values[order], weights[:, order]
    cumulative = np.cumsum(w, axis=1)
    total = cumulative[:, -1]
    target = (total - 1) * quantile
    lo, hi = np.floor(target), np.ceil(target)
    ilo = (cumulative <= lo[:, None]).sum(axis=1)
    ihi = (cumulative <= hi[:, None]).sum(axis=1)
    ilo, ihi = np.minimum(ilo, len(vals)-1), np.minimum(ihi, len(vals)-1)
    output = vals[ilo] + (target-lo) * (vals[ihi]-vals[ilo])
    output[total == 0] = np.nan
    return output


def analyze():
    from run_protocol_attribution import check_prepared, hash_file, rank_key
    prepared = check_prepared()
    run = read_json("test_run.json")
    assert hash_file(OUT / "test_timings.csv") == run["timings_sha256"]
    assert hash_file(OUT / "selected_rules.json") == run["selection_sha256"]
    val_run = read_json("validation_run.json")
    selection = read_json("selected_rules.json")
    assert selection["validation_sha256"] == val_run["timings_sha256"]
    cases = [c for c in read_json("cases.json") if c["role"] == "test"]
    policies = run["policies"]
    rows = read_csv("test_timings.csv")
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["qid"], row["policy"])].append(row)
    case_rows = []
    for case in cases:
        for policy in policies:
            records = grouped[(case["qid"], policy)]
            assert len(records) == run["repeats"]
            actions = {r["admitted"] for r in records}
            active = {r["active"] for r in records}
            case_rows.append({
                "outer": case["outer"], "drift": case["drift"], "qid": case["qid"], "sql_id": case["sql_id"],
                "policy": policy, "median_wall_ms": float(np.median([float(r["wall_ms"]) for r in records])),
                "median_prefix_ms": float(np.median([float(r["prefix_ms"]) for r in records])),
                "admitted": actions == {"True"}, "stable_decision": len(actions) == 1 and len(active) == 1,
                "failed_repeats": sum(r["status"] != "ok" for r in records),
                "statuses": json.dumps(dict(Counter(r["status"] for r in records)), sort_keys=True),
            })
    assert all(r["stable_decision"] for r in case_rows), "Admission decision changed across timing repeats"
    write_csv("test_per_sql.csv", case_rows)
    matrix = {(r["qid"], r["policy"]): r for r in case_rows}
    base = np.array([matrix[(c["qid"], "pg")]["median_wall_ms"] for c in cases])
    frozen_base = np.array([matrix[(c["qid"], "raw")]["median_wall_ms"] for c in cases])
    cluster_ids = sorted({c["sql_id"] for c in cases})
    cluster_lookup = {sid: i for i, sid in enumerate(cluster_ids)}
    columns = np.array([cluster_lookup[c["sql_id"]] for c in cases])
    rng = np.random.default_rng(2026092807)
    n_cluster = len(cluster_ids)
    counts = rng.multinomial(n_cluster, np.full(n_cluster, 1/n_cluster), size=2000)
    weights = counts[:, columns]
    boot_base = weights @ base
    summary, strata, details = [], [], {}
    for policy in policies:
        values = [matrix[(c["qid"], policy)] for c in cases]
        times = np.array([v["median_wall_ms"] for v in values])
        admitted = np.array([v["admitted"] for v in values])
        failures = np.array([v["failed_repeats"] for v in values])
        result = {"policy": policy, **statistics(times, base, admitted, failures),
                  "unique_sql": n_cluster, "accepted_unique_sql": len({c["sql_id"] for c, a in zip(cases, admitted) if a}),
                  "total_ratio_vs_frozen_mscn_injection": float(times.sum()/frozen_base.sum()),
                  "sum_median_prefix_ms": float(sum(v["median_prefix_ms"] for v in values))}
        total_ci = interval((weights @ times) / boot_base)
        cover_ci = interval((weights @ admitted.astype(float)) / weights.sum(axis=1))
        med_ci = interval(weighted_quantile_samples((times/base)[admitted], weights[:, admitted], .5))
        p95_ci = interval(weighted_quantile_samples((times/base)[admitted], weights[:, admitted], .95))
        for field, ci in (("total_ratio", total_ci), ("coverage", cover_ci), ("subset_median", med_ci), ("subset_p95", p95_ci)):
            result[field + "_ci_low"], result[field + "_ci_high"] = ci
        summary.append(result)
        details[policy] = (times, admitted)
        for outer in prepared["config"]["outer_seeds"]:
            for drift in ("all", "column", "table", "value"):
                mask = np.array([c["outer"] == outer and (drift == "all" or c["drift"] == drift) for c in cases])
                strata.append({"outer": outer, "drift": drift, "policy": policy,
                               **statistics(times[mask], base[mask], admitted[mask], failures[mask])})
    comparisons = []
    for name in ("hgb", "rf"):
        full_t, full_a = details["full:"+name]
        runtime_t, runtime_a = details["runtime:"+name]
        ratio_ci = interval((weights @ full_t)/(weights @ runtime_t))
        comparisons.append({"model": name, "full_over_runtime_total_ratio": float(full_t.sum()/runtime_t.sum()),
                            "ci_low": ratio_ci[0], "ci_high": ratio_ci[1],
                            "admission_decisions_identical": bool(np.array_equal(full_a, runtime_a)),
                            "different_admission_cases": int(np.count_nonzero(full_a != runtime_a)),
                            "full_total_ms": float(full_t.sum()), "runtime_total_ms": float(runtime_t.sum())})
    write_csv("policy_summary.csv", summary)
    write_csv("policy_by_split_drift.csv", strata)
    write_csv("full_vs_runtime.csv", comparisons)
    memberships = read_csv("sql_membership.csv")
    fitting = read_csv("fit_gate_results.csv")
    validation_ledger = read_csv("validation_cost_ledger.csv")
    resource_rows = []
    for outer in prepared["config"]["outer_seeds"]:
        fit_sql, calibration_sql = set(), set()
        for drift in ("column", "table", "value"):
            pool = sorted((r["sql_id"] for r in memberships if int(r["outer"]) == outer and r["drift"] == drift and r["calibration_selected"] == "True"),
                          key=lambda sid: rank_key(sid, outer, "calibration"))
            order = np.random.default_rng(1).permutation(pool)
            fit_sql.update(order[:500 if drift == "value" else 2000])
            calibration_sql.update(pool)
        for model in ("hgb", "rf"):
            model_fits = [r for r in fitting if int(r["outer"]) == outer and r["model"] == model]
            first_replica_seconds = sum(float(r["fit_seconds"]) for r in model_fits if r["inner"] == "1")
            all_replica_seconds = sum(float(r["fit_seconds"]) for r in model_fits)
            for mode in ("repair", "estimator", "runtime", "full"):
                uses_gate = mode in ("estimator", "full")
                entry = next((r for r in validation_ledger if int(r["outer"]) == outer and r["model"] == model and r["mode"] == mode), None)
                resource_rows.append({"outer": outer, "model": model, "mode": mode,
                                      "required_unique_calibration_sql": len(calibration_sql if uses_gate else fit_sql),
                                      "deployed_replica_fit_unique_sql": len(fit_sql),
                                      "additional_gate_support_sql": len(calibration_sql-fit_sql) if uses_gate else 0,
                                      "required_fit_seconds": all_replica_seconds if uses_gate else first_replica_seconds,
                                      "required_candidate_validation_sql": int(entry["candidate_validation_sql_required"]) if entry else 0,
                                      "required_candidate_validation_executions": int(entry["candidate_validation_executions_required"]) if entry else 0,
                                      "observed_candidate_validation_ms_for_required_regions": float(entry["candidate_validation_measured_ms_required"]) if entry else 0,
                                      "label_acquisition_ms": "not_measured",
                                      "scope": "Accounting from shared experimental records; excludes gate computation, baseline validation runs, source prediction and label acquisition."})
    write_csv("policy_resource_accounting.csv", resource_rows)
    status_counts = dict(Counter(r["status"] for r in rows))
    result = {"type": "new Windows PG13 executions on existing benchmark with grouped resampling",
              "timed_test_executions": len(rows), "test_cases": len(cases), "unique_test_sql": n_cluster,
              "status_counts": status_counts, "zero_within_split_sql_overlap": prepared["zero_sql_overlap"],
              "pooled_intervals": "2,000 SQL-cluster bootstrap resamples, conditional on frozen selected rules; descriptive, not simultaneous inference or safety guarantees",
              "primary_endpoint": "sum of per-SQL median client request times / native PG reference",
              "full_vs_runtime": comparisons, "summary": summary,
              "hashes": {p.name: hash_file(p) for p in OUT.iterdir() if p.is_file() and p.suffix in (".csv", ".json", ".py") and p.name != "analysis.json"}}
    (OUT / "analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    def fmt(value, digits=3):
        return "--" if value is None else f"{value:.{digits}f}"
    lines = ["# Protocol Attribution Results", "", "## Material Passport", "",
             "- Origin Skill: academic-research-suite / experiment-agent", "- Origin Mode: run + validate",
             "- Origin Date: 2026-09-28", "- Verification Status: ANALYZED; new experiments completed, hashes and split invariants checked",
             "- Version Label: protocol_attribution_v1", "", "## Scope", "",
             f"Three SQL-grouped outer splits; HGB and RF residual models; three randomized timing repetitions. {len(cases)} test split-query cases, {n_cluster} different SQL identities, {len(rows)} measured test executions. Status counts: `{status_counts}`.",
             "", "The native reference is PostgreSQL default. Raw-MSCN injection is reported separately. Cached source-MSCN predictions and offline maps are common inputs; source inference, model fitting, map construction/refresh, and connection setup are not included in request latency. Online residual features/inference, policy work, session setup, required candidate planning, SQL execution and result receipt are included.",
             "", "## All Policies", "",
             "| Policy | Accepted cases | Total time ratio [95% conditional CI] | Saved ms | Subset median/P95 | Whole ratio P95 |",
             "|---|---:|---:|---:|---:|---:|"]
    for s in summary:
        lines.append(f"| {s['policy']} | {s['admitted']}/{s['cases']} | {s['total_ratio']:.3f} [{s['total_ratio_ci_low']:.3f}, {s['total_ratio_ci_high']:.3f}] | {s['saved_ms']:.1f} | {fmt(s['subset_median'])}/{fmt(s['subset_p95'])} | {s['ratio_p95']:.3f} |")
    lines += ["", "## Incremental Effect of the Estimator Gate", ""]
    for c in comparisons:
        lines.append(f"- {c['model'].upper()}: full/runtime total-time ratio {c['full_over_runtime_total_ratio']:.3f}, 95% conditional CI [{c['ci_low']:.3f}, {c['ci_high']:.3f}]; admission decisions differ on {c['different_admission_cases']} test cases.")
    lines += ["", "If admission decisions are identical, timing differences between repeated executions of those policies do not establish a benefit from the extra gate. Report the possible reduction in required validation work separately using `validation_cost_ledger.csv`.",
              "", "The extra estimator gate also consumes additional calibration labels and repeated fits. `policy_resource_accounting.csv` separates those requirements. Fewer runtime candidate executions are not by themselves evidence of lower total acquisition/deployment cost; label acquisition was not timed.",
              "", "## Statistical Audit (11/11 Considered)", "",
              "1. Aggregation reversal: per-split and per-drift results are supplied in policy_by_split_drift.csv; pooled results must be read with them.",
              "2. Ecological inference: the unit is SQL identity; a workload quantile does not guarantee individual safety.",
              "3. Selection bias: the eligible SQL population is the previously selected native subset, not all STATS queries.",
              "4. Collider conditioning: no new timing outcome filters eligibility; earlier executable-label filtering remains part of the scope.",
              "5. Base rates: accepted counts and all-query outcomes are both reported, with no subset-only headline.",
              "6. Regression to the mean: all policies have paired new measurements; old high-error selection limits generalization.",
              "7. Survivorship: errors/timeouts remain in outputs. If present, capped observations are not completed-query latency.",
              "8. Multiple comparisons: report all policies; bootstrap intervals are descriptive, not multiplicity-adjusted superiority tests.",
              "9. Researcher choices: the local protocol/configuration preceded timings; no claim of public preregistration or an untouched benchmark.",
              "10. Causality: randomized policy order supports within-host comparison; shared-host activity, offline map assumptions and resampling restrict generalization.",
              "11. Temporal direction: runtime rules use validation before test; selection hashes are checked and test is not used to retune.",
              "", "## Reproduce", "", "Run `python rebuttal_20260928/protocol_attribution/run_protocol_attribution.py analyze` from .. New DB phases refuse to overwrite existing observations. Raw timing, SQL membership, gate reports, maps and selected rules are preserved."]
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"test_cases": len(cases), "unique_sql": n_cluster, "status_counts": status_counts, "full_vs_runtime": comparisons}, indent=2))


if __name__ == "__main__":
    analyze()
