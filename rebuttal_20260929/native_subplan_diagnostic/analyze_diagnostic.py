"""Paired development-set summaries; never selects a deployment rule."""
from collections import Counter, defaultdict
import math

import numpy as np

from run_diagnostic import OUT, METHODS, PATHS, REPEATS, native, read_json, write_json, write_csv, verify_prepared


def quantile_summary(values):
    x = np.asarray(values, dtype=float)
    return {"median": float(np.median(x)), "p95": float(np.percentile(x, 95)),
            "p99": float(np.percentile(x, 99)), "max": float(np.max(x))}


def qerror(pred, actual):
    p, t = max(1., pred), max(1., actual)
    return max(p / t, t / p)


def main():
    cases = verify_prepared()
    byid = {c["qid"]: c for c in cases}
    measured, profiled = read_json(OUT / "measured.json"), read_json(OUT / "profiled.json")
    assert measured["requests_sha256"] == native.hash_file(OUT / "requests.csv")
    for name, expected in profiled["outputs"].items():
        assert native.hash_file(OUT / name) == expected
    requests = native.csv_rows(OUT / "requests.csv")
    profiles = native.csv_rows(OUT / "profiles.csv")
    groups = defaultdict(list)
    for row in requests:
        groups[row["path"], row["method"], row["qid"]].append(row)
        if row["status"] == "ok":
            assert int(row["result_count"]) == int(byid[row["qid"]]["truth"])
    assert len(requests) == 4320 and len(profiles) == 720
    per_query, lookup = [], {}
    for path in PATHS:
        for method in METHODS:
            for case in cases:
                rows = groups[path, method, case["qid"]]
                assert len(rows) == REPEATS and {int(r["repetition"]) for r in rows} == set(range(REPEATS))
                rec = {"path": path, "method": method, "qid": case["qid"], "sql_id": case["sql_id"],
                       "drift": case["drift"], "tables": case["tables"], "oracle_complete": case["oracle_complete"],
                       "failed_repetitions": sum(r["status"] != "ok" for r in rows)}
                for field in ("wall_ms", "inference_ms", "activation_ms", "candidate_plan_ms", "select_ms"):
                    rec[field] = float(np.median([float(r[field]) for r in rows]))
                per_query.append(rec)
                lookup[path, method, case["qid"]] = rec
    write_csv(OUT / "per_query.csv", per_query)
    rng = np.random.default_rng(2026092907)
    selections = {"all": cases, "oracle_complete": [c for c in cases if c["oracle_complete"]]}
    for drift in ("column", "table", "value"):
        selections[drift] = [c for c in cases if c["drift"] == drift]
    for size in sorted({c["tables"] for c in cases}):
        selections[f"tables_{size}"] = [c for c in cases if c["tables"] == size]
    summaries = []
    for population, selected in selections.items():
        if not selected:
            continue
        weights = rng.multinomial(len(selected), np.full(len(selected), 1 / len(selected)), size=2000)
        for path in PATHS:
            baseline = np.array([lookup[path, "pg", c["qid"]]["wall_ms"] for c in selected])
            for method in METHODS:
                values = np.array([lookup[path, method, c["qid"]]["wall_ms"] for c in selected])
                ratio = values / baseline
                interval = np.percentile((weights @ values) / (weights @ baseline), [2.5, 97.5])
                summaries.append({"population": population, "path": path, "method": method, "cases": len(selected),
                    "oracle_complete_cases": sum(c["oracle_complete"] for c in selected),
                    "base_total_ms": float(baseline.sum()), "method_total_ms": float(values.sum()),
                    "total_ratio": float(values.sum()/baseline.sum()), "total_ratio_ci_low": float(interval[0]),
                    "total_ratio_ci_high": float(interval[1]), "saved_ms": float(baseline.sum()-values.sum()),
                    "win_fraction": float(np.mean(ratio<1)),
                    **{f"ratio_{k}": v for k,v in quantile_summary(ratio).items()},
                    **{f"latency_{k}_ms": v for k,v in quantile_summary(values).items()},
                    "ratio_of_latency_p95": float(np.percentile(values,95)/np.percentile(baseline,95)),
                    "failed_repetitions": sum(lookup[path,method,c["qid"]]["failed_repetitions"] for c in selected)})
    write_csv(OUT / "request_summary.csv", summaries)

    subplans = native.csv_rows(OUT / "subplans.csv")
    errors = []
    for row in subplans:
        if row["count_status"] != "ok":
            continue
        case = byid[row["qid"]]
        actual, pg, hgb = (float(row[k]) for k in ("actual", "pg_rows", "hgb_rows"))
        wanted = math.log(max(1.,actual)/max(1.,pg))
        applied = math.log(case["estimates"]["hgb"]/max(1.,case["raw"]))
        pg_root = math.log(max(1.,case["root_actual"])/max(1.,case["root_pg_rows"])) if case["root_actual"] is not None else None
        source_root = math.log(max(1.,case["root_actual"])/max(1.,case["raw"])) if case["root_actual"] is not None else None
        errors.append({"qid": row["qid"], "drift": row["drift"], "tables": int(row["tables"]),
            "is_root": row["is_root"] == "True", "subquery_id": row["subquery_id"],
            "pg_qerror": qerror(pg,actual), "hgb_qerror": qerror(hgb,actual),
            "hgb_worsens_qerror": qerror(hgb,actual) > qerror(pg,actual),
            "needed_log_ratio": wanted, "applied_root_log_ratio": applied,
            "root_hgb_direction_disagrees": wanted * applied < -1e-12,
            "effective_map_direction_disagrees": wanted * math.log(max(1.,hgb)/max(1.,pg)) < -1e-12,
            "pg_root_direction_disagrees": wanted * pg_root < -1e-12 if pg_root is not None else None,
            "mscn_root_direction_disagrees": wanted * source_root < -1e-12 if source_root is not None else None})
    write_csv(OUT / "subplan_error_detail.csv", errors)
    error_summaries = []
    for kind in ("root", "intermediate"):
        for drift in ("all", "column", "table", "value"):
            selected = [r for r in errors if r["is_root"] == (kind == "root") and (drift == "all" or r["drift"] == drift)]
            if not selected:
                continue
            error_summaries.append({"kind": kind, "drift": drift, "case_subplan_records": len(selected),
                "unique_subqueries": len({r["subquery_id"] for r in selected}),
                "hgb_worsens_fraction": float(np.mean([r["hgb_worsens_qerror"] for r in selected])),
                "opposite_hgb_direction_fraction": float(np.mean([r["root_hgb_direction_disagrees"] for r in selected])) if kind == "intermediate" else None,
                **{f"{m}_{k}": v for m in ("pg", "hgb") for k,v in quantile_summary([r[m+"_qerror"] for r in selected]).items()}})
    write_csv(OUT / "subplan_error_summary.csv", error_summaries)

    finish(cases, requests, profiles, per_query, summaries, error_summaries)


def finish(cases, requests, profiles, per_query, summaries, error_summaries):
    profile_lookup = {(r["qid"], r["method"]): r for r in profiles}
    profile_summaries = []
    for method in METHODS:
        pairs = [(profile_lookup[c["qid"], "pg"], profile_lookup[c["qid"], method]) for c in cases]
        complete = [(a,b) for a,b in pairs if a["status"] == b["status"] == "ok"]
        profile_summaries.append({"method": method, "attempted": len(cases), "paired_complete": len(complete),
            "method_failures": sum(b["status"] != "ok" for a,b in pairs),
            "plan_shape_changes": sum(a["plan_shape_sha256"] != b["plan_shape_sha256"] for a,b in complete),
            "paired_pg_planning_ms": sum(float(a["planning_ms"]) for a,b in complete),
            "paired_method_planning_ms": sum(float(b["planning_ms"]) for a,b in complete),
            "paired_pg_execution_ms": sum(float(a["execution_ms"]) for a,b in complete),
            "paired_method_execution_ms": sum(float(b["execution_ms"]) for a,b in complete)})
    write_csv(OUT / "profile_summary.csv", profile_summaries)
    phases = []
    for path in PATHS:
        for method in METHODS:
            rows = [r for r in per_query if r["path"] == path and r["method"] == method]
            phases.append({"path": path, "method": method, "cases": len(rows),
                **{f"median_{f}": float(np.median([r[f] for r in rows]))
                   for f in ("inference_ms", "activation_ms", "candidate_plan_ms", "select_ms", "wall_ms")}})
    write_csv(OUT / "phase_summary.csv", phases)

    # Only physical join nodes are compared; parameterized-path context is explicit.
    byid = {c["qid"]: c for c in cases}
    maps = {(r["query_id"], r["method"], r["aliases"]): float(r["estimate"]) for r in native.csv_rows(OUT / "estimate_map.csv")}
    join_checks = []
    for entry in read_json(OUT / "plans.json"):
        qid, method = entry["qid"], entry["method"]
        case = byid[qid]
        active = "pg" if method == "oracle_join" and not case["oracle_complete"] else method
        def visit(node, parameterized_context=False):
            aliases = {node["Alias"]} if "Alias" in node else set()
            for i, child in enumerate(node.get("Plans", [])):
                aliases |= visit(child, parameterized_context or (node.get("Node Type") == "Nested Loop" and i == 1))
            if node.get("Node Type") in ("Hash Join", "Merge Join", "Nested Loop"):
                key = " ".join(sorted(aliases))
                expected = maps.get((qid, active, key))
                observed = node.get("Plan Rows")
                join_checks.append({"qid": qid, "method": method, "active": active, "aliases": key,
                    "node_type": node["Node Type"], "parameterized_context": parameterized_context,
                    "expected_map_rows": expected, "observed_path_rows": observed,
                    "matches": abs(observed-expected) <= max(1.,abs(expected)*1e-6) if expected is not None and observed is not None else None})
            return aliases
        visit(entry["plan"]["Plan"])
    write_csv(OUT / "chosen_join_checks.csv", join_checks)

    acquisition = native.csv_rows(OUT / "count_acquisition.csv")
    warmup = native.csv_rows(OUT / "warmup.csv")
    checks = native.csv_rows(OUT / "root_checks.csv")
    assert len(warmup) == len(checks) == 720 and all(r["matches"] == "True" for r in checks)
    summary = {"population": "outer401 runtime-validation, development only", "cases": len(cases),
        "unique_sql": len({c["sql_id"] for c in cases}), "oracle_complete_cases": sum(c["oracle_complete"] for c in cases),
        "fresh_unique_count_attempts": len(acquisition), "count_statuses": dict(Counter(r["count_status"] for r in acquisition)),
        "count_acquisition_ms": sum(float(r["count_ms"]) for r in acquisition),
        "standalone_explain_acquisition_ms": sum(float(r["estimate_ms"]) for r in acquisition),
        "timed_requests": len(requests), "timed_statuses": dict(Counter(r["status"] for r in requests)),
        "warmup_statuses": dict(Counter(r["status"] for r in warmup)), "root_checks_passed": len(checks),
        "profile_statuses": dict(Counter(r["status"] for r in profiles)), "summaries": summaries,
        "chosen_join_comparisons": sum(r["matches"] is not None for r in join_checks),
        "chosen_join_matches": sum(r["matches"] is True for r in join_checks),
        "analysis_sha256": native.hash_file(__file__),
        "output_hashes": {p.name: native.hash_file(p) for p in OUT.glob("*.csv")}}
    write_json(OUT / "analysis.json", summary)
    lines = ["# Native Subplan Diagnostic Results", "", "## Scope", "",
        "Development diagnostic on all 180 outer-401 validation queries, with 60 per drift. No new admission policy is selected. Oracle labels, source predictions and maps are offline inputs; oracle results are not deployable-method performance.", "",
        f"Fresh COUNT attempts: {len(acquisition)}; statuses: {summary['count_statuses']}. Complete oracle parents: {summary['oracle_complete_cases']}/180. Incomplete parents explicitly fall back to PG. Label acquisition took {summary['count_acquisition_ms']/1000:.3f} measured seconds; standalone EXPLAIN acquisition took {summary['standalone_explain_acquisition_ms']/1000:.3f} seconds.", "",
        f"Timed SELECT requests: {len(requests)}; statuses: {summary['timed_statuses']}. Root checks: {len(checks)}/{len(checks)} matched. Separate EXPLAIN ANALYZE statuses: {summary['profile_statuses']}.", "",
        "## Paired Request Results", "",
        "Each SQL contributes its median of three measurements. Total ratio compares sums of those medians; intervals use 2,000 paired query bootstrap samples. P95 ratio is the percentile of query-wise ratios, not a ratio of latency percentiles. Timeout/error observations remain in the primary summaries and must not be interpreted as successful completion times.", "",
        "| Population | Path | Method | Cases | Total ratio [conditional 95% CI] | Median ratio | P95 ratio | Failed repeats |",
        "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in summaries:
        if r["population"] not in ("all", "oracle_complete"):
            continue
        lines.append(f"| {r['population']} | {r['path']} | {r['method']} | {r['cases']} | {r['total_ratio']:.3f} [{r['total_ratio_ci_low']:.3f}, {r['total_ratio_ci_high']:.3f}] | {r['ratio_median']:.3f} | {r['ratio_p95']:.3f} | {r['failed_repetitions']} |")
    lines += ["", "`mapped` includes activation and SELECT completion. `candidate_path` adds HGB inference where applicable and a candidate EXPLAIN for each non-PG condition. PG and incomplete-oracle fallback use the native path. No measured path includes offline labeling or source MSCN inference.", "",
        "## Subplan Error and Transfer", "",
        "The following are label-complete case/subplan records, which can reuse a canonical subquery. Acquisition failures are excluded from error quantiles only and remain in the acquisition ledger. Direction disagreement compares the HGB full-query multiplier with the correction needed by native PG on the intermediate join.", "",
        "| Kind | Drift | Records | PG median / P95 | HGB median / P95 | HGB worsens | Opposite HGB direction |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for r in error_summaries:
        direction = f"{r['opposite_hgb_direction_fraction']:.1%}" if r['opposite_hgb_direction_fraction'] is not None else "n/a (root uses a separate estimate)"
        lines.append(f"| {r['kind']} | {r['drift']} | {r['case_subplan_records']} | {r['pg_median']:.2f} / {r['pg_p95']:.2f} | {r['hgb_median']:.2f} / {r['hgb_p95']:.2f} | {r['hgb_worsens_fraction']:.1%} | {direction} |")
    lines += ["", "## Separate Server Profiles", "",
        "One untimed profiler execution per case/condition; complete matched profiles only. These sums are diagnostic and are not substituted for the randomized direct SELECT measurements.", "",
        "| Method | Paired complete | Plan changes | PG / method planning ms | PG / method execution ms | Method failures |",
        "|---|---:|---:|---:|---:|---:|"]
    for r in profile_summaries:
        lines.append(f"| {r['method']} | {r['paired_complete']} | {r['plan_shape_changes']} | {r['paired_pg_planning_ms']:.1f} / {r['paired_method_planning_ms']:.1f} | {r['paired_pg_execution_ms']:.1f} / {r['paired_method_execution_ms']:.1f} | {r['method_failures']} |")
    lines += ["", "## Interpretation Boundaries", "",
        "- No test set or deployment rule is selected here; previously inspected validation queries remain development data.",
        "- The oracle corrects connected joins and root only, with zero counts floored at 1; base estimates, parameterized-path costing, and other cost-model assumptions remain native.",
        "- pg_map uses standalone estimates in the parent query, so changed plans are an observation rather than an implementation error by definition.",
        "- Conditional intervals describe query composition under the fixed diagnostic; three timing repeats are not independent workloads and no multiplicity correction is applied.",
        "- Shared subqueries and incomplete-oracle selection are recorded. Subplan error summaries are descriptive and do not treat repeated subplans as independent experimental replications.",
        "- Opposite error directions suggest a transfer problem; causal latency attribution additionally requires the oracle, pg_map, and timing-path controls.",
        "- Profile planning/execution times and phase medians use different measurements; phase medians need not sum to the median total.",
        "- All 180 cases, all four conditions, errors/timeouts and fallback remain visible. No favorable-query-only main summary is used.",
        "- A useful oracle result would motivate a separate learned-repair study. It cannot establish practical label cost or held-out policy safety."]
    (OUT / "RESULTS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    native.emit("analysis_complete", cases=180, timed_requests=4320)


if __name__ == "__main__":
    main()
