"""Verify the completed run and summarize the exact native test population."""
from __future__ import annotations

import csv
import hashlib
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


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    prepared = read_json("prepared.json")
    for path, expected in prepared["input_hashes"].items():
        assert sha256(path) == expected, path
    for name, key in (("cases.json", "cases_hash"), ("models.joblib", "models_hash"),
                      ("estimate_map.csv", "map_hash")):
        assert sha256(OUT / name) == prepared[key], name

    role_sets = defaultdict(lambda: defaultdict(set))
    for row in read_csv("sql_membership.csv"):
        role_sets[row["outer"]][row["role"]].add(row["sql_id"])
    for outer, roles in role_sets.items():
        names = sorted(roles)
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                assert not roles[left] & roles[right], (outer, left, right)

    cases = read_json("cases.json")
    statuses, failures = {}, []
    for stage, expected_count in (("validation", 6480), ("test", 16200)):
        run = read_json(stage + "_run.json")
        assert sha256(OUT / (stage + "_timings.csv")) == run["timings_sha256"]
        rows = read_csv(stage + "_timings.csv")
        assert len(rows) == expected_count
        stage_cases = {c["qid"]: c for c in cases if c["role"] == stage}
        expected_keys = {(qid, policy, repeat) for qid in stage_cases
                         for policy in run["policies"] for repeat in range(run["repeats"])}
        actual_keys = {(r["qid"], r["policy"], int(r["repeat"])) for r in rows}
        assert len(actual_keys) == len(rows) and actual_keys == expected_keys
        for row in rows:
            case = stage_cases[row["qid"]]
            assert row["sql_id"] == case["sql_id"]
            assert case["sql_id"] in role_sets[str(case["outer"])][stage]
            if row["status"] == "ok":
                assert int(row["result_count"]) == int(case["truth"])
            else:
                failures.append({k: row[k] for k in ("qid", "policy", "repeat", "status", "error")})
        statuses[stage] = dict(Counter(r["status"] for r in rows))

    selected = read_json("selected_rules.json")
    assert selected["validation_sha256"] == sha256(OUT / "validation_timings.csv")
    assert read_json("test_run.json")["selection_sha256"] == sha256(OUT / "selected_rules.json")
    diagnostics = read_json("callback_diagnostics.json")
    assert diagnostics["map_sha256"] == prepared["map_hash"]
    assert diagnostics["checks"] == diagnostics["root_matches"] == 1620

    native_errors = []
    for outer in prepared["config"]["outer_seeds"]:
        for drift in ("column", "table", "value"):
            group = [c for c in cases if c["role"] == "test" and c["outer"] == outer and c["drift"] == drift]
            truth = np.array([c["truth"] for c in group])
            raw = np.maximum(1., [c["raw"] for c in group])
            raw_q = np.maximum(raw / truth, truth / raw)
            for model in ("hgb", "rf"):
                pred = np.maximum(1., [c["estimates"][model] for c in group])
                q = np.maximum(pred / truth, truth / pred)
                row = {"outer": outer, "drift": drift, "model": model, "native_test_cases": len(group)}
                for pct in (50, 95, 99):
                    rq, pq = float(np.percentile(raw_q, pct)), float(np.percentile(q, pct))
                    row.update({f"raw_p{pct}": rq, f"repair_p{pct}": pq, f"reduction_p{pct}": 1 - pq / rq})
                native_errors.append(row)
    with (OUT / "native_subset_qerror.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(native_errors[0]))
        writer.writeheader()
        writer.writerows(native_errors)

    ledger = read_csv("validation_cost_ledger.csv")
    work = {mode: {
        "candidate_executions": sum(int(r["candidate_validation_executions_required"]) for r in ledger if r["mode"] == mode),
        "observed_candidate_request_ms": sum(float(r["candidate_validation_measured_ms_required"]) for r in ledger if r["mode"] == mode),
    } for mode in ("runtime", "full")}
    report = {
        "verified": True,
        "input_files_hash_checked": len(prepared["input_hashes"]),
        "timed_validation_requests": 6480,
        "timed_test_requests": 16200,
        "timed_total_requests": 22680,
        "status_counts": statuses,
        "non_ok_records": failures,
        "zero_within_outer_sql_role_overlap": True,
        "test_split_query_cases": 540,
        "unique_test_sql": len({c["sql_id"] for c in cases if c["role"] == "test"}),
        "root_checks_passed": diagnostics["root_matches"],
        "candidate_plan_changes": diagnostics["plan_changes"],
        "selected_policies": len(selected["policies"]),
        "selected_global_fallbacks": sum(v["rule"] is None for v in selected["policies"].values()),
        "candidate_validation_accounting": work,
        "candidate_validation_time_fraction_omitted": 1 - work["full"]["observed_candidate_request_ms"] / work["runtime"]["observed_candidate_request_ms"],
        "cost_scope": "Counterfactual requirements from shared measurements; excludes label acquisition and other offline work.",
        "additional_analysis": "Native-subset q-error is descriptive and computed after completion; no policies were changed.",
    }
    (OUT / "completion_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
