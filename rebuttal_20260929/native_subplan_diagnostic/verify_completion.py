"""Audit diagnostic population, coverage, measurements and saved hashes."""
from collections import Counter
from pathlib import Path

from run_diagnostic import OUT, ROOT, PRIOR, METHODS, PATHS, REPEATS, native, read_json, write_json, verify_prepared


def main():
    cases = verify_prepared()
    runner_hash = native.hash_file(OUT / "run_diagnostic.py")
    for filename in ("started.json", "measurement_started.json", "profile_started.json"):
        assert read_json(OUT / filename)["runner_sha256"] == runner_hash
    source = [c for c in read_json(PRIOR / "cases.json") if c["outer"] == 401 and c["role"] == "validation"]
    assert {c["qid"] for c in cases} == {c["qid"] for c in source}
    assert len({c["sql_id"] for c in cases}) == 180
    assert Counter(c["drift"] for c in cases) == {"column": 60, "table": 60, "value": 60}
    byid = {c["qid"]: c for c in cases}
    acquired = native.csv_rows(OUT / "count_acquisition.csv")
    labels = {r["subquery_id"]: r for r in acquired}
    assert len(labels) == len(acquired)
    mapping = {(r["query_id"], r["method"], r["aliases"]): float(r["estimate"])
               for r in native.csv_rows(OUT / "estimate_map.csv")}
    subplans = native.csv_rows(OUT / "subplans.csv")
    for c in cases:
        complete = all(labels[s]["count_status"] == "ok" for s in c["subquery_ids"])
        assert complete == c["oracle_complete"]
        assert c["root_label_matches"]
    for r in subplans:
        key = r["qid"], "oracle_join", r["aliases"]
        if r["count_status"] == "ok":
            assert mapping[key] == max(1., float(r["actual"]))
        else:
            assert key not in mapping
    for filename, expected_count in (("requests.csv", 4320), ("warmup.csv", 720)):
        rows = native.csv_rows(OUT / filename)
        assert len(rows) == expected_count
        keys = []
        for r in rows:
            c = byid[r["qid"]]
            active = "pg" if r["method"] == "oracle_join" and not c["oracle_complete"] else r["method"]
            assert r["active"] == active
            if r["status"] == "ok":
                assert int(r["result_count"]) == int(c["truth"])
            if filename == "requests.csv":
                keys.append((r["path"], r["method"], r["qid"], int(r["repetition"])))
        if keys:
            assert len(set(keys)) == 4320
            assert set(keys) == {(p,m,c["qid"],r) for p in PATHS for m in METHODS for c in cases for r in range(REPEATS)}
    root_checks = native.csv_rows(OUT / "root_checks.csv")
    assert len(root_checks) == 720 and all(r["matches"] == "True" for r in root_checks)
    profiles = native.csv_rows(OUT / "profiles.csv")
    assert len(profiles) == len({(r["qid"],r["method"]) for r in profiles}) == 720
    analysis = read_json(OUT / "analysis.json")
    assert analysis["analysis_sha256"] == native.hash_file(OUT / "analyze_diagnostic.py")
    for name, expected in analysis["output_hashes"].items():
        assert native.hash_file(OUT / name) == expected
    original = ROOT / "rebuttal_20260928/submission_original/PERSEUS-paper/PERSEUS"
    hashes = {"main_kdd2027.pdf": "44fd3345505823434994b332bc255cc33fe07a1495970d5defcce9a29062862a",
              "main_kdd2027.tex": "8e9a374baf8c301996bc1b66452c9eb41a62f23308e3a19cd9c462433f021e3c"}
    for name, expected in hashes.items():
        assert native.hash_file(original / name) == expected
    write_json(OUT / "completion_audit.json", {"status": "passed", "population_preserved": True,
        "unique_sql": 180, "oracle_complete_cases": sum(c["oracle_complete"] for c in cases),
        "timed_requests": 4320, "warmup_requests": 720, "profile_requests": 720,
        "count_statuses": dict(Counter(r["count_status"] for r in acquired)),
        "timed_statuses": analysis["timed_statuses"], "original_submission_unchanged": True,
        "protocol_sha256": native.hash_file(OUT / "PROTOCOL.md"),
        "script_sha256": {p.name: native.hash_file(p) for p in OUT.glob("*.py")}})
    print("Audit passed: 180 fixed SQL, all cases/conditions/repetitions retained, hashes and original submission verified.")


if __name__ == "__main__":
    main()
