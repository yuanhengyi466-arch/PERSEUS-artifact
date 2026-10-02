"""Audit saved supplement inputs, selections, memberships and native requests."""
from collections import defaultdict
from importlib.metadata import version
import math
from pathlib import Path

from run_estimator_controls import OUT, ROOT, FOLDERS, SEEDS, FAMILIES, file_hash, read_json, read_csv, write_json


def main():
    verified = 0
    for drift in FOLDERS:
        base = OUT / drift
        done = read_json(base / "completed.json")
        assert done["protocol_hash"] == file_hash(OUT / "PROTOCOL.md")
        assert done["runner_hash"] == file_hash(OUT / "run_estimator_controls.py")
        assert done["within_drift_source_target_sql_overlap"] == 0
        for name, expected in done["dependencies"].items():
            assert file_hash(ROOT / "experiments" / name) == expected
        for path, expected in done["input_hashes"].items():
            assert file_hash(path) == expected, path
            verified += 1
        for rel, expected in done["output_hashes"].items():
            assert file_hash(base / rel) == expected, rel
            verified += 1
        for seed in SEEDS:
            for family in FAMILIES:
                selected = read_json(base / "controls" / str(seed) / f"{family}_selection.json")
                assert len(selected["candidates"]) == 4
                def rank(row):
                    m = row["tuning"]
                    return (.5 * (math.log(m["p95"]) + math.log(m["p99"])), m["median"], row["config_id"])
                assert min(selected["candidates"], key=rank)["config_id"] == selected["selected_config_id"]

    roles = defaultdict(dict)
    for row in read_csv(OUT / "sql_membership.csv"):
        assigned = roles[row["outer"]].setdefault(row["sql_id"], row["role"])
        assert assigned == row["role"]
    analysis = read_json(OUT / "analysis.json")
    for name, expected in analysis["hashes"].items():
        assert file_hash(OUT / name) == expected

    directory = OUT / "throughput"
    done = read_json(directory / "completed.json")
    assert done["requests"] == 6480
    for name, expected in done["file_hashes"].items():
        assert file_hash(directory / name) == expected
    native = ROOT / "rebuttal_20260928/protocol_attribution"
    expected_hashes = {"protocol_sha256": OUT / "THROUGHPUT_PROTOCOL.md",
                       "runner_sha256": OUT / "run_throughput.py",
                       "native_runner_sha256": native / "run_protocol_attribution.py",
                       "native_selection_sha256": native / "selected_rules.json",
                       "cases_sha256": native / "cases.json"}
    for key, path in expected_hashes.items():
        assert done[key] == file_hash(path)
    cases = {c["qid"]: c for c in read_json(native / "cases.json") if c["role"] == "test"}
    blocks = defaultdict(list)
    rows = read_csv(directory / "requests.csv")
    for row in rows:
        key = (int(row["concurrency"]), int(row["repetition"]), row["policy"])
        blocks[key].append(row)
        assert row["active"] == "pg" and row["admitted"] == "False"
        assert row["sql_id"] == cases[row["qid"]]["sql_id"]
        if row["status"] == "ok":
            assert float(row["result_count"]) == cases[row["qid"]]["truth"]
    assert len(blocks) == 12
    for block in blocks.values():
        assert len(block) == len(cases) == 540
        assert {r["qid"] for r in block} == set(cases)
        assert {int(r["position"]) for r in block} == set(range(540))
    for concurrency in (1, 4):
        for repetition in range(3):
            order = lambda policy: [r["qid"] for r in sorted(blocks[concurrency, repetition, policy], key=lambda r: int(r["position"]))]
            assert order("pg") == order("full:hgb")
    for summary in read_csv(directory / "blocks.csv"):
        key = (int(summary["concurrency"]), int(summary["repetition"]), summary["policy"])
        good = sum(r["status"] == "ok" for r in blocks[key])
        assert int(summary["completed_correctly"]) == good
        assert math.isclose(float(summary["goodput_qps"]), good / float(summary["elapsed_seconds"]), rel_tol=1e-12)
    warmup = read_csv(directory / "warmup.csv")
    assert len(warmup) == 540 and {r["qid"] for r in warmup} == set(cases)
    for row in warmup:
        if row["status"] == "ok":
            assert float(row["result_count"]) == cases[row["qid"]]["truth"]

    submitted = ROOT / "rebuttal_20260928/submission_original/PERSEUS-paper/PERSEUS"
    originals = {"main_kdd2027.pdf": "44fd3345505823434994b332bc255cc33fe07a1495970d5defcce9a29062862a",
                 "main_kdd2027.tex": "8e9a374baf8c301996bc1b66452c9eb41a62f23308e3a19cd9c462433f021e3c"}
    for name, expected in originals.items():
        assert file_hash(submitted / name) == expected
    write_json(OUT / "completion_audit.json", {"status": "passed", "training_selections": 54,
        "configurations_scored": 216, "verified_training_input_output_files": verified,
        "within_outer_sql_role_conflicts": 0, "timed_requests": len(rows),
        "correct": sum(r["status"] == "ok" for r in rows),
        "timeouts": sum(r["status"] == "timeout" for r in rows),
        "warmup_requests": len(warmup), "warmup_correct": sum(r["status"] == "ok" for r in warmup),
        "all_native_test_cases_preserved_each_block": True, "paired_order_verified": True,
        "original_submission_sha256": originals,
        "packages": {name: version(name) for name in ("numpy", "scipy", "scikit-learn", "torch", "joblib", "psutil", "psycopg")},
        "script_sha256": {p.name: file_hash(p) for p in OUT.glob("*.py")}})
    print("Completion audit passed: 54 selections, 216 configs, 6480 timed requests, original submission unchanged.")


if __name__ == "__main__":
    main()
