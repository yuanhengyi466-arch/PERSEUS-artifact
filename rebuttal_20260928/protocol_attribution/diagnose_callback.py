"""Post-timing planner diagnostics; no runtime outcomes are used for selection."""
from __future__ import annotations

import argparse
from collections import Counter
import time

from run_protocol_attribution import (
    OUT, MODELS, activate, check_prepared, connect, explain, hash_file,
    read_json, write_csv, write_json,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args()
    check_prepared()
    cases = [c for c in read_json(OUT / "cases.json") if c["role"] == ("validation" if args.sample else "test")]
    if args.sample:
        cases = [next(c for c in cases if c["outer"] == 401 and c["drift"] == d) for d in ("column", "table", "value")]
    else:
        assert (OUT / "test_run.json").exists(), "Full diagnostics run only after all timings finish"
    rows = []
    conn = connect()
    started = time.perf_counter()
    with conn.cursor() as cur:
        for case in cases:
            activate(cur, case, "pg")
            default = explain(cur, case)
            for method in ("raw", *MODELS):
                activate(cur, case, method)
                stats = explain(cur, case)
                expected = max(1., case["raw"] if method == "raw" else case["estimates"][method])
                observed = stats["root_input_estimate"]
                matches = observed is not None and abs(observed-expected) <= max(1., expected*1e-6)
                rows.append({"outer": case["outer"], "drift": case["drift"], "qid": case["qid"],
                             "sql_id": case["sql_id"], "method": method, "expected_root": expected,
                             "observed_root": observed, "root_matches": matches,
                             "candidate_plan_changed": stats["plan_shape_sha256"] != default["plan_shape_sha256"],
                             "default_plan_sha256": default["plan_shape_sha256"],
                             "candidate_plan_sha256": stats["plan_shape_sha256"],
                             "max_cardinality": stats["max_estimated_cardinality"]})
    conn.close()
    stem = "callback_smoke" if args.sample else "callback_diagnostics"
    write_csv(OUT / (stem + ".csv"), rows)
    summary = {"cases": len(cases), "checks": len(rows), "root_matches": sum(r["root_matches"] for r in rows),
               "plan_changes": dict(Counter(r["method"] for r in rows if r["candidate_plan_changed"])),
               "seconds": time.perf_counter()-started, "map_sha256": hash_file(OUT / "estimate_map.csv"),
               "outside_timed_experiment": True}
    write_json(OUT / (stem + ".json"), summary)
    print(summary)
    if not all(r["root_matches"] for r in rows):
        raise RuntimeError("Some injected root estimates did not match; inspect diagnostic rows before making reachability claims")


if __name__ == "__main__":
    main()
