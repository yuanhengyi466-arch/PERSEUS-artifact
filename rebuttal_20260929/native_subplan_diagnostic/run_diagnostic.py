"""Windows-native connected-join oracle and injection-path diagnostic."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys
import time

import joblib
import numpy as np
import psutil

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
PRIOR = ROOT / "rebuttal_20260928/protocol_attribution"
sys.path.insert(0, str(PRIOR))
import run_protocol_attribution as native

METHODS = ("pg", "pg_map", "hgb_ratio", "oracle_join")
PATHS = ("mapped", "candidate_path")
REPEATS = 3
MAP = OUT / "estimate_map.csv"
read_json, write_json, write_csv = native.read_json, native.write_json, native.write_csv


def emit(stage, **values):
    native.emit(stage, **values)


def connect():
    conn = native.connect()
    conn.prepare_threshold = None
    with conn.cursor() as cur:
        cur.execute("SET application_name='perseus_subplan_diagnostic'")
    return conn


def actual_method(case, method):
    return "pg" if method == "oracle_join" and not case["oracle_complete"] else method


def activate(cur, case, method):
    active = actual_method(case, method)
    if active == "pg":
        cur.execute("SET perseus_estmap_enabled=off")
    else:
        cur.execute("SELECT set_config('perseus_estmap_fname',%s,false), "
                    "set_config('perseus_estmap_query_id',%s,false), "
                    "set_config('perseus_estmap_method',%s,false), "
                    "set_config('perseus_estmap_enabled','on',false)",
                    (MAP.as_posix(), case["qid"], active))
    return active


def subrow(row, subset):
    wanted = set(subset)
    tables = native.subplans.parse_aliases(row[0])
    joins = [j for j, left, right in native.subplans.parse_joins(row[1]) if left in wanted and right in wanted]
    preds, parts = [], row[2].split(",") if row[2] else []
    for i in range(0, len(parts), 3):
        if native.subplans.expr_alias(parts[i]) in wanted:
            preds.extend(parts[i:i + 3])
    return [",".join(f"{tables[a]} {a}" for a in sorted(wanted)), ",".join(joins), ",".join(preds), "0"]


def checkpoint(rows, cases, cache):
    write_csv(OUT / "subplans.csv", rows)
    write_csv(OUT / "count_acquisition.csv", list(cache.values()))
    write_json(OUT / "cases.json", cases)


def prepare():
    if (OUT / "started.json").exists():
        raise RuntimeError("Refusing to overwrite an acquisition; use a new declared version for new observations")
    native.check_prepared()
    cases = [c for c in read_json(PRIOR / "cases.json") if c["outer"] == 401 and c["role"] == "validation"]
    assert len(cases) == len({c["sql_id"] for c in cases}) == 180
    oldmap = {(r["query_id"], r["aliases"]): float(r["estimate"])
              for r in native.csv_rows(PRIOR / "estimate_map.csv") if r["method"] == "hgb"}
    source_files = [PRIOR / n for n in ("cases.json", "estimate_map.csv", "models.joblib", "run_protocol_attribution.py")]
    source_files += [ROOT / "baselines/End-to-End-CardEst-Benchmark/postgresql-13.1/src/backend/optimizer/path/costsize.c"]
    metadata = {"protocol_sha256": native.hash_file(OUT / "PROTOCOL.md"),
                "runner_sha256": native.hash_file(__file__), "cases": 180,
                "inputs": {str(p): native.hash_file(p) for p in source_files},
                "started_unix": time.time(), "command": sys.argv}
    write_json(OUT / "started.json", metadata)
    write_json(OUT / "population_before_acquisition.json", cases)
    conn = connect()
    rows, cache, maps = [], {}, []
    started = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute("SELECT version()")
        metadata["server_version"] = cur.fetchone()[0]
        cur.execute("SELECT name, setting FROM pg_settings WHERE name IN "
                    "('jit','max_parallel_workers_per_gather','statement_timeout','search_path','work_mem',"
                    "'random_page_cost','seq_page_cost','effective_cache_size','join_collapse_limit','geqo_threshold') ORDER BY name")
        metadata["settings"] = dict(cur.fetchall())
        cur.execute("SELECT c.relname,c.reltuples,c.relpages FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
                    "WHERE n.nspname='stats_inj' AND c.relkind='r' ORDER BY c.relname")
        metadata["table_statistics"] = cur.fetchall()
        write_json(OUT / "environment.json", metadata)
        for pos, case in enumerate(cases):
            cur.execute("SET perseus_estmap_enabled=off")
            aliases = tuple(sorted(native.subplans.parse_aliases(case["row"][0])))
            subsets = native.subplans.subplan_subsets(case["row"], 0) + [aliases]
            case["oracle_complete"] = True
            case["subquery_ids"] = []
            for subset in subsets:
                root = subset == aliases
                query = case["sql"] if root else native.subplans.subset_sql(case["row"], subset)
                sid = native.identity(subrow(case["row"], subset))
                if sid not in cache:
                    obs = {"subquery_id": sid, "sql": query, "tables": len(subset), "pg_rows": None,
                           "actual": None, "estimate_status": "pending", "count_status": "pending",
                           "estimate_ms": None, "count_ms": None, "error": ""}
                    tick = time.perf_counter()
                    try:
                        cur.execute("EXPLAIN (FORMAT JSON) " + query)
                        stat = native.injection.collect_plan_stats(native.injection.plan_doc(cur.fetchone()[0]))
                        obs["pg_rows"] = stat["root_input_estimate"]
                        obs["estimate_status"] = "ok" if obs["pg_rows"] is not None else "missing_rows"
                    except Exception as exc:
                        obs["estimate_status"] = "error"
                        obs["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
                    obs["estimate_ms"] = (time.perf_counter() - tick) * 1000
                    tick = time.perf_counter()
                    try:
                        cur.execute(query)
                        obs["actual"] = int(cur.fetchone()[0])
                        obs["count_status"] = "ok"
                    except Exception as exc:
                        obs["count_status"] = "timeout" if isinstance(exc, native.psycopg.errors.QueryCanceled) else "error"
                        obs["error"] += f" {type(exc).__name__}: {str(exc)[:200]}"
                    obs["count_ms"] = (time.perf_counter() - tick) * 1000
                    cache[sid] = obs
                obs = cache[sid]
                case["subquery_ids"].append(sid)
                case["oracle_complete"] &= obs["count_status"] == "ok"
                key = " ".join(subset)
                rows.append({"qid": case["qid"], "drift": case["drift"], "subquery_id": sid,
                             "aliases": key, "tables": len(subset), "is_root": root,
                             "pg_rows": obs["pg_rows"], "actual": obs["actual"], "hgb_rows": oldmap[case["qid"], key],
                             "count_status": obs["count_status"], "estimate_status": obs["estimate_status"]})
                if obs["estimate_status"] != "ok":
                    checkpoint(rows, cases, cache)
                    raise RuntimeError("Native estimate unavailable; retained partial acquisition for inspection")
                for method, value in (("pg_map", obs["pg_rows"]), ("hgb_ratio", oldmap[case["qid"], key])):
                    maps.append({"query_id": case["qid"], "aliases": key, "method": method, "estimate": max(1., value)})
                if obs["count_status"] == "ok":
                    maps.append({"query_id": case["qid"], "aliases": key, "method": "oracle_join", "estimate": max(1, obs["actual"])})
                if root:
                    case["root_actual"] = obs["actual"]
                    case["root_pg_rows"] = obs["pg_rows"]
                    case["root_label_matches"] = obs["count_status"] == "ok" and obs["actual"] == int(case["truth"])
            if (pos + 1) % 10 == 0:
                checkpoint(rows, cases, cache)
                emit("acquisition", completed=pos + 1, cases=len(cases), unique_subqueries=len(cache),
                     count_statuses=dict(Counter(r["count_status"] for r in cache.values())), elapsed_seconds=time.perf_counter()-started)
    conn.close()
    checkpoint(rows, cases, cache)
    write_csv(MAP, maps)
    if any(c["root_actual"] is not None and not c["root_label_matches"] for c in cases):
        raise RuntimeError("Root labels differ from the frozen benchmark; inspect before continuing")
    metadata.update({"seconds": time.perf_counter()-started, "unique_subqueries": len(cache),
                     "subplan_case_entries": len(rows), "oracle_complete_cases": sum(c["oracle_complete"] for c in cases),
                     "root_label_matches": sum(c["root_label_matches"] for c in cases),
                     "outputs": {p.name: native.hash_file(p) for p in (OUT / "cases.json", MAP, OUT / "subplans.csv", OUT / "count_acquisition.csv")}})
    write_json(OUT / "prepared.json", metadata)
    emit("prepared", **{k: metadata[k] for k in ("cases", "unique_subqueries", "oracle_complete_cases", "seconds")})


def verify_prepared():
    prepared = read_json(OUT / "prepared.json")
    assert prepared["protocol_sha256"] == native.hash_file(OUT / "PROTOCOL.md")
    for name, expected in prepared["outputs"].items():
        assert native.hash_file(OUT / name) == expected, name
    for name, expected in prepared["inputs"].items():
        assert native.hash_file(name) == expected, name
    return read_json(OUT / "cases.json")


def root_diagnostics(cases):
    rows = []
    conn = connect()
    with conn.cursor() as cur:
        for case in cases:
            for method in METHODS:
                active = activate(cur, case, method)
                stats = native.explain(cur, case)
                expected = (case["estimates"]["hgb"] if active == "hgb_ratio" else
                            case["root_actual"] if active == "oracle_join" else case["root_pg_rows"])
                observed = stats["root_input_estimate"]
                matches = observed is not None and abs(observed-max(1.,expected)) <= max(1.,abs(expected)*1e-6)
                rows.append({"qid": case["qid"], "method": method, "active": active, "expected": expected,
                             "observed": observed, "matches": matches, "plan_shape_sha256": stats["plan_shape_sha256"]})
    conn.close()
    write_csv(OUT / "root_checks.csv", rows)
    assert all(r["matches"] for r in rows), "Root injection check failed"
    emit("root_checks", checks=len(rows), matches=sum(r["matches"] for r in rows))


def request(cur, case, method, path, models):
    start = time.perf_counter()
    phases = {"inference_ms": 0., "activation_ms": 0., "candidate_plan_ms": 0., "select_ms": 0.}
    active, count, status, error = actual_method(case, method), None, "ok", ""
    phase = "activation_ms"
    tick = start
    try:
        if path == "candidate_path" and active == "hgb_ratio":
            phase = "inference_ms"
            native.infer(case, "hgb", models)
            phases[phase] = (time.perf_counter()-tick)*1000
        phase, tick = "activation_ms", time.perf_counter()
        activate(cur, case, method)
        phases[phase] = (time.perf_counter()-tick)*1000
        if path == "candidate_path" and active != "pg":
            phase, tick = "candidate_plan_ms", time.perf_counter()
            native.explain(cur, case)
            phases[phase] = (time.perf_counter()-tick)*1000
        phase, tick = "select_ms", time.perf_counter()
        cur.execute(case["sql"])
        count = int(cur.fetchone()[0])
        phases[phase] = (time.perf_counter()-tick)*1000
        if count != int(case["truth"]):
            status = "label_mismatch"
    except Exception as exc:
        phases[phase] = (time.perf_counter()-tick)*1000
        status = "timeout" if isinstance(exc, native.psycopg.errors.QueryCanceled) else "error"
        error = f"{type(exc).__name__}: {str(exc)[:200]}"
    return {"qid": case["qid"], "sql_id": case["sql_id"], "drift": case["drift"], "tables": case["tables"],
            "method": method, "path": path, "active": active, "oracle_complete": case["oracle_complete"],
            "status": status, "result_count": count, "wall_ms": (time.perf_counter()-start)*1000, **phases, "error": error}


def measure():
    cases = verify_prepared()
    if (OUT / "measurement_started.json").exists():
        raise RuntimeError("Refusing to overwrite existing measurement")
    write_json(OUT / "measurement_started.json", {"at_unix": time.time(), "runner_sha256": native.hash_file(__file__)})
    root_diagnostics(cases)
    models = joblib.load(PRIOR / "models.joblib")
    conn = connect()
    rows, warmup, blocks = [], [], []
    with conn.cursor() as cur:
        for case in cases:
            for method in METHODS:
                warmup.append(request(cur, case, method, "mapped", models))
        write_csv(OUT / "warmup.csv", warmup)
        emit("warmup_complete", requests=len(warmup), statuses=dict(Counter(r["status"] for r in warmup)))
        for repetition in range(REPEATS):
            paths = PATHS if repetition % 2 == 0 else tuple(reversed(PATHS))
            for path in paths:
                ordered = list(cases)
                random.Random(f"subplan-v1|{repetition}|{path}").shuffle(ordered)
                tick = time.perf_counter()
                psutil.cpu_percent()
                blockrows = []
                for pos, case in enumerate(ordered):
                    methods = list(METHODS)
                    random.Random(f"subplan-v1|{repetition}|{path}|{case['qid']}").shuffle(methods)
                    for method in methods:
                        row = request(cur, case, method, path, models)
                        row.update({"repetition": repetition, "case_position": pos, "method_position": methods.index(method)})
                        blockrows.append(row)
                    if (pos+1) % 30 == 0:
                        emit("measurement", path=path, repetition=repetition, completed=pos+1, total=len(cases))
                        write_csv(OUT / "requests.csv", rows + blockrows)
                rows.extend(blockrows)
                blocks.append({"path": path, "repetition": repetition, "requests": len(blockrows),
                               "seconds": time.perf_counter()-tick, "cpu_percent": psutil.cpu_percent(),
                               "available_memory": psutil.virtual_memory().available})
                write_csv(OUT / "requests.csv", rows)
                write_csv(OUT / "measurement_blocks.csv", blocks)
    conn.close()
    assert len(rows) == 4320
    write_json(OUT / "measured.json", {"requests": len(rows), "statuses": dict(Counter(r["status"] for r in rows)),
               "requests_sha256": native.hash_file(OUT / "requests.csv"), "warmup_requests": len(warmup)})
    emit("measurement_complete", requests=len(rows), statuses=dict(Counter(r["status"] for r in rows)))


def profile():
    cases = verify_prepared()
    assert (OUT / "measured.json").exists()
    if (OUT / "profile_started.json").exists():
        raise RuntimeError("Refusing to overwrite profiling")
    write_json(OUT / "profile_started.json", {"at_unix": time.time(), "runner_sha256": native.hash_file(__file__)})
    conn = connect()
    rows, documents = [], []
    with conn.cursor() as cur:
        for pos, case in enumerate(cases):
            for method in METHODS:
                active = activate(cur, case, method)
                tick = time.perf_counter()
                row = {"qid": case["qid"], "drift": case["drift"], "tables": case["tables"], "method": method,
                       "active": active, "status": "ok", "planning_ms": None, "execution_ms": None,
                       "root_rows": None, "plan_shape_sha256": "", "profile_wall_ms": None, "error": ""}
                try:
                    cur.execute("EXPLAIN (ANALYZE, FORMAT JSON, TIMING OFF, SUMMARY ON) " + case["sql"])
                    doc = native.injection.plan_doc(cur.fetchone()[0])
                    stat = native.injection.collect_plan_stats(doc)
                    row.update({"planning_ms": stat["planning_ms"], "execution_ms": stat["execution_ms"],
                                "root_rows": stat["root_input_estimate"], "plan_shape_sha256": stat["plan_shape_sha256"]})
                    documents.append({"qid": case["qid"], "method": method, "plan": doc})
                except Exception as exc:
                    row["status"] = "timeout" if isinstance(exc, native.psycopg.errors.QueryCanceled) else "error"
                    row["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
                row["profile_wall_ms"] = (time.perf_counter()-tick)*1000
                rows.append(row)
            if (pos+1) % 30 == 0:
                write_csv(OUT / "profiles.csv", rows)
                emit("profile", completed=pos+1, cases=len(cases))
    conn.close()
    write_csv(OUT / "profiles.csv", rows)
    write_json(OUT / "plans.json", documents)
    write_json(OUT / "profiled.json", {"requests": len(rows), "statuses": dict(Counter(r["status"] for r in rows)),
               "outputs": {name: native.hash_file(OUT / name) for name in ("profiles.csv", "plans.json")}})
    emit("profile_complete", requests=len(rows), statuses=dict(Counter(r["status"] for r in rows)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "measure", "profile"))
    globals()[parser.parse_args().phase]()
