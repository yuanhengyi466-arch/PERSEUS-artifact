"""Prospectively specified native optimizer protocol ablation on existing SQL."""
from __future__ import annotations

import os

for env_name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[env_name] = "1"

import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
import math
import platform
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import psutil
import psycopg
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import run_workload_drift_tail_residual_transport as repair
import run_workload_drift_postgres_estimate_injection as injection
import run_workload_drift_postgres_multisubplan_injection as subplans
import summarize_native_callback_policy_grid as original_grid

DRIFTS = ("column", "table", "value")
MODELS = ("hgb", "rf")
SEEDS = (401, 402, 403)
REPEATS = 3
TIMEOUT_MS = 5000
CONFIG = {
    "version": 1, "outer_seeds": list(SEEDS), "inner_seeds": [1, 2, 3],
    "split_proportions": [.6, .2, .2], "native_per_drift_role": 60,
    "timing_repeats": REPEATS, "statement_timeout_ms": TIMEOUT_MS,
    "models": list(MODELS), "injection": "native connected-subplan root-ratio",
    "runtime_class": {"min_support": 20, "median_cap": .98, "p95_cap": 1.10,
                      "loss_cap": .25, "whole_p95_cap": 1.01},
    "bootstrap_resamples": 2000,
}
METRIC_FOLDERS = {
    "column": "workload_drift_mscn_windows_bitmap",
    "table": "workload_drift_mscn_windows_table_bitmap",
    "value": "workload_drift_mscn_windows_value_bitmap",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def hash_file(path):
    return digest(Path(path).read_bytes())


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def emit(stage, **fields):
    print(json.dumps({"at": datetime.now(timezone.utc).isoformat(), "stage": stage, **fields}), flush=True)


def relocated(value):
    value = value.replace("\\", "/")
    prefix = ""
    return ROOT / value[len(prefix):] if value.startswith(prefix) else Path(value)


def identity(row):
    conditions = [x.strip().lower() for x in row[1].split(",") if x.strip()]
    parts = row[2].split(",") if row[2] else []
    conditions += [" ".join(x.strip().lower() for x in parts[i:i + 3]) for i in range(0, len(parts), 3)]
    canonical = json.dumps([sorted(" ".join(t.lower().split()) for t in row[0].split(",")), sorted(conditions)])
    return digest(canonical.encode())


def split_role(sql_id, seed):
    value = int(digest(f"protocol-v1|{seed}|{sql_id}".encode())[:16], 16) / 2**64
    return "calibration" if value < .6 else "validation" if value < .8 else "test"


def rank_key(sql_id, seed, salt):
    return digest(f"{seed}|{salt}|{sql_id}".encode())


def connect():
    conn = psycopg.connect(host="127.0.0.1", port=5434, dbname="postgres", user="postgres",
                           password=os.environ.get("PGPASSWORD"), autocommit=True,
                           prepare_threshold=None, connect_timeout=5,
                           application_name="perseus_protocol_attribution")
    with conn.cursor() as cur:
        cur.execute("SET search_path=stats_inj,public")
        cur.execute(f"SET statement_timeout={TIMEOUT_MS}")
        cur.execute("SET max_parallel_workers_per_gather=0")
        cur.execute("SET jit=off")
        cur.execute("SET plan_cache_mode=force_custom_plan")
        cur.execute("SET join_collapse_limit=8")
        cur.execute("SET from_collapse_limit=8")
        cur.execute("SET perseus_estmap_enabled=off")
    return conn


def activate(cur, case, model):
    if model == "pg":
        cur.execute("SET perseus_estmap_enabled=off")
    else:
        cur.execute("SELECT set_config('perseus_estmap_fname',%s,false), "
                    "set_config('perseus_estmap_query_id',%s,false), "
                    "set_config('perseus_estmap_method',%s,false), "
                    "set_config('perseus_estmap_enabled','on',false)",
                    ((OUT / "estimate_map.csv").as_posix(), case["qid"], model))


def explain(cur, case):
    cur.execute("EXPLAIN (FORMAT JSON) " + case["sql"])
    doc = injection.plan_doc(cur.fetchone()[0])
    stats = injection.collect_plan_stats(doc)
    return stats


def gate_summary(raw, truth, predictions, global_shift):
    raw_q = repair.summarize(repair.qerror(raw, truth))
    cand_q = repair.summarize(repair.qerror(predictions, truth))
    global_q = repair.summarize(repair.qerror(raw * math.exp(global_shift), truth))
    r = {p: 1 - cand_q[p] / raw_q[p] for p in ("p90", "p95", "p99")}
    g = {p: 1 - cand_q[p] / global_q[p] for p in ("p95", "p99")}
    passed = r["p90"] >= .3 and r["p95"] >= .4 and r["p99"] >= .6 and g["p95"] >= .25 and g["p99"] >= .4 and cand_q["median"] <= 1.2 * raw_q["median"]
    return {"passed": bool(passed), "raw": raw_q, "candidate": cand_q, "global": global_q,
            "reductions": r, "vs_global": g}


def make_model(name, seed):
    if name == "hgb":
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=.05,
                                             max_leaf_nodes=31, l2_regularization=.01, random_state=seed)
    return RandomForestRegressor(n_estimators=250, max_depth=14, min_samples_leaf=3,
                                  random_state=seed, n_jobs=1)


def predict(bundle, features, raw):
    shifts = np.clip(bundle["model"].predict(features), *bundle["clip"])
    return np.maximum(1., raw * np.exp(shifts))


def prepare():
    if (OUT / "prepared.json").exists():
        raise RuntimeError("Preparation already exists; use the frozen artifacts or a new version directory.")
    start = time.perf_counter()
    write_json(OUT / "config.json", CONFIG)
    native_path = ROOT / "paper/workload_drift_residual_field/results/native_optimizer_callback_hash_full_seeds123/native_callback_detail.csv"
    native_ids = defaultdict(set)
    for row in csv_rows(native_path):
        if row["method"] == "default":
            native_ids[row["drift"]].add(int(row["query_index"]))
    inputs = {str(native_path): hash_file(native_path), str(OUT / "PROTOCOL.md"): hash_file(OUT / "PROTOCOL.md"),
              str(Path(__file__)): hash_file(__file__)}
    datasets = {}
    for drift in DRIFTS:
        metrics_path = ROOT / "research_plan" / METRIC_FOLDERS[drift] / "seed_1/metrics.json"
        metrics = read_json(metrics_path)
        target_path = metrics_path.parent / "data" / metrics["provenance"]["predict_rel"]
        rows, labels = repair.load_target_rows(target_path)
        prediction_path = relocated(metrics["artifacts"]["predictions"])
        raw = np.loadtxt(prediction_path)
        assert len(rows) == len(raw)
        inputs[str(target_path)] = hash_file(target_path)
        inputs[str(prediction_path)] = hash_file(prediction_path)
        unique = {}
        for index, row in enumerate(rows):
            sid = identity(row)
            if sid not in unique:
                unique[sid] = index
            else:
                assert labels[unique[sid]] == labels[index], "Identical SQL with conflicting labels"
        native_sids = {identity(rows[i]) for i in native_ids[drift]}
        datasets[drift] = {"rows": rows, "labels": labels, "raw": raw, "unique": unique, "native": native_sids}
    models, cases, gates, membership, fit_log, final_error = {}, [], {}, [], [], []
    for outer in SEEDS:
        gates[str(outer)] = {}
        for drift, data in datasets.items():
            rows, labels, raw, unique = (data[k] for k in ("rows", "labels", "raw", "unique"))
            roles = {sid: split_role(sid, outer) for sid in unique}
            pool_n, fit_n = (1000, 500) if drift == "value" else (3000, 2000)
            pool = sorted((sid for sid in unique if roles[sid] == "calibration"), key=lambda x: rank_key(x, outer, "calibration"))[:pool_n]
            if len(pool) < pool_n:
                raise RuntimeError(f"Insufficient unique calibration SQL: {outer} {drift} {len(pool)}")
            vocab = repair.signature_vocab([rows[unique[sid]] for sid in pool])
            features = repair.row_features(rows, raw, vocab)
            pool_idx = np.array([unique[sid] for sid in pool])
            all_test_idx = np.array([i for sid, i in unique.items() if roles[sid] == "test"])
            gates[str(outer)][drift] = {}
            for model_name in MODELS:
                reports = []
                for inner in (1, 2, 3):
                    order = np.random.default_rng(inner).permutation(pool_idx)
                    fit_idx, gate_idx = order[:fit_n], order[fit_n:]
                    residual = np.log(labels[fit_idx]) - np.log(np.maximum(raw[fit_idx], 1))
                    clip = np.percentile(residual, [1, 99])
                    model = make_model(model_name, inner)
                    fit_start = time.perf_counter()
                    model.fit(features[fit_idx], np.clip(residual, *clip))
                    fit_seconds = time.perf_counter() - fit_start
                    bundle = {"model": model, "clip": clip, "vocab": vocab}
                    global_shift = repair.tail_risk_shift(residual, 1.5, .005)
                    gate_pred = predict(bundle, features[gate_idx], raw[gate_idx])
                    report = gate_summary(raw[gate_idx], labels[gate_idx], gate_pred, global_shift)
                    reports.append(report)
                    fit_log.append({"outer": outer, "drift": drift, "model": model_name, "inner": inner,
                                    "fit_unique_sql": len(fit_idx), "gate_unique_sql": len(gate_idx),
                                    "calibration_pool": pool_n, "fit_seconds": fit_seconds,
                                    "fit_indices_sha256": digest(fit_idx.astype(np.int64).tobytes()),
                                    "gate_indices_sha256": digest(gate_idx.astype(np.int64).tobytes()),
                                    "passed": report["passed"], "p95_reduction": report["reductions"]["p95"],
                                    "p99_reduction": report["reductions"]["p99"]})
                    if inner == 1:
                        models[(outer, drift, model_name)] = bundle
                        test_pred = predict(bundle, features[all_test_idx], raw[all_test_idx])
                        test_q = gate_summary(raw[all_test_idx], labels[all_test_idx], test_pred, global_shift)
                        final_error.append({"outer": outer, "drift": drift, "model": model_name,
                                            "test_unique_sql": len(all_test_idx), "p95_reduction": test_q["reductions"]["p95"],
                                            "p99_reduction": test_q["reductions"]["p99"],
                                            "raw_p95": test_q["raw"]["p95"], "repair_p95": test_q["candidate"]["p95"],
                                            "raw_p99": test_q["raw"]["p99"], "repair_p99": test_q["candidate"]["p99"]})
                gates[str(outer)][drift][model_name] = {"pass": all(x["passed"] for x in reports), "splits": reports}
            for sid, i in unique.items():
                membership.append({"outer": outer, "drift": drift, "sql_id": sid, "row_index": i,
                                   "role": roles[sid], "calibration_selected": sid in set(pool)})
            for role in ("validation", "test"):
                already_selected = {c["sql_id"] for c in cases if c["outer"] == outer and c["role"] == role}
                eligible = sorted((sid for sid in data["native"] if roles[sid] == role and sid not in already_selected), key=lambda x: rank_key(x, outer, role))[:CONFIG["native_per_drift_role"]]
                for sid in eligible:
                    i = unique[sid]
                    case = {"outer": outer, "drift": drift, "sql_id": sid, "index": i, "role": role,
                            "qid": f"a{outer}:{drift}:q{i}", "row": rows[i], "sql": injection.query_to_sql(rows[i]),
                            "truth": float(labels[i]), "raw": float(raw[i]),
                            "tables": len(rows[i][0].split(",")),
                            "joins": len([x for x in rows[i][1].split(",") if x]),
                            "predicates": len(rows[i][2].split(",")) // 3 if rows[i][2] else 0,
                            "subplan_count": len(subplans.subplan_subsets(rows[i], 0)), "estimates": {}}
                    for name in MODELS:
                        case["estimates"][name] = float(predict(models[(outer, drift, name)], features[i:i + 1], raw[i:i + 1])[0])
                    cases.append(case)
            emit("fit", outer=outer, drift=drift, gate={m: gates[str(outer)][drift][m]["pass"] for m in MODELS})
    for outer in SEEDS:
        role_sets = defaultdict(set)
        for row in membership:
            if row["outer"] == outer:
                role_sets[row["role"]].add(row["sql_id"])
        for a, b in itertools.combinations(role_sets, 2):
            assert not (role_sets[a] & role_sets[b]), (outer, a, b)
    joblib.dump(models, OUT / "models.joblib")
    write_json(OUT / "cases.json", cases)
    write_json(OUT / "estimator_gates.json", gates)
    write_csv(OUT / "sql_membership.csv", membership)
    write_csv(OUT / "fit_gate_results.csv", fit_log)
    write_csv(OUT / "estimator_final_test.csv", final_error)
    # Collect planner-only native row estimates; no SQL runtime outcomes select cases.
    map_rows, prep = [], []
    conn = connect()
    with conn.cursor() as cur:
        cur.execute("SELECT version()")
        pg_version = cur.fetchone()[0]
        cache = {}
        for pos, case in enumerate(cases):
            activate(cur, case, "pg")
            for alias_set in subplans.subplan_subsets(case["row"], 0):
                query = subplans.subset_sql(case["row"], alias_set)
                if query not in cache:
                    started = time.perf_counter()
                    cur.execute("EXPLAIN (FORMAT JSON) " + query)
                    stats = injection.collect_plan_stats(injection.plan_doc(cur.fetchone()[0]))
                    value = stats["root_input_estimate"]
                    if value is None:
                        raise RuntimeError("Native subplan estimate unavailable")
                    cache[query] = max(1., float(value))
                    prep.append({"sql_sha256": digest(query.encode()), "wall_ms": (time.perf_counter()-started)*1000})
                pg_rows = cache[query]
                for model_name in ("raw", *MODELS):
                    scale = 1. if model_name == "raw" else case["estimates"][model_name] / max(1., case["raw"])
                    map_rows.append({"query_id": case["qid"], "aliases": " ".join(alias_set), "method": model_name, "estimate": max(1., scale * pg_rows)})
            root_aliases = " ".join(sorted(injection.table_aliases(case["row"][0])))
            for name in ("raw", *MODELS):
                map_rows.append({"query_id": case["qid"], "aliases": root_aliases, "method": name,
                                 "estimate": max(1., case["raw"] if name == "raw" else case["estimates"][name])})
            if (pos + 1) % 100 == 0:
                emit("map", completed=pos + 1, total=len(cases))
    conn.close()
    write_csv(OUT / "estimate_map.csv", map_rows)
    write_csv(OUT / "map_preparation_cost.csv", prep)
    write_json(OUT / "prepared.json", {"config": CONFIG, "input_hashes": inputs,
               "prepared_at": datetime.now(timezone.utc).isoformat(), "cases": len(cases),
               "cases_hash": hash_file(OUT / "cases.json"), "models_hash": hash_file(OUT / "models.joblib"),
               "map_hash": hash_file(OUT / "estimate_map.csv"), "pg_version": pg_version,
               "python": sys.version, "platform": platform.platform(),
               "logical_cpus": psutil.cpu_count(), "prepare_seconds": time.perf_counter()-start,
               "native_counts": dict(Counter(f"{c['outer']}/{c['drift']}/{c['role']}" for c in cases)),
               "zero_sql_overlap": True})
    emit("prepared", cases=len(cases), map_rows=len(map_rows), seconds=time.perf_counter()-start)


def infer(case, name, models):
    bundle = models[(case["outer"], case["drift"], name)]
    features = repair.row_features([case["row"]], np.array([case["raw"]]), bundle["vocab"])
    value = float(predict(bundle, features, np.array([case["raw"]]))[0])
    if not np.isclose(value, case["estimates"][name], rtol=1e-10):
        raise RuntimeError("Online prediction differs from immutable map")


def rule_matches(rule, case, maximum):
    return (case["drift"] in rule["drifts"] and maximum <= rule["max_cardinality"]
            and (rule["max_subplans"] is None or case["subplan_count"] <= rule["max_subplans"])
            and (rule["max_joins"] is None or case["joins"] <= rule["max_joins"])
            and (rule["max_predicates"] is None or case["predicates"] <= rule["max_predicates"]))


def request(cur, case, policy, models, gates, selections=None):
    started = time.perf_counter()
    prefix_ms, maximum, shape = 0., None, ""
    admitted, status, error, count = False, "ok", "", None
    active = "pg"
    try:
        if policy == "pg":
            activate(cur, case, "pg")
        elif policy == "raw":
            activate(cur, case, "raw")
            active = "raw"
        else:
            mode, name = policy.split(":")
            est_pass = gates[str(case["outer"])][case["drift"]][name]["pass"]
            rule = selections[f"{case['outer']}/{name}/{mode}"]["rule"] if selections is not None and mode in ("runtime", "full") else None
            need_candidate = mode in ("candidate", "repair") or (mode == "estimator" and est_pass)
            if mode in ("runtime", "full"):
                need_candidate = rule is not None and case["drift"] in rule["drifts"] and (mode != "full" or est_pass)
            if need_candidate:
                infer(case, name, models)
                activate(cur, case, name)
                active, admitted = name, True
                if mode in ("candidate", "runtime", "full"):
                    stats = explain(cur, case)
                    maximum = stats["max_estimated_cardinality"]
                    shape = stats["plan_shape_sha256"]
                    if mode != "candidate" and not rule_matches(rule, case, maximum):
                        activate(cur, case, "pg")
                        active, admitted = "pg", False
            else:
                activate(cur, case, "pg")
        prefix_ms = (time.perf_counter() - started) * 1000
        cur.execute(case["sql"])
        count = int(cur.fetchone()[0])
        if count != int(case["truth"]):
            status = "label_mismatch"
    except Exception as exc:
        status = "timeout" if isinstance(exc, psycopg.errors.QueryCanceled) else "error"
        error = f"{type(exc).__name__}: {str(exc)[:250]}"
    wall_ms = (time.perf_counter() - started) * 1000
    return {"policy": policy, "active": active, "admitted": admitted, "wall_ms": wall_ms,
            "prefix_ms": prefix_ms, "max_cardinality": maximum, "plan_shape_sha256": shape,
            "status": status, "result_count": count, "error": error}


def check_prepared():
    manifest = read_json(OUT / "prepared.json")
    assert manifest["config"] == CONFIG
    for name, key in (("cases.json", "cases_hash"), ("models.joblib", "models_hash"), ("estimate_map.csv", "map_hash")):
        assert hash_file(OUT / name) == manifest[key], name
    assert hash_file(OUT / "PROTOCOL.md") == manifest["input_hashes"][str(OUT / "PROTOCOL.md")]
    return manifest


def measure(role):
    check_prepared()
    selected = None
    if role == "test":
        selected = read_json(OUT / "selected_rules.json")["policies"]
    models = joblib.load(OUT / "models.joblib")
    gates = read_json(OUT / "estimator_gates.json")
    cases = [c for c in read_json(OUT / "cases.json") if c["role"] == role]
    random.Random(20260928 if role == "validation" else 20260929).shuffle(cases)
    policies = ["pg", "raw", "candidate:hgb", "candidate:rf"] if role == "validation" else ["pg", "raw"] + [f"{mode}:{name}" for name in MODELS for mode in ("repair", "estimator", "runtime", "full")]
    output = OUT / f"{role}_timings.csv"
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite {output}; audit any failed partial run before resuming.")
    conn = connect()
    started = time.perf_counter()
    fields = ["outer", "drift", "qid", "sql_id", "repeat", "order", "policy", "active", "admitted", "wall_ms", "prefix_ms", "max_cardinality", "plan_shape_sha256", "status", "result_count", "error"]
    health = []
    with conn.cursor() as cur, output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        # One unscored request loads the immutable estimate map into this backend.
        warm = request(cur, cases[0], "candidate:hgb", models, gates)
        emit("map_warmup", role=role, status=warm["status"], wall_ms=warm["wall_ms"])
        for index, case in enumerate(cases):
            request(cur, case, "pg", models, gates)
            for repeat in range(REPEATS):
                ordered = list(policies)
                random.Random(f"{role}|{case['qid']}|{repeat}").shuffle(ordered)
                for order, policy in enumerate(ordered):
                    result = request(cur, case, policy, models, gates, selected)
                    writer.writerow({"outer": case["outer"], "drift": case["drift"], "qid": case["qid"], "sql_id": case["sql_id"], "repeat": repeat, "order": order, **result})
            stream.flush()
            if (index + 1) % 10 == 0 or index == len(cases) - 1:
                observation = {"completed": index+1, "total": len(cases), "seconds": time.perf_counter()-started,
                               "host_cpu_percent": psutil.cpu_percent(), "available_memory": psutil.virtual_memory().available}
                health.append(observation)
                write_json(OUT / f"{role}_progress.json", observation)
                emit(role, **observation)
    conn.close()
    write_json(OUT / f"{role}_run.json", {"cases": len(cases), "policies": policies, "repeats": REPEATS,
               "seconds": time.perf_counter()-started, "timings_sha256": hash_file(output), "health": health,
               "selection_sha256": hash_file(OUT / "selected_rules.json") if role == "test" else None})
    emit(role + "_complete", seconds=time.perf_counter()-started)


def aggregate_timing(path):
    grouped = defaultdict(list)
    for row in csv_rows(path):
        grouped[(row["qid"], row["policy"])].append(row)
    out = {}
    for key, rows in grouped.items():
        assert len(rows) == REPEATS, (key, len(rows))
        result = {"wall_ms": float(np.median([float(r["wall_ms"]) for r in rows])),
                  "prefix_ms": float(np.median([float(r["prefix_ms"]) for r in rows])),
                  "statuses": dict(Counter(r["status"] for r in rows)),
                  "admitted": all(r["admitted"] == "True" for r in rows)}
        values = [float(r["max_cardinality"]) for r in rows if r["max_cardinality"]]
        result["max_cardinality"] = max(values) if values else None
        out[key] = result
    return out


def select():
    check_prepared()
    assert (OUT / "validation_run.json").exists()
    assert not (OUT / "test_timings.csv").exists(), "Rules cannot be changed after test execution"
    timings = aggregate_timing(OUT / "validation_timings.csv")
    cases = [c for c in read_json(OUT / "cases.json") if c["role"] == "validation"]
    gates = read_json(OUT / "estimator_gates.json")
    rules = [r for r in original_grid.candidate_rules() if r["method"] == "native-multi"]
    assert len(rules) == 2940
    policies, ledger = {}, []
    for outer in SEEDS:
        groups = [c for c in cases if c["outer"] == outer]
        assert len({c["sql_id"] for c in groups}) == len(groups)
        base = np.array([timings[(c["qid"], "pg")]["wall_ms"] for c in groups])
        base_good = np.array([set(timings[(c["qid"], "pg")]["statuses"]) == {"ok"} for c in groups])
        drift_array = np.array([c["drift"] for c in groups])
        for name in MODELS:
            values = [timings[(c["qid"], "candidate:" + name)] for c in groups]
            candidate = np.array([v["wall_ms"] for v in values])
            prefix = np.array([v["prefix_ms"] for v in values])
            maximum = np.array([v["max_cardinality"] if v["max_cardinality"] is not None else math.inf for v in values])
            good = np.array([set(v["statuses"]) == {"ok"} for v in values])
            for mode in ("runtime", "full"):
                allowed = {d for d in DRIFTS if mode == "runtime" or gates[str(outer)][d][name]["pass"]}
                best, best_key, eligible = None, None, 0
                for rule in rules:
                    if not set(rule["drifts"]) <= allowed:
                        continue
                    applies = np.isin(drift_array, rule["drifts"])
                    admitted = applies & (maximum <= rule["max_cardinality"])
                    for field, case_field in (("max_subplans", "subplan_count"), ("max_joins", "joins"), ("max_predicates", "predicates")):
                        if rule[field] is not None:
                            admitted &= np.array([c[case_field] <= rule[field] for c in groups])
                    n = int(admitted.sum())
                    if n < 20 or not np.all((good & base_good)[admitted]):
                        continue
                    ratios = candidate[admitted] / base[admitted]
                    med, p95 = np.percentile(ratios, [50, 95])
                    loss = float(np.mean(ratios > 1.01))
                    policy_time = base.copy()
                    policy_time[applies & ~admitted] += prefix[applies & ~admitted]
                    policy_time[admitted] = candidate[admitted]
                    whole_p95 = float(np.percentile(policy_time / base, 95))
                    if med > .98 or p95 > 1.10 or loss > .25 or whole_p95 > 1.01:
                        continue
                    eligible += 1
                    key = (n, -med, -p95)
                    if best_key is None or key > best_key:
                        best_key = key
                        best = {"rule": rule, "accepted_validation_sql": n, "subset_median": float(med),
                                "subset_p95": float(p95), "loss_fraction": loss, "whole_p95": whole_p95,
                                "validation_total_ratio": float(policy_time.sum()/base.sum())}
                if best is None:
                    best = {"rule": None, "accepted_validation_sql": 0, "reason": "no rule satisfies fixed runtime criteria"}
                best.update({"eligible_rule_count": eligible, "allowed_drifts": sorted(allowed)})
                policies[f"{outer}/{name}/{mode}"] = best
                ledger.append({"outer": outer, "model": name, "mode": mode,
                               "estimator_gate_allowed_drifts": ";".join(sorted(allowed)),
                               "candidate_validation_sql_required": sum(c["drift"] in allowed for c in groups),
                               "candidate_validation_executions_required": REPEATS * sum(c["drift"] in allowed for c in groups),
                               "candidate_validation_measured_ms_required": sum(float(r["wall_ms"]) for r in csv_rows(OUT / "validation_timings.csv") if int(r["outer"]) == outer and r["policy"] == "candidate:"+name and r["drift"] in allowed),
                               "accepted_validation_sql": best["accepted_validation_sql"], "eligible_rules": eligible})
                emit("selection", outer=outer, model=name, mode=mode, accepted=best["accepted_validation_sql"], eligible=eligible)
    write_json(OUT / "selected_rules.json", {"config": CONFIG, "policies": policies,
               "validation_sha256": hash_file(OUT / "validation_timings.csv"), "selected_at": datetime.now(timezone.utc).isoformat()})
    write_csv(OUT / "validation_cost_ledger.csv", ledger)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "validate", "select", "test", "analyze"))
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.phase == "prepare": prepare()
        elif args.phase == "validate": measure("validation")
        elif args.phase == "select": select()
        elif args.phase == "test": measure("test")
        else:
            from analyze_protocol_attribution import analyze
            analyze()


if __name__ == "__main__":
    main()
