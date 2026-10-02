"""Prospective, SQL/subquery-disjoint native PG residual adapter experiment."""
from __future__ import annotations

import os

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import argparse
from collections import Counter, defaultdict
import csv
import itertools
import math
from pathlib import Path
import random
import sys
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
DIAG = OUT.parent / "native_subplan_diagnostic"
sys.path.insert(0, str(DIAG))
import run_diagnostic as diagnostic

n = diagnostic.native
np, joblib = n.np, n.joblib
read_json, write_json, write_csv = n.read_json, n.write_json, n.write_csv
DRIFTS = ("column", "table", "value")
METHODS = ("pg", "pg_map", "root_ratio", "clip_ratio", "sub_hgb", "sub_rf")
POLICIES = ("runtime_hgb", "full_hgb", "runtime_rf", "full_rf")
SEED, REPEATS = 601, 3
MAP = OUT / "estimate_map.csv"


def start(phase):
    path = OUT / f"{phase}_started.json"
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite phase {phase}")
    write_json(path, {"unix": time.time(), "command": sys.argv,
                     "runner_sha256": n.hash_file(__file__),
                     "protocol_sha256": n.hash_file(OUT / "PROTOCOL.md")})


def stamp(phase, files, **other):
    write_json(OUT / f"{phase}_complete.json", {
        "unix": time.time(), "outputs": {x: n.hash_file(OUT / x) for x in files}, **other})


def verify(phase):
    doc = read_json(OUT / f"{phase}_complete.json")
    started = read_json(OUT / f"{phase}_started.json")
    assert started["protocol_sha256"] == n.hash_file(OUT / "PROTOCOL.md")
    assert started["runner_sha256"] == n.hash_file(__file__)
    for name, digest in doc["outputs"].items():
        assert n.hash_file(OUT / name) == digest, name
    return doc


def parts(row):
    root = tuple(sorted(n.subplans.parse_aliases(row[0])))
    return [(a, diagnostic.subrow(row, a)) for a in n.subplans.subplan_subsets(row, 0) + [root]]


def prepare():
    start("prepare")
    diagnostic.verify_prepared()
    dev = read_json(DIAG / "cases.json")
    observations = {r["subquery_id"]: r for r in n.csv_rows(DIAG / "count_acquisition.csv")}
    parent = list(range(len(dev)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first = {}
    for i, case in enumerate(dev):
        for sid in case["subquery_ids"]:
            if sid in first:
                parent[find(i)] = find(first[sid])
            else:
                first[sid] = i
    components = defaultdict(list)
    for i, case in enumerate(dev):
        components[find(i)].append(case)
    allocation = Counter({"fit": 0, "tune": 0, "gate": 0})
    targets = {"fit": .65, "tune": .175, "gate": .175}
    groups = []
    for cases in components.values():
        ids = {s for c in cases for s in c["subquery_ids"]}
        groups.append((cases, ids))
    groups.sort(key=lambda x: (-len(x[1]), n.rank_key(min(c["sql_id"] for c in x[0]), SEED, "dev-components")))
    sample_rows, dev_membership = {}, []
    for cases, ids in groups:
        role = min(targets, key=lambda k: allocation[k] / targets[k])
        allocation[role] += len(ids)
        for case in cases:
            drift = case['drift']
            dev_membership.append({"qid": case["qid"], "sql_id": case["sql_id"], "drift": drift, "role": role})
            for aliases, row in parts(case["row"]):
                sid = n.identity(row)
                assert sid in observations
                obs = observations[sid]
                if sid in sample_rows:
                    assert sample_rows[sid]['role'] == role
                    if drift not in sample_rows[sid]['drifts']:
                        sample_rows[sid]['drifts'].append(drift)
                    continue
                sample_rows[sid] = {"subquery_id": sid, "row": row, "drift": drift, "drifts": [drift], "role": role,
                    "tables": len(aliases), "pg_rows": float(obs["pg_rows"]),
                    "actual": float(obs["actual"]) if obs["count_status"] == "ok" else None,
                    "count_status": obs["count_status"]}

    inputs = {str(DIAG / f): n.hash_file(DIAG / f) for f in ("cases.json", "count_acquisition.csv", "prepared.json")}
    prior_native = defaultdict(set)
    for path in sorted((ROOT / "paper").rglob("native_callback_detail.csv")):
        inputs[str(path)] = n.hash_file(path)
        for row in n.csv_rows(path):
            prior_native[row["drift"]].add(int(row["query_index"]))
    oldcase_path = n.OUT / "cases.json"
    inputs[str(oldcase_path)] = n.hash_file(oldcase_path)
    prior_parents = {c["sql_id"] for c in read_json(oldcase_path)}
    used = set(observations)
    new_cases, memberships, funnels = [], [], []
    for drift, folder in n.METRIC_FOLDERS.items():
        meta_path = ROOT / "research_plan" / folder / "seed_1/metrics.json"
        meta = read_json(meta_path)
        path = meta_path.parent / "data" / meta["provenance"]["predict_rel"]
        inputs[str(path)] = n.hash_file(path)
        rows, _ = n.repair.load_target_rows(path)
        excluded = prior_parents | {n.identity(rows[i]) for i in prior_native[drift]}
        unique = {}
        for i, row in enumerate(rows):
            unique.setdefault(n.identity(row), i)
        counts = Counter(official_rows=len(rows), unique_parents=len(unique))
        count = 0
        for sid, index in sorted(unique.items(), key=lambda kv: n.rank_key(kv[0], SEED, "native-subplan-v1")):
            if sid in excluded:
                counts["prior_native"] += 1
                continue
            row = rows[index]
            subparts = parts(row)
            ids = {n.identity(r) for _, r in subparts}
            if ids & set(observations):
                counts["development_overlap"] += 1
                continue
            if ids & used:
                counts["selected_overlap"] += 1
                continue
            if count >= 120:
                counts["unselected_after_cap"] += 1
                continue
            role = "validation" if count % 2 == 0 else "test"
            case = {"qid": f"s601:{drift}:q{index}", "sql_id": sid, "index": index,
                    "drift": drift, "role": role, "row": row,
                    "sql": n.injection.query_to_sql(row), "truth": float(row[3]),
                    "tables": len(row[0].split(",")), "subquery_ids": sorted(ids)}
            new_cases.append(case)
            for aliases, subrow in subparts:
                memberships.append({"qid": case["qid"], "role": role, "drift": drift,
                    "subquery_id": n.identity(subrow), "aliases": " ".join(aliases),
                    "tables": len(aliases), "is_root": len(aliases) == case["tables"]})
            used.update(ids)
            counts[role] += 1
            count += 1
        assert count == 120, (drift, count)
        funnels.append({"drift": drift, **dict(counts)})
    role_ids = defaultdict(set)
    for row in sample_rows.values():
        role_ids[row["role"]].add(row["subquery_id"])
    for row in memberships:
        role_ids[row["role"]].add(row["subquery_id"])
    for a, b in itertools.combinations(role_ids, 2):
        assert not role_ids[a] & role_ids[b], (a, b)
    write_json(OUT / "development_samples.json", list(sample_rows.values()))
    write_csv(OUT / "development_parents.csv", dev_membership)
    write_json(OUT / "population.json", new_cases)
    write_csv(OUT / "native_membership.csv", memberships)
    write_json(OUT / "eligibility_funnel.json", funnels)
    write_json(OUT / "split_audit.json", {"components": len(groups), "component_sizes": dict(Counter(len(c) for c, _ in groups)),
        "role_unique_subqueries": {r: len(s) for r, s in role_ids.items()}, "all_pairwise_subquery_overlap": 0,
        "native_unique_parents": len({c['sql_id'] for c in new_cases}),
        "successful_development_labels": dict(Counter(r['role'] for r in sample_rows.values() if r['actual'] is not None)),
        "inputs": inputs})
    stamp("prepare", ["development_samples.json", "development_parents.csv", "population.json", "native_membership.csv",
                      "eligibility_funnel.json", "split_audit.json"])
    n.emit("population_frozen", native_parents=len(new_cases), splits=read_json(OUT / "split_audit.json"))


def predict(bundle, rows, pg):
    features = n.repair.row_features(rows, np.asarray(pg), bundle["vocab"])
    shift = np.clip(bundle["model"].predict(features), *bundle["clip"])
    return np.maximum(1., np.maximum(1., pg) * np.exp(shift))


def score(pred, truth):
    q = n.repair.qerror(np.asarray(pred), np.asarray(truth))
    return float(np.mean(np.log(np.percentile(q, [95, 99]))))


def train():
    verify("prepare")
    start("train")
    samples = read_json(OUT / "development_samples.json")
    byrole = {role: [r for r in samples if r['role'] == role and r['actual'] is not None] for role in ("fit", "tune", "gate")}
    fit, tune = byrole["fit"], byrole["tune"]
    vocab = n.repair.signature_vocab([r['row'] for r in fit])
    x = n.repair.row_features([r['row'] for r in fit], np.array([r['pg_rows'] for r in fit]), vocab)
    residual = np.log(np.maximum(1., [r['actual'] for r in fit])) - np.log(np.maximum(1., [r['pg_rows'] for r in fit]))
    clip = np.percentile(residual, [1, 99])
    scalar = float(np.median(residual))
    models, tuning = {}, []
    for name in ("hgb", "rf"):
        configs = ([{"max_leaf_nodes": leaves, "min_samples_leaf": leaf} for leaves in (15, 31) for leaf in (5, 20)] if name == "hgb"
                   else [{"max_depth": depth, "min_samples_leaf": leaf} for depth in (14, None) for leaf in (2, 5)])
        best = None
        for i, config in enumerate(configs):
            model = (n.HistGradientBoostingRegressor(max_iter=300, learning_rate=.05, l2_regularization=.01,
                         early_stopping=False, random_state=SEED, **config) if name == "hgb" else
                     n.RandomForestRegressor(n_estimators=250, n_jobs=1, random_state=SEED, **config))
            tick = time.perf_counter()
            model.fit(x, np.clip(residual, *clip))
            fit_seconds = time.perf_counter() - tick
            bundle = {"model": model, "clip": clip, "vocab": vocab, "scalar": scalar, "config": config}
            pred = predict(bundle, [r['row'] for r in tune], np.array([r['pg_rows'] for r in tune]))
            risk = score(pred, [r['actual'] for r in tune])
            tuning.append({"model": name, "config": i, "parameters": str(config), "fit_seconds": fit_seconds, "tune_score": risk})
            if best is None or risk < best[0]:
                best = (risk, bundle)
        models[name] = best[1]
    lookup = {s['subquery_id']: s for s in samples}
    roles = {r['qid']: r['role'] for r in n.csv_rows(OUT / "development_parents.csv")}
    clip_trials = []
    tune_cases = [c for c in read_json(DIAG / "cases.json") if roles[c['qid']] == "tune"]
    for cap in (2, 4, 10):
        grouped = defaultdict(list)
        for case in tune_cases:
            root = lookup[n.identity(diagnostic.subrow(case['row'], tuple(sorted(n.subplans.parse_aliases(case['row'][0])))))]
            root_pred = predict(models['hgb'], [root['row']], np.array([root['pg_rows']]))[0]
            ratio = np.clip(root_pred / max(1., root['pg_rows']), 1/cap, cap)
            for sid in case['subquery_ids']:
                row = lookup[sid]
                if row['actual'] is not None:
                    grouped[sid].append(math.log(max(1., row['pg_rows'] * ratio)))
        keys = sorted(grouped)
        risk = score([math.exp(np.mean(grouped[k])) for k in keys], [lookup[k]['actual'] for k in keys])
        clip_trials.append({"cap": cap, "tune_score": risk, "unique_tune_labels": len(keys)})
    selected_cap = min(clip_trials, key=lambda x: x['tune_score'])['cap']
    gates = {}
    for name, bundle in models.items():
        gates[name] = {}
        for drift in DRIFTS:
            rows = [r for r in byrole['gate'] if drift in r['drifts']]
            raw, truth = np.array([r['pg_rows'] for r in rows]), np.array([r['actual'] for r in rows])
            pred = predict(bundle, [r['row'] for r in rows], raw)
            report = n.gate_summary(raw, truth, pred, scalar)
            report['unique_subqueries'] = len(rows)
            report['passed'] = bool(report['passed'] and len(rows) >= 30)
            gates[name][drift] = report
    joblib.dump(models, OUT / "models.joblib")
    write_csv(OUT / "tuning.csv", tuning)
    write_json(OUT / "model_selection.json", {"models": {m: b['config'] for m, b in models.items()},
        "clip_trials": clip_trials, "selected_ratio_cap": selected_cap, "residual_clip": clip.tolist(),
        "fit_scalar_shift": scalar, "supervision_counts": {k: len(v) for k, v in byrole.items()}})
    write_json(OUT / "estimator_gates.json", gates)
    stamp("train", ["models.joblib", "tuning.csv", "model_selection.json", "estimator_gates.json"])
    n.emit("trained", gates={m: {d: r['passed'] for d, r in gs.items()} for m, gs in gates.items()},
           selection=read_json(OUT / "model_selection.json"))


def connect():
    conn = n.connect()
    with conn.cursor() as cur:
        cur.execute("SET application_name='perseus_native_subplan_learning'")
    return conn


def maps():
    verify("prepare")
    verify("train")
    start("maps")
    cases = read_json(OUT / "population.json")
    models = joblib.load(OUT / "models.joblib")
    cap = read_json(OUT / "model_selection.json")['selected_ratio_cap']
    observations, maprows, costs = [], [], []
    conn = connect()
    with conn.cursor() as cur:
        cur.execute("SELECT version()")
        version = cur.fetchone()[0]
        cur.execute("SELECT name,setting FROM pg_settings WHERE name IN ('jit','max_parallel_workers_per_gather',"
                    "'statement_timeout','work_mem','plan_cache_mode','join_collapse_limit','random_page_cost') ORDER BY name")
        settings = dict(cur.fetchall())
        cur.execute("SELECT application_name,state FROM pg_stat_activity WHERE pid<>pg_backend_pid() AND state='active'")
        active = cur.fetchall()
        assert not active, active
        for pos, case in enumerate(cases):
            batch, rows, pg = [], [], []
            tick = time.perf_counter()
            for aliases, row in parts(case['row']):
                sql = n.subplans.subset_sql(case['row'], aliases)
                begin = time.perf_counter()
                cur.execute("EXPLAIN (FORMAT JSON) " + sql)
                doc = n.injection.plan_doc(cur.fetchone()[0])
                stat = n.injection.collect_plan_stats(doc)
                estimate = stat['root_input_estimate']
                assert estimate is not None
                obs = {"qid": case['qid'], "subquery_id": n.identity(row), "aliases": " ".join(aliases),
                    "tables": len(aliases), "is_root": len(aliases) == case['tables'], "role": case['role'],
                    "drift": case['drift'], "pg_rows": float(estimate), "sql": sql,
                    "explain_ms": (time.perf_counter()-begin)*1000}
                rows.append(row)
                pg.append(float(estimate))
                batch.append(obs)
            pg = np.maximum(1., pg)
            explain_ms = (time.perf_counter()-tick)*1000
            tick = time.perf_counter()
            estimates = {m: predict(models[m], rows, pg) for m in models}
            inference_ms = (time.perf_counter()-tick)*1000
            ratio = float(estimates['hgb'][-1] / pg[-1])
            case['root_pg_rows'] = float(pg[-1])
            case['root_ratios'] = {m: float(v[-1]/pg[-1]) for m, v in estimates.items()}
            case['root_estimates'] = {}
            for j, obs in enumerate(batch):
                values = {"pg_map": float(pg[j]), "root_ratio": max(1., float(pg[j]*ratio)),
                    "clip_ratio": max(1., float(pg[j]*np.clip(ratio, 1/cap, cap))),
                    "sub_hgb": float(estimates['hgb'][j]), "sub_rf": float(estimates['rf'][j])}
                for method, value in values.items():
                    maprows.append({"query_id": case['qid'], "aliases": obs['aliases'], "method": method, "estimate": value})
                if obs['is_root']:
                    case['root_estimates'] = values
            observations.extend(batch)
            costs.append({"qid": case['qid'], "role": case['role'], "explain_ms": explain_ms,
                          "inference_both_models_ms": inference_ms, "subqueries": len(batch)})
            if (pos+1) % 30 == 0:
                n.emit("map_preparation", parents=pos+1, total=len(cases))
    conn.close()
    write_json(OUT / "cases.json", cases)
    write_csv(OUT / "native_subplans.csv", observations)
    write_csv(MAP, maprows)
    write_csv(OUT / "map_costs.csv", costs)
    write_json(OUT / "environment.json", {"version": version, "settings": settings, "other_active_sessions": active,
        "prepare_threshold": None, "python": sys.version, "sklearn": __import__('sklearn').__version__,
        "callback_source_sha256": n.hash_file(ROOT/'baselines/End-to-End-CardEst-Benchmark/postgresql-13.1/src/backend/optimizer/path/costsize.c')})
    stamp("maps", ["cases.json", "native_subplans.csv", "estimate_map.csv", "map_costs.csv", "environment.json"])


def matches(case, rule, model):
    if rule is None:
        return False
    ratio = case['root_ratios'][model]
    return (case['drift'] in rule['drifts'] and case['tables'] >= rule['min_tables']
            and case['root_pg_rows'] >= rule['min_pg_rows']
            and (rule['direction'] == 'any' or (ratio < 1 if rule['direction'] == 'down' else ratio > 1)))


def activate(cur, case, method):
    if method == 'pg':
        cur.execute("SET perseus_estmap_enabled=off")
    else:
        cur.execute("SELECT set_config('perseus_estmap_fname',%s,false),set_config('perseus_estmap_query_id',%s,false),"
                    "set_config('perseus_estmap_method',%s,false),set_config('perseus_estmap_enabled','on',false)",
                    (MAP.as_posix(), case['qid'], method))


def active_method(case, method, rules):
    if method not in POLICIES:
        return method
    model = method.split('_')[-1]
    return 'sub_' + model if matches(case, rules[method]['rule'], model) else 'pg'


def request(cur, case, method, rules):
    start_time = time.perf_counter()
    active = active_method(case, method, rules)
    decision_ms = (time.perf_counter()-start_time)*1000
    tick = time.perf_counter()
    activation_ms, select_ms = 0., 0.
    status, error, count = 'ok', '', None
    try:
        activate(cur, case, active)
        activation_ms = (time.perf_counter()-tick)*1000
        tick = time.perf_counter()
        cur.execute(case['sql'])
        count = int(cur.fetchone()[0])
        select_ms = (time.perf_counter()-tick)*1000
        if count != int(case['truth']):
            status = 'label_mismatch'
    except Exception as exc:
        select_ms = (time.perf_counter()-tick)*1000
        status = 'timeout' if isinstance(exc, n.psycopg.errors.QueryCanceled) else 'error'
        error = f'{type(exc).__name__}: {str(exc)[:200]}'
    return {'qid': case['qid'], 'sql_id': case['sql_id'], 'role': case['role'], 'drift': case['drift'],
        'method': method, 'active': active, 'status': status, 'count': count,
        'wall_ms': (time.perf_counter()-start_time)*1000, 'decision_ms': decision_ms,
        'activation_ms': activation_ms, 'select_ms': select_ms, 'error': error}


def audit_plans(cur, cases, methods, rules, role):
    checks, joins, documents = [], [], []
    maps_by_key = {(r['query_id'], r['method'], r['aliases']): float(r['estimate']) for r in n.csv_rows(MAP)}
    for case in cases:
        for method in methods:
            active = active_method(case, method, rules)
            activate(cur, case, active)
            cur.execute('EXPLAIN (FORMAT JSON) ' + case['sql'])
            doc = n.injection.plan_doc(cur.fetchone()[0])
            stat = n.injection.collect_plan_stats(doc)
            expected = None if active == 'pg' else case['root_estimates'][active]
            observed = stat['root_input_estimate']
            checks.append({'qid': case['qid'], 'method': method, 'active': active, 'expected': expected,
                'observed': observed, 'matches': expected is None or abs(observed-expected) <= max(1., expected*1e-6),
                'plan_shape_sha256': stat['plan_shape_sha256']})
            def visit(node, parameterized=False):
                aliases = {node['Alias']} if 'Alias' in node else set()
                for i, child in enumerate(node.get('Plans', [])):
                    aliases |= visit(child, parameterized or (node.get('Node Type') == 'Nested Loop' and i == 1))
                if node.get('Node Type') in ('Hash Join', 'Merge Join', 'Nested Loop'):
                    key = ' '.join(sorted(aliases))
                    estimate = maps_by_key.get((case['qid'], active, key))
                    rows = node.get('Plan Rows')
                    joins.append({'qid': case['qid'], 'method': method, 'active': active, 'aliases': key,
                        'parameterized_context': parameterized, 'expected': estimate, 'observed': rows,
                        'matches': None if estimate is None else abs(rows-estimate) <= max(1., estimate*1e-6)})
                return aliases
            visit(doc['Plan'])
            documents.append({'qid': case['qid'], 'method': method, 'plan': doc})
    write_csv(OUT/f'{role}_root_checks.csv', checks)
    write_csv(OUT/f'{role}_join_checks.csv', joins)
    write_json(OUT/f'{role}_plans.json', documents)
    assert all(r['matches'] for r in checks)
    assert all(r['matches'] for r in joins if r['expected'] is not None), 'Inspect join-map mismatch before timing'


def measure(role):
    verify('maps')
    if role == 'test':
        verify('select')
    start(role)
    rules = read_json(OUT/'selected_rules.json') if role == 'test' else {}
    cases = [c for c in read_json(OUT/'cases.json') if c['role'] == role]
    methods = list(METHODS) + (list(POLICIES) if role == 'test' else [])
    rows, warmups, blocks = [], [], []
    conn = connect()
    with conn.cursor() as cur:
        cur.execute("SELECT application_name FROM pg_stat_activity WHERE pid<>pg_backend_pid() AND state='active'")
        assert not cur.fetchall(), 'Another active DB session; do not mix latency measurements'
        audit_plans(cur, cases, methods, rules, role)
        rng = random.Random(SEED + (0 if role == 'validation' else 1))
        order = list(cases)
        rng.shuffle(order)
        for case in order:
            conditions = list(methods)
            rng.shuffle(conditions)
            for method in conditions:
                warmups.append(request(cur, case, method, rules))
        write_csv(OUT/f'{role}_warmup.csv', warmups)
        n.emit('warmup_complete', role=role, requests=len(warmups), statuses=dict(Counter(r['status'] for r in warmups)))
        for repetition in range(REPEATS):
            order = list(cases)
            rng.shuffle(order)
            for pos, case in enumerate(order):
                conditions = list(methods)
                rng.shuffle(conditions)
                for rank, method in enumerate(conditions):
                    row = request(cur, case, method, rules)
                    row.update(repetition=repetition, position=pos, condition_rank=rank)
                    rows.append(row)
                if (pos+1) % 30 == 0:
                    write_csv(OUT/f'{role}_requests.csv', rows)
                    n.emit('measurement', role=role, repetition=repetition, parents=pos+1,
                           statuses=dict(Counter(r['status'] for r in rows)))
            blocks.append({'repetition': repetition, 'unix_finished': time.time(), 'requests_so_far': len(rows)})
    conn.close()
    write_csv(OUT/f'{role}_requests.csv', rows)
    write_json(OUT/f'{role}_blocks.json', blocks)
    stamp(role, [f'{role}_{name}' for name in ('requests.csv', 'warmup.csv', 'root_checks.csv', 'join_checks.csv', 'plans.json', 'blocks.json')],
          requests=len(rows), statuses=dict(Counter(r['status'] for r in rows)))


def aggregate(role):
    groups = defaultdict(list)
    for row in n.csv_rows(OUT/f'{role}_requests.csv'):
        groups[row['qid'], row['method']].append(row)
    result = {}
    for key, rows in groups.items():
        assert len(rows) == REPEATS and {int(r['repetition']) for r in rows} == {0, 1, 2}
        result[key] = {'ms': float(np.median([float(r['wall_ms']) for r in rows])),
                       'failed': sum(r['status'] != 'ok' for r in rows), 'active': rows[0]['active']}
    return result


def candidate_rules():
    for count in (1, 2, 3):
        for drifts in itertools.combinations(DRIFTS, count):
            for tables, pgrows, direction in itertools.product((2, 3, 4, 5), (0, 100, 1000, 10000), ('any', 'down', 'up')):
                yield {'drifts': list(drifts), 'min_tables': tables, 'min_pg_rows': pgrows, 'direction': direction}


def select():
    verify('validation')
    verify('train')
    assert not (OUT/'test_started.json').exists()
    start('select')
    cases = [c for c in read_json(OUT/'cases.json') if c['role'] == 'validation']
    gates = read_json(OUT/'estimator_gates.json')
    lookup = aggregate('validation')
    base = np.array([lookup[c['qid'], 'pg']['ms'] for c in cases])
    searches, selected = [], {}
    for policy in POLICIES:
        model = policy.split('_')[-1]
        values = np.array([lookup[c['qid'], 'sub_'+model]['ms'] for c in cases])
        ratios = values/base
        best = None
        for index, rule in enumerate(candidate_rules()):
            if policy.startswith('full') and not all(gates[model][d]['passed'] for d in rule['drifts']):
                continue
            mask = np.array([matches(c, rule, model) for c in cases])
            support = int(mask.sum())
            if support < 20:
                continue
            selected_cases = [c for c, keep in zip(cases, mask) if keep]
            failures = sum(lookup[c['qid'], m]['failed'] for c in selected_cases for m in ('pg', 'sub_'+model))
            q50, q95 = np.percentile(ratios[mask], [50, 95])
            loss = float(np.mean(ratios[mask] > 1.01))
            whole = np.where(mask, values, base)
            p95 = float(np.percentile(whole, 95)/np.percentile(base, 95))
            savings = float(np.sum(base)-np.sum(whole))
            passed = bool(failures == 0 and q50 <= .98 and q95 <= 1.10 and loss <= .25 and p95 <= 1.01 and savings > 0)
            rec = {'policy': policy, 'index': index, 'rule': rule, 'support': support, 'failures': failures,
                   'median_ratio': float(q50), 'p95_ratio': float(q95), 'loss_fraction': loss,
                   'whole_p95_latency_ratio': p95, 'saved_ms': savings, 'passed': passed}
            searches.append(rec)
            rank = (savings, support, -q95, -index)
            if passed and (best is None or rank > best[0]):
                best = (rank, rec)
        selected[policy] = {'rule': None, 'reason': 'no qualifying validation region'} if best is None else best[1]
    write_json(OUT/'rule_search.json', searches)
    write_json(OUT/'selected_rules.json', selected)
    stamp('select', ['rule_search.json', 'selected_rules.json'],
          validation_sha256=n.hash_file(OUT/'validation_requests.csv'), test_was_absent=True)
    n.emit('policies_frozen', policies=selected)


def counts():
    verify('test')
    start('counts')
    rows = [r for r in n.csv_rows(OUT/'native_subplans.csv') if r['role'] == 'test']
    output = []
    conn = connect()
    with conn.cursor() as cur:
        cur.execute('SET perseus_estmap_enabled=off')
        for pos, row in enumerate(rows):
            tick = time.perf_counter()
            result = {**row, 'actual': None, 'status': 'ok', 'count_ms': None, 'error': ''}
            try:
                cur.execute(row['sql'])
                result['actual'] = int(cur.fetchone()[0])
            except Exception as exc:
                result['status'] = 'timeout' if isinstance(exc, n.psycopg.errors.QueryCanceled) else 'error'
                result['error'] = f'{type(exc).__name__}: {str(exc)[:200]}'
            result['count_ms'] = (time.perf_counter()-tick)*1000
            output.append(result)
            if (pos+1) % 100 == 0:
                write_csv(OUT/'test_subquery_counts.csv', output)
                n.emit('test_count_audit', completed=pos+1, total=len(rows), statuses=dict(Counter(r['status'] for r in output)))
    conn.close()
    write_csv(OUT/'test_subquery_counts.csv', output)
    stamp('counts', ['test_subquery_counts.csv'], statuses=dict(Counter(r['status'] for r in output)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('prepare', 'train', 'maps', 'validation', 'select', 'test', 'counts'))
    phase = parser.parse_args().phase
    if phase in ('validation', 'test'):
        measure(phase)
    else:
        globals()[phase]()
