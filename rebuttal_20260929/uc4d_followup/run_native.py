"""Matched direct/residual models through a complete PG native callback path."""
from __future__ import annotations

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'

import argparse
from collections import Counter, defaultdict
import itertools
from pathlib import Path
import random
import sys
import time

BASE = Path(__file__).resolve().parent
OUT = BASE / 'native'
ROOT = BASE.parents[1]
SOURCE = BASE.parent / 'native_subplan_learning'
sys.path.insert(0, str(SOURCE))
import run_learning as engine
from semantic_audit import identity

n, np, joblib = engine.n, engine.np, engine.joblib
read_json, write_json, write_csv = n.read_json, n.write_json, n.write_csv
SEED, REPEATS = 811, 3
FAMILIES = ('residual_hgb', 'direct_hgb', 'residual_rf', 'direct_rf')
METHODS = ('pg', 'pg_map', *FAMILIES)
POLICIES = tuple(f'{stage}__{family}' for stage in ('runtime', 'full') for family in FAMILIES)
MAP = OUT / 'estimate_map.csv'


def start(phase):
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT/f'{phase}_started.json').exists(), f'Refusing overwrite: {phase}'
    write_json(OUT/f'{phase}_started.json', {'unix': time.time(), 'command': sys.argv,
        'script_sha256': n.hash_file(__file__), 'protocol_sha256': n.hash_file(BASE/'PROTOCOL.md')})


def complete(phase, files, **fields):
    write_json(OUT/f'{phase}_complete.json', {'unix': time.time(),
        'outputs': {f: n.hash_file(OUT/f) for f in files}, **fields})


def verify(phase):
    begin = read_json(OUT/f'{phase}_started.json')
    assert begin['script_sha256'] == n.hash_file(__file__)
    assert begin['protocol_sha256'] == n.hash_file(BASE/'PROTOCOL.md')
    end = read_json(OUT/f'{phase}_complete.json')
    for file, digest in end['outputs'].items():
        assert n.hash_file(OUT/file) == digest, file
    return end


def prepare():
    start('prepare')
    inputs = [SOURCE/f for f in ('cases.json', 'development_samples.json',
        'test_eligibility_amendment.json', 'run_learning.py', 'semantic_audit.py', 'model_selection.json')]
    unique = {}
    for row in read_json(SOURCE/'development_samples.json'):
        if row['actual'] is None:
            continue
        sid = identity(row['row'])
        if sid in unique:
            assert unique[sid]['role'] == row['role'] and unique[sid]['actual'] == row['actual']
            unique[sid]['drifts'] = sorted(set(unique[sid]['drifts']) | set(row['drifts']))
        else:
            unique[sid] = {**row, 'semantic_id': sid}
    samples = list(unique.values())
    accepted = set(read_json(SOURCE/'test_eligibility_amendment.json')['accepted_test_qids'])
    eligible = [c for c in read_json(SOURCE/'cases.json') if c['role'] == 'validation' or c['qid'] in accepted]
    cases = []
    for role in ('validation', 'test'):
        for drift in engine.DRIFTS:
            pool = [c for c in eligible if c['role'] == role and c['drift'] == drift]
            pool.sort(key=lambda c: n.rank_key(identity(c['row']), SEED, 'uc4d-direct-native'))
            assert len(pool) >= 30
            for case in pool[:30]:
                kept = {k: case[k] for k in ('qid', 'row', 'sql', 'role', 'drift', 'tables', 'index')}
                kept.update(sql_id=identity(case['row']), archived_truth=case['truth'])
                cases.append(kept)
    roles = defaultdict(set)
    for row in samples:
        roles[row['role']].add(row['semantic_id'])
    for case in cases:
        roles[case['role']].update(identity(row) for _, row in engine.parts(case['row']))
    overlaps = {f'{a}/{b}': len(roles[a] & roles[b]) for a, b in itertools.combinations(roles, 2)}
    assert not any(overlaps.values()), overlaps
    assert len({c['sql_id'] for c in cases}) == len(cases) == 180
    write_json(OUT/'development.json', samples)
    write_json(OUT/'population.json', cases)
    write_json(OUT/'split_audit.json', {'inputs': {str(p): n.hash_file(p) for p in inputs},
        'overlap_counts': overlaps, 'role_unique_subqueries': {r: len(s) for r, s in roles.items()},
        'development_labels': dict(Counter(r['role'] for r in samples)),
        'cohort': 'hashed attribution subset of previously evaluated, isolated native roles'})
    complete('prepare', ['development.json', 'population.json', 'split_audit.json'])
    n.emit('native_prepared', parents=len(cases), development=dict(Counter(r['role'] for r in samples)))


def predict(bundle, rows, pg):
    raw = np.maximum(1., pg)
    x = n.repair.row_features(rows, raw, bundle['vocab'])
    y = np.exp(np.clip(bundle['model'].predict(x), *bundle['clip']))
    return np.maximum(1., y * raw if bundle['residual'] else y)


def train():
    verify('prepare')
    start('train')
    samples = read_json(OUT/'development.json')
    byrole = {r: [s for s in samples if s['role'] == r] for r in ('fit', 'tune', 'gate')}
    fit, tune = byrole['fit'], byrole['tune']
    vocab = n.repair.signature_vocab([r['row'] for r in fit])
    raw = np.maximum(1., [r['pg_rows'] for r in fit])
    target = np.log(np.maximum(1., [r['actual'] for r in fit]))
    x = n.repair.row_features([r['row'] for r in fit], raw, vocab)
    scalar = float(np.median(target-np.log(raw)))
    models, tuning, selection = {}, [], {}
    for family in FAMILIES:
        residual = family.startswith('residual')
        y = target-np.log(raw) if residual else target
        clip = np.percentile(y, [1, 99])
        is_hgb = family.endswith('hgb')
        configs = ([{'max_leaf_nodes': leaves, 'min_samples_leaf': leaf}
                    for leaves in (15, 31) for leaf in (5, 20)] if is_hgb else
                   [{'max_depth': depth, 'min_samples_leaf': leaf}
                    for depth in (14, None) for leaf in (2, 5)])
        best = None
        for cid, config in enumerate(configs):
            model = (n.HistGradientBoostingRegressor(max_iter=300, learning_rate=.05,
                l2_regularization=.01, early_stopping=False, random_state=SEED, **config) if is_hgb else
                n.RandomForestRegressor(n_estimators=250, n_jobs=1, random_state=SEED, **config))
            tick = time.perf_counter()
            model.fit(x, np.clip(y, *clip))
            bundle = {'model': model, 'clip': clip, 'vocab': vocab, 'residual': residual, 'scalar': scalar}
            predictions = predict(bundle, [s['row'] for s in tune], [s['pg_rows'] for s in tune])
            risk = engine.score(predictions, [s['actual'] for s in tune])
            tuning.append({'family': family, 'config_id': cid, 'parameters': str(config),
                'tune_score': risk, 'fit_tune_seconds': time.perf_counter()-tick})
            if best is None or (risk, cid) < best[0]:
                best = ((risk, cid), bundle, config)
        models[family] = best[1]
        selection[family] = {'config_id': best[0][1], 'parameters': best[2], 'clip': best[1]['clip'].tolist()}
    joblib.dump(models, OUT/'models.joblib')
    write_csv(OUT/'tuning.csv', tuning)
    write_json(OUT/'model_selection.json', selection)
    # Persist the model choice before computing independent gate summaries.
    gates = {}
    for family, bundle in models.items():
        gates[family] = {}
        for drift in engine.DRIFTS:
            rows = [s for s in byrole['gate'] if drift in s['drifts']]
            raw = np.array([s['pg_rows'] for s in rows])
            pred = predict(bundle, [s['row'] for s in rows], raw)
            report = n.gate_summary(raw, np.array([s['actual'] for s in rows]), pred, scalar)
            report['unique_subqueries'] = len(rows)
            report['passed'] = bool(report['passed'] and len(rows) >= 30)
            gates[family][drift] = report
    write_json(OUT/'estimator_gates.json', gates)
    complete('train', ['models.joblib', 'tuning.csv', 'model_selection.json', 'estimator_gates.json'])
    n.emit('native_models_frozen', gates={f: {d: x['passed'] for d, x in v.items()} for f, v in gates.items()})


def connect():
    conn = n.connect()
    with conn.cursor() as cur:
        cur.execute("SET application_name='perseus_uc4d_direct_native'")
        cur.execute("SELECT application_name FROM pg_stat_activity WHERE pid != pg_backend_pid() AND state='active'")
        assert not cur.fetchall(), 'Another active DB session; native timing must be isolated'
    return conn


def maps():
    verify('prepare')
    verify('train')
    start('maps')
    models = joblib.load(OUT/'models.joblib')
    cases = read_json(OUT/'population.json')
    observations, maprows, costs = [], [], []
    conn = connect()
    with conn.cursor() as cur:
        cur.execute('SELECT version()')
        version = cur.fetchone()[0]
        cur.execute("SELECT name,setting FROM pg_settings WHERE name IN ('jit','max_parallel_workers_per_gather',"
                    "'statement_timeout','work_mem','plan_cache_mode','join_collapse_limit','random_page_cost')")
        settings = dict(cur.fetchall())
        for pos, case in enumerate(cases):
            rows, pg, batch = [], [], []
            tick = time.perf_counter()
            for aliases, row in engine.parts(case['row']):
                sql = n.subplans.subset_sql(case['row'], aliases)
                cur.execute('EXPLAIN (FORMAT JSON) ' + sql)
                doc = n.injection.plan_doc(cur.fetchone()[0])
                estimate = n.injection.collect_plan_stats(doc)['root_input_estimate']
                assert estimate is not None
                pg.append(max(1., float(estimate)))
                rows.append(row)
                batch.append({'qid': case['qid'], 'aliases': ' '.join(aliases), 'role': case['role'],
                    'drift': case['drift'], 'semantic_id': identity(row), 'tables': len(aliases),
                    'is_root': len(aliases) == case['tables'], 'pg_rows': pg[-1]})
            explain_ms = (time.perf_counter()-tick)*1000
            tick = time.perf_counter()
            estimates = {family: predict(bundle, rows, pg) for family, bundle in models.items()}
            inference_ms = (time.perf_counter()-tick)*1000
            case['root_pg_rows'] = pg[-1]
            case['root_ratios'] = {f: float(p[-1]/pg[-1]) for f, p in estimates.items()}
            case['root_estimates'] = {'pg_map': pg[-1], **{f: float(p[-1]) for f, p in estimates.items()}}
            for j, record in enumerate(batch):
                for method, value in {'pg_map': pg[j], **{f: float(p[j]) for f, p in estimates.items()}}.items():
                    maprows.append({'query_id': case['qid'], 'aliases': record['aliases'], 'method': method, 'estimate': value})
            observations.extend(batch)
            costs.append({'qid': case['qid'], 'role': case['role'], 'explain_ms': explain_ms,
                'four_model_inference_ms': inference_ms, 'subqueries': len(batch)})
            if (pos+1) % 30 == 0:
                n.emit('native_map_preparation', parents=pos+1)
    conn.close()
    processes = []
    for p in n.psutil.process_iter(['pid', 'name', 'cmdline']):
        if p.info['name'] and p.info['name'].lower().startswith(('python', 'postgres')):
            processes.append({'pid': p.info['pid'], 'name': p.info['name'],
                'entrypoint': next((x for x in (p.info['cmdline'] or [])[1:] if x.endswith('.py')), None)})
    write_json(OUT/'cases.json', cases)
    write_csv(OUT/'native_subplans.csv', observations)
    write_csv(MAP, maprows)
    write_csv(OUT/'map_costs.csv', costs)
    write_json(OUT/'environment.json', {'version': version, 'settings': settings, 'python': sys.version,
        'sklearn': __import__('sklearn').__version__, 'processes': processes,
        'callback_sha256': n.hash_file(ROOT/'baselines/End-to-End-CardEst-Benchmark/postgresql-13.1/src/backend/optimizer/path/costsize.c')})
    complete('maps', ['cases.json', 'native_subplans.csv', 'estimate_map.csv', 'map_costs.csv', 'environment.json'])


def active_method(case, method, rules):
    if method not in POLICIES:
        return method
    family = method.split('__', 1)[1]
    return family if engine.matches(case, rules[method]['rule'], family) else 'pg'


def request(cur, case, method, rules):
    begin = time.perf_counter()
    active = active_method(case, method, rules)
    decision_ms = (time.perf_counter()-begin)*1000
    tick = time.perf_counter()
    activation_ms, select_ms = 0., 0.
    count, status, error = None, 'ok', ''
    try:
        engine.activate(cur, case, active)
        activation_ms = (time.perf_counter()-tick)*1000
        tick = time.perf_counter()
        cur.execute(case['sql'])
        count = int(cur.fetchone()[0])
        select_ms = (time.perf_counter()-tick)*1000
        if case['truth'] is None:
            status = 'reference_unavailable'
        elif count != int(case['truth']):
            status = 'count_conflict'
    except Exception as exc:
        select_ms = (time.perf_counter()-tick)*1000
        status = 'timeout' if isinstance(exc, n.psycopg.errors.QueryCanceled) else 'error'
        error = f'{type(exc).__name__}: {str(exc)[:200]}'
    return {'qid': case['qid'], 'sql_id': case['sql_id'], 'role': case['role'], 'drift': case['drift'],
        'method': method, 'active': active, 'status': status, 'count': count,
        'archived_label_match': None if count is None else count == int(case['archived_truth']),
        'wall_ms': (time.perf_counter()-begin)*1000, 'decision_ms': decision_ms,
        'activation_ms': activation_ms, 'select_ms': select_ms, 'error': error}


def measure(role):
    verify('maps')
    if role == 'test':
        verify('select')
    start(role)
    rules = read_json(OUT/'selected_rules.json') if role == 'test' else {}
    cases = [c for c in read_json(OUT/'cases.json') if c['role'] == role]
    methods = METHODS + (POLICIES if role == 'test' else ())
    rows, warmups, reference, blocks = [], [], [], []
    rng = random.Random(SEED + (role == 'test'))
    conn = connect()
    bind_engine()
    with conn.cursor() as cur:
        for pos, case in enumerate(cases):
            tick = time.perf_counter()
            cur.execute('SET perseus_estmap_enabled=off')
            count, status, error = None, 'ok', ''
            try:
                cur.execute(case['sql'])
                count = int(cur.fetchone()[0])
            except Exception as exc:
                status = 'timeout' if isinstance(exc, n.psycopg.errors.QueryCanceled) else 'error'
                error = f'{type(exc).__name__}: {str(exc)[:200]}'
            case['truth'] = count
            reference.append({'qid': case['qid'], 'sql_id': case['sql_id'], 'role': role,
                'drift': case['drift'], 'actual': count, 'archived': case['archived_truth'], 'status': status,
                'archived_match': None if count is None else count == int(case['archived_truth']),
                'wall_ms': (time.perf_counter()-tick)*1000, 'error': error})
            if (pos+1) % 10 == 0:
                write_csv(OUT/f'{role}_references.csv', reference)
                n.emit('reference_acquisition', role=role, parents=pos+1,
                    statuses=dict(Counter(x['status'] for x in reference)))
        write_csv(OUT/f'{role}_references.csv', reference)
        write_json(OUT/f'{role}_count_cases.json', cases)
        engine.audit_plans(cur, cases, methods, rules, role)
        for repetition in (-1, 0, 1, 2):
            order = list(cases)
            rng.shuffle(order)
            tick = time.perf_counter()
            for pos, case in enumerate(order):
                conditions = list(methods)
                rng.shuffle(conditions)
                for rank, method in enumerate(conditions):
                    record = request(cur, case, method, rules)
                    record.update(repetition=repetition, position=pos, condition_rank=rank)
                    (warmups if repetition == -1 else rows).append(record)
                if (pos+1) % 10 == 0:
                    current = warmups if repetition == -1 else rows
                    write_csv(OUT/f'{role}_{"warmup" if repetition == -1 else "requests"}.csv', current)
                    n.emit('native_measurement', role=role, repetition=repetition, parents=pos+1,
                        requests=len(current), statuses=dict(Counter(x['status'] for x in current)))
            blocks.append({'repetition': repetition, 'elapsed_s': time.perf_counter()-tick, 'unix': time.time()})
    conn.close()
    write_csv(OUT/f'{role}_requests.csv', rows)
    write_csv(OUT/f'{role}_warmup.csv', warmups)
    write_json(OUT/f'{role}_blocks.json', blocks)
    complete(role, [f'{role}_{name}' for name in ('references.csv', 'count_cases.json', 'requests.csv',
        'warmup.csv', 'root_checks.csv', 'join_checks.csv', 'plans.json', 'blocks.json')],
        requests=len(rows), statuses=dict(Counter(r['status'] for r in rows)))


def select():
    verify('validation')
    verify('train')
    assert not (OUT/'test_started.json').exists()
    start('select')
    bind_engine()
    cases = read_json(OUT/'validation_count_cases.json')
    lookup = engine.aggregate('validation')
    base = np.array([lookup[c['qid'], 'pg']['ms'] for c in cases])
    gates = read_json(OUT/'estimator_gates.json')
    searches, selected = [], {}
    for policy in POLICIES:
        stage, family = policy.split('__', 1)
        values = np.array([lookup[c['qid'], family]['ms'] for c in cases])
        ratios, best = values/base, None
        for index, rule in enumerate(engine.candidate_rules()):
            if stage == 'full' and not all(gates[family][d]['passed'] for d in rule['drifts']):
                continue
            mask = np.array([engine.matches(c, rule, family) for c in cases])
            support = int(mask.sum())
            if support < 20:
                continue
            failures = sum(lookup[c['qid'], m]['failed'] for c, yes in zip(cases, mask) if yes for m in ('pg', family))
            q50, q95 = np.percentile(ratios[mask], [50, 95])
            loss = float(np.mean(ratios[mask] > 1.01))
            whole = np.where(mask, values, base)
            p95 = float(np.percentile(whole, 95)/np.percentile(base, 95))
            savings = float(base.sum()-whole.sum())
            passed = bool(not failures and q50 <= .98 and q95 <= 1.10 and loss <= .25 and p95 <= 1.01 and savings > 0)
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
    complete('select', ['rule_search.json', 'selected_rules.json'], test_was_absent=True,
             validation_sha256=n.hash_file(OUT/'validation_requests.csv'))
    n.emit('native_policies_frozen', policies=selected)


def bind_engine():
    # Reuse audited injection/EXPLAIN helpers while isolating every output path.
    engine.OUT, engine.MAP, engine.SEED = OUT, MAP, SEED
    engine.METHODS, engine.POLICIES = METHODS, POLICIES
    engine.active_method = active_method


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('prepare', 'train', 'maps', 'validation', 'select', 'test', 'all'))
    args = parser.parse_args()
    for phase in ('prepare', 'train', 'maps', 'validation', 'select', 'test') if args.phase == 'all' else (args.phase,):
        if phase in ('validation', 'test'):
            measure(phase)
        else:
            globals()[phase]()
