"""Run the structurally corrected final cohort using the frozen native engine."""
import argparse
from collections import Counter
import random
import time

import run_learning as r
from semantic_audit import selected_test


def measure():
    r.verify('maps')
    r.verify('select')
    cases = selected_test()
    r.start('test')
    rules = r.read_json(r.OUT/'selected_rules.json')
    methods = list(r.METHODS + r.POLICIES)
    rows, warmups, blocks = [], [], []
    conn = r.connect()
    with conn.cursor() as cur:
        cur.execute("SELECT application_name FROM pg_stat_activity WHERE pid<>pg_backend_pid() AND state='active'")
        assert not cur.fetchall()
        r.audit_plans(cur, cases, methods, rules, 'test')
        rng = random.Random(r.SEED+1)
        order = list(cases)
        rng.shuffle(order)
        for pos, case in enumerate(order):
            conditions = list(methods)
            rng.shuffle(conditions)
            for method in conditions:
                warmups.append(r.request(cur, case, method, rules))
            if (pos+1) % 30 == 0:
                r.write_csv(r.OUT/'test_warmup.csv', warmups)
                r.n.emit('test_warmup', parents=pos+1, total=len(cases), statuses=dict(Counter(x['status'] for x in warmups)))
        r.write_csv(r.OUT/'test_warmup.csv', warmups)
        for repetition in range(r.REPEATS):
            order = list(cases)
            rng.shuffle(order)
            for pos, case in enumerate(order):
                conditions = list(methods)
                rng.shuffle(conditions)
                for rank, method in enumerate(conditions):
                    row = r.request(cur, case, method, rules)
                    row.update(repetition=repetition, position=pos, condition_rank=rank)
                    rows.append(row)
                if (pos+1) % 30 == 0:
                    r.write_csv(r.OUT/'test_requests.csv', rows)
                    r.n.emit('measurement', role='test', repetition=repetition, parents=pos+1, total=len(cases),
                             statuses=dict(Counter(x['status'] for x in rows)))
            blocks.append({'repetition': repetition, 'unix_finished': time.time(), 'requests_so_far': len(rows)})
    conn.close()
    r.write_csv(r.OUT/'test_requests.csv', rows)
    r.write_json(r.OUT/'test_blocks.json', blocks)
    r.stamp('test', [f'test_{name}' for name in ('requests.csv', 'warmup.csv', 'root_checks.csv', 'join_checks.csv', 'plans.json', 'blocks.json')],
        requests=len(rows), statuses=dict(Counter(x['status'] for x in rows)),
        test_driver_sha256=r.n.hash_file(__file__), amendment_sha256=r.n.hash_file(r.OUT/'test_eligibility_amendment.json'))


def counts():
    r.verify('test')
    cases = selected_test()
    qids = {c['qid'] for c in cases}
    r.start('counts')
    rows = [x for x in r.n.csv_rows(r.OUT/'native_subplans.csv') if x['qid'] in qids]
    output = []
    conn = r.connect()
    with conn.cursor() as cur:
        cur.execute('SET perseus_estmap_enabled=off')
        for pos, row in enumerate(rows):
            tick = time.perf_counter()
            result = {**row, 'actual': None, 'status': 'ok', 'count_ms': None, 'error': ''}
            try:
                cur.execute(row['sql'])
                result['actual'] = int(cur.fetchone()[0])
            except Exception as exc:
                result['status'] = 'timeout' if isinstance(exc, r.n.psycopg.errors.QueryCanceled) else 'error'
                result['error'] = f'{type(exc).__name__}: {str(exc)[:200]}'
            result['count_ms'] = (time.perf_counter()-tick)*1000
            output.append(result)
            if (pos+1) % 100 == 0:
                r.write_csv(r.OUT/'test_subquery_counts.csv', output)
                r.n.emit('test_count_audit', completed=pos+1, total=len(rows), statuses=dict(Counter(x['status'] for x in output)))
    conn.close()
    r.write_csv(r.OUT/'test_subquery_counts.csv', output)
    r.stamp('counts', ['test_subquery_counts.csv'], statuses=dict(Counter(x['status'] for x in output)),
            count_driver_sha256=r.n.hash_file(__file__), amendment_sha256=r.n.hash_file(r.OUT/'test_eligibility_amendment.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('test', 'counts'))
    if parser.parse_args().phase == 'test':
        measure()
    else:
        counts()
