"""Strengthen split audit using equality closure before any native final test."""
from collections import defaultdict
from decimal import Decimal, InvalidOperation
import hashlib
import itertools
import json
import time

import run_learning as r


def identity(row):
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    for expr, _, _ in r.n.subplans.parse_joins(row[1]):
        a, b = [x.strip().lower() for x in expr.split('=')]
        parent[find(a)] = find(b)
    groups = defaultdict(list)
    for key in list(parent):
        groups[find(key)].append(key)
    joins = sorted(sorted(v) for v in groups.values())
    pieces = row[2].split(',') if row[2] else []
    predicates = []
    for i in range(0, len(pieces), 3):
        col, op, value = pieces[i:i+3]
        try:
            value = str(Decimal(value).normalize())
        except InvalidOperation:
            pass
        predicates.append((col.strip().lower(), op.strip(), value))
    canonical = [sorted(x.lower().strip() for x in row[0].split(',')), joins, sorted(predicates)]
    return hashlib.sha256(json.dumps(canonical).encode()).hexdigest()


def selected_test():
    doc = r.read_json(r.OUT/'test_eligibility_amendment.json')
    assert doc['audit_source_sha256'] == r.n.hash_file(__file__)
    excluded = {x['qid'] for x in doc['excluded_test_parents']}
    return [c for c in r.read_json(r.OUT/'cases.json') if c['role'] == 'test' and c['qid'] not in excluded]


def audit():
    assert not (r.OUT/'test_started.json').exists()
    assert not (r.OUT/'test_eligibility_amendment.json').exists()
    r.verify('prepare')
    roles = defaultdict(set)
    samples = r.read_json(r.OUT/'development_samples.json')
    for sample in samples:
        roles[sample['role']].add(identity(sample['row']))
    cases = r.read_json(r.OUT/'cases.json')
    for case in cases:
        if case['role'] == 'validation':
            roles['validation'].update(identity(row) for _, row in r.parts(case['row']))
    for a, b in itertools.combinations(roles, 2):
        assert not roles[a] & roles[b], (a, b)
    excluded, accepted = [], []
    for case in cases:
        if case['role'] != 'test':
            continue
        ids = {identity(row) for _, row in r.parts(case['row'])}
        overlap = {role: sorted(ids & used) for role, used in roles.items() if ids & used}
        if overlap:
            excluded.append({'qid': case['qid'], 'sql_id': case['sql_id'], 'drift': case['drift'], 'overlaps': overlap})
        else:
            accepted.append(case['qid'])
            roles['test'].update(ids)
    for a, b in itertools.combinations(roles, 2):
        assert not roles[a] & roles[b], (a, b)
    full_queries = {role: [identity(c['row']) for c in cases if c['role'] == role and (role != 'test' or c['qid'] in accepted)]
                    for role in ('validation', 'test')}
    assert all(len(x) == len(set(x)) for x in full_queries.values())
    r.write_json(r.OUT/'test_eligibility_amendment.json', {
        'unix': time.time(), 'audit_source_sha256': r.n.hash_file(__file__),
        'protocol_amendment_sha256': r.n.hash_file(r.OUT/'AMENDMENT_EQUIVALENCE.md'),
        'selected_before_test': True, 'selection_uses_only_sql_structure': True,
        'excluded_test_parents': excluded, 'accepted_test_qids': accepted,
        'final_test_parents': len(accepted), 'validation_parents': len(full_queries['validation']),
        'semantic_unique_subqueries_by_role': {role: len(ids) for role, ids in roles.items()},
        'all_cross_role_equality_normalized_overlap': 0,
        'fit_successful_records': sum(x['role'] == 'fit' and x['actual'] is not None for x in samples),
        'fit_successful_semantic_unique_labels': len({identity(x['row']) for x in samples if x['role'] == 'fit' and x['actual'] is not None}),
        'original_population_sha256': r.n.hash_file(r.OUT/'population.json'),
        'cases_sha256': r.n.hash_file(r.OUT/'cases.json')})
    r.n.emit('test_semantic_eligibility_frozen', excluded=excluded, final_test_parents=len(accepted))


if __name__ == '__main__':
    audit()
