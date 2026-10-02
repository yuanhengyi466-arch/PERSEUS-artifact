"""Verify measured artifacts, matched membership and unchanged submission."""
from collections import Counter, defaultdict
from pathlib import Path
import re
import time

import run_budget as b
import run_native as n


def main():
    verified = 0
    for drift in b.BUDGETS:
        directory = b.OUT/'budget'/drift
        for rel, digest in b.read_json(directory/'complete.json')['outputs'].items():
            assert b.old.file_hash(directory/rel) == digest, rel
            verified += 1
    for name in ('budget_analysis_complete.json', 'native/analysis_complete.json'):
        manifest_path = b.OUT/name
        manifest = b.read_json(manifest_path)
        for rel, digest in manifest['outputs'].items():
            assert b.old.file_hash(manifest_path.parent/rel) == digest, rel
            verified += 1
    for phase in ('prepare', 'train', 'maps', 'validation', 'select', 'test'):
        n.verify(phase)
    assert b.read_json(n.OUT/'select_complete.json')['unix'] < b.read_json(n.OUT/'test_started.json')['unix']
    policies = b.read_json(n.OUT/'selected_rules.json')
    cases = {c['qid']: c for c in b.read_json(n.OUT/'cases.json')}
    timed_count = 0
    for role in ('validation', 'test'):
        methods = n.METHODS+(n.POLICIES if role == 'test' else ())
        for kind, repeats in (('requests', (0, 1, 2)), ('warmup', (-1,))):
            rows = n.n.csv_rows(n.OUT/f'{role}_{kind}.csv')
            grouped = defaultdict(list)
            for row in rows:
                assert cases[row['qid']]['role'] == role
                assert row['method'] in methods and int(row['repetition']) in repeats
                assert float(row['wall_ms']) > 0
                grouped[row['qid'], row['repetition']].append(row)
                expected = n.active_method(cases[row['qid']], row['method'], policies)
                assert row['active'] == expected
            assert len(grouped) == 90*len(repeats)
            for group in grouped.values():
                assert Counter(x['method'] for x in group) == Counter(methods)
                assert sorted(int(x['condition_rank']) for x in group) == list(range(len(methods)))
            if kind == 'requests':
                timed_count += len(rows)
    assert timed_count == 5400
    budget = n.n.csv_rows(b.OUT/'budget_summary.csv')
    keyed = {(r['drift'], r['budget'], r['method']): r for r in budget}
    for row in budget:
        if not row['method'].endswith('_gated'):
            continue
        comparator = 'raw' if row['gate_passes'] == '0' else row['method'].removesuffix('_gated') if row['gate_passes'] == '3' else None
        if comparator:
            ref = keyed[row['drift'], row['budget'], comparator]
            assert all(float(row[x]) == float(ref[x]) for x in ('median', 'p95', 'p99'))
    documents = list(b.OUT.glob('*.md'))+list((b.OUT.parent/'reviewer_uc4d').glob('*.md'))
    link_count = 0
    for path in documents:
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
            if re.match(r'^[A-Za-z]:[/\\]', target):
                target = re.sub(r':\d+$', '', target)
                assert Path(target).exists(), (path.name, target)
                link_count += 1
    original = b.ROOT/'rebuttal_20260928/submission_original/PERSEUS-paper/PERSEUS'
    for file, digest in {'main_kdd2027.pdf': '44fd3345505823434994b332bc255cc33fe07a1495970d5defcce9a29062862a',
                         'main_kdd2027.tex': '8e9a374baf8c301996bc1b66452c9eb41a62f23308e3a19cd9c462433f021e3c'}.items():
        assert b.old.file_hash(original/file) == digest
    b.write_json(b.OUT/'completion_audit.json', {'unix': time.time(), 'verified_output_hashes': verified,
        'local_links_verified': link_count, 'timed_native_requests': timed_count,
        'method_repetition_pairing_and_active_routes_verified': True,
        'budget_extreme_pass_routing_verified': True, 'original_submission_unchanged': True,
        'budget_models': 144, 'budget_configuration_fits': 576, 'native_configuration_fits': 16,
        'script_hashes': {p.name: b.old.file_hash(p) for p in b.OUT.glob('*.py')},
        'protocol_sha256': b.old.file_hash(b.OUT/'PROTOCOL.md')})
    print({'verified_hashes': verified, 'links': link_count, 'requests': timed_count})


if __name__ == '__main__':
    main()
