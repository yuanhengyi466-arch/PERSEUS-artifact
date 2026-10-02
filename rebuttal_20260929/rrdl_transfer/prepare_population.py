"""Select fresh native parents using SQL structure only."""
from __future__ import annotations

from collections import Counter
import itertools
from pathlib import Path
import sys
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(OUT.parent/'uc4d_followup'))
import run_native as source

n, identity, engine = source.n, source.identity, source.engine
read_json, write_json, write_csv = n.read_json, n.write_json, n.write_csv
TARGETS = {'column': (10,20), 'table': (20,40), 'value': (30,60)}
SEED = 912


def main():
    assert not (OUT/'population.json').exists(), 'Refusing population overwrite'
    inputs, datasets = {}, {}
    for drift, folder in n.METRIC_FOLDERS.items():
        metapath = ROOT/'research_plan'/folder/'seed_1/metrics.json'
        meta = read_json(metapath)
        target = metapath.parent/'data'/meta['provenance']['predict_rel']
        rows, _ = n.repair.load_target_rows(target)
        datasets[drift] = rows
        inputs[str(target)] = n.hash_file(target)
        inputs[str(metapath)] = n.hash_file(metapath)
    old_parents, old_subqueries = set(), set()

    def register(row):
        sid = identity(row)
        if sid in old_parents:
            return
        old_parents.add(sid)
        old_subqueries.update(identity(subrow) for _, subrow in engine.parts(row))

    for path in sorted((ROOT/'paper').rglob('native_callback_detail.csv')):
        inputs[str(path)] = n.hash_file(path)
        for row in n.csv_rows(path):
            register(datasets[row['drift']][int(row['query_index'])])
    paths = [n.OUT/'cases.json', OUT.parent/'native_subplan_diagnostic/cases.json',
        OUT.parent/'native_subplan_learning/population.json',
        OUT.parent/'native_subplan_strength/population.json',
        OUT.parent/'uc4d_followup/native/population.json']
    for path in paths:
        inputs[str(path)] = n.hash_file(path)
        for case in read_json(path):
            register(case['row'])
    for path in (OUT.parent/'uc4d_followup/native/development.json',
                 OUT.parent/'native_subplan_learning/development_samples.json'):
        inputs[str(path)] = n.hash_file(path)
        for sample in read_json(path):
            old_subqueries.add(identity(sample['row']))
    selected, membership, funnels = [], [], []
    used = set(old_subqueries)
    role_ids = {'validation': set(), 'test': set()}
    for drift, dataset in datasets.items():
        unique = {}
        for i, row in enumerate(dataset):
            unique.setdefault(identity(row), i)
        funnel = Counter(official_rows=len(dataset), unique_sql=len(unique))
        sizes = TARGETS[drift]
        pos, available = 0, 0
        for sid, index in sorted(unique.items(), key=lambda item: n.rank_key(item[0], SEED, 'rrdl-transfer')):
            row = dataset[index]
            if sid in old_parents:
                funnel['previous_native_parent'] += 1
                continue
            parts = engine.parts(row)
            ids = {identity(r) for _, r in parts}
            if ids & old_subqueries:
                funnel['previous_connected_subquery'] += 1
                continue
            available += 1
            if ids & used:
                funnel['new_selected_subquery_overlap'] += 1
                continue
            if pos >= sum(sizes):
                funnel['unselected_after_cap'] += 1
                continue
            role = 'validation' if pos%3 == 2 else 'test'
            case = {'qid': f'rr912:{drift}:q{index}', 'index': index, 'sql_id': sid,
                'drift': drift, 'role': role, 'row': row, 'sql': n.injection.query_to_sql(row),
                'tables': len(row[0].split(',')), 'archived_truth': float(row[3])}
            selected.append(case)
            for aliases, subrow in parts:
                membership.append({'qid': case['qid'], 'role': role, 'drift': drift,
                    'semantic_id': identity(subrow), 'aliases': ' '.join(aliases), 'is_root': len(aliases)==case['tables']})
            role_ids[role].update(ids)
            used.update(ids)
            funnel[role] += 1
            pos += 1
        funnel['structurally_novel_before_new_overlap'] = available
        funnels.append({'drift': drift, **dict(funnel)})
        n.emit('transfer_structural_selection', drift=drift, selected=pos, needed=sum(sizes), funnel=dict(funnel))
    write_json(OUT/'eligibility_funnel.json', funnels)
    enough = all(sum(c['role']==role and c['drift']==drift for c in selected)==count
                 for drift, sizes in TARGETS.items() for role,count in zip(('validation','test'),sizes))
    write_json(OUT/'structural_preflight.json', {'unix': time.time(), 'sufficient_population': enough,
        'source_sha256': n.hash_file(__file__), 'preflight_plan_sha256': n.hash_file(OUT/'PREFLIGHT.md'),
        'inputs': inputs, 'prior_native_unique_parents': len(old_parents),
        'prior_native_unique_connected_queries': len(old_subqueries), 'selected_parents': len(selected),
        'role_unique_subqueries': {r: len(ids) for r,ids in role_ids.items()},
        'cross_role_subquery_overlap': len(role_ids['validation'] & role_ids['test'])})
    assert enough, 'Insufficient structural availability; no labels or measurements collected'
    assert not role_ids['validation'] & role_ids['test']
    assert not (role_ids['validation'] | role_ids['test']) & old_subqueries
    write_json(OUT/'population.json', selected)
    write_csv(OUT/'membership.csv', membership)
    write_json(OUT/'population_complete.json', {'unix': time.time(),
        'outputs': {f: n.hash_file(OUT/f) for f in ('population.json','membership.csv','eligibility_funnel.json','structural_preflight.json')}})


if __name__ == '__main__':
    main()
