"""Total-supervision curves with fixed SQL-grouped test sets."""
from __future__ import annotations

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '2'

import argparse
from collections import defaultdict
import itertools
from pathlib import Path
import sys
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(OUT.parent / 'reviewer_4udk'))
import run_estimator_controls as old
sys.path.insert(0, str(OUT.parent / 'native_subplan_learning'))
from semantic_audit import identity

np = old.np
SEEDS = (711, 712, 713)
FAMILIES = ('residual_hgb', 'direct_hgb', 'residual_rf', 'direct_rf')
BUDGETS = {'column': (500, 1000, 2000, 3000, 4000),
           'table': (500, 1000, 2000, 3000, 4000), 'value': (500, 1000)}
write_json, read_json, write_csv = old.write_json, old.read_json, old.write_csv


def load(drift):
    base = ROOT / 'research_plan' / old.FOLDERS[drift] / 'seed_1'
    meta_path = base / 'metrics.json'
    meta = read_json(meta_path)
    target = base / 'data' / meta['provenance']['predict_rel']
    source = base / 'data' / meta['provenance']['train_rel']
    rows, truth = old.repair.load_target_rows(target)
    prediction = old.relocated(meta['artifacts']['predictions'])
    raw = np.maximum(1., np.loadtxt(prediction, dtype=np.float64))
    assert len(raw) == len(rows)
    sids = np.array([identity(r) for r in rows])
    first = {}
    for i, sid in enumerate(sids):
        if sid in first:
            assert truth[i] == truth[first[sid]], 'Conflicting equivalent SQL labels'
        first.setdefault(sid, i)
    source_rows, _ = old.repair.load_target_rows(source)
    assert not set(sids) & {identity(r) for r in source_rows}
    return {'rows': rows, 'labels': truth, 'raw': raw, 'sids': sids,
            'unique': list(first.values()), 'inputs': [meta_path, target, source, prediction]}


def roles_for(data, seed):
    roles = {r: [] for r in ('fit', 'tune', 'gate', 'test')}
    for i in data['unique']:
        sid = data['sids'][i]
        u = int(old.digest(f'uc4d-budget-v1|{seed}|{sid}'.encode())[:16], 16) / 2**64
        role = 'fit' if u < .5 else 'tune' if u < .65 else 'gate' if u < .8 else 'test'
        roles[role].append(i)
    for indices in roles.values():
        indices.sort(key=lambda i: old.digest(f'prefix|{seed}|{data["sids"][i]}'.encode()))
    return roles


def run(drift):
    directory = OUT / 'budget' / drift
    directory.mkdir(parents=True, exist_ok=True)
    assert not (directory / 'started.json').exists(), 'Refusing observation overwrite'
    data = load(drift)
    plans = {}
    for seed in SEEDS:
        roles = roles_for(data, seed)
        for budget in BUDGETS[drift]:
            sizes = {'fit': budget * 3 // 5, 'tune': budget // 5, 'gate': budget // 5}
            assert sum(sizes.values()) == budget
            assert all(len(roles[r]) >= count for r, count in sizes.items()), (drift, seed, sizes)
            splits = {r: indices[:sizes[r]] if r in sizes else indices for r, indices in roles.items()}
            plans[f'{seed}/{budget}'] = splits
    hashes = {str(p): old.file_hash(p) for p in data['inputs']}
    write_json(directory / 'started.json', {'unix': time.time(), 'command': sys.argv,
        'inputs': hashes, 'protocol_sha256': old.file_hash(OUT/'PROTOCOL.md'),
        'script_sha256': old.file_hash(__file__),
        'helper_sha256': old.file_hash(old.__file__), 'identity_sha256': old.file_hash(OUT.parent/'native_subplan_learning/semantic_audit.py'),
        'python': sys.version, 'sklearn': __import__('sklearn').__version__})
    write_json(directory/'all_splits.json', plans)
    metrics, ledger, membership, gates = [], [], [], []
    with old.threadpool_limits(limits=2):
        for seed in SEEDS:
            for budget in BUDGETS[drift]:
                cell = directory/str(seed)/str(budget)
                cell.mkdir(parents=True, exist_ok=True)
                split = {r: np.asarray(idx, dtype=np.int64) for r, idx in plans[f'{seed}/{budget}'].items()}
                test, gate = split['test'], split['gate']
                write_json(cell/'splits.json', {r: a.tolist() for r, a in split.items()})
                scalar_shift = float(np.median(np.log(data['labels'][split['fit']]) - np.log(data['raw'][split['fit']])))
                scalar = np.maximum(1., data['raw'] * np.exp(scalar_shift))
                stored = {'truth': data['labels'][test], 'raw': data['raw'][test],
                          'scalar': scalar[test], 'sql_id': data['sids'][test]}
                for role, indices in split.items():
                    membership.extend({'drift': drift, 'seed': seed, 'budget': budget, 'role': role,
                        'index': int(i), 'sql_id': str(data['sids'][i])} for i in indices)
                for name, prediction in (('raw', data['raw'][test]), ('scalar', scalar[test])):
                    metrics.append({'drift': drift, 'seed': seed, 'budget': budget, 'method': name,
                        'mode': 'candidate', 'passed': None, **old.metrics(prediction, data['labels'][test])})
                for family in FAMILIES:
                    start = time.perf_counter()
                    prediction, selection = old.train_trees(data, split, family, seed, cell)
                    seconds = time.perf_counter() - start
                    # Selection is persisted by train_trees before observing gate/test.
                    passed = old.gate(data['raw'][gate], scalar[gate], prediction[gate], data['labels'][gate], 'default')
                    gates.append({'drift': drift, 'seed': seed, 'budget': budget, 'method': family,
                        'passed': passed, 'gate_sql': len(gate), 'raw': old.metrics(data['raw'][gate], data['labels'][gate]),
                        'scalar': old.metrics(scalar[gate], data['labels'][gate]),
                        'candidate': old.metrics(prediction[gate], data['labels'][gate])})
                    for mode, p in (('candidate', prediction[test]), ('gated', prediction[test] if passed else data['raw'][test])):
                        metrics.append({'drift': drift, 'seed': seed, 'budget': budget, 'method': family,
                            'mode': mode, 'passed': passed, **old.metrics(p, data['labels'][test])})
                        stored[family + ('_gated' if mode == 'gated' else '')] = p
                    ledger.append({'drift': drift, 'seed': seed, 'budget': budget, 'method': family,
                        'fit_labels': len(split['fit']), 'tune_labels': len(split['tune']),
                        'gate_labels': len(gate), 'total_unique_labels': budget, 'test_sql': len(test),
                        'configurations': len(selection['candidates']), 'selected_config': selection['selected_config_id'],
                        'fit_select_predict_seconds': seconds, 'acquisition_time': 'historical_not_recorded'})
                np.savez_compressed(cell/'test_predictions.npz', **stored)
                old.emit('budget_cell_complete', drift=drift, seed=seed, total_labels=budget,
                         gate_passes={r['method']: r['passed'] for r in gates[-4:]})
                write_csv(directory/'per_split.csv', metrics)
                write_csv(directory/'costs.csv', ledger)
                write_json(directory/'gates.json', gates)
    write_csv(directory/'membership.csv', membership)
    assert all(old.file_hash(p) == sha for p, sha in hashes.items())
    outputs = {str(p.relative_to(directory)): old.file_hash(p) for p in directory.rglob('*') if p.is_file()}
    write_json(directory/'complete.json', {'unix': time.time(), 'outputs': outputs,
        'selected_models': len(ledger), 'configuration_fits': sum(r['configurations'] for r in ledger)})
    old.emit('budget_drift_complete', drift=drift, models=len(ledger))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('drift', choices=tuple(BUDGETS) + ('all',))
    args = parser.parse_args()
    for drift in BUDGETS if args.drift == 'all' else (args.drift,):
        run(drift)
