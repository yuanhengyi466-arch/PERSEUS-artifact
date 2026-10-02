"""Read-only source audit for UC4d; outputs stay in this supplement directory."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path, PureWindowsPath
import sys

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(OUT.parent / 'native_subplan_learning'))
from semantic_audit import identity


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(name, rows):
    assert rows, name
    with (OUT / name).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def index_hash(indices):
    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()


def local_path(saved):
    parts = PureWindowsPath(saved).parts
    index = next(i for i, part in enumerate(parts) if part in ('research_plan', 'paper'))
    path = ROOT.joinpath(*parts[index:]).resolve()
    assert path.is_relative_to(ROOT.resolve()) and path.is_file(), path
    return path


def main():
    inputs = {}
    cache = {}

    def track(path):
        inputs[str(path)] = digest(path)
        return path

    def workload(path):
        if path not in cache:
            track(path)
            with path.open(encoding='utf-8', newline='') as handle:
                rows = list(csv.reader(handle, delimiter='#'))
            assert all(len(row) == 4 for row in rows)
            keys = [identity(row) for row in rows]
            truths = defaultdict(set)
            for row, key in zip(rows, keys):
                truths[key].add(int(row[3]))
            cache[path] = (rows, keys, sum(len(values) > 1 for values in truths.values()))
        return cache[path]

    original = ROOT / 'paper/workload_drift_residual_field/results'
    budget_rows = read_csv(track(original / 'budget_sweep.csv'))
    detail, summary, targets = [], [], {}
    for record in budget_rows:
        drift, method, budget = record['drift'], record['method'], int(record['adaptation_rows'])
        path = original / 'gate_sweep' / f'{drift}_{method}_{budget}.json'
        report = read_json(track(path))
        assert report['passed'] == (record['passed'] == 'True')
        target = local_path(report['target_workload'])
        targets[drift] = target
        rows, keys, conflicts = workload(target)
        fit_union, eval_union = set(), set()
        for split in report['split_reports']:
            order = np.arange(len(rows))
            np.random.default_rng(split['seed']).shuffle(order)
            fit, evaluation = order[:budget], order[budget:]
            assert split.get('fit_rows', budget) == len(fit)
            assert split.get('gate_rows', 0) == 0
            assert split['eval_rows'] == len(evaluation)
            assert split['adaptation_indices_sha256'] == index_hash(fit)
            assert split['eval_indices_sha256'] == index_hash(evaluation)
            fit_ids, eval_ids = {keys[i] for i in fit}, {keys[i] for i in evaluation}
            fit_union.update(int(i) for i in fit)
            eval_union.update(int(i) for i in evaluation)
            detail.append({'drift': drift, 'method': method, 'budget': budget, 'seed': split['seed'],
                'fit_rows': len(fit), 'evaluation_rows_used_for_admission': len(evaluation),
                'total_label_rows': len(rows), 'fit_unique_sql': len(fit_ids),
                'admission_eval_unique_sql': len(eval_ids), 'fit_eval_sql_overlap': len(fit_ids & eval_ids),
                'total_unique_sql': len(fit_ids | eval_ids), 'independent_final_test_rows': 0,
                'identity_groups_with_conflicting_archived_labels': conflicts,
                'split_hashes_verified': True})
        union = fit_union | eval_union
        assert len(union) == len(rows)
        summary.append({'drift': drift, 'method': method, 'fit_rows_per_seed': budget,
            'admission_evaluation_rows_per_seed': len(rows)-budget,
            'total_label_rows_per_seed': len(rows), 'unique_label_rows_across_seeds': len(union),
            'unique_fit_rows_across_seeds': len(fit_union), 'unique_eval_rows_across_seeds': len(eval_union),
            'cross_seed_fit_eval_overlap_rows': len(fit_union & eval_union),
            'total_unique_sql': len(set(keys)), 'independent_final_test_rows': 0,
            'p95_reduction': record['p95_reduction'], 'p99_reduction': record['p99_reduction'],
            'all_seed_gate_passed': report['passed'], 'measured_historical_acquisition_seconds': '',
            'interpretation': 'Fitting allocation at fixed fully labeled target pool; evaluation also makes admission decision.'})
    write_csv('budget_total_labels_by_seed.csv', detail)
    write_csv('budget_total_labels.csv', summary)

    nested = read_json(track(original / 'nested_final_test.json'))
    nested_rows = []
    for drift, splits in nested['split_reports'].items():
        rows, keys, conflicts = workload(targets[drift])
        fit_union, gate_union = set(), set()
        for split in splits:
            nfit, ngate = split['fit_rows'], split['gate_rows']
            assert nfit + ngate + split['final_rows'] == len(rows)
            order = np.arange(len(rows))
            np.random.default_rng(split['seed']).shuffle(order)
            parts = {'fit': order[:nfit], 'gate': order[nfit:nfit+ngate], 'final': order[nfit+ngate:]}
            for role, indices in parts.items():
                assert index_hash(indices) == split[f'{role}_indices_sha256']
            fit_union.update(int(i) for i in parts['fit'])
            gate_union.update(int(i) for i in parts['gate'])
        labels = fit_union | gate_union
        nested_rows.append({'drift': drift, 'fit_rows_per_split': nfit, 'gate_rows_per_split': ngate,
            'unique_fit_rows_across_seeds': len(fit_union), 'unique_gate_rows_across_seeds': len(gate_union),
            'fit_gate_union_rows': len(labels), 'cross_seed_fit_gate_overlap_rows': len(fit_union & gate_union),
            'fit_gate_union_unique_sql': len({keys[i] for i in labels}),
            'identity_groups_with_conflicting_archived_labels_in_full_target': conflicts,
            'all_index_hashes_verified': True, 'scope': 'Supervision union across repeated splits; final-test labels excluded.'})
    write_csv('nested_total_labels.csv', nested_rows)

    sscard = read_csv(track(ROOT / 'rebuttal_20260928/audit/sscard_budget_gate_recheck.csv'))
    for row in sscard:
        path = ROOT / 'research_plan' / row['report_file']
        assert digest(track(path)) == row['sha256']
    write_csv('sscard_eq5_passing_reports.csv', [row for row in sscard if row['eq5_all_splits_passed'] == 'True'])

    learning = OUT.parent / 'native_subplan_learning'
    errors = read_csv(track(learning / 'test_subplan_error_detail.csv'))
    opposites = []
    for drift in ('all', 'column', 'table', 'value'):
        for size in (0, 2, 3, 4, 5, 6):
            eligible = [row for row in errors if row['method'] == 'pg' and row['is_root'] == 'False'
                and row['root_needed_opposite'] in ('True', 'False')
                and (drift == 'all' or row['drift'] == drift)
                and (size == 0 or int(row['tables']) == size)]
            if not eligible:
                continue
            opposite = [row for row in eligible if row['root_needed_opposite'] == 'True']
            opposites.append({'drift': drift, 'subplan_tables': size or 'all',
                'labeled_intermediates_with_labeled_root': len(eligible), 'opposite_direction_records': len(opposite),
                'opposite_fraction': len(opposite)/len(eligible),
                'observed_parent_count': len({row['qid'] for row in eligible}),
                'parents_with_any_observed_opposite': len({row['qid'] for row in opposite}),
                'scope': 'Only observed root/intermediate COUNT pairs; no imputation for timeout labels.'})
    write_csv('heldout_root_intermediate_directions.csv', opposites)
    size_rows = read_csv(track(learning / 'test_subplan_error_summary.csv'))
    write_csv('error_by_subplan_size.csv', [row for row in size_rows
        if row['drift'] == 'all' and row['kind'] == 'intermediate' and row['size'] != '0'])
    runtime = read_csv(track(learning / 'test_runtime_summary.csv'))
    write_csv('runtime_by_parent_size.csv', [row for row in runtime if row['population'].startswith('tables_')])
    original_submission = ROOT / 'rebuttal_20260928/submission_original/PERSEUS-paper/PERSEUS'
    for filename, expected in {
        'main_kdd2027.pdf': '44fd3345505823434994b332bc255cc33fe07a1495970d5defcce9a29062862a',
        'main_kdd2027.tex': '8e9a374baf8c301996bc1b66452c9eb41a62f23308e3a19cd9c462433f021e3c',
    }.items():
        assert digest(track(original_submission / filename)) == expected
    output_names = ('budget_total_labels_by_seed.csv', 'budget_total_labels.csv', 'nested_total_labels.csv',
        'sscard_eq5_passing_reports.csv', 'heldout_root_intermediate_directions.csv',
        'error_by_subplan_size.csv', 'runtime_by_parent_size.csv')
    audit = {'completed': True, 'kind': 'Existing evidence audit; no model fitting or SQL execution',
        'budget_points': len(summary), 'budget_split_hash_pairs_verified': len(detail),
        'budget_unique_totals': {d: sorted({r['total_label_rows_per_seed'] for r in summary if r['drift'] == d}) for d in targets},
        'nested_labels': nested_rows, 'sscard_reports_verified': len(sscard),
        'sscard_eq5_passing_reports': sum(row['eq5_all_splits_passed'] == 'True' for row in sscard),
        'observed_heldout_directions': opposites[0], 'original_submission_unchanged': True,
        'inputs': inputs, 'outputs': {name: digest(OUT / name) for name in output_names},
        'script_sha256': digest(__file__)}
    (OUT / 'audit_complete.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in audit.items() if key not in ('inputs', 'outputs')}, indent=2))


if __name__ == '__main__':
    main()
