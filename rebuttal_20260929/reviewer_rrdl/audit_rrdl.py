"""Read-only rrdL utility, cap-sensitivity and scope audit."""
from __future__ import annotations

import csv
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
NATIVE = OUT.parent/'uc4d_followup/native'
sys.path.insert(0, str(ROOT/'experiments'))
import summarize_native_callback_policy_grid as grid


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def save(name, obj):
    (OUT/name).write_text(json.dumps(obj, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


def table(name, data):
    assert data
    with (OUT/name).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def inputs():
    result = ROOT/'paper/workload_drift_residual_field/results'
    paths = [result/'native_optimizer_callback_hash_full_seeds123/native_callback_detail.csv',
        result/'native_callback_policy_grid_manifest.json',
        ROOT/'experiments/summarize_native_callback_policy_grid.py',
        ROOT/'rebuttal_20260928/audit/native_original_policy_recomputed.csv',
        OUT.parent/'uc4d_followup/budget_gate_passes.csv',
        OUT.parent/'uc4d_followup/budget_summary.csv',
        OUT.parent/'reviewer_uc4d/sscard_eq5_passing_reports.csv']
    paths += [NATIVE/file for file in ('rule_search.json', 'selected_rules.json', 'cases.json',
        'validation_requests.csv', 'test_requests.csv', 'estimator_gates.json',
        'environment.json', 'cost_ledger.json', 'analysis_complete.json')]
    for folder in ('native_subplan_learning', 'native_subplan_strength'):
        paths.append(OUT.parent/folder/'environment.json')
    return [p for p in paths if p.exists()]


def utility():
    result = ROOT/'paper/workload_drift_residual_field/results'
    detail = result/'native_optimizer_callback_hash_full_seeds123/native_callback_detail.csv'
    manifest = load(result/'native_callback_policy_grid_manifest.json')
    groups = grid.load_groups(detail, grid.SPLIT_SEED)
    prior = {(x['policy_class'], x['split']): x for x in rows(ROOT/'rebuttal_20260928/audit/native_original_policy_recomputed.csv')}
    result_rows = []
    for policy in manifest['selected']:
        rule = policy['rule']
        for split in ('validation', 'test'):
            selected = [g for g in groups if g['split'] == split]
            accepted = [g for g in selected if grid.rule_applies(rule, g)]
            old = prior[policy['policy_class'], split]
            assert len(selected) == int(old['groups']) and len(accepted) == int(old['deployed'])
            base = sum(float(g['methods']['default']['execution_ms']) for g in selected)
            accepted_base = sum(float(g['methods']['default']['execution_ms']) for g in accepted)
            accepted_time = sum(float(g['methods'][rule['method']]['execution_ms']) for g in accepted)
            saved = accepted_base-accepted_time
            assert np.isclose(base, float(old['execution_ms_raw_sum']))
            assert np.isclose(base-saved, float(old['execution_ms_policy_sum']))
            result_rows.append({'policy_class': policy['policy_class'], 'split': split,
                'groups': len(selected), 'admitted_groups': len(accepted), 'coverage': len(accepted)/len(selected),
                'admitted_unique_sql': int(old['deployed_unique_sql']), 'baseline_total_execution_ms': base,
                'admitted_baseline_execution_ms': accepted_base,
                'admitted_baseline_time_fraction': accepted_base/base,
                'admitted_candidate_execution_ms': accepted_time,
                'saved_execution_ms': saved, 'whole_execution_saving_fraction': saved/base,
                'saved_planning_execution_ms': float(old['planning_plus_execution_ms_raw_sum'])-float(old['planning_plus_execution_ms_policy_sum']),
                'saved_client_wall_ms': 1000*(float(old['wall_seconds_raw_sum'])-float(old['wall_seconds_policy_sum'])),
                'subset_median_ratio': float(old['subset_median_ratio']),
                'subset_p95_ratio': float(old['subset_p95_ratio']),
                'policy_ratio_p99': float(old['policy_p99_ratio']),
                'policy_ratio_of_latency_p99': float(old['execution_ms_ratio_of_p99'])})
    table('original_policy_utility.csv', result_rows)
    return result_rows


def matches(case, rule, family):
    if rule is None:
        return False
    ratio = case['root_ratios'][family]
    return (case['drift'] in rule['drifts'] and case['tables'] >= rule['min_tables']
        and case['root_pg_rows'] >= rule['min_pg_rows']
        and (rule['direction'] == 'any' or (ratio < 1 if rule['direction'] == 'down' else ratio > 1)))


def cap_sensitivity():
    search = load(NATIVE/'rule_search.json')
    original = load(NATIVE/'selected_rules.json')
    selections, validation = [], []
    for cap in (1.05, 1.10, 1.25):
        for policy in original:
            eligible = [r for r in search if r['policy'] == policy]
            passing = [r for r in eligible if r['failures'] == 0 and r['median_ratio'] <= .98
                and r['p95_ratio'] <= cap and r['loss_fraction'] <= .25
                and r['whole_p95_latency_ratio'] <= 1.01 and r['saved_ms'] > 0]
            best = max(passing, key=lambda r: (r['saved_ms'], r['support'], -r['p95_ratio'], -r['index'])) if passing else None
            rule = best['rule'] if best else None
            if cap == 1.10:
                assert rule == original[policy]['rule']
            entry = {'cap': cap, 'policy': policy, 'rule': rule,
                'eligible_support_regions': len(eligible), 'passing_regions': len(passing),
                'selected_validation_support': best['support'] if best else 0}
            selections.append(entry)
            validation.append({k: v for k, v in entry.items() if k != 'rule'})
    save('cap_selected_rules.json', {'kind': 'retrospective one-factor validation sensitivity',
        'unix': time.time(), 'rules': selections, 'all_other_limits_fixed': True})
    table('cap_validation_sensitivity.csv', validation)
    # Hold selection fixed when evaluating the archived final-test request records.
    cases = [c for c in load(NATIVE/'cases.json') if c['role'] == 'test']
    requests = rows(NATIVE/'test_requests.csv')
    grouped = defaultdict(list)
    for row in requests:
        grouped[row['qid'], row['method']].append(row)
    medians, failures = {}, {}
    for key, values in grouped.items():
        assert len(values) == 3 and {int(v['repetition']) for v in values} == {0, 1, 2}
        medians[key] = float(np.median([float(v['wall_ms']) for v in values]))
        failures[key] = sum(v['status'] != 'ok' for v in values)
    base = np.array([medians[c['qid'], 'pg'] for c in cases])
    outputs = []
    for selection in selections:
        policy, rule = selection['policy'], selection['rule']
        family = policy.split('__', 1)[1]
        active = [family if matches(c, rule, family) else 'pg' for c in cases]
        accepted = np.array([m != 'pg' for m in active])
        values = np.array([medians[c['qid'], m] for c, m in zip(cases, active)])
        ratio = values/base
        same = rule == original[policy]['rule']
        actual = np.array([medians[c['qid'], policy] for c in cases]) if same else None
        draws = np.random.default_rng(811).integers(0, len(cases), size=(2000, len(cases)))
        replay_ci = np.percentile(values[draws].sum(axis=1)/base[draws].sum(axis=1), [2.5, 97.5])
        coverage_ci = np.percentile(accepted[draws].mean(axis=1), [2.5, 97.5])
        subset = ratio[accepted]
        subset_ci = (None, None)
        if len(subset):
            subset_draws = np.random.default_rng(811).integers(0, len(subset), size=(2000, len(subset)))
            subset_ci = np.percentile(np.percentile(subset[subset_draws], 95, axis=1), [2.5, 97.5])
        actual_ci = np.percentile(actual[draws].sum(axis=1)/base[draws].sum(axis=1), [2.5, 97.5]) if same else (None, None)
        outputs.append({'cap': selection['cap'], 'policy': policy, 'test_sql': len(cases),
            'admitted_sql': int(accepted.sum()), 'coverage': float(accepted.mean()),
            'coverage_ci_low': coverage_ci[0], 'coverage_ci_high': coverage_ci[1],
            'trace_replay_total_ratio': float(values.sum()/base.sum()),
            'trace_replay_total_ci_low': replay_ci[0], 'trace_replay_total_ci_high': replay_ci[1],
            'trace_replay_whole_ratio_p95': float(np.percentile(ratio, 95)),
            'trace_replay_whole_ratio_p99': float(np.percentile(ratio, 99)),
            'trace_replay_ratio_of_latency_p95': float(np.percentile(values, 95)/np.percentile(base, 95)),
            'trace_replay_accepted_ratio_p95': float(np.percentile(ratio[accepted], 95)) if accepted.any() else None,
            'trace_replay_accepted_ratio_p50': float(np.median(subset)) if len(subset) else None,
            'trace_replay_accepted_p95_ci_low': subset_ci[0], 'trace_replay_accepted_p95_ci_high': subset_ci[1],
            'trace_replay_accepted_worst_ratio': float(subset.max()) if len(subset) else None,
            'trace_replay_accepted_failed_requests': sum(failures[c['qid'], m] for c, m, yes in zip(cases, active, accepted) if yes),
            'trace_replay_failed_requests': sum(failures[c['qid'], m] for c, m in zip(cases, active)),
            'same_as_actually_executed_policy': same,
            'actual_policy_total_ratio': float(actual.sum()/base.sum()) if same else None,
            'actual_policy_total_ci_low': actual_ci[0], 'actual_policy_total_ci_high': actual_ci[1],
            'actual_policy_ratio_p95': float(np.percentile(actual/base, 95)) if same else None})
    table('cap_test_replay.csv', outputs)
    return selections


def evidence_scope():
    budget = rows(OUT.parent/'uc4d_followup/budget_gate_passes.csv')
    table('estimator_budget_passes.csv', budget)
    native = load(NATIVE/'estimator_gates.json')
    entries = []
    for family, drifts in native.items():
        for drift, doc in drifts.items():
            entries.append({'family': family, 'drift': drift, 'passed': doc['passed'],
                'gate_subqueries': doc['unique_subqueries'], 'raw_p95': doc['raw']['p95'],
                'candidate_p95': doc['candidate']['p95'], 'raw_p99': doc['raw']['p99'],
                'candidate_p99': doc['candidate']['p99'],
                'scope': 'PG-relative subplan target; not the same labels/unit as MSCN root gate'})
    table('pg_adapter_gate_scope.csv', entries)
    sscard = rows(OUT.parent/'reviewer_uc4d/sscard_eq5_passing_reports.csv')
    for row in sscard:
        assert sha(ROOT/'research_plan'/row['report_file']) == row['sha256']
        assert row['eq5_all_splits_passed'] == 'True'
    env = []
    for path in inputs():
        if path.name == 'environment.json':
            doc = load(path)
            env.append({'file': str(path), 'version': doc.get('version'), 'settings': doc.get('settings')})
    save('scope_evidence.json', {'native_environments': env, 'sscard_eq5_archive_passes': len(sscard),
        'sscard_scope': 'Existing artifact reclassification, no unchanged-rule native transfer',
        'independent_cross_workload_transfer': 'not measured',
        'native_coverage_by_total_labels': 'not measured',
        'online_refresh_catalog_invalidation_concurrent_compilation': 'not measured'})


def main():
    snapshot = {str(p): sha(p) for p in inputs()}
    for file, digest in load(NATIVE/'analysis_complete.json')['outputs'].items():
        assert sha(NATIVE/file) == digest, file
    metrics = utility()
    selections = cap_sensitivity()
    evidence_scope()
    for path, expected in snapshot.items():
        assert sha(path) == expected, path
    original = ROOT/'rebuttal_20260928/submission_original/PERSEUS-paper/PERSEUS'
    for name, expected in {'main_kdd2027.pdf': '44fd3345505823434994b332bc255cc33fe07a1495970d5defcce9a29062862a',
        'main_kdd2027.tex': '8e9a374baf8c301996bc1b66452c9eb41a62f23308e3a19cd9c462433f021e3c'}.items():
        assert sha(original/name) == expected
    outputs = ('original_policy_utility.csv', 'cap_selected_rules.json', 'cap_validation_sensitivity.csv',
        'cap_test_replay.csv', 'estimator_budget_passes.csv', 'pg_adapter_gate_scope.csv', 'scope_evidence.json')
    save('audit_complete.json', {'unix': time.time(), 'analysis_kind': 'existing-evidence audit, no training or SQL execution',
        'inputs': snapshot, 'outputs': {f: sha(OUT/f) for f in outputs},
        'script_sha256': sha(__file__), 'analysis_plan_sha256': sha(OUT/'ANALYSIS_PLAN.md'),
        'original_submission_unchanged': True, 'cap_policy_conditions': len(selections),
        'cap_conditions_with_nonzero_validation_admission': sum(x['rule'] is not None for x in selections)})
    print(json.dumps({'cap_policy_conditions': len(selections),
        'cap_selected_nonzero': sum(x['rule'] is not None for x in selections),
        'test_utility': [r for r in metrics if r['split'] == 'test']}, indent=2))


if __name__ == '__main__':
    main()
