"""Audit complete native attribution; keep censoring separate from speedups."""
from __future__ import annotations

from collections import Counter, defaultdict
import itertools
import time

import run_native as r

np, OUT = r.np, r.OUT


def ci(values):
    return [float(v) for v in np.percentile(values, [2.5, 97.5])]


def stats(cases, lookup, method, draws):
    baseline = np.array([lookup[c['qid'], 'pg']['ms'] for c in cases])
    values = np.array([lookup[c['qid'], method]['ms'] for c in cases])
    ratio = values/baseline
    injected = np.array([lookup[c['qid'], method]['active'] != 'pg' for c in cases])
    accepted = injected if method in r.POLICIES else np.zeros(len(cases), dtype=bool)
    valid = np.array([lookup[c['qid'], method]['failed'] == 0 and lookup[c['qid'], 'pg']['failed'] == 0 for c in cases])
    total_ci = ci(values[draws].sum(axis=1)/baseline[draws].sum(axis=1))
    coverage_ci = ci(accepted[draws].mean(axis=1))
    record = {'method': method, 'parents': len(cases), 'injected_parents': int(injected.sum()),
        'admitted_parents': int(accepted.sum()), 'coverage': float(accepted.mean()),
        'coverage_ci_low': coverage_ci[0], 'coverage_ci_high': coverage_ci[1],
        'baseline_total_ms': float(baseline.sum()), 'total_ms': float(values.sum()),
        'total_ratio': float(values.sum()/baseline.sum()), 'total_ci_low': total_ci[0], 'total_ci_high': total_ci[1],
        'win_fraction_all_attempts': float((ratio < 1).mean()), 'worst_ratio': float(ratio.max()),
        'failed_repetitions': sum(lookup[c['qid'], method]['failed'] for c in cases),
        'baseline_failed_repetitions': sum(lookup[c['qid'], 'pg']['failed'] for c in cases),
        'paired_correct_parents': int(valid.sum()),
        'paired_correct_total_ratio': float(values[valid].sum()/baseline[valid].sum()) if valid.any() else None}
    for pct in (50, 95, 99):
        record[f'ratio_p{pct}'] = float(np.percentile(ratio, pct))
        record[f'latency_p{pct}_ms'] = float(np.percentile(values, pct))
        record[f'ratio_of_latency_p{pct}'] = float(np.percentile(values, pct)/np.percentile(baseline, pct))
    subset = ratio[accepted]
    for pct in (50, 95):
        interval = [None, None]
        if len(subset):
            boot = np.random.default_rng(r.SEED).integers(0, len(subset), size=(2000, len(subset)))
            interval = ci(np.percentile(subset[boot], pct, axis=1))
        record[f'accepted_ratio_p{pct}'] = float(np.percentile(subset, pct)) if len(subset) else None
        record[f'accepted_ratio_p{pct}_ci_low'], record[f'accepted_ratio_p{pct}_ci_high'] = interval
    record['accepted_worst_ratio'] = float(subset.max()) if len(subset) else None
    record['accepted_win_fraction'] = float((subset < 1).mean()) if len(subset) else None
    record['accepted_failed_repetitions'] = sum(lookup[c['qid'], method]['failed'] for c, flag in zip(cases, accepted) if flag)
    return record


def main():
    for phase in ('prepare', 'train', 'maps', 'validation', 'select', 'test'):
        r.verify(phase)
    assert r.read_json(OUT/'select_complete.json')['unix'] <= r.read_json(OUT/'test_started.json')['unix']
    for path, digest in r.read_json(OUT/'split_audit.json')['inputs'].items():
        assert r.n.hash_file(path) == digest
    r.bind_engine()
    summaries, pairs, per_query, statuses, reconciliation = [], [], [], [], []
    phase_costs = {}
    counts_total = 0
    for role in ('validation', 'test'):
        cases = r.read_json(OUT/f'{role}_count_cases.json')
        methods = r.METHODS + (r.POLICIES if role == 'test' else ())
        requests = r.n.csv_rows(OUT/f'{role}_requests.csv')
        warmups = r.n.csv_rows(OUT/f'{role}_warmup.csv')
        references = r.n.csv_rows(OUT/f'{role}_references.csv')
        assert len(cases) == 90 and len(requests) == 90*len(methods)*3
        assert len(warmups) == 90*len(methods)
        assert len({c['sql_id'] for c in cases}) == len(cases)
        lookup = r.engine.aggregate(role)
        case_by_qid = {c['qid']: c for c in cases}
        groups = defaultdict(list)
        for row in requests + warmups:
            case = case_by_qid[row['qid']]
            groups[row['qid']].append(row)
            if row['status'] == 'ok':
                assert int(row['count']) == int(case['truth'])
        for case in cases:
            returned = {int(x['count']) for x in groups[case['qid']] if x['count'] != ''}
            reconciliation.append({'role': role, 'qid': case['qid'], 'drift': case['drift'],
                'archived': case['archived_truth'], 'pg_reference': case['truth'],
                'observed_counts': str(sorted(returned)), 'cross_method_conflict': len(returned) > 1,
                'archived_discrepancy': bool(returned and returned != {int(case['archived_truth'])}),
                'reference_available': case['truth'] is not None})
        for method in methods:
            selection = [x for x in requests if x['method'] == method]
            counts = Counter(x['status'] for x in selection)
            statuses.append({'role': role, 'method': method, 'requests': len(selection),
                **{key: counts[key] for key in ('ok', 'timeout', 'reference_unavailable', 'count_conflict', 'error')}})
        for case in cases:
            for method in methods:
                value = lookup[case['qid'], method]
                per_query.append({'role': role, 'qid': case['qid'], 'sql_id': case['sql_id'], 'drift': case['drift'],
                    'tables': case['tables'], 'method': method, 'active': value['active'],
                    'median_wall_ms': value['ms'], 'pg_median_ms': lookup[case['qid'], 'pg']['ms'],
                    'ratio': value['ms']/lookup[case['qid'], 'pg']['ms'], 'failed_repetitions': value['failed']})
        pops = {'all': cases, **{d: [c for c in cases if c['drift'] == d] for d in r.engine.DRIFTS}}
        for population, selected in pops.items():
            draws = np.random.default_rng(r.SEED).integers(0, len(selected), size=(2000, len(selected)))
            for method in methods:
                summaries.append({'role': role, 'population': population, **stats(selected, lookup, method, draws)})
            for family in ('hgb', 'rf'):
                for stage in ('candidate', 'runtime', 'full') if role == 'test' else ('candidate',):
                    prefix = '' if stage == 'candidate' else stage+'__'
                    a, b = prefix+'residual_'+family, prefix+'direct_'+family
                    va = np.array([lookup[c['qid'], a]['ms'] for c in selected])
                    vb = np.array([lookup[c['qid'], b]['ms'] for c in selected])
                    ratios = va/vb
                    interval = ci(va[draws].sum(axis=1)/vb[draws].sum(axis=1))
                    pairs.append({'role': role, 'population': population, 'stage': stage,
                        'numerator': a, 'denominator': b, 'total_ratio': float(va.sum()/vb.sum()),
                        'total_ci_low': interval[0], 'total_ci_high': interval[1],
                        'ratio_p50': float(np.median(ratios)), 'ratio_p95': float(np.percentile(ratios, 95)),
                        'numerator_failures': sum(lookup[c['qid'], a]['failed'] for c in selected),
                        'denominator_failures': sum(lookup[c['qid'], b]['failed'] for c in selected)})
        phase_costs[role] = {'reference_attempts': len(references),
            'reference_successes': sum(x['status'] == 'ok' for x in references),
            'reference_ms': sum(float(x['wall_ms']) for x in references),
            'timed_requests': len(requests), 'timed_ms': sum(float(x['wall_ms']) for x in requests),
            'warmup_requests': len(warmups), 'warmup_ms': sum(float(x['wall_ms']) for x in warmups)}
        counts_total += len(requests)
    roots, joins = [], []
    for role in ('validation', 'test'):
        roots += r.n.csv_rows(OUT/f'{role}_root_checks.csv')
        joins += r.n.csv_rows(OUT/f'{role}_join_checks.csv')
    checked = [x for x in roots+joins if x['expected']]
    assert all(x['matches'] == 'True' for x in checked)
    assert not any(x['cross_method_conflict'] for x in reconciliation), 'COUNT conflicts invalidate evidence'
    dev = r.read_json(OUT/'development.json')
    mapcosts = r.n.csv_rows(OUT/'map_costs.csv')
    tuning = r.n.csv_rows(OUT/'tuning.csv')
    oldcounts = r.n.csv_rows(r.SOURCE.parent/'native_subplan_diagnostic/count_acquisition.csv')
    ledger = {'development_unique_labels': len(dev),
        'development_roles': dict(Counter(x['role'] for x in dev)),
        'shared_historical_development_count_attempts': len(oldcounts),
        'shared_historical_count_ms': sum(float(x['count_ms']) for x in oldcounts),
        'new_count_and_request_costs': phase_costs,
        'new_model_fits': len(tuning), 'fit_tune_seconds': sum(float(x['fit_tune_seconds']) for x in tuning),
        'map_explain_ms': sum(float(x['explain_ms']) for x in mapcosts),
        'map_four_model_inference_ms': sum(float(x['four_model_inference_ms']) for x in mapcosts),
        'cached_map_deployment': True, 'online_model_inference_in_request': False}
    ledger['predeployment_successful_unique_labels'] = len(dev)+phase_costs['validation']['reference_successes']
    ledger['predeployment_count_attempts_including_reused_acquisition'] = len(oldcounts)+phase_costs['validation']['reference_attempts']
    ledger['listed_predeployment_ms'] = (ledger['shared_historical_count_ms']+
        phase_costs['validation']['reference_ms']+phase_costs['validation']['timed_ms']+
        phase_costs['validation']['warmup_ms']+ledger['fit_tune_seconds']*1000+
        sum(float(x['explain_ms'])+float(x['four_model_inference_ms']) for x in mapcosts if x['role'] == 'validation'))
    rejection_counts = []
    for policy in r.POLICIES:
        searched = [x for x in r.read_json(OUT/'rule_search.json') if x['policy'] == policy]
        rejection_counts.append({'policy': policy, 'regions_with_support': len(searched),
            'passes': sum(x['passed'] for x in searched),
            'fails_correctness': sum(x['failures'] > 0 for x in searched),
            'fails_median': sum(x['median_ratio'] > .98 for x in searched),
            'fails_subset_p95': sum(x['p95_ratio'] > 1.10 for x in searched),
            'fails_loss_fraction': sum(x['loss_fraction'] > .25 for x in searched),
            'fails_whole_p95': sum(x['whole_p95_latency_ratio'] > 1.01 for x in searched),
            'fails_positive_savings': sum(x['saved_ms'] <= 0 for x in searched)})
    for name, rows in (('runtime_summary.csv', summaries), ('paired_models.csv', pairs),
                      ('per_query.csv', per_query), ('request_statuses.csv', statuses),
                      ('count_reconciliation.csv', reconciliation), ('validation_rejection_counts.csv', rejection_counts)):
        r.write_csv(OUT/name, rows)
    r.write_json(OUT/'cost_ledger.json', ledger)
    original = r.ROOT/'rebuttal_20260928/submission_original/PERSEUS-paper/PERSEUS'
    for file, digest in {'main_kdd2027.pdf': '44fd3345505823434994b332bc255cc33fe07a1495970d5defcce9a29062862a',
                         'main_kdd2027.tex': '8e9a374baf8c301996bc1b66452c9eb41a62f23308e3a19cd9c462433f021e3c'}.items():
        assert r.n.hash_file(original/file) == digest
    files = ['runtime_summary.csv', 'paired_models.csv', 'per_query.csv', 'request_statuses.csv',
             'count_reconciliation.csv', 'cost_ledger.json', 'validation_rejection_counts.csv']
    r.write_json(OUT/'analysis_complete.json', {'unix': time.time(), 'script_sha256': r.n.hash_file(__file__),
        'timed_requests': counts_total, 'mapped_root_checks': sum(bool(x['expected']) for x in roots),
        'mapped_join_checks': sum(bool(x['expected']) for x in joins), 'all_injection_checks_passed': True,
        'cross_method_count_conflicts': 0, 'original_submission_unchanged': True,
        'outputs': {f: r.n.hash_file(OUT/f) for f in files},
        'interpretation': 'Historical held-out attribution replay; timeout-inclusive request totals are censored.'})
    r.n.emit('native_analysis_complete', requests=counts_total)


if __name__ == '__main__':
    main()
