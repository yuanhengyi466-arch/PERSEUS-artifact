"""Analyze the frozen native test without changing models or admission rules."""
from collections import Counter, defaultdict
import itertools
import math

from run_learning import (OUT, ROOT, DIAG, METHODS, POLICIES, SEED, n, np,
                          read_json, write_json, write_csv, verify, aggregate)
from semantic_audit import identity as semantic_identity, selected_test


def ci(x):
    return [float(v) for v in np.percentile(x, [2.5, 97.5])]


def summary(cases, lookup, method, draws):
    baseline = np.array([lookup[c['qid'], 'pg']['ms'] for c in cases])
    values = np.array([lookup[c['qid'], method]['ms'] for c in cases])
    accepted = np.array([lookup[c['qid'], method]['active'] != 'pg' for c in cases])
    ratios = values / baseline
    complete = np.array([lookup[c['qid'], method]['failed'] == 0 and lookup[c['qid'], 'pg']['failed'] == 0 for c in cases])
    total_ci = ci(values[draws].sum(axis=1)/baseline[draws].sum(axis=1))
    rec = {'method': method, 'parents': len(cases), 'admitted': int(accepted.sum()),
           'coverage': float(accepted.mean()), 'coverage_ci_low': ci(accepted[draws].mean(axis=1))[0],
           'coverage_ci_high': ci(accepted[draws].mean(axis=1))[1],
           'baseline_total_ms': float(baseline.sum()), 'method_total_ms': float(values.sum()),
           'total_ratio': float(values.sum()/baseline.sum()), 'total_ci_low': total_ci[0], 'total_ci_high': total_ci[1],
           'saved_ms': float(baseline.sum()-values.sum()), 'win_fraction': float((ratios<1).mean()),
           'worst_ratio': float(ratios.max()),
           'failed_repetitions': sum(lookup[c['qid'], method]['failed'] for c in cases),
           'baseline_failed_repetitions': sum(lookup[c['qid'], 'pg']['failed'] for c in cases),
           'paired_complete_parents': int(complete.sum()),
           'paired_complete_total_ratio': float(values[complete].sum()/baseline[complete].sum()) if complete.any() else None,
           'accepted_failed_repetitions': sum(lookup[c['qid'], method]['failed'] for c, yes in zip(cases, accepted) if yes)}
    for q in (50, 95, 99):
        rec[f'ratio_p{q}'] = float(np.percentile(ratios, q))
        rec[f'latency_p{q}_ms'] = float(np.percentile(values, q))
        rec[f'ratio_of_latency_p{q}'] = float(np.percentile(values, q)/np.percentile(baseline, q))
    subset = ratios[accepted]
    for q in (50, 95):
        rec[f'accepted_ratio_p{q}'] = float(np.percentile(subset, q)) if len(subset) else None
        if len(subset):
            idx = np.random.default_rng(SEED).integers(0, len(subset), size=(2000, len(subset)))
            interval = ci(np.percentile(subset[idx], q, axis=1))
        else:
            interval = [None, None]
        rec[f'accepted_ratio_p{q}_ci_low'], rec[f'accepted_ratio_p{q}_ci_high'] = interval
    rec['accepted_worst_ratio'] = float(subset.max()) if len(subset) else None
    rec['accepted_win_fraction'] = float((subset<1).mean()) if len(subset) else None
    return rec


def main():
    for phase in ('prepare', 'train', 'maps', 'validation', 'select', 'test', 'counts'):
        verify(phase)
    assert read_json(OUT/'select_complete.json')['unix'] <= read_json(OUT/'test_started.json')['unix']
    assert read_json(OUT/'test_complete.json')['unix'] <= read_json(OUT/'counts_started.json')['unix']
    original_cases = read_json(OUT/'cases.json')
    tests = selected_test()
    test_qids = {c['qid'] for c in tests}
    cases = [c for c in original_cases if c['role'] == 'validation' or c['qid'] in test_qids]
    amendment = read_json(OUT/'test_eligibility_amendment.json')
    assert amendment['unix'] < read_json(OUT/'test_started.json')['unix']
    assert amendment['protocol_amendment_sha256'] == n.hash_file(OUT/'AMENDMENT_EQUIVALENCE.md')
    for phase, key in (('test', 'test_driver_sha256'), ('counts', 'count_driver_sha256')):
        complete = read_json(OUT/f'{phase}_complete.json')
        assert complete[key] == n.hash_file(OUT/'run_final_test.py')
        assert complete['amendment_sha256'] == n.hash_file(OUT/'test_eligibility_amendment.json')
    byqid = {c['qid']: c for c in cases}
    members = [x for x in n.csv_rows(OUT/'native_membership.csv') if x['role'] == 'validation' or x['qid'] in test_qids]
    dev = read_json(OUT/'development_samples.json')
    ids = defaultdict(set)
    for row in dev + members:
        ids[row['role']].add(row['subquery_id'])
    overlaps = {f'{a}/{b}': len(ids[a] & ids[b]) for a, b in itertools.combinations(ids, 2)}
    assert not any(overlaps.values())
    assert len({c['sql_id'] for c in cases}) == len(cases) == 180 + len(tests)
    semantic_roles = defaultdict(set)
    for row in dev:
        semantic_roles[row['role']].add(semantic_identity(row['row']))
    import run_learning as engine
    for case in cases:
        semantic_roles[case['role']].update(semantic_identity(row) for _, row in engine.parts(case['row']))
    semantic_overlaps = {f'{a}/{b}': len(semantic_roles[a] & semantic_roles[b]) for a, b in itertools.combinations(semantic_roles, 2)}
    assert not any(semantic_overlaps.values())
    requests = {role: n.csv_rows(OUT/f'{role}_requests.csv') for role in ('validation', 'test')}
    statuses, warmup_statuses = {}, {}
    status_rows = []
    for role, rows in requests.items():
        expected = 180 * 3 * 6 if role == 'validation' else len(tests) * 3 * 10
        assert len(rows) == expected
        for row in rows:
            assert byqid[row['qid']]['role'] == role
            if row['status'] == 'ok':
                assert int(row['count']) == int(byqid[row['qid']]['truth'])
        statuses[role] = dict(Counter(r['status'] for r in rows))
        for method in sorted({r['method'] for r in rows}):
            selected = [r for r in rows if r['method'] == method]
            counts = Counter(r['status'] for r in selected)
            status_rows.append({'role': role, 'method': method, 'requests': len(selected),
                'ok': counts['ok'], 'timeout': counts['timeout'], 'error': counts['error'],
                'label_mismatch': counts['label_mismatch']})
        warmup_statuses[role] = dict(Counter(r['status'] for r in n.csv_rows(OUT/f'{role}_warmup.csv')))
    root_checks, join_checks = [], []
    for role in ('validation', 'test'):
        root_checks += n.csv_rows(OUT/f'{role}_root_checks.csv')
        join_checks += n.csv_rows(OUT/f'{role}_join_checks.csv')
    mapped_roots = [r for r in root_checks if r['expected']]
    mapped_joins = [r for r in join_checks if r['expected']]
    assert all(r['matches'] == 'True' for r in mapped_roots + mapped_joins)
    original = ROOT/'rebuttal_20260928/submission_original/PERSEUS-paper/PERSEUS'
    submission = {'main_kdd2027.pdf': '44fd3345505823434994b332bc255cc33fe07a1495970d5defcce9a29062862a',
                  'main_kdd2027.tex': '8e9a374baf8c301996bc1b66452c9eb41a62f23308e3a19cd9c462433f021e3c'}
    for name, digest in submission.items():
        assert n.hash_file(original/name) == digest
    for path, digest in read_json(OUT/'split_audit.json')['inputs'].items():
        assert n.hash_file(path) == digest
    lookup = aggregate('test')
    write_csv(OUT/'request_statuses.csv', status_rows)
    reconciliation = []
    for role, rows in requests.items():
        grouped = defaultdict(list)
        for row in rows:
            grouped[row['qid']].append(row)
        for qid, attempts in grouped.items():
            returned = {int(r['count']) for r in attempts if r['count'] != ''}
            native_returned = {int(r['count']) for r in attempts if r['method'] == 'pg' and r['count'] != ''}
            reconciliation.append({'role': role, 'qid': qid, 'drift': byqid[qid]['drift'],
                'archived_truth': int(byqid[qid]['truth']), 'observed_counts': str(sorted(returned)),
                'pg_observed_counts': str(sorted(native_returned)),
                'cross_method_count_conflict': len(returned) > 1,
                'observed_archived_label_mismatch': bool(returned and returned != {int(byqid[qid]['truth'])}),
                'returned_requests': sum(r['count'] != '' for r in attempts),
                'requests_without_count': sum(r['count'] == '' for r in attempts)})
    write_csv(OUT/'archived_label_reconciliation.csv', reconciliation)
    per_query = []
    for case in tests:
        for method in METHODS + POLICIES:
            row = lookup[case['qid'], method]
            per_query.append({'qid': case['qid'], 'sql_id': case['sql_id'], 'drift': case['drift'],
                'tables': case['tables'], 'method': method, 'active': row['active'], 'wall_ms': row['ms'],
                'pg_ms': lookup[case['qid'], 'pg']['ms'], 'ratio': row['ms']/lookup[case['qid'], 'pg']['ms'],
                'failed_repetitions': row['failed']})
    write_csv(OUT/'test_per_query.csv', per_query)
    summaries = []
    populations = {'all': tests, **{d: [c for c in tests if c['drift'] == d] for d in ('column', 'table', 'value')}}
    for size in sorted({c['tables'] for c in tests}):
        populations[f'tables_{size}'] = [c for c in tests if c['tables'] == size]
    for pop, selected in populations.items():
        draws = np.random.default_rng(SEED).integers(0, len(selected), size=(2000, len(selected)))
        for method in METHODS + POLICIES:
            summaries.append({'population': pop, **summary(selected, lookup, method, draws)})
    write_csv(OUT/'test_runtime_summary.csv', summaries)
    runtime_pairs = []
    for pop, selected in populations.items():
        draws = np.random.default_rng(SEED).integers(0, len(selected), size=(2000, len(selected)))
        for method, control in (('sub_hgb', 'pg_map'), ('sub_hgb', 'root_ratio'), ('sub_hgb', 'clip_ratio'),
                                ('sub_rf', 'pg_map'), ('sub_hgb', 'sub_rf'), ('full_hgb', 'runtime_hgb'),
                                ('full_rf', 'runtime_rf')):
            values = np.array([lookup[c['qid'], method]['ms'] for c in selected])
            reference = np.array([lookup[c['qid'], control]['ms'] for c in selected])
            interval = ci(values[draws].sum(axis=1)/reference[draws].sum(axis=1))
            runtime_pairs.append({'population': pop, 'method': method, 'control': control, 'parents': len(selected),
                'total_ratio': float(values.sum()/reference.sum()), 'ci_low': interval[0], 'ci_high': interval[1],
                'method_failed_repetitions': sum(lookup[c['qid'], method]['failed'] for c in selected),
                'control_failed_repetitions': sum(lookup[c['qid'], control]['failed'] for c in selected)})
    write_csv(OUT/'test_runtime_paired_controls.csv', runtime_pairs)
    predictions = {(r['query_id'], r['aliases'], r['method']): float(r['estimate']) for r in n.csv_rows(OUT/'estimate_map.csv')}
    counts = n.csv_rows(OUT/'test_subquery_counts.csv')
    counts_by_key = {(r['qid'], r['aliases']): r for r in counts}
    assert len(counts) == len(ids['test'])
    errors = []
    for row in counts:
        if row['status'] != 'ok':
            continue
        qid, alias = row['qid'], row['aliases']
        actual, pg = max(1., float(row['actual'])), max(1., float(row['pg_rows']))
        root = counts_by_key[qid, ' '.join(sorted(n.subplans.parse_aliases(byqid[qid]['row'][0])))]
        needed = math.log(actual/pg)
        opposite = None
        if root['status'] == 'ok':
            opposite = needed * math.log(max(1., float(root['actual']))/max(1., float(root['pg_rows']))) < -1e-12
        for method in METHODS:
            pred = pg if method == 'pg' else predictions[qid, alias, method]
            q = max(pred/actual, actual/pred)
            errors.append({'qid': qid, 'subquery_id': row['subquery_id'], 'drift': row['drift'],
                'tables': int(row['tables']), 'is_root': row['is_root'] == 'True', 'method': method,
                'prediction': pred, 'actual': float(row['actual']), 'qerror': q,
                'pg_qerror': max(pg/actual, actual/pg), 'worsens_pg': q > max(pg/actual, actual/pg),
                'wrong_direction': needed * math.log(pred/pg) < -1e-12,
                'root_needed_opposite': opposite})
    write_csv(OUT/'test_subplan_error_detail.csv', errors)
    error_summaries = []
    for kind in ('all', 'root', 'intermediate'):
        for drift in ('all', 'column', 'table', 'value'):
            for method in METHODS:
                for size in [0] + sorted({r['tables'] for r in errors}):
                    subset = [r for r in errors if r['method'] == method and (drift == 'all' or r['drift'] == drift)
                              and (kind == 'all' or r['is_root'] == (kind == 'root')) and (size == 0 or r['tables'] == size)]
                    if not subset:
                        continue
                    qs = np.array([r['qerror'] for r in subset])
                    error_summaries.append({'kind': kind, 'drift': drift, 'size': size, 'method': method,
                        'unique_subqueries': len(subset), 'median': float(np.median(qs)),
                        'p95': float(np.percentile(qs, 95)), 'p99': float(np.percentile(qs, 99)),
                        'max': float(qs.max()), 'worsens_fraction': float(np.mean([r['worsens_pg'] for r in subset])),
                        'wrong_direction_fraction': float(np.mean([r['wrong_direction'] for r in subset]))})
    write_csv(OUT/'test_subplan_error_summary.csv', error_summaries)
    # Query clusters preserve dependence among a parent's connected subqueries.
    error_comparisons = []
    for kind in ('all', 'intermediate', 'root'):
        rows = [r for r in errors if r['method'] == 'pg' and (kind == 'all' or r['is_root'] == (kind == 'root'))]
        qids = sorted({r['qid'] for r in rows})
        indices = {qid: [i for i, r in enumerate(rows) if r['qid'] == qid] for qid in qids}
        raw = np.array([r['qerror'] for r in rows])
        key = {(r['qid'], r['subquery_id'], r['method']): r['qerror'] for r in errors}
        boot = np.random.default_rng(SEED).integers(0, len(qids), size=(2000, len(qids)))
        for method in ('root_ratio', 'clip_ratio', 'sub_hgb', 'sub_rf'):
            values = np.array([key[r['qid'], r['subquery_id'], method] for r in rows])
            ratios = []
            for draw in boot:
                take = np.concatenate([indices[qids[i]] for i in draw])
                ratios.append(np.percentile(values[take], [50, 95, 99])/np.percentile(raw[take], [50, 95, 99]))
            lower, upper = np.percentile(ratios, [2.5, 97.5], axis=0)
            observed = np.percentile(values, [50, 95, 99])/np.percentile(raw, [50, 95, 99])
            for j, q in enumerate((50, 95, 99)):
                error_comparisons.append({'kind': kind, 'method': method, 'quantile': q, 'ratio_to_pg': float(observed[j]),
                    'ci_low': float(lower[j]), 'ci_high': float(upper[j]), 'parent_clusters': len(qids), 'subqueries': len(rows)})
    write_csv(OUT/'test_error_paired_intervals.csv', error_comparisons)
    map_costs = n.csv_rows(OUT/'map_costs.csv')
    development_counts = n.csv_rows(DIAG/'count_acquisition.csv')
    ledger = {'development_count_attempts': len(development_counts),
        'development_count_successes': sum(r['count_status'] == 'ok' for r in development_counts),
        'development_count_ms': sum(float(r['count_ms']) for r in development_counts),
        'fit_tune_gate_labels': read_json(OUT/'model_selection.json')['supervision_counts'],
        'unique_archived_validation_root_labels_for_correctness': 180,
        'unique_archived_test_root_labels_for_correctness_only': len(tests),
        'fit_successful_semantic_unique_labels': amendment['fit_successful_semantic_unique_labels'],
        'predeployment_canonical_successful_development_plus_validation_root_labels':
            sum(r['count_status'] == 'ok' for r in development_counts) + 180,
        'predeployment_equality_normalized_successful_development_plus_validation_root_labels':
            len({semantic_identity(r['row']) for r in dev if r['actual'] is not None}) + 180,
        'validation_distinct_parents_with_any_successful_measured_count':
            len({r['qid'] for r in requests['validation'] if r['status'] == 'ok'}),
        'validation_timed_executions': len(requests['validation']),
        'validation_warmup_executions': len(n.csv_rows(OUT/'validation_warmup.csv')),
        'all_eight_model_fits_ms': sum(float(r['fit_seconds'])*1000 for r in n.csv_rows(OUT/'tuning.csv')),
        'map_native_explain_ms': sum(float(r['explain_ms']) for r in map_costs),
        'map_both_model_inference_ms': sum(float(r['inference_both_models_ms']) for r in map_costs),
        'validation_timed_request_ms': sum(float(r['wall_ms']) for r in requests['validation']),
        'validation_warmup_ms': sum(float(r['wall_ms']) for r in n.csv_rows(OUT/'validation_warmup.csv')),
        'test_count_attempts_diagnostic_only': len(counts), 'test_count_statuses': dict(Counter(r['status'] for r in counts)),
        'test_count_ms_diagnostic_only': sum(float(r['count_ms']) for r in counts),
        'cached_features_and_maps': True, 'online_inference_measured': False,
        'cost_boundary': 'Listed acquisition, fit, map and validation components; not exhaustive production setup.'}
    ledger['listed_development_and_setup_ms'] = sum(ledger[k] for k in ('development_count_ms', 'all_eight_model_fits_ms',
        'map_native_explain_ms', 'map_both_model_inference_ms', 'validation_timed_request_ms', 'validation_warmup_ms'))
    all_summary = [r for r in summaries if r['population'] == 'all']
    ledger['descriptive_break_even_workload_replays'] = {
        r['method']: ledger['listed_development_and_setup_ms']/r['saved_ms'] if r['saved_ms'] > 0 else None
        for r in all_summary if r['method'] in POLICIES}
    write_json(OUT/'cost_ledger.json', ledger)
    audit = {'completed': True, 'timed_requests': sum(len(r) for r in requests.values()), 'statuses': statuses,
        'warmup_statuses': warmup_statuses, 'native_parents': len(cases), 'test_parents': len(tests),
        'mapped_root_checks': len(mapped_roots), 'mapped_join_checks': len(mapped_joins), 'all_mapped_checks_pass': True,
        'all_role_subquery_intersections': overlaps, 'test_count_statuses': ledger['test_count_statuses'],
        'all_equality_normalized_role_intersections': semantic_overlaps,
        'initial_selected_native_parents': len(original_cases),
        'pretest_structure_only_excluded_parents': len(amendment['excluded_test_parents']),
        'selected_before_test': True, 'test_subplan_truth_acquired_after_test': True,
        'original_submission_unchanged': True, 'all_preparation_inputs_unchanged': True,
        'cross_method_count_conflict_parents': sum(r['cross_method_count_conflict'] for r in reconciliation),
        'archived_label_mismatch_parents': {role: sum(r['observed_archived_label_mismatch'] for r in reconciliation if r['role'] == role)
                                          for role in ('validation', 'test')},
        'analysis_sha256': n.hash_file(__file__), 'conditional_bootstrap_draws': 2000}
    write_json(OUT/'completion_audit.json', audit)
    results = ['# Native subplan learning: measured results', '',
        'PG-anchored adapter; one prospectively fixed grouped split on STATS. '
        'Times include cached-map policy selection, activation, planning, execution and returned COUNT. '
        'Offline acquisition/inference and validation are separately charged. '
        'If a condition has timeouts, its all-population total is time to response or timeout; '
        'it is not uncensored completion latency. Paired-complete descriptive totals and denominators '
        'are provided separately in the CSV and cannot replace the full-population result.', '',
        '| Condition | Injected/test | Total request ratio [95% CI] | Injected median / P95 | Whole P95 / P99 latency ratios |',
        '|---|---:|---:|---:|---:|']
    for r in all_summary:
        subset = '-' if r['accepted_ratio_p50'] is None else f"{r['accepted_ratio_p50']:.3f} / {r['accepted_ratio_p95']:.3f}"
        results.append(f"| {r['method']} | {r['admitted']}/{r['parents']} | {r['total_ratio']:.3f} [{r['total_ci_low']:.3f}, {r['total_ci_high']:.3f}] | {subset} | {r['ratio_of_latency_p95']:.3f} / {r['ratio_of_latency_p99']:.3f} |")
    results += ['', 'Candidate conditions are forced controls. Only runtime/full rows represent validation-selected admission. '
                'The CSV admitted/accepted fields count active injection; they are not gate approval for forced controls.', '',
                'Intervals are paired unique-parent bootstraps, conditional on the frozen split/policy. '
                'They do not establish selection stability or generalization to another dataset. '
                'Injected medians/P95 are percentiles of per-query ratios; whole P95/P99 are ratios of latency percentiles.',
                '', '## Held-out intermediate cardinality error', '',
                '| Method | Labeled subqueries | Median | P95 | P99 |', '|---|---:|---:|---:|---:|']
    for r in error_summaries:
        if r['kind'] == 'intermediate' and r['drift'] == 'all' and r['size'] == 0:
            results.append(f"| {r['method']} | {r['unique_subqueries']} | {r['median']:.3f} | {r['p95']:.3f} | {r['p99']:.3f} |")
    results += ['', '## Execution and resource accounting', '',
                f"- Timed requests: {audit['timed_requests']}; statuses: {statuses}.",
                f"- Native population: 180 validation + {len(tests)} test parents; "
                f"{len(amendment['excluded_test_parents'])} test parents excluded before execution by the stronger equality-closure audit.",
                f"- Warmups: {warmup_statuses}.",
                f"- All {len(mapped_roots)} mapped root and {len(mapped_joins)} mapped chosen-join checks passed.",
                f"- Final-test subquery labels: {ledger['test_count_statuses']}; acquired after policy freeze and timing.",
                f"- Development: {ledger['fit_tune_gate_labels']}; {ledger['development_count_attempts']} original COUNT attempts, "
                f"{ledger['development_count_ms']/1000:.3f} seconds acquisition.",
                f"- Validation timed requests: {ledger['validation_timed_request_ms']/1000:.3f} s; "
                f"warmup: {ledger['validation_warmup_ms']/1000:.3f} s.",
                f"- Both-role map preparation: native EXPLAIN {ledger['map_native_explain_ms']/1000:.3f} s, "
                f"both models' inference {ledger['map_both_model_inference_ms']/1000:.3f} s.",
                '- See `selected_rules.json` for frozen regions and `rule_search.json` for all evaluated supported regions.',
                '- Failed COUNTs do not remove their parent from the runtime population. Model-error summaries require observed labels.',
                '- No throughput, online feature acquisition or online map refresh claim is made.', '']
    (OUT/'RESULTS.md').write_text('\n'.join(results), encoding='utf-8')
    n.emit('analysis_complete', audit=audit, all_test=all_summary)


if __name__ == '__main__':
    main()
