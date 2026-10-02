"""Analyze paired native transfer, keeping censored attempts and correct pairs separate."""
from __future__ import annotations

from collections import Counter, defaultdict
import itertools
import time

import run_transfer as r

np, OUT = r.np, r.OUT
BOOT = 2000


def interval(values):
    return [float(x) for x in np.percentile(values, [2.5, 97.5])]


def stats(cases, lookup, method, draws):
    baseline = np.array([lookup[c['qid'], 'pg']['ms'] for c in cases])
    values = np.array([lookup[c['qid'], method]['ms'] for c in cases])
    ratio = values / baseline
    injected = np.array([lookup[c['qid'], method]['active'] != 'pg' for c in cases])
    admitted = injected if method in r.POLICIES else np.zeros(len(cases), dtype=bool)
    correct = np.array([lookup[c['qid'], method]['failed'] == 0 and lookup[c['qid'], 'pg']['failed'] == 0
                        for c in cases])
    result = {'method': method, 'parents': len(cases), 'injected_parents': int(injected.sum()),
        'admitted_parents': int(admitted.sum()), 'coverage': float(admitted.mean()),
        'baseline_time_coverage': float(baseline[admitted].sum() / baseline.sum()),
        'baseline_total_ms': float(baseline.sum()), 'total_ms': float(values.sum()),
        'total_ratio': float(values.sum() / baseline.sum()), 'saved_ms': float(baseline.sum() - values.sum()),
        'all_query_ratio_max': float(ratio.max()), 'all_win_fraction': float((ratio < 1).mean()),
        'failed_repetitions': sum(lookup[c['qid'], method]['failed'] for c in cases),
        'pg_failed_repetitions': sum(lookup[c['qid'], 'pg']['failed'] for c in cases),
        'correct_pairs': int(correct.sum()),
        'correct_pair_total_ratio': float(values[correct].sum() / baseline[correct].sum()) if correct.any() else None}
    for key, samples in (
        ('total_ratio', values[draws].sum(axis=1) / baseline[draws].sum(axis=1)),
        ('coverage', admitted[draws].mean(axis=1)),
        ('baseline_time_coverage', (baseline * admitted)[draws].sum(axis=1) / baseline[draws].sum(axis=1))):
        result[key + '_ci_low'], result[key + '_ci_high'] = interval(samples)
    rng = np.random.default_rng(r.SEED)
    if correct.any():
        cb, cv = baseline[correct], values[correct]
        idx = rng.integers(0, len(cb), (BOOT, len(cb)))
        result['correct_pair_total_ci_low'], result['correct_pair_total_ci_high'] = interval(
            cv[idx].sum(axis=1) / cb[idx].sum(axis=1))
    else:
        result['correct_pair_total_ci_low'] = result['correct_pair_total_ci_high'] = None
    for pct in (50, 95, 99):
        result[f'query_ratio_p{pct}'] = float(np.percentile(ratio, pct))
        result[f'latency_p{pct}_ms'] = float(np.percentile(values, pct))
        result[f'ratio_of_latency_p{pct}'] = float(np.percentile(values, pct) / np.percentile(baseline, pct))
    for prefix, mask in (('accepted', admitted), ('accepted_correct', admitted & correct)):
        subset = ratio[mask]
        result[prefix + '_parents'] = len(subset)
        result[prefix + '_pg_ms'] = float(baseline[mask].sum())
        result[prefix + '_win_fraction'] = float((subset < 1).mean()) if len(subset) else None
        result[prefix + '_loss_fraction_over_1p01'] = float((subset > 1.01).mean()) if len(subset) else None
        result[prefix + '_failed_repetitions'] = sum(lookup[c['qid'], method]['failed']
            for c, keep in zip(cases, mask) if keep)
        result[prefix + '_pg_failed_repetitions'] = sum(lookup[c['qid'], 'pg']['failed']
            for c, keep in zip(cases, mask) if keep)
        idx = rng.integers(0, len(subset), (BOOT, len(subset))) if len(subset) else None
        for name, func in [('p50', lambda x, axis=None: np.percentile(x, 50, axis=axis)),
                           ('p95', lambda x, axis=None: np.percentile(x, 95, axis=axis)),
                           ('p99', lambda x, axis=None: np.percentile(x, 99, axis=axis)),
                           ('max', np.max)]:
            result[f'{prefix}_ratio_{name}'] = float(func(subset)) if len(subset) else None
            ci = interval(func(subset[idx], axis=1)) if len(subset) else [None, None]
            result[f'{prefix}_ratio_{name}_ci_low'], result[f'{prefix}_ratio_{name}_ci_high'] = ci
    return result


def compare(cases, lookup, numerator, denominator, draws):
    a = np.array([lookup[c['qid'], numerator]['ms'] for c in cases])
    b = np.array([lookup[c['qid'], denominator]['ms'] for c in cases])
    limits = interval(a[draws].sum(axis=1) / b[draws].sum(axis=1))
    return {'numerator': numerator, 'denominator': denominator, 'parents': len(cases),
        'total_ratio': float(a.sum() / b.sum()), 'total_ci_low': limits[0], 'total_ci_high': limits[1]}


def populations(cases, role):
    result = {'all': cases, **{d: [c for c in cases if c['drift'] == d] for d in r.engine.DRIFTS}}
    if role == 'test':
        result['source_like60'] = [c for d in r.engine.DRIFTS for c in result[d][:20]]
        assert len(result['source_like60']) == 60
    return result


def main():
    for phase in ('population', 'freeze', 'maps', 'validation', 'select', 'test'):
        r.verify(phase)
    assert r.read_json(OUT/'select_complete.json')['unix'] < r.read_json(OUT/'test_started.json')['unix']
    preflight = r.read_json(OUT/'structural_preflight.json')
    assert r.n.hash_file(OUT/'prepare_population.py') == preflight['source_sha256']
    assert r.n.hash_file(OUT/'PREFLIGHT.md') == preflight['preflight_plan_sha256']
    inputs = {str(OUT/f'{p}_complete.json'): r.n.hash_file(OUT/f'{p}_complete.json')
              for p in ('population', 'freeze', 'maps', 'validation', 'select', 'test')}
    for source in ('structural_preflight.json', 'source_model.json'):
        for path, digest in r.read_json(OUT/source)['inputs'].items():
            assert r.n.hash_file(path) == digest, path
            inputs[path] = digest
    rules = r.read_json(OUT/'selected_rules.json')
    resolved = {}
    for config,selections in rules.items():
        resolved[config] = {}
        for method,item in selections.items():
            resolved[config][method] = {'rule':item['rule'],'subset_p95_cap':item['cap'],
                'validation_origin':'source' if method.startswith('source') else 'target_'+config,
                'qualifying_rules':item['passing_rules'],'nonempty_selection':item['rule'] is not None,
                'selected_region_support':item.get('support',0),
                'selected_region_ratio_p95':item.get('p95_ratio'),
                'fallback':'native_pg','source_estimator_gates_unchanged':True}
    r.write_json(OUT/'policy_specification.json',{
        'source_model':'Frozen PG-relative residual HGB; 1340 fit/tune/gate labels; no target fitting.',
        'source_estimator_gate':{'raw_reductions':{'p90':.30,'p95':.40,'p99':.60},
            'scalar_reductions':{'p95':.25,'p99':.40},'median_over_raw_cap':1.20,
            'target_recomputed':False,'source_passed_drifts':['column','table','value']},
        'runtime_contract':{'minimum_validation_parents':20,'all_selected_repetitions_correct':True,
            'accepted_ratio_median_cap':.98,'accepted_ratio_p95_caps':{'primary':1.10,'sensitivity':1.25},
            'fraction_ratio_above_1p01_cap':.25,'ratio_of_whole_latency_p95_cap':1.01,
            'positive_total_saved_ms_required':True,'grid_size':336,
            'ranking':['larger_saved_ms','larger_support','lower_subset_p95','lower_grid_index'],
            'features':['drift','table_count','native_root_rows','root_correction_direction']},
        'fallback_precedence':'Failed source estimator gate excludes its drift; no qualified native region or unmatched query uses native PG. Only a matched frozen region activates repair.',
        'selections':resolved})
    summary, per_query, statuses, reconciliation, paired, configuration_pairs = [], [], [], [], [], []
    repetition_totals = []
    plan_changes, role_costs, root_checks, join_checks = [], {}, [], []
    unique_reference_parents = {}
    parent_roles = defaultdict(set)
    requests_total = 0
    for role, expected_parents in (('validation', 60), ('test', 120)):
        cases = r.read_json(OUT/f'{role}_count_cases.json')
        methods = r.METHODS + (r.POLICIES if role == 'test' else ())
        lookup = r.aggregate(role)
        requests = r.n.csv_rows(OUT/f'{role}_requests.csv')
        warmups = r.n.csv_rows(OUT/f'{role}_warmup.csv')
        references = r.n.csv_rows(OUT/f'{role}_references.csv')
        assert len(cases) == expected_parents * 2
        assert len(requests) == len(cases) * len(methods) * 3
        assert len(warmups) == len(cases) * len(methods)
        assert len(references) == len(cases)
        for c in cases:
            parent_roles[role].add(c['sql_id'])
        assert len(parent_roles[role]) == expected_parents
        case_by_id = {c['qid']: c for c in cases}
        returned = defaultdict(set)
        req_groups = defaultdict(list)
        for row in requests + warmups:
            case = case_by_id[row['qid']]
            assert row['active'] == r.active_method(case, row['method'], rules if role == 'test' else {})
            assert row['config'] == case['config'] and row['parent_id'] == case['parent_id']
            if row['count']:
                returned[case['parent_id']].add(int(row['count']))
            if row['status'] == 'ok':
                assert int(row['count']) == int(case['truth'])
            req_groups[row['qid'], row['method']].append(row)
        for rows in req_groups.values():
            assert sorted(int(x['repetition']) for x in rows) == [-1, 0, 1, 2]
        for ref in references:
            if ref['actual']:
                returned[ref['parent_id']].add(int(ref['actual']))
        for case in cases:
            counts = returned[case['parent_id']]
            reconciliation.append({'role': role, 'qid': case['qid'], 'parent_id': case['parent_id'],
                'config': case['config'], 'drift': case['drift'], 'archived_truth': case['archived_truth'],
                'reference': case['truth'], 'returned_counts': str(sorted(counts)),
                'cross_method_or_config_conflict': len(counts) > 1,
                'archived_discrepancy': bool(counts and counts != {int(case['archived_truth'])})})
            for method in methods:
                item = lookup[case['qid'], method]
                per_query.append({'role': role, 'config': case['config'], 'parent_id': case['parent_id'],
                    'qid': case['qid'], 'drift': case['drift'], 'tables': case['tables'],
                    'method': method, 'active': item['active'], 'median_ms': item['ms'],
                    'pg_median_ms': lookup[case['qid'], 'pg']['ms'],
                    'query_ratio': item['ms']/lookup[case['qid'], 'pg']['ms'],
                    'failed_repetitions': item['failed'], 'pg_failed_repetitions': lookup[case['qid'], 'pg']['failed']})
        role_costs[role] = {}
        unique_reference_parents[role] = len({x['parent_id'] for x in references if x['status'] == 'ok'})
        configs = {}
        for config in r.CONFIGS:
            cfg_cases = [c for c in cases if c['config'] == config]
            cfg_requests = [x for x in requests if x['config'] == config]
            cfg_warmup = [x for x in warmups if x['config'] == config]
            cfg_refs = [x for x in references if x['config'] == config]
            switches = [x for x in r.n.csv_rows(OUT/f'{role}_config_switches.csv') if x['config'] == config]
            assert len(cfg_cases) == expected_parents
            configs[config] = populations(cfg_cases, role)
            for method in methods:
                selected = [x for x in cfg_requests if x['method'] == method]
                counter = Counter(x['status'] for x in selected)
                statuses.append({'role': role, 'config': config, 'method': method, 'requests': len(selected),
                    **{key: counter[key] for key in ('ok', 'timeout', 'reference_unavailable', 'count_conflict', 'error')}})
                for repetition in range(3):
                    group = [x for x in selected if int(x['repetition'])==repetition]
                    base = [x for x in cfg_requests if x['method']=='pg' and int(x['repetition'])==repetition]
                    assert len(group)==len(base)==expected_parents
                    pg_ms = sum(float(x['wall_ms']) for x in base)
                    measured_ms = sum(float(x['wall_ms']) for x in group)
                    repetition_totals.append({'role':role,'config':config,'method':method,'repetition':repetition,
                        'requests':len(group),'correct':sum(x['status']=='ok' for x in group),
                        'timeouts':sum(x['status']=='timeout' for x in group),'request_ms':measured_ms,
                        'pg_request_ms':pg_ms,'ratio_of_sums':measured_ms/pg_ms,
                        'decision_ms':sum(float(x['decision_ms']) for x in group),
                        'activation_ms':sum(float(x['activation_ms']) for x in group),
                        'select_ms':sum(float(x['select_ms']) for x in group)})
            for population, selected in configs[config].items():
                draws = np.random.default_rng(r.SEED).integers(0, len(selected), (BOOT, len(selected)))
                for method in methods:
                    summary.append({'role': role, 'config': config, 'population': population,
                                    **stats(selected, lookup, method, draws)})
                if role == 'test':
                    for numerator, denominator in (('target110', 'source110'), ('target125', 'source125'),
                                                   ('source125', 'repair'), ('target110', 'repair')):
                        paired.append({'config': config, 'population': population,
                            **compare(selected, lookup, numerator, denominator, draws)})
            role_costs[role][config] = {'reference_attempts': len(cfg_refs),
                'reference_successes': sum(x['status'] == 'ok' for x in cfg_refs),
                'reference_ms': sum(float(x['wall_ms']) for x in cfg_refs),
                'timed_requests': len(cfg_requests), 'timed_ms': sum(float(x['wall_ms']) for x in cfg_requests),
                'warmup_requests': len(cfg_warmup), 'warmup_ms': sum(float(x['wall_ms']) for x in cfg_warmup),
                'selection_pg_repair_timed_ms': sum(float(x['wall_ms']) for x in cfg_requests if x['method'] in ('pg','repair')),
                'selection_pg_repair_warmup_ms': sum(float(x['wall_ms']) for x in cfg_warmup if x['method'] in ('pg','repair')),
                'configuration_switch_ms': sum(float(x['switch_ms']) for x in switches),
                'decision_ms': sum(float(x['decision_ms']) for x in cfg_requests),
                'activation_ms': sum(float(x['activation_ms']) for x in cfg_requests)}
            roots = r.n.csv_rows(OUT/config/f'{role}_root_checks.csv')
            joins = r.n.csv_rows(OUT/config/f'{role}_join_checks.csv')
            root_checks.extend(roots)
            join_checks.extend(joins)
            fingerprints = {(x['qid'], x['method']): x['plan_shape_sha256'] for x in roots}
            for method in methods:
                plan_changes.append({'role': role, 'config': config, 'method': method, 'parents': len(cfg_cases),
                    'changed_vs_pg': sum(fingerprints[c['qid'], method] != fingerprints[c['qid'], 'pg'] for c in cfg_cases)})
        for population in configs['pg4']:
            a, b = configs['pg11'][population], configs['pg4'][population]
            assert [c['parent_id'] for c in a] == [c['parent_id'] for c in b]
            draws = np.random.default_rng(r.SEED).integers(0, len(a), (BOOT, len(a)))
            for method in methods:
                av = np.array([lookup[c['qid'], method]['ms'] for c in a])
                bv = np.array([lookup[c['qid'], method]['ms'] for c in b])
                apg = np.array([lookup[c['qid'], 'pg']['ms'] for c in a])
                bpg = np.array([lookup[c['qid'], 'pg']['ms'] for c in b])
                absolute = av[draws].sum(axis=1) / bv[draws].sum(axis=1)
                relative = (av[draws].sum(axis=1)/apg[draws].sum(axis=1)) / (bv[draws].sum(axis=1)/bpg[draws].sum(axis=1))
                ci1, ci2 = interval(absolute), interval(relative)
                configuration_pairs.append({'role': role, 'population': population, 'method': method, 'parents': len(a),
                    'pg11_over_pg4_total': float(av.sum()/bv.sum()), 'absolute_ci_low': ci1[0], 'absolute_ci_high': ci1[1],
                    'pg_normalized_config_ratio': float((av.sum()/apg.sum())/(bv.sum()/bpg.sum())),
                    'relative_ci_low': ci2[0], 'relative_ci_high': ci2[1]})
        requests_total += len(requests)
    assert not (parent_roles['validation'] & parent_roles['test'])
    assert not any(x['cross_method_or_config_conflict'] for x in reconciliation)
    checked_roots = [x for x in root_checks if x['expected']]
    checked_joins = [x for x in join_checks if x['expected']]
    assert all(x['matches'] == 'True' for x in checked_roots + checked_joins)
    maps = r.n.csv_rows(OUT/'estimate_map.csv')
    subplans = r.n.csv_rows(OUT/'native_subplans.csv')
    map_lookup = {(x['query_id'], x['aliases'], x['method']): float(x['estimate']) for x in maps}
    assert len(map_lookup) == len(maps)
    subplan_pairs = []
    for row in subplans:
        if row['config'] != 'pg4':
            continue
        q4, q11 = row['qid'], 'pg11|' + row['parent_id']
        item = {k: row[k] for k in ('parent_id','role','drift','aliases','tables','is_root')}
        for method in ('pg_map','repair'):
            v4, v11 = map_lookup[q4,row['aliases'],method], map_lookup[q11,row['aliases'],method]
            item[method + '_pg4'] = v4
            item[method + '_pg11'] = v11
            item[method + '_changed'] = not np.isclose(v4,v11,rtol=1e-10,atol=1e-10)
        subplan_pairs.append(item)
    config_plan_changes = []
    for role in ('validation','test'):
        lookup4 = {(x['qid'].split('|',1)[1],x['method']): x['plan_shape_sha256']
                   for x in r.n.csv_rows(OUT/'pg4'/f'{role}_root_checks.csv')}
        lookup11 = {(x['qid'].split('|',1)[1],x['method']): x['plan_shape_sha256']
                    for x in r.n.csv_rows(OUT/'pg11'/f'{role}_root_checks.csv')}
        for method in r.METHODS + (r.POLICIES if role=='test' else ()):
            keys = [k for k in lookup4 if k[1] == method]
            config_plan_changes.append({'role':role,'method':method,'parents':len(keys),
                'pg11_vs_pg4_plan_changes':sum(lookup4[k]!=lookup11[k] for k in keys)})
    rejection = []
    search = r.read_json(OUT/'target_rule_search.json')
    for config, cap in itertools.product(r.CONFIGS, r.CAPS.values()):
        rows = [x for x in search if x['config']==config]
        selected = r.choose(rows,cap)
        assert selected['rule'] == rules[config]['target110' if cap==1.1 else 'target125']['rule']
        rejection.append({'config':config,'cap':cap,'all_grid_rules':336,'rules_with_support':len(rows),
            'passing_rules':selected['passing_rules'],'fails_correctness':sum(x['failures']>0 for x in rows),
            'fails_median':sum(x['median_ratio']>.98 for x in rows),'fails_subset_p95':sum(x['p95_ratio']>cap for x in rows),
            'fails_loss_fraction':sum(x['loss_fraction']>.25 for x in rows),
            'fails_whole_p95':sum(x['whole_p95_latency_ratio']>1.01 for x in rows),
            'fails_saving':sum(x['saved_ms']<=0 for x in rows)})
    source_costs = r.read_json(r.SOURCE/'cost_ledger.json')
    source_requests = r.n.csv_rows(r.SOURCE/'validation_requests.csv')
    source_warmup = r.n.csv_rows(r.SOURCE/'validation_warmup.csv')
    map_costs = r.n.csv_rows(OUT/'map_costs.csv')
    ledger = {'source_historical_ledger':source_costs, 'source_model_refitted':False,
        'source_validation_pg_residual_hgb_timed_ms':sum(float(x['wall_ms']) for x in source_requests if x['method'] in ('pg','residual_hgb')),
        'source_validation_pg_residual_hgb_warmup_ms':sum(float(x['wall_ms']) for x in source_warmup if x['method'] in ('pg','residual_hgb')),
        'source_residual_hgb_fit_tune_seconds':sum(float(x['fit_tune_seconds']) for x in r.n.csv_rows(r.SOURCE/'tuning.csv') if x['family']=='residual_hgb'),
        'new_costs':role_costs, 'new_unique_successful_reference_parents':unique_reference_parents,
        'map_costs':[{ 'role':role,'config':config,'parents':sum(x['role']==role and x['config']==config for x in map_costs),
            'explain_ms':sum(float(x['explain_ms']) for x in map_costs if x['role']==role and x['config']==config),
            'inference_ms':sum(float(x['inference_ms']) for x in map_costs if x['role']==role and x['config']==config)}
            for role,config in itertools.product(('validation','test'),r.CONFIGS)],
        'phase_elapsed_s':{p:r.read_json(OUT/f'{p}_complete.json')['unix']-r.read_json(OUT/f'{p}_started.json')['unix']
            for p in ('freeze','maps','validation','select','test')},
        'accounting':'Historical acquisition shared; no new fitting. Config copies share labels. Test COUNT and all test timing are evaluation only. Source-only rule needs no target validation. Target comparator uses reference + PG/repair validation costs; pg_map and other comparisons are audit costs. Phase wall time includes listed components and must not be added to them.',
        'request_boundary':'Already configured session, cached map, policy lookup + activation + planning + SQL return; map construction/inference and config switching separately measured.'}
    descriptors = []
    source_cases = r.read_json(r.SOURCE/'validation_count_cases.json')
    cohorts = {'source_validation_pg4':source_cases}
    for config,role in itertools.product(r.CONFIGS,('validation','test')):
        cohorts[f'target_{role}_{config}'] = [c for c in r.read_json(OUT/'cases.json')
                                             if c['config']==config and c['role']==role]
    for cohort,cases in cohorts.items():
        for drift in ('all',*r.engine.DRIFTS):
            selected = [c for c in cases if drift=='all' or c['drift']==drift]
            item = {'cohort':cohort,'drift':drift,'parents':len(selected),
                    'upward_root_corrections':sum(c['root_ratios']['residual_hgb']>1 for c in selected)}
            for feature in ('tables','root_pg_rows'):
                for pct in (50,95):
                    item[f'{feature}_p{pct}'] = float(np.percentile([c[feature] for c in selected],pct))
            descriptors.append(item)
    outputs = [('runtime_summary.csv',summary),('per_query.csv',per_query),('request_statuses.csv',statuses),
        ('count_reconciliation.csv',reconciliation),('paired_policies.csv',paired),('paired_configurations.csv',configuration_pairs),
        ('plan_changes.csv',plan_changes),('configuration_plan_changes.csv',config_plan_changes),
        ('subplan_feature_changes.csv',subplan_pairs),('validation_rejection_counts.csv',rejection),
        ('population_characteristics.csv',descriptors),('repetition_totals.csv',repetition_totals)]
    for file,rows in outputs:
        r.write_csv(OUT/file,rows)
    r.write_json(OUT/'cost_ledger.json',ledger)
    original = r.ROOT/'rebuttal_20260928/submission_original/PERSEUS-paper/PERSEUS'
    for file,digest in {'main_kdd2027.pdf':'44fd3345505823434994b332bc255cc33fe07a1495970d5defcce9a29062862a',
        'main_kdd2027.tex':'8e9a374baf8c301996bc1b66452c9eb41a62f23308e3a19cd9c462433f021e3c'}.items():
        assert r.n.hash_file(original/file)==digest
        inputs[str(original/file)] = digest
    r.write_json(OUT/'analysis_complete.json',{'unix':time.time(),'script_sha256':r.n.hash_file(__file__),
        'inputs':inputs,'outputs':{file:r.n.hash_file(OUT/file) for file in [x[0] for x in outputs]+['cost_ledger.json','policy_specification.json']},
        'timed_requests':requests_total,'warmup_requests':2040,'reference_attempts':360,
        'mapped_root_checks':len(checked_roots),'mapped_join_checks':len(checked_joins),
        'all_injection_checks_passed':True,'returned_count_conflicts':0,'original_submission_unchanged':True,
        'archived_label_discrepant_parents':len({x['parent_id'] for x in reconciliation if x['archived_discrepancy']}),
        'reference_verified_unique_parents':unique_reference_parents,
        'changed_pg_subplan_features':sum(x['pg_map_changed'] for x in subplan_pairs),
        'changed_repair_maps':sum(x['repair_changed'] for x in subplan_pairs),
        'paired_subplans':len(subplan_pairs),'bootstrap_draws':BOOT,
        'uncertainty':'Paired SQL-parent bootstrap, conditional on frozen model/rules and measured repetitions. Correct-pair and accepted intervals condition on their subsets; bootstrap maximum describes observed support, not an upper bound on future worst-case latency.',
        'source_record_schema':'The inherited source125 passed=false field is the original 1.10 flag. source125 cap=1.25 and passing_rules=4 define its frozen sensitivity selection. No route uses the inherited passed flag.'})
    r.n.emit('transfer_analysis_complete',requests=requests_total)


if __name__ == '__main__':
    main()
