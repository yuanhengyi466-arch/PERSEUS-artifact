"""Render measured transfer summaries; never infer deployment success from coverage."""
from __future__ import annotations

import csv
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent


def read(name):
    with (OUT/name).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def load(name):
    return json.loads((OUT/name).read_text(encoding='utf-8'))


def number(value, digits=3):
    return 'NA' if value in ('',None) else f'{float(value):.{digits}f}'


def ci(row, key, low=None, high=None):
    low = low or key+'_ci_low'
    high = high or key+'_ci_high'
    return f'{number(row[key])} [{number(row[low])}, {number(row[high])}]'


def main():
    rows = read('runtime_summary.csv')
    test = [r for r in rows if r['role']=='test' and r['population']=='all']
    audit = load('analysis_complete.json')
    ledger = load('cost_ledger.json')
    rules = load('selected_rules.json')
    status = read('request_statuses.csv')
    lines = ['# Frozen-rule native transfer: measured results', '',
        'Completed Windows-native PG13.1 follow-up for rrdL RR.1/RR.4. '
        'The original submission remains unchanged. See [protocol](rebuttal_20260929/rrdl_transfer/PROTOCOL.md) '
        'and [reproduction](rebuttal_20260929/rrdl_transfer/README.md).', '',
        '## Design and verification', '',
        'The frozen source PG-relative residual HGB uses 1,340 unique development labels '
        '(866 fit / 237 tune / 237 gate). Source runtime validation has 30/30/30 '
        'column/table/value parents. New target validation has 10/20/30; new test has '
        '20/40/60. Target parents and represented equality-normalized connected '
        'subqueries are disjoint from enumerated previous native cohorts and each other. '
        'This is a new target cohort within STATS, not a second dataset.', '',
        'The same 180 parents run at random_page_cost=4 (pg4) and 1.1 (pg11). '
        'Source weights, clipping, vocabulary and rules stay fixed; maps are regenerated '
        'from configuration-specific PG inputs. Target runtime revalidation has its own '
        '60-parent validation set and freezes regions before test reference counts or '
        'timing. It retains the source estimator gate and does not retrain the model.', '',
        f"Completed {audit['timed_requests']:,} timed requests, {audit['warmup_requests']:,} warmups "
        f"and {audit['reference_attempts']} reference COUNT attempts. "
        f"All {audit['mapped_root_checks']:,} mapped-root and {audit['mapped_join_checks']:,} "
        f"mapped-join checks pass. Returned count conflicts: {audit['returned_count_conflicts']}. "
        f"Parents with a returned count different from the archived label: {audit['archived_label_discrepant_parents']}.", '',
        '| Role | Configuration | Correct timed requests | Timeouts | Reference unavailable | Other errors |',
        '|---|---|---:|---:|---:|---:|']
    for role in ('validation','test'):
        for config in ('pg4','pg11'):
            group = [r for r in status if r['role']==role and r['config']==config]
            values = [sum(int(r[k]) for r in group) for k in ('ok','timeout','reference_unavailable')]
            errors = sum(int(r['error'])+int(r['count_conflict']) for r in group)
            lines.append(f'| {role} | {config} | {values[0]} | {values[1]} | {values[2]} | {errors} |')
    lines += ['', '## Frozen choices', '',
        'Primary cap: accepted-subset ratio P95 <= 1.10. Separate sensitivity cap: 1.25. '
        'All other limits remain unchanged. The inherited `passed=false` in the source125 '
        'record is its old 1.10 flag; its explicit 1.25 selection uses `passing_rules=4`.', '',
        '| Configuration | Policy | Validation support | Qualifying rules | Frozen region |',
        '|---|---|---:|---:|---|']
    for config, selections in rules.items():
        for method,item in selections.items():
            rule = 'PG fallback' if item['rule'] is None else json.dumps(item['rule'],separators=(',',':'))
            lines.append(f"| {config} | {method} | {item.get('support',0)} | {item['passing_rules']} | {rule} |")
    lines += ['', '## Actual final-test policy executions', '',
        'Total ratios divide sums of three-repeat per-parent median request times. '
        'Every policy, including fallback, is actually executed. The totals retain '
        '5-second-censored attempts and are not completed-workload speedups. '
        'Square brackets are conditional 95% intervals from 2,000 paired SQL-parent '
        'bootstrap draws. Parent copies remain paired across configurations.', '',
        '| Config | Method | Admitted / 120 | Baseline-time coverage | Total request ratio [CI] | Fully correct pairs | Correct-pair ratio [CI] |',
        '|---|---|---:|---:|---|---:|---|']
    for row in test:
        lines.append(f"| {row['config']} | {row['method']} | {row['admitted_parents']} | "
            f"{100*float(row['baseline_time_coverage']):.2f}% | {ci(row,'total_ratio')} | {row['correct_pairs']} | "
            f"{ci(row,'correct_pair_total_ratio','correct_pair_total_ci_low','correct_pair_total_ci_high')} |")
    lines += ['', 'PG-map and unconditional repair are controls; admitted count applies only to '
        'the four selected policies. Correct-pair results condition on all six timed '
        'PG/policy repetitions being correct for a parent; selection can bias that sensitivity.', '',
        '| Config | Policy | Coverage [CI] | Accepted median [CI] | Accepted ratio P95 [CI] | Accepted worst | Accepted correct / all |',
        '|---|---|---|---|---|---:|---:|']
    for row in test:
        if row['method'] not in ('source110','source125','target110','target125'):
            continue
        lines.append(f"| {row['config']} | {row['method']} | {ci(row,'coverage')} | "
            f"{ci(row,'accepted_ratio_p50')} | {ci(row,'accepted_ratio_p95')} | "
            f"{number(row['accepted_ratio_max'])} | {row['accepted_correct_parents']} / {row['accepted_parents']} |")
    lines += ['', 'An empty accepted set has undefined accepted tails. A null policy does not '
        'demonstrate useful transfer. Total savings under a 1.25 rule cannot establish the '
        'primary 1.10 objective; test tails are measured outcomes, not guaranteed bounds.', '',
        '## Tail definitions and strata', '',
        '| Config | Method | Q95(query ratio) | Q99(query ratio) | Ratio of latency P95 | Ratio of latency P99 |',
        '|---|---|---:|---:|---:|---:|']
    for row in test:
        lines.append(f"| {row['config']} | {row['method']} | {number(row['query_ratio_p95'])} | "
            f"{number(row['query_ratio_p99'])} | {number(row['ratio_of_latency_p95'])} | {number(row['ratio_of_latency_p99'])} |")
    lines += ['', 'Full drift-specific results and the prespecified nested source-like 60-parent '
        'view are in `runtime_summary.csv`. These views are not independent experiments. '
        '`paired_policies.csv` compares source transfer with target runtime revalidation; '
        '`paired_configurations.csv` preserves parent pairing when comparing costs.', '',
        '## Input and plan stability', '',
        f"Across {audit['paired_subplans']} paired connected subqueries, "
        f"{audit['changed_pg_subplan_features']} PG row features and {audit['changed_repair_maps']} "
        'repair-map entries change between configurations. Cost changes can alter plans '
        'even when cardinality features agree.', '',
        '| Role | Method | Paired parents | Plans changed between pg4 / pg11 |',
        '|---|---|---:|---:|']
    for row in read('configuration_plan_changes.csv'):
        lines.append(f"| {row['role']} | {row['method']} | {row['parents']} | {row['pg11_vs_pg4_plan_changes']} |")
    lines += ['', '## Cost ledger', '',
        'Historical source label acquisition, source fitting and source runtime validation '
        'are shared, already-paid costs, reported in `cost_ledger.json`. No new fit occurs. '
        'Configuration copies do not create distinct supervised labels. Test labels and '
        'test executions below are evaluation costs, not target runtime-selection inputs.', '',
        '| Role | Config | Reference success / attempts | Reference seconds | Timed seconds | Warmup seconds | PG/repair timed seconds |',
        '|---|---|---:|---:|---:|---:|---:|']
    for role, configs in ledger['new_costs'].items():
        for config,item in configs.items():
            lines.append(f"| {role} | {config} | {item['reference_successes']} / {item['reference_attempts']} | "
                f"{item['reference_ms']/1000:.3f} | {item['timed_ms']/1000:.3f} | {item['warmup_ms']/1000:.3f} | "
                f"{item['selection_pg_repair_timed_ms']/1000:.3f} |")
    lines += ['', 'Source-only transfer requires no target runtime validation. The comparator '
        'uses target reference acquisition and PG/repair validation measurements; PG-map '
        'replay is an audit control. Map generation, offline inference and configuration '
        'switches are separately costed. Do not add phase wall time to its measured '
        'components or add the same historical source cost once per configuration.', '',
        '## Interpretation boundary', '',
        'These are actual within-STATS mixture/configuration-transfer measurements. '
        'The new test is prospectively selected after earlier evidence motivated this '
        'design. Source-model/rule-selection uncertainty, independent source replications, '
        'cross-dataset transfer, native coverage versus development budget, and production '
        'map-refresh/concurrency lifecycles are not measured by this study.', '',
        'The PG-relative adapter predicts subplan residuals. This setting differs from '
        'the submitted frozen-MSCN root residual experiment; it cannot establish that '
        'MSCN structural/value gate outcomes are universal.', '']
    sensitivity = read('first_two_repetitions.csv')
    lines += ['## Timing-context diagnostic', '',
        'Post-run process inspection found another Python workload starting during the '
        'last approximately 84 seconds of the final test repetition. No unrelated process '
        'was stopped. The primary three-repeat observations remain unchanged. A '
        'retrospective sensitivity uses the first two complete repetitions for all '
        '120 parents and every policy, with no re-selection. It removes the entire '
        'repetition overlapping that observed workload; post-run enumeration cannot '
        'prove that all other system noise was absent.', '',
        '| Config | Source125 total ratio [CI], first two repetitions | Accepted ratio P95 |',
        '|---|---|---:|']
    for row in sensitivity:
        if row['method']=='source125':
            lines.append(f"| {row['config']} | {ci(row,'total_ratio')} | {number(row['accepted_ratio_p95'])} |")
    lines += ['', 'Both accepted-tail point estimates still exceed 1.25. '
        '`postrun_process_observation.json` preserves the original observation; '
        '`first_two_repetitions.csv` retains every method. '
        '`paired_fallback_sensitivity.csv` also compares the source125 rule with the '
        'actually executed source110 fallback control. Small timing differences among '
        'null policies are measurement variation, not repair benefits. '
        '`repetition_totals.csv` provides actual per-repetition request sums as a '
        'descriptive companion to the primary sums of per-query medians.', '']
    (OUT/'RESULTS_EN.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__ == '__main__':
    main()
