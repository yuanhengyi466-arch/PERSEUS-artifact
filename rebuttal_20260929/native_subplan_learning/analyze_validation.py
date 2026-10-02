"""Explain frozen validation rejections; never selects a different test policy."""
from collections import defaultdict
import json

import run_learning as r


def main():
    r.verify('validation')
    r.verify('select')
    rules = r.read_json(r.OUT/'rule_search.json')
    cases = [c for c in r.read_json(r.OUT/'cases.json') if c['role'] == 'validation']
    lookup = r.aggregate('validation')
    rejection = []
    for policy in r.POLICIES:
        rows = [x for x in rules if x['policy'] == policy]
        correct = [x for x in rows if x['failures'] == 0]
        model = policy.split('_')[-1]
        masks = {tuple(c['qid'] for c in cases if r.matches(c, x['rule'], model)) for x in correct}
        best = min(correct, key=lambda x: x['p95_ratio']) if correct else None
        rejection.append({'policy': policy, 'supported_rules': len(rows), 'correct_supported_rules': len(correct),
            'unique_correct_supported_regions': len(masks), 'passing_rules': sum(x['passed'] for x in rows),
            'best_p95_rule': best, 'p95_cap': 1.10,
            'rejection_interpretation': 'P95 alone excludes every correct supported region' if best and best['p95_ratio'] > 1.10
                                         else 'Inspect joint constraints'})
    shapes = {(x['qid'], x['method']): x['plan_shape_sha256'] for x in r.n.csv_rows(r.OUT/'validation_root_checks.csv')}
    summaries = []
    for drift in ('all', 'column', 'table', 'value'):
        for method in r.METHODS:
            for changed in ('all', False, True):
                selected = [c for c in cases if (drift == 'all' or c['drift'] == drift)
                            and (changed == 'all' or (shapes[c['qid'], 'pg'] != shapes[c['qid'], method]) == changed)
                            and lookup[c['qid'], 'pg']['failed'] == 0 and lookup[c['qid'], method]['failed'] == 0]
                if not selected:
                    continue
                baseline = r.np.array([lookup[c['qid'], 'pg']['ms'] for c in selected])
                values = r.np.array([lookup[c['qid'], method]['ms'] for c in selected])
                ratio = values/baseline
                summaries.append({'drift': drift, 'method': method, 'plan_changed': changed, 'paired_correct_parents': len(selected),
                    'base_total_ms': float(baseline.sum()), 'method_total_ms': float(values.sum()),
                    'total_ratio': float(values.sum()/baseline.sum()), 'saved_ms': float(baseline.sum()-values.sum()),
                    'median_ratio': float(r.np.median(ratio)), 'p95_ratio': float(r.np.percentile(ratio, 95)),
                    'p99_ratio': float(r.np.percentile(ratio, 99)), 'worst_ratio': float(ratio.max()),
                    'loss_fraction_gt_1_01': float(r.np.mean(ratio > 1.01))})
    r.write_json(r.OUT/'validation_rejection_diagnostic.json', rejection)
    r.write_csv(r.OUT/'validation_plan_change_diagnostic.csv', summaries)
    r.write_json(r.OUT/'validation_diagnostic_provenance.json', {
        'analysis_sha256': r.n.hash_file(__file__),
        'validation_requests_sha256': r.n.hash_file(r.OUT/'validation_requests.csv'),
        'selected_rules_sha256': r.n.hash_file(r.OUT/'selected_rules.json'),
        'scope': 'Post-selection validation-only explanation. Does not use test outcomes or alter any policy.'})
    print(json.dumps(rejection, indent=2))


if __name__ == '__main__':
    main()
