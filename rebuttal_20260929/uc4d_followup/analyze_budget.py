"""Verify supervision accounting, estimate conditional uncertainty, and plot."""
from __future__ import annotations

from collections import defaultdict
import gc
import itertools
import sys
import time

import run_budget as r
from analyze_estimator_controls import weighted_quantiles, weights_for

np, OUT = r.np, r.OUT


def interval(values):
    return np.percentile(values, [2.5, 97.5]).tolist()


def analyze():
    all_members, costs, per_split, summaries, comparisons, pass_rows = [], [], [], [], [], []
    for dindex, drift in enumerate(r.BUDGETS):
        directory = OUT/'budget'/drift
        started, complete = r.read_json(directory/'started.json'), r.read_json(directory/'complete.json')
        assert started['protocol_sha256'] == r.old.file_hash(OUT/'PROTOCOL.md')
        assert started['script_sha256'] == r.old.file_hash(r.__file__)
        assert started['helper_sha256'] == r.old.file_hash(r.old.__file__)
        for path, digest in started['inputs'].items():
            assert r.old.file_hash(path) == digest
        for path, digest in complete['outputs'].items():
            assert r.old.file_hash(directory/path) == digest
        all_members += r.old.read_csv(directory/'membership.csv')
        costs += r.old.read_csv(directory/'costs.csv')
        per_split += r.old.read_csv(directory/'per_split.csv')
        gates = r.read_json(directory/'gates.json')
        previous = {}
        for budget in r.BUDGETS[drift]:
            combined = defaultdict(list)
            for seed in r.SEEDS:
                cell = directory/str(seed)/str(budget)
                with np.load(cell/'test_predictions.npz') as saved:
                    data = {k: saved[k] for k in saved.files}
                splits = r.read_json(cell/'splits.json')
                assert sum(len(splits[k]) for k in ('fit', 'tune', 'gate')) == budget
                if seed in previous:
                    old = previous[seed]
                    assert old['test'] == splits['test']
                    for role in ('fit', 'tune', 'gate'):
                        assert splits[role][:len(old[role])] == old[role]
                previous[seed] = splits
                for key, val in data.items():
                    combined[key].append(val)
            combined = {k: np.concatenate(v) for k, v in combined.items()}
            weights = weights_for(combined['sql_id'], 20260929 + dindex*100 + budget)
            boots, observed = {}, {}
            for method in (k for k in combined if k not in ('sql_id', 'truth')):
                q = r.old.qerror(combined[method], combined['truth'])
                boots[method] = weighted_quantiles(q, weights, (.5, .95, .99))
                observed[method] = r.old.metrics(combined[method], combined['truth'])
            for method, stats in observed.items():
                family = method.removesuffix('_gated')
                passed = [x['passed'] for x in gates if x['budget'] == budget and x['method'] == family]
                rec = {'drift': drift, 'budget': budget, 'method': method,
                    'test_cases': len(combined['truth']), 'unique_test_sql': len(set(combined['sql_id'])),
                    'gate_passes': sum(passed) if passed else None, 'gate_trials': len(passed) if passed else None, **stats}
                for j, pct in enumerate((50, 95, 99)):
                    key = 'median' if pct == 50 else f'p{pct}'
                    ci = interval(boots[method][:, j]/boots['raw'][:, j])
                    absolute = interval(boots[method][:, j])
                    rec.update({f'p{pct}_ratio_vs_raw': stats[key]/observed['raw'][key],
                        f'p{pct}_ratio_ci_low': ci[0], f'p{pct}_ratio_ci_high': ci[1],
                        f'p{pct}_ci_low': absolute[0], f'p{pct}_ci_high': absolute[1]})
                summaries.append(rec)
            for family in r.FAMILIES:
                selected = [x for x in gates if x['budget'] == budget and x['method'] == family]
                pass_rows.append({'drift': drift, 'budget': budget, 'method': family,
                    'passes': sum(x['passed'] for x in selected), 'trials': len(selected),
                    'gate_labels_per_trial': budget//5})
            for left, right in (('residual_hgb', 'direct_hgb'), ('residual_rf', 'direct_rf'),
                                ('residual_hgb', 'residual_rf'), ('residual_hgb', 'scalar')):
                for mode in ('candidate', 'gated'):
                    if right == 'scalar' and mode == 'gated':
                        continue
                    a, b = [m+('_gated' if mode == 'gated' else '') for m in (left, right)]
                    rec = {'drift': drift, 'budget': budget, 'mode': mode, 'numerator': left, 'denominator': right}
                    for j, pct in enumerate((50, 95, 99)):
                        key = 'median' if pct == 50 else f'p{pct}'
                        ci = interval(boots[a][:, j]/boots[b][:, j])
                        rec.update({f'p{pct}_ratio': observed[a][key]/observed[b][key],
                            f'p{pct}_ci_low': ci[0], f'p{pct}_ci_high': ci[1]})
                    comparisons.append(rec)
            r.old.emit('budget_analysis', drift=drift, total_labels=budget)
            del weights, combined, boots
            gc.collect()
    unions = defaultdict(lambda: defaultdict(set))
    for row in all_members:
        unions[row['seed'], row['budget']][row['role']].add(row['sql_id'])
    union_rows = []
    for (seed, budget), roles in unions.items():
        for a, b in itertools.combinations(roles, 2):
            assert not roles[a] & roles[b], (seed, budget, a, b)
        union_rows.append({'seed': seed, 'per_drift_budget': budget,
            **{role+'_unique_sql': len(ids) for role, ids in roles.items()},
            'supervision_union': len(roles['fit'] | roles['tune'] | roles['gate'])})
    repeated = defaultdict(lambda: defaultdict(set))
    for row in all_members:
        repeated[row['drift'], row['budget']][row['role']].add(row['sql_id'])
    repeat_rows = []
    for (drift, budget), roles in repeated.items():
        # Outer replications can exchange roles; charge shared labels once.
        repeat_rows.append({'drift': drift, 'budget_per_trial': budget, 'outer_trials': len(r.SEEDS),
            **{role+'_union_sql': len(ids) for role, ids in roles.items()},
            'supervision_union_across_trials': len(roles['fit'] | roles['tune'] | roles['gate']),
            'cross_trial_fit_gate_overlap': len(roles['fit'] & roles['gate'])})
    for name, rows in (('budget_summary.csv', summaries), ('budget_paired.csv', comparisons),
                      ('budget_gate_passes.csv', pass_rows), ('budget_cross_drift_unions.csv', union_rows),
                      ('budget_cross_trial_unions.csv', repeat_rows),
                      ('budget_costs.csv', costs), ('budget_per_split.csv', per_split)):
        r.write_csv(OUT/name, rows)
    plot(summaries)
    outputs = [p for p in OUT.glob('budget_*') if p.is_file() and p.name != 'budget_analysis_complete.json']
    r.write_json(OUT/'budget_analysis_complete.json', {'unix': time.time(),
        'script_sha256': r.old.file_hash(__file__), 'model_selections': len(costs),
        'configuration_fits': sum(int(x['configurations']) for x in costs),
        'outputs': {p.name: r.old.file_hash(p) for p in outputs},
        'paired_bootstrap': '2000 SQL-cluster draws; conditional on fitted models and gates',
        'nested_budgets_and_fixed_test_verified': True, 'cross_drift_role_overlap': 0})
    r.old.emit('budget_analysis_complete', model_selections=len(costs))


def plot(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 9, 'axes.labelsize': 9,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8, 'text.color': 'black',
        'axes.labelcolor': 'black', 'xtick.color': 'black', 'ytick.color': 'black',
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    styles = [('raw', 'Raw MSCN', '#555555', ':', None), ('scalar', 'Scalar', '#ad8e4b', '--', 'x'),
        ('residual_hgb', 'Residual HGB', '#467684', '-', 'o'), ('direct_hgb', 'Direct HGB', '#467684', '--', '^'),
        ('residual_rf', 'Residual RF', '#a35c57', '-', 's'), ('direct_rf', 'Direct RF', '#a35c57', '--', 'd')]
    for mode in ('candidate', 'gated'):
        fig, axes = plt.subplots(2, 3, figsize=(7.3, 4.9), layout='constrained')
        for col, drift in enumerate(r.BUDGETS):
            for row, pct in enumerate((95, 99)):
                ax = axes[row, col]
                for name, label, color, ls, marker in styles:
                    # Scalar remains an ungated reference in both views.
                    key = name+'_gated' if mode == 'gated' and name in r.FAMILIES else name
                    series = [x for x in rows if x['drift'] == drift and x['method'] == key]
                    ax.plot([x['budget'] for x in series], [x[f'p{pct}'] for x in series],
                        label=label, color=color, linestyle=ls, marker=marker, markersize=3, linewidth=1.1)
                ax.set_yscale('log')
                ax.grid(axis='y', color='#dddddd', linewidth=.5)
                ax.set_axisbelow(True)
                ax.set_xticks(r.BUDGETS[drift])
                ax.set_xticklabels([str(b) for b in r.BUDGETS[drift]], rotation=40, ha='right')
                ax.set_ylabel(f'P{pct} q-error')
                letter = chr(ord('a')+row*3+col)
                ax.set_xlabel(f'Fit + tune + gate labels\n({letter}) {drift.capitalize()} drift, P{pct}')
                if drift == 'value':
                    from matplotlib.ticker import ScalarFormatter
                    ax.yaxis.set_major_formatter(ScalarFormatter())
                    ax.yaxis.set_minor_formatter(ScalarFormatter())
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside lower center', ncol=3, frameon=False)
        fig.savefig(OUT/f'budget_{mode}.pdf')
        fig.savefig(OUT/f'budget_{mode}.png', dpi=220)
        plt.close(fig)


if __name__ == '__main__':
    analyze()
