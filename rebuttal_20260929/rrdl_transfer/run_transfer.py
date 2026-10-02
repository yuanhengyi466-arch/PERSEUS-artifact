"""Actual source-frozen and target-revalidated policies under two PG costs."""
from __future__ import annotations

import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import random
import sys
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(OUT.parent/'uc4d_followup'))
import run_native as old

n, np, engine = old.n, old.np, old.engine
read_json, write_json, write_csv = n.read_json, n.write_json, n.write_csv
SOURCE = OUT.parent/'uc4d_followup/native'
CONFIGS = {'pg4': '4', 'pg11': '1.1'}
METHODS = ('pg','pg_map','repair')
POLICIES = ('source110','source125','target110','target125')
CAPS = {'110':1.10, '125':1.25}
SEED, REPEATS = 912, 3
MAP = OUT/'estimate_map.csv'


def start(phase):
    assert not (OUT/f'{phase}_started.json').exists(), f'Refusing overwrite: {phase}'
    write_json(OUT/f'{phase}_started.json', {'unix': time.time(), 'command': sys.argv,
        'runner_sha256': n.hash_file(__file__), 'protocol_sha256': n.hash_file(OUT/'PROTOCOL.md')})


def complete(phase, files, **fields):
    write_json(OUT/f'{phase}_complete.json', {'unix': time.time(),
        'outputs': {p:n.hash_file(OUT/p) for p in files}, **fields})


def verify(phase):
    if phase!='population':
        began = read_json(OUT/f'{phase}_started.json')
        assert began['runner_sha256']==n.hash_file(__file__)
        assert began['protocol_sha256']==n.hash_file(OUT/'PROTOCOL.md')
    ended = read_json(OUT/f'{phase}_complete.json')
    for file, expected in ended['outputs'].items():
        assert n.hash_file(OUT/file)==expected, file
    return ended


def choose(records, cap):
    passed = [r for r in records if r['support']>=20 and r['failures']==0 and r['median_ratio']<=.98
        and r['p95_ratio']<=cap and r['loss_fraction']<=.25 and r['whole_p95_latency_ratio']<=1.01
        and r['saved_ms']>0]
    if not passed:
        return {'rule':None, 'reason':'no qualifying validation region', 'cap':cap, 'passing_rules':0}
    best = max(passed,key=lambda r:(r['saved_ms'],r['support'],-r['p95_ratio'],-r['index']))
    return {**best, 'cap':cap, 'passing_rules':len(passed)}


def freeze():
    verify('population')
    start('freeze')
    for phase in ('prepare','train','maps','validation','select'):
        old.verify(phase)
    paths = [SOURCE/f for f in ('models.joblib','model_selection.json','estimator_gates.json',
        'development.json','rule_search.json','selected_rules.json','validation_requests.csv',
        'validation_count_cases.json','cost_ledger.json')]
    paths += [OUT.parent/'uc4d_followup/run_native.py',OUT.parent/'native_subplan_learning/run_learning.py',
        OUT.parent/'native_subplan_learning/semantic_audit.py',OUT.parent/'reviewer_rrdl/cap_selected_rules.json']
    prior = read_json(OUT.parent/'reviewer_rrdl/cap_selected_rules.json')['rules']
    records = [r for r in read_json(SOURCE/'rule_search.json') if r['policy']=='full__residual_hgb']
    selected = {f'source{name}':choose(records,cap) for name,cap in CAPS.items()}
    for name, cap in CAPS.items():
        expected = next(x['rule'] for x in prior if x['cap']==cap and x['policy']=='full__residual_hgb')
        assert selected[f'source{name}']['rule']==expected
    gates = read_json(SOURCE/'estimator_gates.json')['residual_hgb']
    assert all(x['passed'] for x in gates.values())
    dev_roles = {c['role'] for c in read_json(SOURCE/'development.json')}
    assert dev_roles=={'fit','tune','gate'}
    write_json(OUT/'source_rules.json',selected)
    write_json(OUT/'source_model.json',{'source_path':str(SOURCE/'models.joblib'),'family':'residual_hgb',
        'selection':read_json(SOURCE/'model_selection.json')['residual_hgb'],'estimator_gates':gates,
        'inputs':{str(p):n.hash_file(p) for p in paths}, 'source_validation_mix':{'column':30,'table':30,'value':30},
        'unchanged_source_validation_and_model':True})
    complete('freeze',['source_rules.json','source_model.json'])
    n.emit('source_policies_frozen',rules=selected)


def configure(cur, config):
    cur.execute("SELECT set_config('random_page_cost',%s,false)",(CONFIGS[config],))


def connect():
    conn = n.connect()
    with conn.cursor() as cur:
        cur.execute("SET application_name='perseus_rrdl_transfer'")
        cur.execute("SET work_mem='4MB'")
        cur.execute("SELECT application_name,state FROM pg_stat_activity WHERE pid!=pg_backend_pid() AND state='active'")
        assert not cur.fetchall(), 'Do not mix native measurements with another active DB session'
    return conn


def maps():
    verify('freeze')
    verify('population')
    start('maps')
    model = n.joblib.load(SOURCE/'models.joblib')['residual_hgb']
    parents = read_json(OUT/'population.json')
    cases, observations, maprows, costs = [],[],[],[]
    environment = {'configurations':{},'python':sys.version,'cached_maps':True,
        'callback_sha256':n.hash_file(ROOT/'baselines/End-to-End-CardEst-Benchmark/postgresql-13.1/src/backend/optimizer/path/costsize.c')}
    conn=connect()
    with conn.cursor() as cur:
        cur.execute('SELECT version()')
        environment['version']=cur.fetchone()[0]
        cur.execute("SELECT c.relname,c.reltuples,c.relpages FROM pg_class c JOIN pg_namespace ns ON ns.oid=c.relnamespace WHERE ns.nspname='stats_inj' AND c.relkind='r' ORDER BY c.relname")
        environment['catalog']=cur.fetchall()
        for config in CONFIGS:
            configure(cur,config)
            cur.execute("SELECT name,setting FROM pg_settings WHERE name IN ('jit','max_parallel_workers_per_gather','statement_timeout','work_mem','plan_cache_mode','join_collapse_limit','from_collapse_limit','random_page_cost')")
            environment['configurations'][config]=dict(cur.fetchall())
            for pos,parent in enumerate(parents):
                case={**parent,'parent_id':parent['qid'],'qid':f"{config}|{parent['qid']}",'config':config}
                sqlrows, pg, batch = [],[],[]
                tick=time.perf_counter()
                for aliases,row in engine.parts(case['row']):
                    cur.execute('EXPLAIN (FORMAT JSON) '+n.subplans.subset_sql(case['row'],aliases))
                    plan=n.injection.plan_doc(cur.fetchone()[0])
                    value=n.injection.collect_plan_stats(plan)['root_input_estimate']
                    assert value is not None
                    pg.append(max(1.,float(value)))
                    sqlrows.append(row)
                    batch.append({'qid':case['qid'],'parent_id':case['parent_id'],'config':config,'role':case['role'],
                        'drift':case['drift'],'semantic_id':old.identity(row),'aliases':' '.join(aliases),
                        'tables':len(aliases),'is_root':len(aliases)==case['tables'],'pg_rows':pg[-1]})
                explain_ms=(time.perf_counter()-tick)*1000
                tick=time.perf_counter()
                pred=old.predict(model,sqlrows,pg)
                inference_ms=(time.perf_counter()-tick)*1000
                case['root_pg_rows']=pg[-1]
                case['root_ratios']={'residual_hgb':float(pred[-1]/pg[-1])}
                case['root_estimates']={'pg_map':pg[-1],'repair':float(pred[-1])}
                cases.append(case)
                for j,record in enumerate(batch):
                    for method,value in (('pg_map',pg[j]),('repair',float(pred[j]))):
                        maprows.append({'query_id':case['qid'],'aliases':record['aliases'],'method':method,'estimate':value})
                observations.extend(batch)
                costs.append({'qid':case['qid'],'config':config,'role':case['role'],'explain_ms':explain_ms,
                    'inference_ms':inference_ms,'subqueries':len(batch)})
                if (pos+1)%30==0:
                    n.emit('transfer_maps',config=config,parents=pos+1,total=len(parents))
    conn.close()
    environment['processes']=[]
    for proc in n.psutil.process_iter(['pid','name','cmdline']):
        info=proc.info
        if info['name'] and info['name'].lower().startswith(('python','postgres')):
            environment['processes'].append({'pid':info['pid'],'name':info['name'],
                'entrypoint':next((x for x in (info['cmdline'] or [])[1:] if x.endswith('.py')),None)})
    for config,settings in environment['configurations'].items():
        assert float(settings['random_page_cost'])==float(CONFIGS[config])
    reference={k:v for k,v in environment['configurations']['pg4'].items() if k!='random_page_cost'}
    assert reference=={k:v for k,v in environment['configurations']['pg11'].items() if k!='random_page_cost'}
    write_json(OUT/'cases.json',cases)
    write_csv(OUT/'native_subplans.csv',observations)
    write_csv(MAP,maprows)
    write_csv(OUT/'map_costs.csv',costs)
    write_json(OUT/'environment.json',environment)
    complete('maps',['cases.json','native_subplans.csv','estimate_map.csv','map_costs.csv','environment.json'])


def active_method(case,method,rules):
    if method not in POLICIES:
        return method
    rule=rules[case['config']][method]['rule']
    return 'repair' if engine.matches(case,rule,'residual_hgb') else 'pg'


def bind_engine():
    engine.MAP=MAP
    engine.active_method=active_method


def request(cur,case,method,rules):
    began=time.perf_counter()
    active=active_method(case,method,rules)
    decision_ms=(time.perf_counter()-began)*1000
    tick=time.perf_counter()
    activation_ms,select_ms=0.,0.
    count,status,error=None,'ok',''
    try:
        engine.activate(cur,case,active)
        activation_ms=(time.perf_counter()-tick)*1000
        tick=time.perf_counter()
        cur.execute(case['sql'])
        count=int(cur.fetchone()[0])
        select_ms=(time.perf_counter()-tick)*1000
        if case['truth'] is None:
            status='reference_unavailable'
        elif count!=int(case['truth']):
            status='count_conflict'
    except Exception as exc:
        select_ms=(time.perf_counter()-tick)*1000
        status='timeout' if isinstance(exc,n.psycopg.errors.QueryCanceled) else 'error'
        error=f'{type(exc).__name__}: {str(exc)[:200]}'
    return {'qid':case['qid'],'parent_id':case['parent_id'],'sql_id':case['sql_id'],'config':case['config'],
        'role':case['role'],'drift':case['drift'],'method':method,'active':active,'status':status,'count':count,
        'archived_label_match':None if count is None else count==int(case['archived_truth']),
        'wall_ms':(time.perf_counter()-began)*1000,'decision_ms':decision_ms,
        'activation_ms':activation_ms,'select_ms':select_ms,'error':error}


def measure(role):
    verify('maps')
    verify('freeze')
    if role=='test':
        verify('select')
    start(role)
    bind_engine()
    rules=read_json(OUT/'selected_rules.json') if role=='test' else {}
    cases=[c for c in read_json(OUT/'cases.json') if c['role']==role]
    methods=METHODS+(POLICIES if role=='test' else ())
    byparent=defaultdict(dict)
    for case in cases:
        byparent[case['parent_id']][case['config']]=case
    references,warmups,requests,switches,blocks=[],[],[],[],[]
    rng=random.Random(SEED+(role=='test'))
    conn=connect()
    with conn.cursor() as cur:
        order=list(byparent)
        rng.shuffle(order)
        for pos,parent in enumerate(order):
            configs=list(CONFIGS)
            rng.shuffle(configs)
            observed=[]
            for config in configs:
                case=byparent[parent][config]
                configure(cur,config)
                cur.execute('SET perseus_estmap_enabled=off')
                tick=time.perf_counter()
                count,status,error=None,'ok',''
                try:
                    cur.execute(case['sql'])
                    count=int(cur.fetchone()[0])
                    observed.append(count)
                except Exception as exc:
                    status='timeout' if isinstance(exc,n.psycopg.errors.QueryCanceled) else 'error'
                    error=f'{type(exc).__name__}: {str(exc)[:200]}'
                case['truth']=count
                references.append({'qid':case['qid'],'parent_id':parent,'config':config,'role':role,
                    'drift':case['drift'],'actual':count,'archived_truth':case['archived_truth'],
                    'status':status,'wall_ms':(time.perf_counter()-tick)*1000,'error':error})
            assert len(set(observed))<=1, f'Cross-configuration reference conflict {parent}'
            if (pos+1)%10==0:
                write_csv(OUT/f'{role}_references.csv',references)
                n.emit('transfer_references',role=role,parents=pos+1,total=len(order),
                    statuses=dict(Counter(x['status'] for x in references)))
        write_csv(OUT/f'{role}_references.csv',references)
        write_json(OUT/f'{role}_count_cases.json',cases)
        audit_files=[]
        for config in CONFIGS:
            directory=OUT/config
            directory.mkdir(exist_ok=True)
            engine.OUT=directory
            configure(cur,config)
            engine.audit_plans(cur,[c for c in cases if c['config']==config],methods,rules,role)
            audit_files.extend(f'{config}/{role}_{s}' for s in ('root_checks.csv','join_checks.csv','plans.json'))
        for repeat in (-1,0,1,2):
            order=list(byparent)
            rng.shuffle(order)
            start_block=time.perf_counter()
            for pos,parent in enumerate(order):
                configs=list(CONFIGS)
                rng.shuffle(configs)
                for config_rank,config in enumerate(configs):
                    case=byparent[parent][config]
                    tick=time.perf_counter()
                    configure(cur,config)
                    switches.append({'role':role,'repetition':repeat,'parent_id':parent,'config':config,
                        'config_order':config_rank,'switch_ms':(time.perf_counter()-tick)*1000})
                    conditions=list(methods)
                    rng.shuffle(conditions)
                    for rank,method in enumerate(conditions):
                        row=request(cur,case,method,rules)
                        row.update(repetition=repeat,position=pos,config_order=config_rank,condition_rank=rank)
                        (warmups if repeat==-1 else requests).append(row)
                if (pos+1)%10==0:
                    current=warmups if repeat==-1 else requests
                    write_csv(OUT/f'{role}_{"warmup" if repeat==-1 else "requests"}.csv',current)
                    n.emit('transfer_measurement',role=role,repetition=repeat,parents=pos+1,total=len(order),
                        requests=len(current),statuses=dict(Counter(x['status'] for x in current)))
            blocks.append({'repetition':repeat,'wall_s':time.perf_counter()-start_block,'unix':time.time()})
    conn.close()
    write_csv(OUT/f'{role}_requests.csv',requests)
    write_csv(OUT/f'{role}_warmup.csv',warmups)
    write_csv(OUT/f'{role}_config_switches.csv',switches)
    write_json(OUT/f'{role}_blocks.json',blocks)
    complete(role,audit_files+[f'{role}_{s}' for s in ('requests.csv','warmup.csv','config_switches.csv',
        'blocks.json','references.csv','count_cases.json')],requests=len(requests),
        statuses=dict(Counter(x['status'] for x in requests)))


def aggregate(role):
    grouped=defaultdict(list)
    for row in n.csv_rows(OUT/f'{role}_requests.csv'):
        grouped[row['qid'],row['method']].append(row)
    result={}
    for key,rows in grouped.items():
        assert len(rows)==REPEATS and {int(r['repetition']) for r in rows}=={0,1,2}
        result[key]={'ms':float(np.median([float(r['wall_ms']) for r in rows])),
            'failed':sum(r['status']!='ok' for r in rows),'active':rows[0]['active']}
    return result


def select():
    verify('validation')
    verify('freeze')
    assert not (OUT/'test_started.json').exists()
    start('select')
    lookup=aggregate('validation')
    source_rules=read_json(OUT/'source_rules.json')
    gates=read_json(OUT/'source_model.json')['estimator_gates']
    selected,search={},[]
    for config in CONFIGS:
        cases=[c for c in read_json(OUT/'validation_count_cases.json') if c['config']==config]
        base=np.array([lookup[c['qid'],'pg']['ms'] for c in cases])
        values=np.array([lookup[c['qid'],'repair']['ms'] for c in cases])
        ratios=values/base
        records=[]
        for index,rule in enumerate(engine.candidate_rules()):
            if not all(gates[d]['passed'] for d in rule['drifts']):
                continue
            mask=np.array([engine.matches(c,rule,'residual_hgb') for c in cases])
            support=int(mask.sum())
            if support<20:
                continue
            failures=sum(lookup[c['qid'],m]['failed'] for c,yes in zip(cases,mask) if yes for m in ('pg','repair'))
            q50,q95=np.percentile(ratios[mask],[50,95])
            whole=np.where(mask,values,base)
            records.append({'config':config,'index':index,'rule':rule,'support':support,'failures':failures,
                'median_ratio':float(q50),'p95_ratio':float(q95),'loss_fraction':float(np.mean(ratios[mask]>1.01)),
                'whole_p95_latency_ratio':float(np.percentile(whole,95)/np.percentile(base,95)),
                'saved_ms':float(base.sum()-whole.sum())})
        selected[config]={**source_rules,**{f'target{name}':choose(records,cap) for name,cap in CAPS.items()}}
        search.extend(records)
    write_json(OUT/'target_rule_search.json',search)
    write_json(OUT/'selected_rules.json',selected)
    complete('select',['target_rule_search.json','selected_rules.json'],test_was_absent=True,
        validation_sha256=n.hash_file(OUT/'validation_requests.csv'))
    n.emit('target_rules_frozen',rules=selected)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=('freeze','maps','validation','select','test','all'))
    args=parser.parse_args()
    for phase in ('freeze','maps','validation','select','test') if args.phase=='all' else (args.phase,):
        if phase in ('validation','test'):
            measure(phase)
        else:
            globals()[phase]()
