"""Post-run diagnostic for an observed late unrelated Python workload."""
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import time

import analyze_transfer as a

r, OUT, np = a.r, a.OUT, a.np


def main():
    r.verify('test')
    records = r.n.csv_rows(OUT/'test_requests.csv')
    cases = r.read_json(OUT/'test_count_cases.json')
    grouped = defaultdict(list)
    for row in records:
        if int(row['repetition']) in (0,1):
            grouped[row['qid'],row['method']].append(row)
    lookup = {}
    for key,rows in grouped.items():
        assert len(rows)==2 and {int(x['repetition']) for x in rows}=={0,1}
        lookup[key] = {'ms':float(np.median([float(x['wall_ms']) for x in rows])),
            'failed':sum(x['status']!='ok' for x in rows),'active':rows[0]['active']}
    summaries, contrasts = [], []
    for config in r.CONFIGS:
        selected = [c for c in cases if c['config']==config]
        draws = np.random.default_rng(r.SEED).integers(0,len(selected),(a.BOOT,len(selected)))
        for method in (*r.METHODS,*r.POLICIES):
            summaries.append({'config':config,'repetitions':'0,1','role':'test','population':'all',
                              **a.stats(selected,lookup,method,draws)})
        contrasts.append({'config':config,'repetitions':'0,1',
                          **a.compare(selected,lookup,'source125','source110',draws)})
        contrasts.append({'config':config,'repetitions':'0,1,2',
                          **a.compare(selected,r.aggregate('test'),'source125','source110',draws)})
    start = r.read_json(OUT/'test_started.json')['unix']
    end = r.read_json(OUT/'test_complete.json')['unix']
    blocks = r.read_json(OUT/'test_blocks.json')
    final_start = blocks[-1]['unix']-blocks[-1]['wall_s']
    processes = []
    for proc in r.n.psutil.process_iter(['pid','ppid','name','cmdline','create_time','cpu_times']):
        info = proc.info
        if info['name'] and info['name'].lower().startswith('python') and start<info['create_time']<end:
            entry = next((Path(x).name for x in (info['cmdline'] or []) if x.endswith('.py')),None)
            processes.append({'pid':info['pid'],'parent_pid':info['ppid'],'name':info['name'],
                'script_basename':entry,'created_unix':info['create_time'],
                'created_utc':datetime.fromtimestamp(info['create_time'],timezone.utc).isoformat(),
                'cpu_seconds_at_observation':sum(info['cpu_times'][:2]) if info['cpu_times'] else None,
                'possible_overlap_seconds':end-info['create_time']})
    observation = {'observed_unix':time.time(),'test_start_unix':start,'test_end_unix':end,
        'final_repetition_start_unix':final_start,'observed_processes_started_during_test':processes,
        'interpretation':'A separate Python workload was observed post-run to have started late in the last repetition. No process was terminated. Post-run enumeration cannot recover exited processes or quantify interference. Primary three-repeat results are retained. The first-two-repeat sensitivity retains all 120 parents and all policies with no new selection; it removes the entire repetition overlapping the observed workload.'}
    r.write_csv(OUT/'first_two_repetitions.csv',summaries)
    r.write_csv(OUT/'paired_fallback_sensitivity.csv',contrasts)
    observation_path = OUT/'postrun_process_observation.json'
    if observation_path.exists():
        observation = r.read_json(observation_path)
        assert observation['test_start_unix']==start and observation['test_end_unix']==end
    else:
        r.write_json(observation_path,observation)
    files = ['first_two_repetitions.csv','paired_fallback_sensitivity.csv','postrun_process_observation.json']
    r.write_json(OUT/'timing_sensitivity_complete.json',{'unix':time.time(),'script_sha256':r.n.hash_file(__file__),
        'test_sha256':r.n.hash_file(OUT/'test_requests.csv'),'primary_results_unchanged':True,
        'retrospective_diagnostic':True,'outputs':{f:r.n.hash_file(OUT/f) for f in files}})
    r.n.emit('timing_sensitivity_complete',retained_requests=len(grouped)*2,
             observed_processes=len(observation['observed_processes_started_during_test']))


if __name__=='__main__':
    main()
