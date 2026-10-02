"""Verify transfer provenance, complete execution, and reviewer-response traceability."""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
import time

import run_transfer as r

OUT = r.OUT
EXPECTED_COMMENTS = [
    'b1b1ddef0e5bdc6b6ac443dc4f95c9b08e300db413d7ec777d45baf4f659d5f8',
    'a7cd3cf58b9db7f959f56c1b1be558fa54b3185e38b1d90a8dec83e5627dd85f',
    'f185696b5caabc61f5f23c30dc87813e30c17b9f20169055e167379b569b9674',
    'f7933c1561f37fe5b51254b5f2a7a5b2b9fee9bfc7e786a51dfc497536800185',
    'b5fc21686e2c6c18185c3e6ad734d2b4476a6177dac96506145db64553808aa2',
]


def main():
    hashes = 0
    for phase in ('population','freeze','maps','validation','select','test'):
        hashes += len(r.verify(phase)['outputs'])
    analysis = r.read_json(OUT/'analysis_complete.json')
    assert analysis['script_sha256'] == r.n.hash_file(OUT/'analyze_transfer.py')
    for file,digest in analysis['inputs'].items():
        assert r.n.hash_file(file) == digest, file
        hashes += 1
    for file,digest in analysis['outputs'].items():
        assert r.n.hash_file(OUT/file) == digest, file
        hashes += 1
    sensitivity = r.read_json(OUT/'timing_sensitivity_complete.json')
    assert sensitivity['script_sha256']==r.n.hash_file(OUT/'audit_timing_sensitivity.py')
    assert sensitivity['test_sha256']==r.n.hash_file(OUT/'test_requests.csv')
    for file,digest in sensitivity['outputs'].items():
        assert r.n.hash_file(OUT/file)==digest, file
        hashes += 1
    previous = OUT.parent/'reviewer_rrdl'
    historic = r.read_json(previous/'audit_complete.json')
    for file,digest in historic['outputs'].items():
        assert r.n.hash_file(previous/file) == digest, file
        hashes += 1
    reply = previous/'RESPONSE_RRDL_EN.md'
    body = reply.read_text(encoding='utf-8')
    quotes = re.findall(r'^> (.+)$',body,re.MULTILINE)
    actual = [hashlib.sha256(re.sub(r'\s+',' ',x.strip()).encode()).hexdigest() for x in quotes]
    assert actual == EXPECTED_COMMENTS, 'Preserve all four questions and W3 exactly'
    assert 'draft_with_placeholders' in body
    assert 'timing is in progress' not in body
    assert '[EXPERIMENT_NEEDED: frozen source rule' not in body
    files = sorted(OUT.glob('*.md')) + [previous/x for x in
        ('README.md','REVIEWER_RRDL_ZH.md','RESPONSE_RRDL_EN.md','PENDING_EXPERIMENTS.md')]
    files += [r.ROOT/'rebuttal_20260928'/x for x in
              ('EVIDENCE_INDEX.md','REBUTTAL_DRAFT_EN.md','REBUTTAL_PLAN_ZH.md')]
    link_checks = set()
    for path in files:
        assert path.exists(), path
        text = path.read_text(encoding='utf-8-sig')
        assert '[MEASURED_RESULTS_PENDING:' not in text, path
        for link in re.findall(r'\]\((E:/[^)]+)\)',text):
            target = re.sub(r':\d+$','',link)
            assert Path(target).exists(), (path,link)
            link_checks.add(target)
    membership = r.n.csv_rows(OUT/'membership.csv')
    roles = {role:{x['semantic_id'] for x in membership if x['role']==role} for role in ('validation','test')}
    assert not roles['validation'] & roles['test']
    cases = r.read_json(OUT/'cases.json')
    assert len(cases)==360 and len({c['parent_id'] for c in cases})==180
    assert analysis['timed_requests']==6120 and analysis['warmup_requests']==2040
    assert analysis['all_injection_checks_passed'] and analysis['returned_count_conflicts']==0
    for role,ncases,methods in (('validation',120,3),('test',240,7)):
        requests = r.n.csv_rows(OUT/f'{role}_requests.csv')
        assert len(requests)==ncases*methods*3
        assert len({(x['qid'],x['method'],x['repetition']) for x in requests})==len(requests)
    inputs = {str(p):r.n.hash_file(p) for p in files}
    for filename in ('prepare_population.py','run_transfer.py','analyze_transfer.py','render_results.py',
                     'verify_completion.py','analysis_complete.json','selected_rules.json','policy_specification.json',
                     'audit_timing_sensitivity.py','timing_sensitivity_complete.json'):
        inputs[str(OUT/filename)] = r.n.hash_file(OUT/filename)
    r.write_json(OUT/'completion_audit.json',{'unix':time.time(),'checks_passed':True,
        'verified_hashes':hashes,'verified_local_link_targets':len(link_checks),'reviewer_comments_preserved':5,
        'timed_requests':6120,'warmups':2040,'reference_count_attempts':360,
        'native_parents':180,'configurations':2,'test_parents_per_configuration':120,
        'role_subquery_overlap':0,'original_submission_unchanged':True,'historical_audit_outputs_unchanged':True,
        'inputs':inputs,'package_status':'draft_with_placeholders',
        'remaining_scope':'Native coverage-by-budget, cross-dataset transfer and production lifecycle are not measured.'})
    r.n.emit('transfer_completion_verified',hashes=hashes,local_links=len(link_checks),timed_requests=6120)


if __name__ == '__main__':
    main()
