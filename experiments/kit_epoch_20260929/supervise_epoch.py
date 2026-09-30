"""Bounded two-job KIT supervisor. Resume is explicit; failures stay inspectable."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
STOP = False

def utc():
    return datetime.now(timezone.utc).isoformat()

def write(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    os.replace(tmp, path)

def preflight_jobs(run, plan, gate, allow_resume):
    from train_kit import canonical_sha, locked_config, sha
    checks = {item['job']: item for item in gate['checks']}
    assert len(checks) == len(gate['checks']) == len(plan['jobs'])
    assert set(checks) == {job['id'] for job in plan['jobs']}, 'READY jobs differ'
    prepared, audit = [], []
    for job in plan['jobs']:
        checked = checks[job['id']]
        assert checked['PASS'] is True
        assert checked['generated_samplers'] == job['validation_samplers'], 'Missing sampler check'
        manifest_path = Path(checked['restart']).parent / 'continuous/manifest.json'
        manifest = json.loads(manifest_path.read_text())
        config = locked_config(json.loads(Path(job['config']).read_text()), 'base')
        assert manifest['smoke'] is True and manifest['stage'] == 'base'
        assert canonical_sha(config) == manifest['config_sha256'], 'Checked configuration changed: ' + job['id']
        code = Path(config['paths']['code_root']).resolve()
        current = {str(p.relative_to(code)): sha(p) for p in sorted(code.rglob('*.py'))
                   if '.git' not in p.parts and '__pycache__' not in p.parts}
        current.update({'isolated/' + p.name: sha(p) for p in sorted(HERE.glob('*.py'))})
        expected = dict(manifest['source_hashes'])
        # Only this scheduler was authorized to change after the real smoke.
        key = 'isolated/supervise_epoch.py'
        before, after = expected.pop(key, None), current.pop(key, None)
        assert current == expected, 'Checked model/experiment source changed: ' + job['id']
        out = Path(job['output'])
        complete, resume = (out / 'COMPLETE').exists(), (out / 'last.pt').exists()
        if out.exists() and not complete and not (allow_resume and resume):
            raise FileExistsError('Inspect existing incomplete run before explicit resume: ' + str(out))
        if out.exists() and not complete:
            with (out / 'run.lock').open('a+') as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepared.append((job, out, complete, resume))
        audit.append(dict(job=job['id'], smoke_manifest=str(manifest_path),
                          config_sha256=manifest['config_sha256'], source_identity=canonical_sha(current),
                          supervisor_change=dict(checked_sha256=before, launch_sha256=after,
                                                 reason='authorized scheduler-only hardening after smoke')))
    write(run / 'launch_preflight.json', dict(PASS=True, utc=utc(), jobs=audit))
    return prepared

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run = args.run.resolve()
    guard = (run / 'supervisor.lock').open('a+')
    fcntl.flock(guard.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = json.loads((run / 'experiment_plan.json').read_text())
    gate = json.loads((run / 'checks/READY.json').read_text())
    assert gate['PASS'] is True, 'Real-data checks required'
    assert gate['plan_sha256'] == hashlib.sha256((run / 'experiment_plan.json').read_bytes()).hexdigest()
    for name, expected in gate['source_hashes'].items():
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == expected, 'Checked source changed: ' + name
    prepared = preflight_jobs(run, plan, gate, args.resume)
    state = dict(state='running', pid=os.getpid(), started_utc=utc(), jobs={})
    children = {}
    def stop(_s, _f):
        global STOP
        STOP = True
        for process, _log, _out in list(children.values()):
            if process.poll() is None:
                try:
                    process.send_signal(signal.SIGTERM)
                except ProcessLookupError:
                    pass
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    for job, out, complete, resume in prepared:
        if complete:
            state['jobs'][job['id']] = dict(state='complete', existing=True)
            continue
        if STOP:
            state['jobs'][job['id']] = dict(state='stopped', reason='stop requested before launch')
            continue
        command = [sys.executable, '-u', str(HERE / 'train_kit.py'), '--config', job['config'],
                   '--stage', 'base', '--output', str(out)]
        if resume:
            command.append('--resume')
        log = None
        try:
            log = (run / (job['id'] + '.log')).open('a')
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        except Exception as exc:
            if log is not None:
                log.close()
            state['jobs'][job['id']] = dict(state='failed', command=command, output=str(out),
                                            error=repr(exc), finished_utc=utc())
            write(run / 'supervisor_status.json', state)
            continue
        children[job['id']] = (child, log, out)
        state['jobs'][job['id']] = dict(state='running', pid=child.pid, command=command, output=str(out))
        write(run / (job['id'] + '_launch.json'), state['jobs'][job['id']])
        if STOP:
            stop(None, None)
    while children:
        for name, (process, log, out) in list(children.items()):
            rc = process.poll()
            if rc is not None:
                status = 'complete' if (out / 'COMPLETE').exists() and rc == 0 else 'stopped' if STOP and rc == 0 else 'failed'
                state['jobs'][name].update(state=status, exit_code=rc, finished_utc=utc())
                log.close()
                del children[name]
        state['heartbeat_utc'] = utc()
        write(run / 'supervisor_status.json', state)
        if children:
            time.sleep(10)
    state['state'] = 'complete' if all(j['state'] == 'complete' for j in state['jobs'].values()) else 'needs_attention'
    state['heartbeat_utc'] = utc()
    write(run / 'supervisor_status.json', state)
    print(json.dumps(state), flush=True)

if __name__ == '__main__':
    main()
