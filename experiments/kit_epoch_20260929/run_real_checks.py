"""Real KIT gradient, exact restart and all-three-sampler generation checks."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent

def stamp():
    return datetime.now(timezone.utc).isoformat()

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temp, path)

def execute(command, output, name):
    start = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    write(output / (name + '.launch.json'), dict(command=command, utc=stamp()))
    with (output / (name + '.log')).open('w') as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        write(output / (name + '.pid.json'), dict(pid=proc.pid, utc=stamp()))
        rc = proc.wait()
    write(output / (name + '.exit.json'), dict(exit_code=rc, elapsed_seconds=time.monotonic()-start, utc=stamp()))
    if rc:
        raise RuntimeError(str(output / (name + '.log')))

def check_job(job, checks):
    out = checks / job['id']
    common = [sys.executable, '-u', str(HERE / 'train_kit.py'), '--config', job['config'],
              '--stage', 'base', '--smoke', '--skip-validation']
    execute(common + ['--output', str(out / 'continuous'), '--stop-after-updates', '8'], out, 'continuous8')
    execute(common + ['--output', str(out / 'split'), '--stop-after-updates', '4'], out, 'split4')
    execute(common + ['--output', str(out / 'split'), '--stop-after-updates', '8', '--resume'], out, 'resume8')
    execute([sys.executable, str(HERE / 'check_kit_resume.py'), '--continuous', str(out / 'continuous/last.pt'),
             '--resumed', str(out / 'split/last.pt'), '--output', str(out / 'restart_comparison.json')], out, 'compare')
    comparisons = json.loads((out / 'restart_comparison.json').read_text())
    assert comparisons['PASS']
    for sampler in job['validation_samplers']:
        dest = out / ('generation_' + sampler)
        execute([sys.executable, '-u', str(HERE / 'evaluate_control.py'), '--config', job['config'],
                 '--checkpoint', str(out / 'continuous/last.pt'), '--output', str(dest), '--split', 'val',
                 '--seed', '3407', '--sampler', sampler, '--base-training', '--mode', 'smoke', '--smoke-batches', '1'],
                out, 'generation_' + sampler)
        assert (dest / 'COMPLETE').is_file()
    progress = json.loads((out / 'continuous/progress.json').read_text())
    result = dict(PASS=True, job=job['id'], real_data_identity=comparisons['real_data_identity'],
                  gradients=comparisons['gradient_observed'], actual_peak_allocated=progress['peak_allocated'],
                  actual_peak_reserved=progress['peak_reserved'], generated_samplers=job['validation_samplers'],
                  restart=str(out / 'restart_comparison.json'), utc=stamp())
    write(out / 'PASS.json', result)
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--tag', default='real_checks_v1')
    args = parser.parse_args()
    run = args.run.resolve()
    plan = json.loads((run / 'experiment_plan.json').read_text())
    checks = run / 'checks' / args.tag
    checks.mkdir(parents=True, exist_ok=False)
    sources = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in
               ('train_kit.py', 'eval_common.py', 'evaluate_control.py', 'control_guidance.py', 'check_kit_resume.py')}
    write(checks / 'source_lock.json', sources)
    results, failures = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        pending = {pool.submit(check_job, job, checks): job['id'] for job in plan['jobs']}
        for future in concurrent.futures.as_completed(pending):
            try:
                results.append(future.result())
            except Exception as exc:
                failures.append(dict(job=pending[future], error=repr(exc)))
            write(checks / 'status.json', dict(utc=stamp(), passed=results, failures=failures))
    if failures:
        raise RuntimeError(json.dumps(failures))
    actual = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in sources}
    assert actual == sources, 'Source changed during checks'
    write(run / 'checks/READY.json', dict(PASS=True, utc=stamp(), checks=results, source_hashes=sources,
                                         plan_sha256=hashlib.sha256((run / 'experiment_plan.json').read_bytes()).hexdigest()))
    print(json.dumps(dict(PASS=True, checks=results)), flush=True)

if __name__ == '__main__':
    main()
