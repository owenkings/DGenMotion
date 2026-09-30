"""Create a new KIT run using verified old assets; never overwrite old runs."""
import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True,
                        help='Local asset workspace containing runs/paper_controls_20260920; '
                             'new runs are created under this workspace')
    args = parser.parse_args()
    ROOT = args.root.expanduser().resolve()
    RUN = ROOT / 'runs/kit_epoch_20260929'
    if RUN.exists():
        raise FileExistsError('Existing new run must be inspected, not prepared again')
    old = ROOT / 'runs/paper_controls_20260920'
    cfg = json.loads((old / 'KIT_base/config.json').read_text())
    ae_complete = json.loads((old / 'KIT_tokenizer/COMPLETE').read_text())
    assert ae_complete['state'] == 'complete' and ae_complete['epochs_complete'] == 50
    assert sha(cfg['paths']['ae_checkpoint']) == cfg['hashes']['ae_checkpoint'] == ae_complete['best_checkpoint_sha256']
    source = Path(cfg['paths']['code_root']).resolve()
    target = HERE / 'code'
    if target.exists():
        raise FileExistsError(target)
    if source == target.resolve() or source in target.resolve().parents:
        raise ValueError('The source code_root must be separate from the target experiment/code directory')
    shutil.copytree(source, target, symlinks=True, ignore=shutil.ignore_patterns('__pycache__', '.git'))
    source_hashes = {str(p.relative_to(source)): sha(p) for p in source.rglob('*.py') if '__pycache__' not in p.parts}
    target_hashes = {str(p.relative_to(target)): sha(p) for p in target.rglob('*.py') if '__pycache__' not in p.parts}
    assert source_hashes == target_hashes, 'Code backup differs'
    (RUN / 'configs').mkdir(parents=True)
    (RUN / 'checks').mkdir()
    retained = ('dataset', 'paths', 'hashes', 'model', 'sampling', 'evaluation', 'data_manifest')
    base = {k: copy.deepcopy(cfg[k]) for k in retained}
    base.update(schema=1, seed=3407, cpu_threads=4, tf32=True, initialization='from_scratch_KIT_no_HumanML_weights',
                objective='native_from_scratch', free_space_floor_bytes=30 * 1024**3)
    base['paths']['code_root'] = str(target)
    base['paths'].pop('base_checkpoint', None)
    base['hashes'].pop('base_checkpoint', None)
    base['evaluation'].pop('steps', None)
    base['evaluation'].update(split='val', seed=3407, selection='minimum_validation_generation_fid_earlier_tie',
                              frequency='every_completed_epoch', separate_rng_process=True)
    base['kit_base'] = copy.deepcopy(cfg['kit_base'])
    base['kit_base'].update(epochs=500, validation_every=274, validation_every_epoch=True,
                            selection='minimum_validation_generation_fid_earlier_tie')
    plans = []
    for ident, model, samplers in [('K_FSQ', 'FSQ-MARDM-SiT-XL', ['native_cfg']),
                                   ('K_FSQ_JiT', 'FSQ-MARDM-DiT-XL', ['native_cfg', 'direct_apg'])]:
        item = copy.deepcopy(base)
        item['experiment_id'] = ident
        item['model'].update(name=model, generator=model, generator_model=model)
        item['kit_base']['validation_samplers'] = samplers
        path = RUN / 'configs' / (ident + '.json')
        path.write_text(json.dumps(item, indent=2) + '\n')
        plans.append(dict(id=ident, config=str(path), output=str(RUN / ident),
                          epochs=500, expected_updates_per_epoch=274, expected_total_updates=137000,
                          validation_samplers=samplers, initialization='random; shared frozen KIT AE'))
    plan = dict(schema=1, root=str(ROOT), experiment=str(HERE), run=str(RUN), jobs=plans,
                user_decision='2026-09-29: retrain both generators from scratch; validate every epoch',
                reused_tokenizer=dict(path=base['paths']['ae_checkpoint'], sha256=base['hashes']['ae_checkpoint'],
                                      completed_epochs=50),
                max_concurrent_training_jobs=2, effective_batch=16, training_seed=3407,
                validation_seed=3407, maximum_generation_validations=1500,
                checkpoint_policy='full last.pt + previous; per-sampler best; final; bounded pending snapshot',
                stop_policy='500 epochs; no FID early stop; stop affected job on numerical/data/checkpoint failure',
                comparisons='FSQ SiT CFG, FSQ JiT CFG, SAME JiT checkpoint with direct APG',
                compute_caveat='SiT retains adaptive dopri5; JiT uses 50 fixed iterations; NFE is measured, not equated',
                speed_benchmark_policy='No isolated latency claim while existing workloads overlap',
                old_runs_preserved=True, source_copy_sha256=source_hashes)
    (RUN / 'experiment_plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    print(json.dumps({k:plan[k] for k in ('run','jobs','reused_tokenizer','maximum_generation_validations')}))

if __name__ == '__main__':
    main()
