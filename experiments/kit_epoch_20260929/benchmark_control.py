"""Synchronized text-to-decoded-motion latency; never infer speed from NFE.

The parent queue must finish/pause its own GPU workers before launching this
script. Other users' processes are only observed and never stopped. Only fixed
step samplers may share a structural timing across same-architecture weights;
adaptive SiT checkpoints require separate timing runs.
"""
import argparse
import csv
import gc
import json
import os
from pathlib import Path
import re
import time
import traceback

import numpy as np
import torch

from eval_common import (PairedDataset, append_jsonl, build_dataset, configure_precision,
    device_snapshot, generate_motion, json_sha, load_models, load_stats, read_config,
    seed_sampling, sha_file, source_manifest, utc_now, write_json)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GPU_ENTRYPOINTS = frozenset((
    'train_kit.py', 'train_control.py', 'train_MARDM.py', 'train_AE.py',
    'evaluate_control.py', 'benchmark_control.py', 'evaluation_MARDM.py',
    'evaluation_AE.py', 'quick_eval.py', 'quick_evaluation.py',
    'comprehensive_evaluation.py',
))


def python_entrypoint(args, cwd):
    """Resolve a Python script/module entrypoint, never a later script argument."""
    if not args or not re.fullmatch(r'(?:python|pypy)(?:\d+(?:\.\d+)*)?(?:\.exe)?', Path(args[0]).name):
        return None
    index = 1
    while index < len(args):
        arg = args[index]
        if arg.startswith('-c') or arg == '-':
            return None
        if arg == '-m':
            if index + 1 == len(args):
                return None
            return (cwd / Path(*args[index + 1].split('.'))).with_suffix('.py').resolve()
        if arg in ('-W', '-X', '--check-hash-based-pycs'):
            index += 2
            continue
        if arg == '--':
            index += 1
            break
        if not arg.startswith('-'):
            break
        index += 1
    if index >= len(args):
        return None
    script = Path(args[index])
    return (script if script.is_absolute() else cwd / script).resolve()


def own_gpu_workers(code_root=None, proc_root=Path('/proc')):
    """Observe project Python workers in /proc; never signal any process.

    The configured code snapshot and this checkout both belong to the run.
    Relative script paths are resolved against each process's actual cwd.
    """
    roots = [PROJECT_ROOT]
    if code_root is not None:
        roots.append(Path(code_root).resolve())
    workers = []
    for proc in proc_root.glob('[0-9]*'):
        if int(proc.name) == os.getpid():
            continue
        try:
            args = [os.fsdecode(a) for a in (proc / 'cmdline').read_bytes().split(b'\0') if a]
            script = python_entrypoint(args, (proc / 'cwd').resolve(strict=True))
            if (script is not None and script.name in GPU_ENTRYPOINTS
                    and any(script.is_relative_to(root) for root in roots)):
                workers.append(dict(pid=int(proc.name), command=' '.join(args),
                                    argv=args, entrypoint=str(script)))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return workers


def architecture_signature(model, ae):
    def signature(module):
        return json_sha([(name, list(value.shape), str(value.dtype))
                         for name, value in sorted(module.state_dict().items())])
    return dict(generator=signature(model), ae=signature(ae),
                head_type=type(model.DiffMLPs).__name__,
                generator_type=type(model).__name__, ae_type=type(ae).__name__)


def benchmark_inputs(config, dataset):
    benchmark = config.get('benchmark', {})
    if list(benchmark.get('batch_sizes', [1, 32])) != [1, 32]:
        raise ValueError('Preregistered latency batches are 1 and 32')
    lengths = list(benchmark.get('lengths', [60, 120, 196]))
    minimum = 40 if config['dataset'] == 't2m' else 24
    if any(not minimum <= length <= 196 or length % 4 for length in lengths):
        raise ValueError('Benchmark length illegal for repository dataset/unit4 protocol')
    if lengths != [60, 120, 196] and not benchmark.get('length_substitution_reason'):
        raise ValueError('A changed length set needs a preregistered substitution reason')
    paired = PairedDataset(dataset, int(benchmark.get('seed', 20000)))
    if len(paired) < 32:
        raise ValueError('Need at least 32 real caption entries')
    entries = [paired[index][1] for index in range(32)]
    return dict(dataset=config['dataset'], seed=int(benchmark.get('seed', 20000)),
                entries=entries, batch_sizes=[1, 32], lengths=lengths,
                source_split='val', requested_length_source='predeclared lengths, not data caption lengths',
                caption_selection='first32 length-sorted validation entries, fixed per-entry caption RNG',
                same_text_cache_policy='no text embedding cache; CLIP tokenization/encoding occurs inside every timed call',
                length_substitution_reason=benchmark.get('length_substitution_reason'))


@torch.no_grad()
def timed_call(config, model, ae, controller, captions, frames, seed):
    seed_sampling(seed)
    lengths = torch.full((len(captions),), frames, dtype=torch.long, device='cuda')
    controller.reset()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    motion = generate_motion(config, model, ae, captions, lengths)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    allocated = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    # These checks are outside the timer and never masquerade as generation.
    if tuple(motion.shape) != (len(captions), frames, 67 if config['dataset'] == 't2m' else 64):
        raise ValueError('Actual returned frame/feature dimensions differ: ' + str(tuple(motion.shape)))
    if not bool(torch.isfinite(motion).all()) or seconds <= 0 or not np.isfinite(seconds):
        raise FloatingPointError('Nonfinite benchmark output/duration')
    budget = controller.check_budget(1)
    result = dict(latency_seconds=seconds, seed=seed, batch_size=len(captions), requested_frames=frames,
        actual_returned_frames=int(motion.shape[1]), output_shape=list(motion.shape),
        samples_per_second=len(captions) / seconds, peak_allocated_bytes=allocated,
        peak_reserved_bytes=reserved, **budget)
    del motion, lengths
    return result


def write_summaries(out, rows, manifest):
    summaries = []
    for batch in manifest['inputs']['batch_sizes']:
        for frames in manifest['inputs']['lengths']:
            selected = [row for row in rows if row['batch_size'] == batch and row['requested_frames'] == frames]
            if not selected:
                continue
            latency = np.array([row['latency_seconds'] for row in selected])
            median = float(np.median(latency))
            summary = dict(dataset=manifest['dataset'], sampler=manifest['sampler'],
                checkpoint_sha256=manifest['checkpoint_sha256'], batch_size=batch, requested_frames=frames,
                actual_returned_frames=selected[0]['actual_returned_frames'], timed_calls=len(selected),
                warmup_calls=manifest['warmups'], median_seconds=median,
                p95_seconds=float(np.percentile(latency, 95)), samples_per_second=batch / median,
                peak_allocated_bytes=max(row['peak_allocated_bytes'] for row in selected),
                peak_reserved_bytes=max(row['peak_reserved_bytes'] for row in selected),
                mar_calls=selected[0]['mar_calls'], mar_token_forwards=selected[0]['mar_token_forwards'])
            for label, key in (('head_nfe', 'net_calls'), ('head_token_forwards', 'net_token_forwards')):
                values = [int(row[key]) for row in selected]
                # Keep the legacy scalar only when every timed draw agrees.
                summary[label] = values[0] if min(values) == max(values) else None
                summary.update({label + '_values': values, label + '_min': min(values),
                    label + '_max': max(values), label + '_mean': float(np.mean(values)),
                    label + '_median': float(np.median(values)),
                    label + '_p95': float(np.percentile(values, 95))})
            summaries.append(summary)
    summary_name = 'smoke_summary.json' if manifest['smoke_only'] else 'summary.json'
    write_json(out / summary_name, dict(configurations=summaries, timing_boundary=manifest['timing_boundary'],
        smoke_only=manifest['smoke_only'], formal_benchmark=not manifest['smoke_only'],
        shared_gpu_limit='Other loads observed, not controlled; no exclusive hardware claim',
        nfe_definition='actual head forward invocations per timed generation; values preserve timed draw order',
        nfe_scalar_definition='legacy scalar is null when timed draws differ; use the recorded distribution',
        percentile_definition='numpy percentile95 linear interpolation',
        throughput_definition='batch size / median end-to-end seconds'))
    if summaries and not manifest['smoke_only']:
        with (out / 'latency_summary.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
            writer.writeheader()
            for summary in summaries:
                writer.writerow({key: json.dumps(value) if isinstance(value, list) else value
                                 for key, value in summary.items()})


def require_fixed_alternate(controller, alternate):
    if alternate is not None and controller.config.get('adaptive_nfe') is not False:
        raise ValueError('Alternate checkpoint timing reuse requires a verified fixed-step sampler; '
                         'benchmark each adaptive SiT checkpoint separately')


def main(args):
    cfg_path, out = args.config.resolve(), args.output.resolve()
    checkpoint = args.checkpoint.resolve() if args.checkpoint else None
    alternate = args.alternate_checkpoint.resolve() if args.alternate_checkpoint else None
    if args.smoke and alternate:
        raise ValueError('Smoke validates one sampler path; alternate weight mapping belongs to formal timing')
    config, config_info = read_config(cfg_path)
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock_stream = (out / 'run.lock').open('a+')
    fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    controller = None
    try:
        workers = own_gpu_workers(config['paths']['code_root'])
        if workers:
            write_json(out / 'BLOCKED_OWN_GPU_WORKERS.json', workers)
            raise RuntimeError('Parent queue must isolate its own training/evaluation workers before timing')
        precision = configure_precision(config)
        stats, stat_assets = load_stats(config)
        dataset, data_audit = build_dataset(config, stats, 'val')
        inputs = benchmark_inputs(config, dataset)
        if args.smoke:
            inputs.update(batch_sizes=[1], lengths=[60], entries=inputs['entries'][:1],
                          caption_selection='first validation entry with fixed caption RNG; smoke only')
        model, ae, payload, assets, controller = load_models(config, checkpoint, args.sampler)
        require_fixed_alternate(controller, alternate)
        signature = architecture_signature(model, ae)
        warmups, iterations = ((1, 2) if args.smoke else
            (int(config.get('benchmark', {}).get('warmups', 3)), int(config.get('benchmark', {}).get('iterations', 20))))
        if not args.smoke and (warmups < 3 or iterations < 20):
            raise ValueError('Formal benchmark requires at least3 warmups and20 timed iterations')
        manifest = dict(schema=2, created_utc=utc_now(), **config_info, dataset=config['dataset'],
            smoke_only=args.smoke, mode='smoke' if args.smoke else 'benchmark',
            sampler=controller.sampler, checkpoint_sha256=assets['checkpoint']['sha256'],
            model_assets=assets, stats=stat_assets, inputs=inputs, inputs_sha256=json_sha(inputs),
            data_audit_sha256=json_sha(data_audit), precision=precision, architecture=signature,
            sampler_config=controller.config, warmups=warmups, iterations=iterations,
            source_sha256=source_manifest(config),
            timing_boundary=dict(start='immediately before model.generate(raw text, lengths)',
                stop='after ae.decode_from_fsq and torch.cuda.synchronize',
                included=['text tokenization', 'CLIP text encoding', 'outer MAR', 'inner generation head', 'final grid alignment', 'AE decoding'],
                excluded=['model/checkpoint/data loading', 'length tensor allocation', 'input validation', 'metric embeddings', 'disk writing', 'rendering', 'denormalization'],
                output='train-normalized motion features [B,T,67|64]', common_text_cache='none', cuda_sync_before_and_after=True),
            alternate_checkpoint=None if alternate is None else dict(path=str(alternate), sha256=sha_file(alternate)),
            own_workers_at_start=workers)
        manifest['identity_sha256'] = json_sha({k: v for k, v in manifest.items() if k != 'created_utc'})
        if (out / 'manifest.json').exists():
            previous = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
            if previous['identity_sha256'] != manifest['identity_sha256']:
                raise ValueError('Benchmark identity differs; choose new output directory')
            manifest = previous
            if not args.resume and not (out / 'COMPLETE').exists():
                raise FileExistsError('Use --resume for an interrupted benchmark')
        else:
            write_json(out / 'manifest.json', manifest)
            write_json(out / 'benchmark_inputs.json', inputs)
            write_json(out / 'device_start.json', device_snapshot())
        if (out / 'COMPLETE').exists():
            complete = json.loads((out / 'COMPLETE').read_text())
            if (complete['smoke_only'] != args.smoke
                    or complete['latency_raw_sha256'] != sha_file(out / ('smoke_latency_raw.jsonl' if args.smoke else 'latency_raw.jsonl'))
                    or complete['summary_sha256'] != sha_file(out / ('smoke_summary.json' if args.smoke else 'summary.json'))
                    or complete['manifest_sha256'] != sha_file(out / 'manifest.json')
                    or (not args.smoke and complete['csv_sha256'] != sha_file(out / 'latency_summary.csv'))
                    or (alternate and complete['alternate_check_sha256'] != sha_file(out / 'alternate_check.json'))):
                raise RuntimeError('Completed benchmark integrity failed')
            print(json.dumps(dict(state='already_complete', output=str(out))))
            return
        rows = []
        for batch in inputs['batch_sizes']:
            captions = [item['caption'] for item in inputs['entries'][:batch]]
            for frames in inputs['lengths']:
                if own_gpu_workers(config['paths']['code_root']):
                    raise RuntimeError('Own GPU workers appeared during benchmark')
                configuration = out / 'configurations' / f'b{batch}_f{frames}'
                configuration.mkdir(parents=True, exist_ok=True)
                write_json(configuration / 'device_before.json', device_snapshot())
                if (configuration / 'COMPLETE').exists():
                    complete = json.loads((configuration / 'COMPLETE').read_text())
                    if complete['raw_sha256'] != sha_file(configuration / 'raw.jsonl'):
                        raise RuntimeError('Benchmark configuration hash differs')
                    selected = [json.loads(line) for line in (configuration / 'raw.jsonl').read_text().splitlines()]
                    if len(selected) != iterations:
                        raise ValueError('Completed benchmark configuration has wrong repetition count')
                    rows.extend(selected)
                    continue
                for old_name in ('raw.jsonl', 'warmups.jsonl'):
                    old = configuration / old_name
                    if old.exists():
                        old.replace(configuration / (old.name + '.partial.' + str(time.time_ns())))
                for index in range(warmups):
                    seed = inputs['seed'] + 100000 + frames * 100 + batch * 1000 + index
                    warm = timed_call(config, model, ae, controller, captions, frames, seed)
                    warm.update(warmup=True, repetition=index)
                    append_jsonl(configuration / 'warmups.jsonl', warm)
                for index in range(iterations):
                    seed = inputs['seed'] + frames * 100 + batch * 1000 + index
                    row = timed_call(config, model, ae, controller, captions, frames, seed)
                    row.update(dataset=config['dataset'], sampler=controller.sampler,
                        checkpoint_sha256=manifest['checkpoint_sha256'], repetition=index,
                        smoke_only=args.smoke, formal_benchmark=not args.smoke,
                        inputs_sha256=inputs['entries'][:batch] and json_sha(captions),
                        manifest_identity_sha256=manifest['identity_sha256'])
                    append_jsonl(configuration / 'raw.jsonl', row)
                    rows.append(row)
                    write_json(out / 'progress.json', dict(utc=utc_now(), state='running',
                        batch_size=batch, frames=frames, timed_iteration=index + 1,
                        completed_timed_calls=len(rows), total_timed_calls=len(inputs['batch_sizes']) * len(inputs['lengths']) * iterations))
                write_json(configuration / 'device_after.json', device_snapshot())
                write_json(configuration / 'COMPLETE', dict(raw_sha256=sha_file(configuration / 'raw.jsonl'),
                    warmups_sha256=sha_file(configuration / 'warmups.jsonl')))
                write_summaries(out, rows, manifest)
                print(json.dumps(dict(batch_size=batch, frames=frames, completed_timed_calls=len(rows))), flush=True)
        raw_text = ''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in rows)
        raw_name = 'smoke_latency_raw.jsonl' if args.smoke else 'latency_raw.jsonl'
        summary_name = 'smoke_summary.json' if args.smoke else 'summary.json'
        (out / raw_name).write_text(raw_text, encoding='utf-8')
        write_summaries(out, rows, manifest)
        if alternate:
            sampler = controller.sampler
            controller.remove()
            controller = None
            del model, ae, payload
            gc.collect()
            torch.cuda.empty_cache()
            alt_model, alt_ae, alt_payload, alt_assets, controller = load_models(config, alternate, sampler)
            require_fixed_alternate(controller, alternate)
            if architecture_signature(alt_model, alt_ae) != signature:
                raise ValueError('Alternate checkpoint architecture differs; needs separate benchmark')
            smoke = []
            for batch, frames in ((1, 60), (32, 196)):
                captions = [item['caption'] for item in inputs['entries'][:batch]]
                row = timed_call(config, alt_model, alt_ae, controller, captions, frames, inputs['seed'])
                # One call checks behavior only, never reported as a speed estimate.
                row.pop('latency_seconds')
                row.pop('samples_per_second')
                row['timing_claim'] = False
                smoke.append(row)
            write_json(out / 'alternate_check.json', dict(PASS=True, primary_checkpoint_sha256=manifest['checkpoint_sha256'],
                alternate_checkpoint_sha256=alt_assets['checkpoint']['sha256'], architecture=signature,
                sampler=sampler, representative_batch_checks=smoke,
                mapping='fixed-step same-architecture weights share the structural timing; generated values need not be identical'))
        write_json(out / 'device_end.json', device_snapshot())
        write_json(out / 'COMPLETE', dict(utc=utc_now(), checkpoint_sha256=manifest['checkpoint_sha256'],
            mode='smoke' if args.smoke else 'benchmark', smoke_only=args.smoke,
            raw_file=raw_name, summary_file=summary_name,
            latency_raw_sha256=sha_file(out / raw_name), summary_sha256=sha_file(out / summary_name),
            csv_sha256=None if args.smoke else sha_file(out / 'latency_summary.csv'),
            manifest_sha256=sha_file(out / 'manifest.json'),
            alternate_check_sha256=sha_file(out / 'alternate_check.json') if alternate else None))
        write_json(out / 'progress.json', dict(utc=utc_now(), state='complete', timed_calls=len(rows)))
    except BaseException as exc:
        write_json(out / ('FAILED_' + str(time.time_ns()) + '.json'), dict(utc=utc_now(),
            exception=repr(exc), traceback=traceback.format_exc()))
        raise
    finally:
        if controller:
            controller.remove()
        lock_stream.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--alternate-checkpoint', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--sampler', choices=['native_cfg', 'direct_apg', 'distilled', 'cfg', 'apg'])
    p.add_argument('--resume', action='store_true')
    p.add_argument('--smoke', action='store_true', help='Separate real-data B1/60-frame path check:1 warmup+2 timings, not a formal benchmark')
    main(p.parse_args())
