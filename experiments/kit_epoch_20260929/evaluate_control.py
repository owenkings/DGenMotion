"""Paired validation/test and separate multimodality/reconstruction evaluation.

Examples (all GPU work must be queued by the parent scheduler):
  python evaluate_control.py --config H_CFG_joint.json --checkpoint ema_005000.pt --output val5000
  python evaluate_control.py --config ref.json --sampler native_cfg --split test \
      --selection-lock selection_lock.json --output test_cfg --seeds 10000..10019
  python evaluate_control.py --config kit.json --checkpoint kit_ae.pt --output ae_eval --mode reconstruction
  python evaluate_control.py --config kit.json --checkpoint kit_base.pt --output smoke --mode smoke --smoke-batches 2

Test selection is external and locked before any test result is generated.
Twenty generation seeds are not twenty independent training runs.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from eval_common import (append_jsonl, batch_fingerprint, build_dataset, build_evaluators,
    configure_precision, device_snapshot, finite_embeddings, generate_motion, json_sha,
    load_models, load_stats, lock_test_candidate, make_loader, normalize_generated,
    read_config, rng_fingerprint, seed_sampling, sha_file, source_manifest, utc_now, write_json)


METRICS = ('fid', 'r1', 'r2', 'r3', 'matching', 'diversity', 'clip_score', 'mm',
           'reconstruction_l1', 'reconstruction_smooth_l1', 'mpjpe')


def parse_seeds(value):
    if '..' in value:
        start, end = map(int, value.split('..'))
        values = list(range(start, end + 1))
    else:
        values = [int(x) for x in value.split(',')]
    if not values or len(values) != len(set(values)) or any(x < 0 for x in values):
        raise ValueError('Expected unique nonnegative seeds')
    return values


def summarize(rows, manifest):
    result = dict(n=len(rows), dataset=manifest['dataset'], split=manifest['split'],
                  mode=manifest['mode'], sampler=manifest['sampler'], step=manifest['step'],
                  seeds=[r['seed'] for r in rows],
                  checkpoint_sha256=manifest['checkpoint_sha256'],
                  training_replicates=1, seed_variation='generation and paired evaluation data only')
    for metric in METRICS:
        values = [r[metric] for r in rows if r.get(metric) is not None]
        result[metric] = (dict(mean=float(np.mean(values)),
                               sample_sd=float(np.std(values, ddof=1)) if len(values) > 1 else None)
                          if len(values) == len(rows) and values else None)
    result['evaluated_samples_per_seed'] = [r.get('evaluated_samples', 0) for r in rows]
    result['elapsed_seconds'] = sum(r['elapsed_seconds'] for r in rows)
    if manifest['mode'] == 'smoke':
        result.update(smoke_only=True, formal_quality_metrics=False,
                      embedding_shapes=rows[0].get('embedding_shapes'),
                      normalization_roundtrip_max_abs=rows[0].get('normalization_roundtrip_max_abs'))
    return result


def write_aggregate(out, rows, manifest):
    summary = summarize(rows, manifest)
    # Both names are intentional: metrics.jsonl remains compatible with trainer
    # selection, while metrics_raw.jsonl is the required paper-delivery artifact.
    contents = ''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in rows)
    for filename in ('metrics.jsonl', 'metrics_raw.jsonl'):
        path = out / filename
        temp = path.with_name(path.name + '.tmp')
        temp.write_text(contents, encoding='utf-8')
        os.replace(temp, path)
    write_json(out / 'summary.json', summary)
    fields = ['dataset', 'split', 'mode', 'sampler', 'step', 'checkpoint_sha256', 'n']
    for name in METRICS:
        fields.extend([name + '_mean', name + '_sample_sd'])
    flat = {key: summary[key] for key in fields if key in summary}
    for name in METRICS:
        value = summary[name]
        flat[name + '_mean'] = '' if value is None else value['mean']
        flat[name + '_sample_sd'] = '' if value is None or value['sample_sd'] is None else value['sample_sd']
    with (out / 'quality_summary.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow(flat)
    return summary


@torch.no_grad()
def evaluate_seed(config, model, ae, wrapper, dataset, stats, controller, seed, mode,
                  seed_dir, smoke_batches=2):
    from utils.eval_utils import (calculate_R_precision, euclidean_distance_matrix,
        calculate_activation_statistics, calculate_frechet_distance, calculate_diversity,
        calculate_multimodality, calculate_mpjpe)
    from utils.motion_process import recover_from_ric
    seed_sampling(seed)
    loader = make_loader(config, dataset, seed)
    if mode == 'multimodality' and len(loader) < 3:
        raise ValueError('Standard multimodality needs three complete batches of 32 prompts')
    if len(loader) == 0:
        raise ValueError('No complete standard retrieval batch')
    if controller:
        controller.reset()
    before = rng_fingerprint()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.monotonic()
    predicted_embeddings, real_embeddings, multimodal_embeddings = [], [], []
    r_total, r_real = np.zeros(3), np.zeros(3)
    matching, matching_real, clip_total, clip_real = 0.0, 0.0, 0.0, 0.0
    l1_sum, smooth_sum, element_count, mpjpe_sum, pose_count = 0.0, 0.0, 0, 0.0, 0
    samples, generate_calls, fingerprints, input_records = 0, 0, [], []
    embedding_shapes = None
    source_ids, captions_seen = set(), set()
    roundtrip_max = 0.0
    inputs_path = seed_dir / 'inputs.jsonl'
    if inputs_path.exists():
        inputs_path.replace(seed_dir / ('inputs.partial.' + str(time.time_ns()) + '.jsonl'))
    for batch_index, (batch, identifiers) in enumerate(loader):
        if mode == 'smoke' and batch_index >= smoke_batches:
            break
        if mode == 'multimodality' and batch_index >= 3:
            break
        words, pos, captions, sent_len, motion, lengths, tokens = batch
        if motion.shape[-1] != stats['train_mean'].shape[0] or len(captions) != 32:
            raise ValueError('Motion dimensions or retrieval batch changed')
        if not torch.isfinite(motion).all() or (lengths % 4).any() or (lengths < 4).any():
            raise FloatingPointError('Invalid real batch')
        lengths = lengths.to('cuda')
        observed = batch_fingerprint(batch)
        fingerprints.append(observed)
        source_ids.update(i['entry'] if i['entry'][0] not in 'ABCDEFGHIJKLMNOPQRSTUVW' or '_' not in i['entry']
                          else i['entry'].split('_', 1)[1] for i in identifiers)
        captions_seen.update(captions)
        batch_before = rng_fingerprint()
        real = wrapper.get_co_embeddings(words, pos, sent_len, captions, motion.to('cuda').float(), lengths)
        finite_embeddings(real)
        (et, em), (et_clip, em_clip) = real
        repetitions = 30 if mode == 'multimodality' else 1
        batch_mm = []
        for draw in range(repetitions):
            if mode == 'reconstruction':
                raw_real = motion.numpy() * stats['eval_std'] + stats['eval_mean']
                train_real = (raw_real - stats['train_mean']) / stats['train_std']
                pred = ae(torch.from_numpy(train_real).to('cuda'))
            else:
                pred = generate_motion(config, model, ae, captions, lengths)
                generate_calls += 1
            if pred.ndim != 3 or pred.shape[0] != len(captions) or pred.shape[-1] != motion.shape[-1]:
                raise ValueError('Invalid decoded shape: ' + str(tuple(pred.shape)))
            if pred.shape[1] < int(lengths.max()) or not bool(torch.isfinite(pred).all()):
                raise FloatingPointError('Nonfinite or short decoded motion')
            pred_eval, raw_prediction = normalize_generated(pred, stats)
            generated = wrapper.get_co_embeddings(words, pos, sent_len, captions, pred_eval, lengths)
            finite_embeddings(generated)
            (et_pred, em_pred), (et_pred_clip, em_pred_clip) = generated
            embedding_shapes = dict(real=[list(v.shape) for pair in real for v in pair],
                                    generated=[list(v.shape) for pair in generated for v in pair],
                                    decoded=list(pred.shape), raw_motion=list(motion.shape))
            if mode == 'multimodality':
                batch_mm.append(em_pred.cpu().unsqueeze(1))
        if mode == 'multimodality':
            multimodal_embeddings.append(torch.cat(batch_mm, dim=1))
        else:
            predicted_embeddings.append(em_pred.cpu())
            real_embeddings.append(em.cpu())
            r_total += calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), 3, sum_all=True)
            r_real += calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), 3, sum_all=True)
            matching += float(euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace())
            matching_real += float(euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace())
            clip_total += float((et_pred_clip * em_pred_clip).sum(dim=1).sum())
            clip_real += float((et_clip * em_clip).sum(dim=1).sum())
        if mode == 'reconstruction':
            prediction_numpy = pred.detach().cpu().numpy()
            for index, length in enumerate(lengths.tolist()):
                difference = prediction_numpy[index, :length] - train_real[index, :length]
                absolute = np.abs(difference)
                l1_sum += float(absolute.sum())
                smooth_sum += float(np.where(absolute < 1, .5 * absolute ** 2, absolute - .5).sum())
                element_count += absolute.size
                joints = 22 if config['dataset'] == 't2m' else 21
                gt_xyz = recover_from_ric(torch.from_numpy(raw_real[index, :length]).float(), joints)
                pred_xyz = recover_from_ric(torch.from_numpy(raw_prediction[index, :length]).float(), joints)
                mpjpe_sum += float(calculate_mpjpe(gt_xyz, pred_xyz).sum())
                pose_count += length
        if mode == 'smoke':
            # Exercise the AE encoding/reconstruction and both normalization
            # directions on the same genuine dataset batch as generation.
            raw_real = motion.numpy() * stats['eval_std'] + stats['eval_mean']
            train_real = (raw_real - stats['train_mean']) / stats['train_std']
            latent, coords = ae.encode_with_fsq(torch.from_numpy(train_real).to('cuda'))
            reconstructed = ae.decode_from_fsq(coords)
            reconstructed_eval, _ = normalize_generated(reconstructed, stats)
            reconstruction_emb = wrapper.get_co_embeddings(words, pos, sent_len, captions, reconstructed_eval, lengths)
            finite_embeddings(reconstruction_emb)
            if not all(bool(torch.isfinite(v).all()) for v in (latent, coords, reconstructed)):
                raise FloatingPointError('Nonfinite smoke encode/reconstruction')
            embedding_shapes.update(latent=list(latent.shape), coordinates=list(coords.shape),
                reconstruction=list(reconstructed.shape),
                reconstruction_embeddings=[list(v.shape) for pair in reconstruction_emb for v in pair])
            normalized_again = (train_real * stats['train_std'] + stats['train_mean'] - stats['eval_mean']) / stats['eval_std']
            roundtrip_max = max(roundtrip_max, float(np.abs(normalized_again - motion.numpy()).max()))
        samples += len(captions)
        row = dict(batch=batch_index, seed=seed, inputs_sha256=observed, entries=identifiers,
                   rng_before=batch_before, rng_after=rng_fingerprint(), generated_draws=repetitions,
                   returned_max_frames=int(pred.shape[1]), effective_lengths=lengths.tolist())
        append_jsonl(inputs_path, row)
        input_records.append(row)
        write_json(seed_dir / 'progress.json', dict(utc=utc_now(), seed=seed, mode=mode,
            completed_batches=batch_index + 1, total_batches=min(len(loader), 3) if mode == 'multimodality' else len(loader),
            generated_sequences=generate_calls * 32, elapsed_seconds=time.monotonic() - started))
    if not samples:
        raise ValueError('Empty evaluation')
    metrics = {name: None for name in METRICS}
    if mode == 'multimodality':
        activations = torch.cat(multimodal_embeddings).numpy()
        metrics['mm'] = float(calculate_multimodality(activations, 10))
    elif mode != 'smoke':
        pred_np = torch.cat(predicted_embeddings).numpy()
        real_np = torch.cat(real_embeddings).numpy()
        diversity_pairs = 300 if samples > 300 else 100
        if samples <= diversity_pairs:
            raise ValueError('Too few samples for repository standard diversity; use --mode smoke')
        mean_gt, cov_gt = calculate_activation_statistics(real_np)
        mean_pred, cov_pred = calculate_activation_statistics(pred_np)
        # Keep repository ordering: real diversity draws precede generated ones.
        diversity_real = float(calculate_diversity(real_np, diversity_pairs))
        metrics.update(fid=float(calculate_frechet_distance(mean_gt, cov_gt, mean_pred, cov_pred)),
            diversity=float(calculate_diversity(pred_np, diversity_pairs)),
            r1=float(r_total[0] / samples), r2=float(r_total[1] / samples), r3=float(r_total[2] / samples),
            matching=matching / samples, clip_score=clip_total / samples)
        if mode == 'reconstruction':
            metrics.update(reconstruction_l1=l1_sum / element_count,
                reconstruction_smooth_l1=smooth_sum / element_count, mpjpe=mpjpe_sum / pose_count)
    if any(value is not None and not np.isfinite(value) for value in metrics.values()):
        raise FloatingPointError('Nonfinite final metric')
    budget = controller.check_budget(generate_calls) if controller else dict(mar_calls=0, net_calls=0, net_token_forwards=0)
    torch.cuda.synchronize()
    row = dict(metrics, seed=seed, mode=mode, evaluated_samples=samples,
        evaluated_batches=len(fingerprints), evaluated_unique_source_motions=len(source_ids),
        evaluated_unique_captions=len(captions_seen), generated_sequences=generate_calls * 32,
        input_batches_sha256=fingerprints, inputs_file_sha256=sha_file(inputs_path),
        input_manifest_sha256=json_sha(input_records), rng_before=before, rng_after=rng_fingerprint(),
        elapsed_seconds=time.monotonic() - started, **budget,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        drop_last=True, shuffle=True, batch_size=32,
        caption_length_source='paired repository Text2MotionDataset caption/crop from this split',
        standard_retrieval_candidate_pool=32, sample_std_interpretation='generation replicates, not training replicates')
    if mode == 'multimodality':
        row.update(mm_protocol=dict(prompt_batches=3, prompts=96, generations_per_prompt=30,
                                    comparison_pairs=10, metric='repository calculate_multimodality'),
                   evaluated_quality_metrics=False)
    elif mode == 'smoke':
        row.update(smoke_only=True, embedding_shapes=embedding_shapes,
                   normalization_roundtrip_max_abs=roundtrip_max)
    else:
        row['real_reference'] = dict(diversity=diversity_real, r1=float(r_real[0] / samples),
            r2=float(r_real[1] / samples), r3=float(r_real[2] / samples),
            matching=matching_real / samples, clip_score=clip_real / samples)
    return row


def main(args):
    # Resolve CLI paths before read_config changes cwd to the isolated code copy.
    config_path, out = args.config.resolve(), args.output.resolve()
    checkpoint = args.checkpoint.resolve() if args.checkpoint else None
    lock_path = args.selection_lock.resolve() if args.selection_lock else None
    config, config_info = read_config(config_path)
    split = args.split or config.get('evaluation', {}).get('split', 'val')
    if split not in ('val', 'test'):
        raise ValueError('Generation selection must use val; final evaluation uses test')
    if args.mode == 'smoke' and (split != 'val' or not 1 <= args.smoke_batches <= 2):
        raise ValueError('Smoke is val-only with 1 or 2 genuine batches')
    if args.seeds and args.seed is not None:
        raise ValueError('Choose --seed or --seeds')
    seeds = parse_seeds(args.seeds) if args.seeds else ([args.seed] if args.seed is not None else
            list(range(10000, 10020)) if split == 'test' and args.mode == 'quality'
            else [int(config.get('evaluation', {}).get('seed', 3407))])
    if split == 'test' and args.mode == 'quality' and seeds != list(range(10000, 10020)):
        raise ValueError('Formal test quality uses the predeclared 20 seeds 10000..10019')
    out.mkdir(parents=True, exist_ok=True)
    # Kernel-owned lock is released even after a crash; the readable lock file is
    # retained as evidence and is safe to reopen during exact continuation.
    import fcntl
    lock_stream = (out / 'run.lock').open('a+')
    fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    controller = None
    try:
        precision = configure_precision(config)
        stats, stat_assets = load_stats(config)
        model, ae, payload, model_assets, controller = load_models(config, checkpoint, args.sampler,
                                                                  args.mode == 'reconstruction', args.base_training)
        actual_sampler = 'reconstruction' if controller is None else controller.sampler
        checkpoint_sha = model_assets['ae_checkpoint' if controller is None else 'checkpoint']['sha256']
        selection = None
        if split == 'test':
            lock_path = lock_path or config.get('evaluation', {}).get('selection_lock')
            selection = lock_test_candidate(lock_path, config['dataset'], checkpoint_sha, actual_sampler,
                                            model_assets['ae_checkpoint']['sha256'])
        dataset, data_audit = build_dataset(config, stats, split)
        wrapper, evaluator_assets = build_evaluators(config, torch.device('cuda'))
        manifest = dict(schema=1, created_utc=utc_now(), **config_info, dataset=config['dataset'],
            split=split, mode=args.mode, sampler=actual_sampler, seeds=seeds,
            step=payload.get('steps', payload.get('step', payload.get('total_it', 0))),
            checkpoint_sha256=checkpoint_sha, model_assets=model_assets,
            stats=stat_assets, evaluators=evaluator_assets, selection_lock=selection,
            source_sha256=source_manifest(config), sampling=config['sampling'], precision=precision,
            sampler_config=None if controller is None else controller.config,
            evaluation=dict(batch_size=32, num_workers=config.get('evaluation', {}).get('num_workers', 4),
                drop_last=True, shuffle=True, unit_length=4, max_motion_length=196,
                max_text_length=20, hard_pseudo_reorder=False,
                data_rng='per-seed/per-entry Python and NumPy; independent DataLoader generator',
                sampling_rng='reset torch CPU/CUDA to each declared seed, independent of loader',
                historical_random_stream_identical=False),
            data_audit_sha256=json_sha(data_audit), smoke_batches=args.smoke_batches if args.mode == 'smoke' else None)
        identity = {key: value for key, value in manifest.items() if key != 'created_utc'}
        manifest['identity_sha256'] = json_sha(identity)
        existing = out / 'manifest.json'
        if existing.exists():
            previous = json.loads(existing.read_text(encoding='utf-8'))
            if previous['identity_sha256'] != manifest['identity_sha256']:
                raise ValueError('Existing evaluation identity differs; use a new output directory')
            if not args.resume and not (out / 'COMPLETE').exists():
                raise FileExistsError('Incomplete evaluation exists; use --resume after inspecting its failure')
            manifest = previous
        else:
            write_json(existing, manifest)
            write_json(out / 'data_audit.json', data_audit)
            write_json(out / 'device_start.json', device_snapshot())
        if (out / 'COMPLETE').exists():
            complete = json.loads((out / 'COMPLETE').read_text())
            if (complete['checkpoint_sha256'] != checkpoint_sha
                    or complete['summary_sha256'] != sha_file(out / 'summary.json')
                    or complete['metrics_sha256'] != sha_file(out / 'metrics.jsonl')
                    or complete['raw_sha256'] != sha_file(out / 'metrics_raw.jsonl')
                    or complete['manifest_sha256'] != sha_file(out / 'manifest.json')):
                raise RuntimeError('Completed evaluation artifacts are damaged')
            print(json.dumps(dict(state='already_complete', output=str(out))), flush=True)
            return
        rows = []
        for seed in seeds:
            seed_dir = out / 'seeds' / str(seed)
            seed_dir.mkdir(parents=True, exist_ok=True)
            completion = seed_dir / 'COMPLETE'
            if completion.exists():
                complete = json.loads(completion.read_text())
                if complete['metrics_sha256'] != sha_file(seed_dir / 'metrics.json'):
                    raise RuntimeError('Per-seed result hash mismatch')
                row = json.loads((seed_dir / 'metrics.json').read_text())
                if row['seed'] != seed or row['checkpoint_sha256'] != checkpoint_sha:
                    raise RuntimeError('Per-seed result identity mismatch')
                if row['inputs_file_sha256'] != sha_file(seed_dir / 'inputs.jsonl'):
                    raise RuntimeError('Per-seed input/caption manifest hash mismatch')
            else:
                row = evaluate_seed(config, model, ae, wrapper, dataset, stats, controller,
                                    seed, args.mode, seed_dir, args.smoke_batches)
                row.update(dataset=config['dataset'], split=split, sampler=actual_sampler,
                           step=manifest['step'], checkpoint_sha256=checkpoint_sha,
                           ae_checkpoint_sha256=model_assets['ae_checkpoint']['sha256'],
                           evaluator_sha256=evaluator_assets['evaluator']['sha256'],
                           evaluator_clip_sha256=evaluator_assets['evaluator_clip']['sha256'],
                           manifest_identity_sha256=manifest['identity_sha256'])
                write_json(seed_dir / 'metrics.json', row)
                write_json(completion, dict(seed=seed, metrics_sha256=sha_file(seed_dir / 'metrics.json')))
            rows.append(row)
            write_aggregate(out, rows, manifest)
            write_json(out / 'progress.json', dict(utc=utc_now(), completed_seeds=len(rows),
                total_seeds=len(seeds), last_seed=seed, state='running'))
            print(json.dumps(dict(seed=seed, mode=args.mode, fid=row.get('fid'),
                r1=row.get('r1'), mm=row.get('mm'), elapsed_seconds=row['elapsed_seconds'])), flush=True)
        summary = write_aggregate(out, rows, manifest)
        write_json(out / 'device_end.json', device_snapshot())
        write_json(out / 'COMPLETE', dict(utc=utc_now(), mode=args.mode, seeds=seeds,
            smoke_only=args.mode == 'smoke', checkpoint_sha256=checkpoint_sha,
            summary_sha256=sha_file(out / 'summary.json'), metrics_sha256=sha_file(out / 'metrics.jsonl'),
            manifest_sha256=sha_file(out / 'manifest.json'), raw_sha256=sha_file(out / 'metrics_raw.jsonl')))
        write_json(out / 'progress.json', dict(utc=utc_now(), completed_seeds=len(rows),
            total_seeds=len(seeds), state='complete'))
        print(json.dumps(summary), flush=True)
    except BaseException as exc:
        write_json(out / ('FAILED_' + str(time.time_ns()) + '.json'), dict(utc=utc_now(),
            exception=repr(exc), traceback=traceback.format_exc(), mode=args.mode, split=split))
        raise
    finally:
        if controller:
            controller.remove()
        lock_stream.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--output', '--out', type=Path, required=True)
    p.add_argument('--split', choices=['val', 'test'])
    p.add_argument('--seed', type=int)
    p.add_argument('--seeds')
    p.add_argument('--sampler', choices=['native_cfg', 'direct_apg', 'distilled', 'cfg', 'apg'])
    p.add_argument('--mode', choices=['quality', 'multimodality', 'reconstruction', 'smoke'], default='quality')
    p.add_argument('--smoke-batches', type=int, default=2)
    p.add_argument('--selection-lock', type=Path)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--base-training', action='store_true', help='KIT foundation validation before a final base is selected')
    main(p.parse_args())
