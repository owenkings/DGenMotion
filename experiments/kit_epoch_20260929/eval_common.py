"""Explicit assets, paired data, and audited samplers for paper controls.

No network downloads, checkpoint selection, or training occurs in this module.
The repository's metric functions and evaluator architectures are retained.
"""
import contextlib
import copy
import datetime
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import types

import numpy as np
import torch


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def metadata_scalar(value):
    """JSON numbers must be native scalars; never stringify unknown objects."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError('Unsupported JSON metadata type: ' + type(value).__name__)


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, default=metadata_scalar,
                                    allow_nan=False).encode('utf-8')).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False, default=metadata_scalar)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def append_jsonl(path, value):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False, default=metadata_scalar) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def read_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding='utf-8-sig'))
    if config['dataset'] not in ('t2m', 'kit'):
        raise ValueError('dataset must be t2m or kit')
    width = 67 if config['dataset'] == 't2m' else 64
    if config.get('model', {}).get('input_width', width) != width:
        raise ValueError('Dataset/input_width mismatch')
    sampling = config.setdefault('sampling', {})
    sampling.setdefault('outer_steps', 18)
    sampling.setdefault('inner_steps', 50)
    sampling.setdefault('cfg', 4.5 if width == 67 else 2.5)
    sampling.setdefault('temperature', sampling.get('temp', 1.0))
    if sampling['outer_steps'] != 18 or sampling['inner_steps'] != 50:
        raise ValueError('This preregistration requires outer18/inner50')
    if float(sampling['cfg']) == 1 or not math.isfinite(float(sampling['cfg'])):
        raise ValueError('Legacy outer MAR requires paired cfg != 1')
    if float(sampling['cfg']) != (4.5 if width == 67 else 2.5):
        raise ValueError('Locked HumanML/KIT CFG must be4.5/2.5')
    if sampling.get('apg', dict(beta=-.5, eta=0., norm_threshold=0.)) != dict(beta=-.5, eta=0., norm_threshold=0.):
        raise ValueError('The preregistered APG parameters must not change')
    if float(sampling['temperature']) != 1:
        raise ValueError('Preregistered temperature is 1; legacy MAR also hardcodes head temperature1')
    paths = config['paths']
    for key, value in list(paths.items()):
        if value is not None:
            resolved = Path(value).expanduser()
            if not resolved.is_absolute():
                resolved = path.parent / resolved
            paths[key] = str(resolved.resolve())
    code = Path(paths['code_root'])
    if not (code / 'models/MARDM.py').is_file():
        raise FileNotFoundError(code / 'models/MARDM.py')
    sys.path.insert(0, str(code))
    os.chdir(code)
    return config, dict(config_file=str(path), config_file_sha256=sha_file(path),
                        effective_config_sha256=json_sha(config))


def verify_asset(config, key, override=None, require_locked=True):
    path = Path(override or config['paths'][key]).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Missing required {key}: {path}')
    actual = sha_file(path)
    expected = config.get('hashes', {}).get(key)
    if isinstance(expected, dict):
        expected = expected.get('sha256')
    # CLI-selected snapshots are selected/locked separately, never confused with
    # a different AE/base path in the training configuration.
    configured_path = config['paths'].get(key)
    same_config_path = not override or bool(configured_path and path == Path(configured_path).resolve())
    if same_config_path and expected and actual != expected:
        raise ValueError(f'{key} SHA-256 mismatch: {path}')
    if require_locked and same_config_path and not expected:
        raise ValueError(f'Lock hashes.{key} before running evaluation')
    return dict(path=str(path), sha256=actual, bytes=path.stat().st_size,
                configured_hash_verified=bool(same_config_path and expected))


def source_manifest(config):
    code = Path(config['paths']['code_root'])
    files = list((code / 'models').rglob('*.py'))
    files.extend((code / 'diffusions').rglob('*.py'))
    files.extend(code / x for x in ('utils/datasets.py', 'utils/evaluators.py', 'utils/eval_utils.py',
                                  'utils/glove.py', 'utils/motion_process.py'))
    files.extend(Path(__file__).parent / name for name in
                 ('eval_common.py', 'evaluate_control.py', 'benchmark_control.py', 'control_guidance.py'))
    return {str(p): sha_file(p) for p in files if p.is_file()}


def local_clip_only(config):
    """Resolve official CLIP cache without allowing implicit network downloads."""
    import importlib
    sys.path.insert(0, str(Path(config['paths']['code_root']) / 'CLIP'))
    clip = importlib.import_module('clip')
    namespace = clip.load.__globals__
    if '_download' not in namespace or '_MODELS' not in namespace:
        raise RuntimeError('Unrecognized CLIP loader; verify its offline asset path first')
    def cached_only(url, root):
        path = Path(root).expanduser() / url.rsplit('/', 1)[-1]
        if not path.is_file():
            raise FileNotFoundError('Missing official CLIP cache; obtain explicitly before retry: ' + str(path))
        if sha_file(path) != url.split('/')[-2]:
            raise ValueError('CLIP cache SHA-256 differs from official URL: ' + str(path))
        return str(path)
    namespace['_download'] = cached_only
    assets = {}
    for version in ('ViT-B/32', 'ViT-B/16'):
        url = namespace['_MODELS'][version]
        path = Path.home() / '.cache/clip' / url.rsplit('/', 1)[-1]
        cached_only(url, path.parent)
        assets[version] = dict(path=str(path), sha256=url.split('/')[-2])
    return assets


def seed_sampling(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@contextlib.contextmanager
def isolated_data_seed(seed):
    py, np_state = random.getstate(), np.random.get_state()
    try:
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        yield
    finally:
        random.setstate(py)
        np.random.set_state(np_state)


def rng_fingerprint():
    return dict(cpu=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                cuda=[hashlib.sha256(s.cpu().numpy().tobytes()).hexdigest()
                      for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [])


def configure_precision(config):
    precision = config.get('precision', config.get('training', {}).get('precision', 'fp32'))
    if isinstance(precision, dict):
        precision = precision.get('dtype', 'fp32')
    if precision not in ('fp32', 'float32', 'float32_tf32'):
        raise ValueError('Only verified fp32 generator/AE path is supported; CLIP retains repository dtype')
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if not torch.cuda.is_available():
        raise RuntimeError('Real generation requires CUDA; no silent CPU benchmark fallback')
    return dict(generator='float32', ae='float32', generator_clip='repository converted weights',
                evaluator_clip='repository MotionCLIP', allow_tf32=True, cudnn_benchmark=False)


def device_snapshot():
    result = dict(utc=utc_now(), executable=sys.executable, torch=torch.__version__,
                  cuda_version=torch.version.cuda, pid=os.getpid(),
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        free, total = torch.cuda.mem_get_info()
        result.update(device_name=props.name, device_total_bytes=props.total_memory,
                      free_bytes=free, total_bytes=total, device_count=torch.cuda.device_count())
    for key, command in [('nvidia_smi', ['nvidia-smi', '-L']),
                         ('gpu_load', ['nvidia-smi', '--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total', '--format=csv']),
                         ('gpu_processes', ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv'])]:
        try:
            r = subprocess.run(command, capture_output=True, text=True, timeout=10)
            result[key] = dict(returncode=r.returncode, stdout=r.stdout[-12000:], stderr=r.stderr[-3000:])
        except (OSError, subprocess.TimeoutExpired) as exc:
            result[key] = dict(unavailable=repr(exc))
    result['exclusive_gpu_claim'] = False
    return result


def load_stats(config):
    assets, stats = {}, {}
    width = 67 if config['dataset'] == 't2m' else 64
    for key in ('train_mean', 'train_std', 'eval_mean', 'eval_std'):
        assets[key] = verify_asset(config, key)
        value = np.load(assets[key]['path'], allow_pickle=False)
        if value.shape != (width,) or not np.isfinite(value).all():
            raise ValueError(f'{key} must be finite [{width}], got {value.shape}')
        if key.endswith('std') and np.any(value <= 0):
            raise ValueError(f'{key} has zero/negative scales')
        stats[key] = value
    return stats, assets


def audit_split(config, split):
    """Read every declared source; disclose missing/filtering before legacy loader."""
    data = Path(config['paths']['data_root'])
    split_path = Path(config['paths'].get(split + '_split', data / (split + '.txt')))
    split_asset = verify_asset(config, split + '_split', split_path)
    ids = [x.strip() for x in split_path.read_text(encoding='utf-8').splitlines() if x.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate split IDs are not supported')
    train_path = config['paths'].get('train_split')
    if train_path:
        verify_asset(config, 'train_split')
        train_ids = set(Path(train_path).read_text(encoding='utf-8').split())
        if train_ids.intersection(ids):
            raise ValueError('Evaluation split overlaps the configured training split')
    width = 67 if config['dataset'] == 't2m' else 64
    minimum = 40 if width == 67 else 24
    motions, texts = data / 'new_joint_vecs', data / 'texts'
    rows, expected_entries = [], 0
    for ident in ids:
        mp, tp = motions / (ident + '.npy'), texts / (ident + '.txt')
        row = dict(id=ident, motion=str(mp), text=str(tp))
        missing = [str(p) for p in (mp, tp) if not p.is_file()]
        if missing:
            row.update(status='missing', missing=missing)
            rows.append(row)
            continue
        motion = np.load(mp, allow_pickle=False)
        if motion.ndim != 2 or motion.shape[1] not in (width, 263 if width == 67 else 251):
            raise ValueError(f'Unexpected raw dimension {ident}: {motion.shape}')
        if not np.isfinite(motion).all():
            raise FloatingPointError('Nonfinite motion: ' + ident)
        row.update(raw_shape=list(motion.shape), motion_sha256=sha_file(mp), text_sha256=sha_file(tp))
        if not minimum <= len(motion) < 200:
            row.update(status='filtered_length', required_min=minimum, required_max_exclusive=200)
            rows.append(row)
            continue
        full_caption, segment_entries, caption_count = False, 0, 0
        for line in tp.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            parts = line.strip().split('#')
            if len(parts) != 4 or not parts[0] or not parts[1]:
                raise ValueError(f'Malformed caption line in {tp}')
            start, end = float(parts[2]), float(parts[3])
            start = 0.0 if np.isnan(start) else start
            end = 0.0 if np.isnan(end) else end
            if not np.isfinite([start, end]).all():
                raise ValueError(f'Nonfinite caption interval in {tp}')
            if start == 0 and end == 0:
                full_caption = True
                caption_count += 1
            elif minimum <= len(motion[int(start * 20):int(end * 20)]) < 200:
                segment_entries += 1
                caption_count += 1
        entries = segment_entries + int(full_caption)
        expected_entries += entries
        row.update(status='accepted' if entries else 'filtered_no_valid_caption',
                   entries=entries, valid_captions=caption_count)
        rows.append(row)
    return dict(split=split, split_file=str(split_path), split_sha256=split_asset['sha256'],
                split_items=len(ids), expected_entries=expected_entries,
                valid_source_motions=sum(r['status'] == 'accepted' for r in rows),
                valid_captions=sum(r.get('valid_captions', 0) for r in rows),
                rows=rows, feature_slice=f'raw_motion[:, :{width}]',
                segment_time_to_frame='repository int(tag*20), retained for both datasets',
                original_split_modified=False)


def build_dataset(config, stats, split):
    from utils import datasets as ds
    audit = audit_split(config, split)
    data = Path(config['paths']['data_root'])
    original_glove = ds.GloVe
    glove = Path(config['paths'].get('glove', Path(config['paths']['code_root']) / 'glove'))
    ds.GloVe = lambda _root, prefix: original_glove(str(glove), prefix)
    try:
        with isolated_data_seed(int(config.get('evaluation', {}).get('dataset_seed', 3407))):
            dataset = ds.Text2MotionDataset(stats['eval_mean'], stats['eval_std'], audit['split_file'],
                config['dataset'], str(data / 'new_joint_vecs'), str(data / 'texts'),
                4, 196, 20, evaluation=True)
    finally:
        ds.GloVe = original_glove
    if len(dataset) != audit['expected_entries']:
        raise RuntimeError(f'Loader silently filtered entries: expected {audit["expected_entries"]}, got {len(dataset)}')
    audit.update(loaded_entries=len(dataset), pointer=int(dataset.pointer),
                 ordered_entries_sha256=json_sha(list(dataset.name_list)),
                 glove_files={str(p): sha_file(p) for p in sorted(glove.glob('our_vab*')) if p.is_file()})
    if not audit['glove_files']:
        raise FileNotFoundError('Missing GloVe evaluation vocabulary')
    return dataset, audit


class PairedDataset(torch.utils.data.Dataset):
    """Same repository caption/crop choices, isolated per seed+entry for pairing."""
    def __init__(self, dataset, seed):
        self.dataset, self.seed = dataset, seed

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        local_seed = int(hashlib.sha256(f'paper-controls-data-v1:{self.seed}:{index}'.encode()).hexdigest()[:8], 16)
        with isolated_data_seed(local_seed):
            item = self.dataset[index]
        return item, dict(index=int(index), entry=self.dataset.name_list[self.dataset.pointer + index],
                          caption=item[2], length=int(item[5]), data_seed=local_seed)


def paired_collate(items):
    from torch.utils.data._utils.collate import default_collate
    items.sort(key=lambda item: item[0][3], reverse=True)
    return default_collate([item[0] for item in items]), [item[1] for item in items]


def make_loader(config, dataset, seed):
    ev = config.get('evaluation', {})
    if int(ev.get('batch_size', 32)) != 32 or not ev.get('drop_last', True) or not ev.get('shuffle', True):
        raise ValueError('Standard retrieval protocol requires batch32, drop_last=True, shuffle=True')
    return torch.utils.data.DataLoader(PairedDataset(dataset, seed), batch_size=32,
        num_workers=int(ev.get('num_workers', 4)), shuffle=True, drop_last=True,
        collate_fn=paired_collate, generator=torch.Generator().manual_seed(seed),
        persistent_workers=False, pin_memory=False)


def build_evaluators(config, device):
    """Original two evaluator networks with explicit checkpoint locations."""
    from utils import evaluators as ev
    from utils.glove import POS_enumerator
    width = 67 if config['dataset'] == 't2m' else 64
    assets = {k: verify_asset(config, k) for k in ('evaluator', 'evaluator_clip')}
    assets['clip_weights'] = local_clip_only(config)
    checkpoint = torch.load(assets['evaluator']['path'], map_location='cpu', weights_only=False)
    checkpoint_clip = torch.load(assets['evaluator_clip']['path'], map_location='cpu', weights_only=False)
    wrapper = ev.Evaluators.__new__(ev.Evaluators)
    wrapper.device, wrapper.unit_length = device, 4
    wrapper.movement_encoder = ev.MovementConvEncoder(width, 512, 512)
    wrapper.text_encoder = ev.TextEncoderBiGRUCo(300, len(POS_enumerator), 512, 512, device)
    wrapper.motion_encoder = ev.MotionEncoderBiGRUCo(512, 1024, 512, device)
    wrapper.contrast_model = ev.MotionCLIP(width)
    wrapper.movement_encoder.load_state_dict(checkpoint['movement_encoder'], strict=True)
    wrapper.text_encoder.load_state_dict(checkpoint['text_encoder'], strict=True)
    wrapper.motion_encoder.load_state_dict(checkpoint['motion_encoder'], strict=True)
    wrapper.contrast_model.load_state_dict(checkpoint_clip['contrast_model'], strict=True)
    for model in (wrapper.movement_encoder, wrapper.text_encoder, wrapper.motion_encoder, wrapper.contrast_model):
        model.to(device).eval().requires_grad_(False)
    assets['protocol'] = dict(input_width=width, architecture='repository MARDM evaluator',
                              generator_clip='ViT-B/32', evaluator_clip='ViT-B/16',
                              optional_clip_decoupled=False, unit_length=4)
    return wrapper, assets


def load_models(config, checkpoint_path=None, sampler=None, reconstruction=False, base_training=False):
    from models.AE import AE_models
    from models.MARDM import MARDM_models
    width = 67 if config['dataset'] == 't2m' else 64
    device = torch.device('cuda')
    assets = {}
    if not reconstruction:
        assets['clip_weights'] = local_clip_only(config)
    assets['ae_checkpoint'] = verify_asset(config, 'ae_checkpoint', checkpoint_path if reconstruction else None,
                                            require_locked=not reconstruction)
    ae = AE_models[config.get('model', {}).get('ae_name', 'FSQ_AE_High')](input_width=width)
    ae_payload = torch.load(assets['ae_checkpoint']['path'], map_location='cpu', weights_only=False)
    ae.load_state_dict(ae_payload['ae'], strict=True)
    if int(ae.fsq_dim) != 6 or list(ae.fsq_levels) != [8, 8, 8, 5, 5, 5] or ae.output_emb_width != 512:
        raise ValueError('Unexpected FSQ architecture')
    ae.to(device).eval().requires_grad_(False)
    if reconstruction:
        return None, ae, ae_payload, assets, None
    if base_training:
        if config['dataset'] != 'kit' or not checkpoint_path:
            raise ValueError('--base-training only accepts an explicit KIT foundation snapshot')
        assets['base_checkpoint'] = verify_asset(config, 'base_checkpoint', checkpoint_path, require_locked=False)
    else:
        assets['base_checkpoint'] = verify_asset(config, 'base_checkpoint')
    selected = Path(checkpoint_path or assets['base_checkpoint']['path']).resolve()
    assets['checkpoint'] = dict(path=str(selected), sha256=sha_file(selected), bytes=selected.stat().st_size)
    payload = torch.load(selected, map_location='cpu', weights_only=False)
    if base_training and ('ema_mardm' not in payload or payload.get('objective') == 'trajectory_distill'
                          or payload.get('inference') in ('distilled_single_conditional', 'distilled')):
        raise ValueError('Foundation validation cannot use a distilled student as original KIT base')
    if payload.get('ema_mardm_alias_no_ema', False):
        raise ValueError('Checkpoint advertises a fake EMA alias')
    if 'dataset' in payload and payload['dataset'] != config['dataset']:
        raise ValueError('Snapshot dataset differs')
    if payload.get('source_checkpoint_sha256', assets['base_checkpoint']['sha256']) != assets['base_checkpoint']['sha256']:
        raise ValueError('Student original-base provenance differs')
    declared_ae = payload.get('ae_checkpoint_sha256', payload.get('ae_sha256'))
    if declared_ae and declared_ae != assets['ae_checkpoint']['sha256']:
        raise ValueError('Student tokenizer differs')
    if payload.get('frozen_unchanged') is False or payload.get('real_ema') is False:
        raise ValueError('Invalid frozen/EMA provenance')
    name = config.get('model', {}).get('name', 'FSQ-MARDM-DiT-XL')
    model = MARDM_models[name](ae_dim=ae.output_emb_width, fsq_dim=ae.fsq_dim, cond_mode='text')
    missing = []
    if 'ema_model' in payload:
        model.load_state_dict(payload['ema_model'], strict=True)
        weight_key = 'ema_model'
    elif 'ema_mardm' in payload:
        missing, unexpected = model.load_state_dict(payload['ema_mardm'], strict=False)
        if unexpected or any(not k.startswith('clip_model.') for k in missing):
            raise ValueError('Base snapshot has missing/non-CLIP or unexpected keys')
        weight_key = 'ema_mardm'
    elif 'ema_head' in payload:
        base = torch.load(assets['base_checkpoint']['path'], map_location='cpu', weights_only=False)
        missing, unexpected = model.load_state_dict(base['ema_mardm'], strict=False)
        if unexpected or any(not k.startswith('clip_model.') for k in missing):
            raise ValueError('Head snapshot base keys differ')
        model.DiffMLPs.net.load_state_dict(payload['ema_head'], strict=True)
        weight_key = 'ema_head'
    else:
        raise ValueError('Need ema_model/ema_mardm/ema_head, never silently select student weights')
    inferred = payload.get('inference')
    if sampler is None:
        sampler = 'distilled' if inferred in ('distilled_single_conditional', 'distilled') else 'native_cfg'
    sampler = {'cfg': 'native_cfg', 'apg': 'direct_apg'}.get(sampler, sampler)
    if inferred in ('distilled_single_conditional', 'distilled') and sampler != 'distilled':
        raise ValueError('Do not reapply guidance to a distilled student')
    if payload.get('objective') == 'native_continue' and sampler == 'distilled':
        raise ValueError('Native continuation requires native guidance inference')
    model.to(device).eval().requires_grad_(False)
    controller = SamplerAudit(model, sampler, config['sampling']['cfg'])
    if controller.config['inner_steps'] != config['sampling']['inner_steps']:
        controller.remove()
        raise ValueError('Generator steps/output grid differ from locked protocol')
    assets['selected_weights'] = dict(key=weight_key, missing_frozen_clip_keys=missing,
        steps=payload.get('steps', payload.get('step', payload.get('total_it', 0))),
        epoch=payload.get('epoch', payload.get('ep')), inference=sampler,
        training_config_sha256=payload.get('config_sha256'), train_scope=payload.get('train_scope'))
    return model, ae, payload, assets, controller


class SamplerAudit:
    """Count actual forwards, including native SiT's direct net.forward calls.

    SiT retains its original adaptive ODE sampler. Its 50 requested output
    points are not 50 model evaluations, and CFG branches share one forward.
    """
    def __init__(self, model, sampler, cfg):
        if sampler not in ('native_cfg', 'direct_apg', 'distilled'):
            raise ValueError(sampler)
        self.model, self.head, self.sampler, self.cfg = model, model.DiffMLPs, sampler, float(cfg)
        self.native_sit = type(self.head).__name__ == 'DiffMLPs_SiT'
        if self.native_sit:
            if sampler != 'native_cfg':
                raise ValueError('Native SiT only supports its original velocity/ODE CFG sampler; no APG or distillation')
            parameters = inspect.signature(self.head.gen_diffusion.sample_ode).parameters
            ode_defaults = {key: parameters[key].default for key in
                            ('sampling_method', 'num_steps', 'atol', 'rtol', 'reverse')}
            expected = dict(sampling_method='dopri5', num_steps=50,
                            atol=1e-6, rtol=1e-3, reverse=False)
            if ode_defaults != expected:
                raise ValueError('Native SiT ODE defaults differ from the original sampler')
            inner_steps = int(ode_defaults['num_steps'])
        else:
            if not hasattr(self.head, 'num_sampling_steps'):
                raise ValueError('Unsupported diffusion head: ' + type(self.head).__name__)
            inner_steps = int(self.head.num_sampling_steps)
        self.original_sample, self.original_forward = self.head.sample, model.forward
        self.had_sample, self.had_forward = 'sample' in self.head.__dict__, 'forward' in model.__dict__
        self.original_net_forward = self.head.net.forward
        self.had_net_forward = 'forward' in self.head.net.__dict__
        self.config = dict(sampler=sampler, outer_cfg_envelope=float(cfg),
            head_effective_cfg=1.0 if sampler == 'distilled' else float(cfg),
            extra_guidance_on_student=False, outer_mar_branches=2,
            inner_steps=inner_steps,
            head_class=type(self.head).__name__,
            prediction_target='velocity' if self.native_sit else 'clean_fsq_coordinates',
            integrator='dopri5' if self.native_sit else 'repository_xpred_interpolation',
            inner_steps_meaning='interpolated_ode_output_points' if self.native_sit else 'fixed_model_prediction_steps',
            adaptive_nfe=self.native_sit,
            head_cfg_branches_per_forward=2 if self.native_sit else 1,
            net_counting='actual_net_forward_invocations',
            native_sample_preserved=sampler == 'native_cfg',
            beta=-0.5 if sampler == 'direct_apg' else None,
            eta=0.0 if sampler == 'direct_apg' else None,
            norm_threshold=0.0 if sampler == 'direct_apg' else None)
        if self.native_sit:
            self.config['ode_defaults'] = ode_defaults
        self.reset()
        def mar_forward(owner, *args, **kwargs):
            self.stats['mar_calls'] += 1
            self.stats['mar_token_forwards'] += int(args[0].shape[0] * args[0].shape[1])
            return self.original_forward(*args, **kwargs)
        def net_forward(owner, *args, **kwargs):
            self.stats['net_calls'] += 1
            state = args[0] if args else kwargs['x']
            self.stats['net_token_forwards'] += int(state.shape[0])
            return self.original_net_forward(*args, **kwargs)
        self.head.net.forward = types.MethodType(net_forward, self.head.net)
        model.forward = types.MethodType(mar_forward, model)
        def sample(owner, z, temperature=1.0, cfg=1.0):
            if float(cfg) != self.cfg or cfg == 1 or len(z) % 2:
                raise ValueError('Preserve locked paired outer MAR interface')
            self.stats['sample_calls'] += 1
            before = self.stats['net_calls']
            if self.sampler == 'native_cfg':
                generated = self.original_sample(z, temperature, cfg)
            elif self.sampler == 'distilled':
                generated = self.original_sample(z[:len(z) // 2], temperature, 1.0)
                generated = torch.cat([generated, generated], dim=0)
            else:
                generated = self._apg_sample(owner, z, temperature, cfg)
            self.stats['net_calls_per_sample'].append(self.stats['net_calls'] - before)
            return generated
        self.head.sample = types.MethodType(sample, self.head)

    def reset(self):
        self.stats = dict(mar_calls=0, mar_token_forwards=0, sample_calls=0,
                          net_calls=0, net_token_forwards=0, momentum_updates=0,
                          net_calls_per_sample=[])

    def _apg_sample(self, head, z, temperature, cfg):
        from control_guidance import combine_predictions, CONFIGS
        count = len(z) // 2
        state = torch.randn(count, head.in_channels, device=z.device) * temperature
        initial = state.clone()
        times = torch.linspace(.02, .98, head.num_sampling_steps, device=z.device)
        momentum = None
        for index, current in enumerate(times):
            time = torch.full((count,), current.item(), device=z.device)
            cond = head.net(state, time, z[:count])
            uncond = head.net(state, time, z[count:])
            prediction, momentum, _ = combine_predictions(cond, uncond, cfg, CONFIGS['apg'], current.item(), momentum)
            self.stats['momentum_updates'] += 1
            if index == len(times) - 1:
                state = prediction
            else:
                estimate = (state - current * prediction) / (1 - current) if current > .01 else initial
                state = times[index + 1] * prediction + (1 - times[index + 1]) * estimate
        return torch.cat([state, state], dim=0)

    def check_budget(self, generate_calls, outer=18, inner=50):
        expected = dict(mar_calls=generate_calls * outer * 2, sample_calls=generate_calls * outer)
        if not self.native_sit:
            expected['net_calls'] = generate_calls * outer * inner * (1 if self.sampler == 'distilled' else 2)
        if self.sampler == 'direct_apg':
            expected['momentum_updates'] = generate_calls * outer * inner
        for key, value in expected.items():
            if self.stats[key] != value:
                raise RuntimeError(f'Sampler budget differs: {key}={self.stats[key]}, expected {value}')
        observed = self.stats['net_calls_per_sample']
        if (len(observed) != self.stats['sample_calls'] or any(n <= 0 for n in observed)
                or sum(observed) != self.stats['net_calls']):
            raise RuntimeError('Incomplete actual head-forward accounting per sample')
        return copy.deepcopy(self.stats)

    def remove(self):
        for owner, name, original, existed in ((self.model, 'forward', self.original_forward, self.had_forward),
                                                (self.head, 'sample', self.original_sample, self.had_sample),
                                                (self.head.net, 'forward', self.original_net_forward, self.had_net_forward)):
            if existed:
                setattr(owner, name, original)
            else:
                del owner.__dict__[name]


def generate_motion(config, model, ae, captions, lengths):
    sampling = config['sampling']
    coordinates = model.generate(captions, lengths // 4, sampling['outer_steps'], sampling['cfg'],
        temperature=sampling['temperature'], hard_pseudo_reorder=False, ae=ae)
    return ae.decode_from_fsq(coordinates)


def lock_test_candidate(lock_path, dataset, checkpoint_sha256, sampler, ae_checkpoint_sha256=None):
    if not lock_path:
        raise ValueError('Test requires a pre-test selection_lock with all frozen candidates')
    path = Path(lock_path).resolve()
    lock = json.loads(path.read_text(encoding='utf-8'))
    if not lock.get('locked_before_test', False):
        raise ValueError('Selection lock must explicitly assert locked_before_test=true')
    matches = [r for r in lock['candidates'] if r['dataset'] == dataset
               and r['checkpoint_sha256'] == checkpoint_sha256 and r['sampler'] == sampler
               and (ae_checkpoint_sha256 is None or r.get('ae_checkpoint_sha256') == ae_checkpoint_sha256)]
    if not matches:
        raise ValueError('Weight+sampler identity missing from pre-test selection lock')
    return dict(path=str(path), sha256=sha_file(path), candidates=matches)


def batch_fingerprint(batch):
    h = hashlib.sha256(json.dumps(list(batch[2]), ensure_ascii=False).encode('utf-8'))
    for item in (batch[4], batch[5]):
        h.update(item.contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def finite_embeddings(outputs):
    for pair in outputs:
        for value in pair:
            if value.ndim != 2 or value.shape[1] != 512 or not bool(torch.isfinite(value).all()):
                raise FloatingPointError(f'Invalid evaluator embedding {tuple(value.shape)}')


def normalize_generated(prediction, stats):
    decoded = prediction.detach().cpu().numpy()
    raw = decoded * stats['train_std'] + stats['train_mean']
    evaluated = (raw - stats['eval_mean']) / stats['eval_std']
    if not np.isfinite(evaluated).all():
        raise FloatingPointError('Nonfinite decoder/evaluator normalization')
    return torch.from_numpy(evaluated).to(prediction.device), raw
