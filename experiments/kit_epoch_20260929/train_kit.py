"""Independent KIT FSQ/FSQ+JiT training with per-epoch paired validation.

This file never imports a HumanML checkpoint or changes the original dataset.
The optimizer schedules follow train_AE.py/train_MARDM.py and the KIT README;
the data enumeration, validation, checkpointing and frozen-CLIP EMA are explicit.
"""
import argparse
import bisect
import copy
import datetime
import hashlib
import json
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate


HERE = Path(__file__).resolve().parent
STOP_REQUESTED = False
AE_DEFAULTS = dict(epochs=50, batch_size=512, micro_batch=512, window=64,
                   lr=2e-4, betas=[0.9, 0.99], weight_decay=0.0,
                   warmup_updates=2000, milestones=[150000, 250000], lr_decay=0.1,
                   aux_loss_joints=1.0, grad_clip=None, validation_batch=32,
                   selection='minimum_validation_reconstruction_loss_earlier_tie')
BASE_DEFAULTS = dict(epochs=500, batch_size=16, micro_batch=16,
                     lr=2e-4, betas=[0.9, 0.99], weight_decay=1e-5,
                     warmup_updates=2000, milestones=[20000], lr_decay=0.1,
                     ema=0.9999, grad_clip=None, validation_every='epoch',
                     selection='minimum_validation_generation_fid_earlier_tie')


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def tensor_sha(tensor):
    tensor = tensor.detach().cpu().contiguous()
    h = hashlib.sha256(str(tensor.dtype).encode() + str(tuple(tensor.shape)).encode())
    h.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def module_sha(model):
    return canonical_sha({name: tensor_sha(value) for name, value in model.state_dict().items()})


def cpu_weights(model):
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.pending')
    with temp.open('w', encoding='utf-8') as f:
        json.dump(value, f, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_save(path, value, previous=False):
    path = Path(path)
    temp = path.with_name(path.name + '.pending')
    with temp.open('wb') as f:
        torch.save(value, f)
        f.flush()
        os.fsync(f.fileno())
    if previous and path.exists():
        prev = path.with_name('last.prev.pt')
        pending = path.with_name('last.prev.pending')
        pending.unlink(missing_ok=True)
        os.link(path, pending)
        os.replace(pending, prev)
    os.replace(temp, path)
    if os.name != 'nt':
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


class EpochStream:
    """Single-process data stream. Each epoch drops only its final partial batch."""
    def __init__(self, dataset, batch_size, seed):
        if len(dataset) < batch_size:
            raise ValueError('KIT dataset contains fewer examples than the effective batch')
        self.dataset, self.batch_size = dataset, batch_size
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(len(dataset), generator=self.generator)
        self.epoch, self.cursor = 0, 0
        self.updates_per_epoch = len(dataset) // batch_size

    def next(self):
        if self.cursor + self.batch_size > len(self.order):
            self.epoch += 1
            self.cursor = 0
            self.order = torch.randperm(len(self.dataset), generator=self.generator)
        indices = self.order[self.cursor:self.cursor + self.batch_size].tolist()
        value = default_collate([self.dataset[i] for i in indices])
        self.cursor += self.batch_size
        return value, indices

    def state_dict(self):
        return dict(epoch=self.epoch, cursor=self.cursor, order=self.order.clone(),
                    generator=self.generator.get_state(), size=len(self.dataset),
                    batch_size=self.batch_size)

    def load_state_dict(self, value):
        if value['size'] != len(self.dataset) or value['batch_size'] != self.batch_size:
            raise ValueError('Dataset size or effective batch changed during resume')
        if not torch.equal(value['order'].sort().values, torch.arange(len(self.dataset))):
            raise ValueError('Invalid checkpoint data permutation')
        if not 0 <= value['cursor'] <= len(self.dataset) or value['cursor'] % self.batch_size:
            raise ValueError('Invalid checkpoint data cursor')
        self.order = value['order'].clone()
        self.epoch, self.cursor = value['epoch'], value['cursor']
        self.generator.set_state(value['generator'])


class KitCorpus:
    """Strict source loading, with every exclusion recorded against the real split."""
    def __init__(self, config, split, text=False):
        paths = config['paths']
        root = Path(paths['data_root'])
        motion_dir = Path(paths.get('motion_dir', root / 'new_joint_vecs'))
        text_dir = Path(paths.get('text_dir', root / 'texts'))
        split_key = ('ae_' + split + '_split') if not text and ('ae_' + split + '_split') in paths else split + '_split'
        split_file = Path(paths[split_key])
        names = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
        if len(names) != len(set(names)):
            raise ValueError('Duplicate IDs in KIT ' + split + ' split')
        allowed_missing = set(config.get('data', {}).get('allowed_missing_motion', []))
        self.mean = np.load(paths['train_mean'])
        self.std = np.load(paths['train_std'])
        if self.mean.shape != (64,) or self.std.shape != (64,):
            raise ValueError('KIT training statistics must have exactly 64 features')
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or (self.std <= 0).any():
            raise ValueError('Invalid KIT training normalization statistics')
        self.records, excluded, errors, motion_hashes, text_hashes = [], [], [], {}, {}
        for name in names:
            motion_path = motion_dir / (name + '.npy')
            if not motion_path.is_file():
                record = dict(id=name, reason='missing_motion', path=str(motion_path))
                (excluded if name in allowed_missing else errors).append(record)
                continue
            try:
                motion = np.load(motion_path, allow_pickle=False)
                if motion.ndim != 2 or motion.shape[1] not in (64, 251):
                    raise ValueError('Expected KIT [frames,251] or prepared [frames,64], got ' + str(motion.shape))
                if not np.isfinite(motion).all():
                    raise ValueError('Non-finite motion')
                motion_hashes[name] = sha(motion_path)
                if not text:
                    self.records.append(dict(id=name, motion=motion[:, :64].astype(np.float32),
                                             source_frames=len(motion), source_width=motion.shape[1]))
                    continue
                if len(motion) < 24 or len(motion) >= 200:
                    excluded.append(dict(id=name, reason='native_motion_length_filter', frames=len(motion)))
                    continue
                text_path = text_dir / (name + '.txt')
                if not text_path.is_file():
                    raise FileNotFoundError(text_path)
                text_hashes[name] = sha(text_path)
                full_captions = []
                for line_index, line in enumerate(text_path.read_text(encoding='utf-8').splitlines()):
                    if not line.strip():
                        continue
                    parts = line.strip().split('#')
                    if len(parts) != 4:
                        raise ValueError('Malformed caption line ' + str(line_index + 1))
                    caption, tokens = parts[:2]
                    start, end = map(float, parts[2:])
                    start = 0.0 if math.isnan(start) else start
                    end = 0.0 if math.isnan(end) else end
                    if not math.isfinite(start) or not math.isfinite(end):
                        raise ValueError('Non-finite caption interval')
                    if start == 0.0 and end == 0.0:
                        full_captions.append(caption)
                    else:
                        segment = motion[int(start * 20):int(end * 20), :64]
                        segment_id = name + '__caption_' + str(line_index + 1)
                        if len(segment) < 24 or len(segment) >= 200:
                            excluded.append(dict(id=segment_id, reason='native_segment_length_filter', frames=len(segment)))
                        else:
                            self.records.append(dict(id=segment_id, source_id=name,
                                motion=segment.astype(np.float32), captions=[caption], source_frames=len(segment)))
                if full_captions:
                    self.records.append(dict(id=name, source_id=name, motion=motion[:, :64].astype(np.float32),
                                             captions=full_captions, source_frames=len(motion)))
                elif not any(record.get('source_id') == name for record in self.records):
                    excluded.append(dict(id=name, reason='no_usable_caption'))
            except Exception as exc:
                errors.append(dict(id=name, reason=type(exc).__name__, error=str(exc)))
        self.audit = dict(dataset='kit', split=split, split_path=str(split_file), split_sha256=sha(split_file),
            source_entries=len(names), loaded_records=len(self.records), excluded=excluded, errors=errors,
            motion_hashes_sha256=canonical_sha(motion_hashes), text_hashes_sha256=canonical_sha(text_hashes),
            motion_hashes=motion_hashes, text_hashes=text_hashes,
            record_order_sha256=canonical_sha([record['id'] for record in self.records]),
            raw_widths=sorted(set(record.get('source_width', 251) for record in self.records)),
            feature_selection='first_64 = root_features_4 + 20_nonroot_xyz_60',
            statistics_source=config.get('data', {}).get('training_statistics_source', 'published_frozen_file'),
            train_mean_sha256=sha(paths['train_mean']), train_std_sha256=sha(paths['train_std']))
        if errors:
            exc = ValueError('KIT asset errors: ' + json.dumps(errors[:10], ensure_ascii=False))
            exc.asset_audit = self.audit
            raise exc
        if not self.records:
            raise ValueError('No usable KIT records in ' + split)


class WindowDataset:
    def __init__(self, corpus, window):
        self.corpus, self.window = corpus, window
        self.records, counts, excluded = [], [], []
        for record in corpus.records:
            count = len(record['motion']) - window + 1
            if count <= 0:
                excluded.append(dict(id=record['id'], reason='shorter_than_AE_window', frames=len(record['motion'])))
            else:
                self.records.append(record)
                counts.append(count)
        self.ends = np.cumsum(counts).tolist()
        corpus.audit.update(window=window, usable_motion_count=len(self.records),
            windows=self.ends[-1] if self.ends else 0, window_excluded=excluded,
            window_enumeration='all valid inclusive sliding windows L-W+1; repairs legacy duplicate-first/omitted-last indexing')

    def __len__(self):
        return self.ends[-1] if self.ends else 0

    def __getitem__(self, index):
        motion_index = bisect.bisect_right(self.ends, int(index))
        start = int(index) - (self.ends[motion_index - 1] if motion_index else 0)
        value = self.records[motion_index]['motion'][start:start + self.window]
        return ((value - self.corpus.mean) / self.corpus.std).astype(np.float32)


class CaptionDataset:
    def __init__(self, corpus):
        self.corpus = corpus

    def __len__(self):
        return len(self.corpus.records)

    def __getitem__(self, index):
        record = self.corpus.records[index]
        motion = record['motion']
        caption = random.choice(record['captions'])
        coin = np.random.choice(['single', 'single', 'double'])
        length = (len(motion) // 4 - (coin == 'double')) * 4
        start = random.randint(0, len(motion) - length)
        value = (motion[start:start + length] - self.corpus.mean) / self.corpus.std
        if length > 196 or length < 20:
            raise ValueError('KIT native crop length outside [20,196]')
        value = np.pad(value, ((0, 196 - length), (0, 0))).astype(np.float32)
        return caption, value, length


def locked_config(config, stage):
    config = copy.deepcopy(config)
    if config.get('dataset') != 'kit' or config.get('model', {}).get('input_width', 64) != 64:
        raise ValueError('train_kit.py only permits independent KIT 64D training')
    config.setdefault('seed', config.get('training', {}).get('seed', 3407))
    model = config.setdefault('model', {})
    name = model.get('name', model.get('generator', 'FSQ-MARDM-DiT-XL'))
    if name not in ('FSQ-MARDM-SiT-XL', 'FSQ-MARDM-DiT-XL'):
        raise ValueError('KIT epoch experiment requires native FSQ SiT-XL or DiT-XL')
    model.update(input_width=64, joints=21, name=name,
        ae='FSQ_AE_High', generator=name, fsq_levels=[8, 8, 8, 5, 5, 5], lift_width=512)
    params = dict(AE_DEFAULTS if stage == 'tokenizer' else BASE_DEFAULTS)
    aliases = dict(micro_batch_size='micro_batch', window_size='window', gamma='lr_decay',
                   joint_loss_weight='aux_loss_joints', ema_decay='ema', eval_every='validation_every',
                   gradient_clip='grad_clip')
    native = config.get('kit_training', {}).get(stage, {})
    params.update({aliases.get(key, key): value for key, value in native.items()})
    params.update(config.get('kit_' + stage, {}))
    # This experiment explicitly supersedes the historical 5k-update cadence.
    params['validation_every'] = 'epoch'
    if params['batch_size'] % params['micro_batch'] or params['micro_batch'] <= 0:
        raise ValueError('micro_batch must divide effective batch')
    expected_epochs = 50 if stage == 'tokenizer' else 500
    if params['epochs'] != expected_epochs:
        raise ValueError('KIT full budget must remain ' + str(expected_epochs) + ' epochs')
    config['kit_' + stage] = params
    config['stage'] = stage
    config['checkpoint_every'] = 500
    config['initialization'] = 'from_scratch_KIT_no_HumanML_weights'
    config.setdefault('sampling', dict(outer_steps=18, inner_steps=50, cfg=2.5, temperature=1.0))
    config.setdefault('evaluation', {}).update(seed=3407, split='val', cadence='every_epoch',
        samplers=['native_cfg', 'direct_apg'] if name == 'FSQ-MARDM-DiT-XL' else ['native_cfg'])
    if stage == 'base':
        params['validation_every_epoch'] = True
        params['validation_samplers'] = list(config['evaluation']['samplers'])
    if config['sampling']['outer_steps'] != 18 or config['sampling']['inner_steps'] != 50:
        raise ValueError('KIT sampling protocol must remain outer18/inner50')
    if float(config['sampling']['cfg']) != 2.5 or float(config['sampling']['temperature']) != 1.0:
        raise ValueError('KIT epoch experiment locks CFG=2.5 and temperature=1')
    return config


def verify_assets(config, stage):
    required = ['train_split', 'val_split', 'train_mean', 'train_std']
    if stage == 'tokenizer':
        required = [('ae_' + key) if key.endswith('_split') and ('ae_' + key) in config['paths'] else key
                    for key in required]
    if stage == 'base':
        required += ['ae_checkpoint', 'eval_mean', 'eval_std', 'evaluator', 'evaluator_clip']
    result = {}
    for key in required:
        path = Path(config['paths'][key])
        expected = config.get('hashes', {}).get(key)
        if not expected:
            raise ValueError('A locked asset SHA-256 is required for ' + key)
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha(path)
        if actual != expected:
            raise ValueError('Asset hash mismatch: ' + key)
        result[key] = dict(path=str(path.resolve()), sha256=actual)
    train_key = 'ae_train_split' if stage == 'tokenizer' and 'ae_train_split' in config['paths'] else 'train_split'
    val_key = 'ae_val_split' if stage == 'tokenizer' and 'ae_val_split' in config['paths'] else 'val_split'
    train_ids = set(Path(config['paths'][train_key]).read_text().split())
    val_ids = set(Path(config['paths'][val_key]).read_text().split())
    if train_ids & val_ids:
        raise ValueError('KIT train/val splits overlap')
    return result


@torch.no_grad()
def ema_update(student, ema, decay):
    source = dict(student.named_parameters())
    for name, target in ema.named_parameters():
        if source[name].requires_grad:
            target.mul_(decay).add_(source[name], alpha=1.0 - decay)
        else:
            target.copy_(source[name])
    for name, target in ema.named_buffers():
        target.copy_(dict(student.named_buffers())[name])


def gradient_norm(parameters):
    values = [parameter.grad.detach().float().norm().square() for parameter in parameters if parameter.grad is not None]
    result = torch.stack(values).sum().sqrt() if values else torch.tensor(0.)
    if not torch.isfinite(result):
        raise FloatingPointError('Non-finite gradient norm')
    return float(result)


class KitTrainer:
    def __init__(self, config, output, stage, resume=False, smoke=False):
        self.config, self.out, self.stage = locked_config(config, stage), Path(output), stage
        self.params = self.config['kit_' + stage]
        self.model_name = self.config['model']['name']
        self.validation_samplers = self.config['evaluation']['samplers'] if stage == 'base' else ['reconstruction']
        self.smoke = smoke
        self.out.mkdir(parents=True, exist_ok=True)
        self.identity = canonical_sha(self.config)
        if (self.out / 'COMPLETE').exists():
            raise FileExistsError('This KIT stage is COMPLETE; it must not be restarted')
        if (self.out / 'config.json').exists():
            if canonical_sha(read_json(self.out / 'config.json')) != self.identity:
                raise ValueError('Existing KIT directory has a different locked configuration')
            if not resume:
                raise FileExistsError('Existing stage requires explicit --resume')
        elif resume:
            raise FileNotFoundError('Cannot resume without the locked run config')
        else:
            atomic_json(self.out / 'config.json', self.config)
        self.assets = verify_assets(self.config, stage)
        code = Path(self.config['paths']['code_root']).resolve()
        os.chdir(code)
        sys.path.insert(0, str(code))
        self.source = {str(path.relative_to(code)): sha(path) for path in sorted(code.rglob('*.py'))
                       if '.git' not in path.parts and '__pycache__' not in path.parts}
        for source_path in sorted(HERE.glob('*.py')):
            self.source['isolated/' + source_path.name] = sha(source_path)
        seed_all(self.config['seed'])
        torch.set_num_threads(int(self.config.get('cpu_threads', 4)))
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = bool(self.config.get('tf32', True))
        torch.backends.cudnn.allow_tf32 = bool(self.config.get('tf32', True))
        if not torch.cuda.is_available():
            raise RuntimeError('Real KIT training requires the audited CUDA GPU')
        self.device = torch.device('cuda')
        try:
            train = KitCorpus(self.config, 'train', text=(stage == 'base'))
            validation = KitCorpus(self.config, 'val', text=False) if stage == 'tokenizer' else None
        except Exception as exc:
            if hasattr(exc, 'asset_audit'):
                atomic_json(self.out / 'asset_error_audit.json', exc.asset_audit)
            raise
        self.dataset = WindowDataset(train, self.params['window']) if stage == 'tokenizer' else CaptionDataset(train)
        self.validation = WindowDataset(validation, self.params['window']) if validation else None
        self.data_audits = {'train': train.audit}
        if validation:
            self.data_audits['val'] = validation.audit
        atomic_json(self.out / 'data_audit.json', self.data_audits)
        self.data_identity = canonical_sha(self.data_audits)
        self.stream = EpochStream(self.dataset, self.params['batch_size'], self.config['seed'] + 31)
        self.budget = self.params['epochs'] * self.stream.updates_per_epoch
        from models.AE import AE_models
        if stage == 'base':
            # Replace CLIP's downloader before constructing a model: cache misses fail closed.
            from eval_common import local_clip_only
            self.assets['clip_weights'] = local_clip_only(self.config)
        from models.MARDM import MARDM_models
        self.ae = AE_models['FSQ_AE_High'](input_width=64).to(self.device)
        if list(self.ae.fsq_levels) != [8, 8, 8, 5, 5, 5] or self.ae.output_emb_width != 512:
            raise ValueError('Unexpected KIT tokenizer architecture')
        if stage == 'tokenizer':
            self.model, self.ema = self.ae, None
            self.frozen = {}
        else:
            checkpoint = torch.load(self.config['paths']['ae_checkpoint'], map_location='cpu', weights_only=False)
            if checkpoint.get('dataset') != 'kit' or checkpoint.get('stage') != 'tokenizer':
                raise ValueError('KIT base requires the identity-verified independent KIT tokenizer checkpoint')
            if checkpoint['config']['hashes']['train_mean'] != self.config['hashes']['train_mean'] or checkpoint['config']['hashes']['train_std'] != self.config['hashes']['train_std']:
                raise ValueError('KIT base normalization differs from its trained tokenizer')
            if not smoke:
                ae_directory = Path(self.config['paths']['ae_checkpoint']).parent
                completion = read_json(ae_directory / 'COMPLETE')
                if completion.get('state') != 'complete' or completion.get('epochs_complete') != 50:
                    raise ValueError('Formal KIT base requires completed 50-epoch tokenizer training')
                if completion['best_checkpoint_sha256'] != self.config['hashes']['ae_checkpoint']:
                    raise ValueError('KIT base must use the preregistered selected tokenizer')
            self.ae.load_state_dict(checkpoint['ae'], strict=True)
            self.ae.eval().requires_grad_(False)
            self.model = MARDM_models[self.model_name](ae_dim=512, fsq_dim=6, cond_mode='text').to(self.device)
            if self.model_name == 'FSQ-MARDM-DiT-XL':
                self.model.DiffMLPs.num_sampling_steps = 50
            self.model.clip_model.eval().requires_grad_(False)
            self.ema = copy.deepcopy(self.model).eval().requires_grad_(False)
            self.frozen = dict(ae=module_sha(self.ae), clip=module_sha(self.model.clip_model))
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(self.parameters, lr=self.params['lr'],
            betas=tuple(self.params['betas']), weight_decay=self.params['weight_decay'])
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
            milestones=self.params['milestones'], gamma=self.params['lr_decay'])
        self.step, self.elapsed_before = 0, 0.0
        self.selection = dict(best_value=None, best_step=None, best_epoch=None, last_validation_step=-1,
                              rule=self.params['selection'])
        self.apg_selection = copy.deepcopy(self.selection) if 'direct_apg' in self.validation_samplers else None
        self.initial_sha = module_sha(self.model)
        self.gradient_observed = dict(ae=False, mar=False, head=False)
        self.validation_history = []
        self.started = time.monotonic()
        manifest = dict(utc=utc(), dataset='kit', stage=stage, config_sha256=self.identity,
            data_sha256=self.data_identity, source_hashes=self.source, assets=self.assets,
            updates_per_epoch=self.stream.updates_per_epoch, total_updates=self.budget,
            epochs=self.params['epochs'], effective_batch=self.params['batch_size'],
            micro_batch=self.params['micro_batch'], accumulation=self.params['batch_size'] // self.params['micro_batch'],
            train_examples=len(self.dataset), drop_last=True, num_workers=0, frozen=self.frozen,
            initial_state_sha256=self.initial_sha, precision='float32 except frozen pretrained CLIP float16',
            tf32=self.config.get('tf32', True), smoke=smoke,
            model_name=self.model_name,
            objective=('SmoothL1 full64D + auxiliary local xyz' if stage == 'tokenizer' else
                       'native clean-coordinate MSE, no teacher' if self.model_name == 'FSQ-MARDM-DiT-XL' else
                       'native SiT linear-transport velocity loss, no teacher'),
            native_schedule='warmup lr*(step+1)/(2001) for step<2000; AE scheduler steps from2000, base every update',
            selection=self.params['selection'], validation_rng='AE deterministic/all_windows; generator separate process',
            native_training_scope='AE encoder/FSQ projections/decoder' if stage == 'tokenizer' else 'MAR+native generation head; AE+CLIP frozen',
            validation_samplers=self.validation_samplers, validation_every='epoch',
            sampler_protocol=('DiT fixed50 clean-prediction updates; paired CFG/APG on identical EMA weights'
                if self.model_name == 'FSQ-MARDM-DiT-XL' else
                'native SiT adaptive dopri5, 50 output grid points, atol1e-6/rtol1e-3; NFE is measured, not fixed50'),
            checkpoint_policy='last.pt is latest full recovery; previous every500 and epoch/final boundaries; independent CFG/APG best; bounded temporary EMA snapshots',
            no_transfer_from_humanml=True)
        if resume:
            old_manifest = read_json(self.out / 'manifest.json')
            if old_manifest['source_hashes'] != self.source or old_manifest['data_sha256'] != self.data_identity:
                raise ValueError('Source or KIT data changed during resume')
            if old_manifest['smoke'] != smoke:
                raise ValueError('Smoke and formal training may not be mixed')
            self.restore(self.out / 'last.pt')
            self.initial_sha = old_manifest['initial_state_sha256']
        else:
            atomic_json(self.out / 'manifest.json', manifest)
        torch.cuda.reset_peak_memory_stats()
        self.started = time.monotonic()
        self.log = (self.out / 'train.jsonl').open('a', encoding='utf-8', buffering=1)
        atomic_json(self.out / 'status.json', dict(state='running', step=self.step, utc=utc(), pid=os.getpid()))

    def verify_frozen(self):
        if self.stage == 'base':
            if module_sha(self.ae) != self.frozen['ae'] or module_sha(self.model.clip_model) != self.frozen['clip']:
                raise RuntimeError('Frozen KIT AE or CLIP changed')
            if module_sha(self.ema.clip_model) != self.frozen['clip']:
                raise RuntimeError('Frozen EMA CLIP changed')
        return True

    def inference(self):
        payload = dict(dataset='kit', stage=self.stage, step=self.step,
            epoch=self.step / self.stream.updates_per_epoch, config_sha256=self.identity,
            data_sha256=self.data_identity, config=self.config, initial_state_sha256=self.initial_sha)
        if self.stage == 'tokenizer':
            payload['ae'] = cpu_weights(self.ae)
        else:
            payload.update(ema_mardm=cpu_weights(self.ema), ae_checkpoint=self.config['paths']['ae_checkpoint'],
                           ae_sha256=self.config['hashes']['ae_checkpoint'], sampler='native_cfg',
                            ema_updates=self.step, ema_decay=self.params['ema'], model_name=self.model_name)
        return payload

    def checkpoint(self):
        self.verify_frozen()
        floor = int(self.config.get('free_space_floor_bytes', 30 * 1024 ** 3))
        previous_bytes = (self.out / 'last.pt').stat().st_size if (self.out / 'last.pt').exists() else 0
        if shutil.disk_usage(self.out).free < floor + previous_bytes:
            raise OSError('Insufficient space for atomic KIT checkpoint; previous checkpoint preserved')
        payload = self.inference()
        payload.update(student=cpu_weights(self.model), optimizer=self.optimizer.state_dict(),
            scheduler=self.scheduler.state_dict(), rng=rng_state(), stream=self.stream.state_dict(),
            selection=copy.deepcopy(self.selection), apg_selection=copy.deepcopy(self.apg_selection),
            validation_history=copy.deepcopy(self.validation_history),
            elapsed_seconds=self.elapsed_before + time.monotonic() - self.started,
            gradient_observed=self.gradient_observed, frozen=self.frozen, full_resume=True,
            source_hashes=self.source, safe_boundary='complete optimizer update; no prefetched batches')
        atomic_save(self.out / 'last.pt', payload, previous=True)
        atomic_json(self.out / 'last.json', dict(step=self.step, bytes=(self.out / 'last.pt').stat().st_size,
            utc=utc(), epoch=self.step / self.stream.updates_per_epoch, full_resume=True))
        atomic_json(self.out / 'latest.json', dict(checkpoint=str(self.out / 'last.pt'),
            previous=str(self.out / 'last.prev.pt'), step=self.step, full_resume=True,
            note='last.pt is the latest complete recovery checkpoint; best files are inference-only'))

    def restore(self, path):
        value = torch.load(path, map_location='cpu', weights_only=False)
        if value['config_sha256'] != self.identity or value['data_sha256'] != self.data_identity:
            raise ValueError('Checkpoint configuration/data mismatch')
        if not value.get('full_resume') or value['frozen'] != self.frozen:
            raise ValueError('Not a valid full KIT recovery checkpoint')
        self.model.load_state_dict(value['student'], strict=True)
        if self.ema:
            self.ema.load_state_dict(value['ema_mardm'], strict=True)
        self.optimizer.load_state_dict(value['optimizer'])
        self.scheduler.load_state_dict(value['scheduler'])
        self.stream.load_state_dict(value['stream'])
        self.selection, self.validation_history = value['selection'], value['validation_history']
        self.apg_selection = value.get('apg_selection')
        if ('direct_apg' in self.validation_samplers) != (self.apg_selection is not None):
            raise ValueError('Resume checkpoint has incompatible validation sampler state')
        self.step, self.elapsed_before = value['step'], value['elapsed_seconds']
        self.gradient_observed = value['gradient_observed']
        restore_rng(value['rng'])
        # Discard only uncommitted tail rows belonging to this new experiment.
        log_path = self.out / 'train.jsonl'
        if log_path.exists():
            lines = log_path.read_text(encoding='utf-8').splitlines()
            keep = [line for line in lines if json.loads(line)['step'] <= self.step]
            if len(keep) != len(lines):
                shutil.copy2(log_path, self.out / ('train_uncommitted_' + str(time.time_ns()) + '.jsonl'))
                log_path.write_text('\n'.join(keep) + ('\n' if keep else ''), encoding='utf-8')
        self.verify_frozen()

    def train_update(self):
        params = self.params
        update = self.step + 1
        if update < params['warmup_updates']:
            lr = params['lr'] * (update + 1) / (params['warmup_updates'] + 1)
            for group in self.optimizer.param_groups:
                group['lr'] = lr
        self.model.train()
        if self.stage == 'base':
            self.ae.eval()
            self.model.clip_model.eval()
        batch, indices = self.stream.next()
        self.optimizer.zero_grad(set_to_none=True)
        losses, token_forwards, input_sha = [], 0, None
        torch.cuda.synchronize()
        started = time.monotonic()
        if self.stage == 'tokenizer':
            input_sha = tensor_sha(batch)
            for chunk in batch.split(params['micro_batch']):
                motion = chunk.float().to(self.device)
                pred = self.ae(motion)
                loss_rec = F.smooth_l1_loss(pred, motion)
                loss_joint = F.smooth_l1_loss(pred[..., 4:64], motion[..., 4:64])
                loss = loss_rec + params['aux_loss_joints'] * loss_joint
                if not torch.isfinite(loss):
                    raise FloatingPointError('KIT AE loss is non-finite')
                (loss * (len(chunk) / params['batch_size'])).backward()
                losses.append(float(loss) * len(chunk) / params['batch_size'])
                token_forwards += len(chunk) * (motion.shape[1] // 4)
        else:
            captions, motion, lengths = batch
            input_sha = canonical_sha(dict(captions=list(captions), motion=tensor_sha(motion), lengths=tensor_sha(lengths)))
            counts = []
            handle = self.model.DiffMLPs.net.register_forward_pre_hook(
                lambda _module, inputs: counts.append(inputs[0].shape[0]))
            try:
                for offset in range(0, params['batch_size'], params['micro_batch']):
                    sl = slice(offset, offset + params['micro_batch'])
                    values = motion[sl].float().to(self.device)
                    lens = lengths[sl].long().to(self.device) // 4
                    with torch.no_grad():
                        latent, target = self.ae.encode_with_fsq(values)
                    loss = self.model.forward_loss(latent, target, captions[sl], lens)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('KIT native generation loss is non-finite')
                    (loss * (len(values) / params['batch_size'])).backward()
                    losses.append(float(loss) * len(values) / params['batch_size'])
            finally:
                handle.remove()
            token_forwards = sum(counts)
        grad = gradient_norm(self.parameters)
        if grad <= 0:
            raise RuntimeError('No finite nonzero gradient in KIT ' + self.stage)
        if self.stage == 'base':
            head_grad = gradient_norm(self.model.DiffMLPs.parameters())
            mar_grad = gradient_norm([p for name, p in self.model.named_parameters()
                                      if not name.startswith(('DiffMLPs.', 'clip_model.'))])
            self.gradient_observed['mar'] |= mar_grad > 0
            self.gradient_observed['head'] |= head_grad > 0
            if update >= 8 and (head_grad <= 0 or mar_grad <= 0):
                raise RuntimeError('KIT base MAR/head gradient disconnected after initial zero-output warmup')
        else:
            head_grad = mar_grad = None
            self.gradient_observed['ae'] = True
        if params['grad_clip'] is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters, params['grad_clip'], error_if_nonfinite=True)
        lr_used = self.optimizer.param_groups[0]['lr']
        self.optimizer.step()
        if self.stage == 'base' or update >= params['warmup_updates']:
            self.scheduler.step()
        if self.ema:
            ema_update(self.model, self.ema, params['ema'])
        torch.cuda.synchronize()
        self.step = update
        row = dict(step=update, epoch=(update - 1) // self.stream.updates_per_epoch + 1,
            epoch_update=(update - 1) % self.stream.updates_per_epoch + 1, loss=sum(losses),
            gradient=grad, mar_gradient=mar_grad, head_gradient=head_grad,
            lr=lr_used, next_lr=self.optimizer.param_groups[0]['lr'], input_sha256=input_sha,
            data_indices_sha256=canonical_sha(indices), data_epoch=self.stream.epoch,
            data_cursor=self.stream.cursor, student_token_forwards=token_forwards,
            motion_exposures=params['batch_size'], update_seconds=time.monotonic() - started,
            elapsed_seconds=self.elapsed_before + time.monotonic() - self.started,
            peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved(), utc=utc())
        self.log.write(json.dumps(row, allow_nan=False) + '\n')
        atomic_json(self.out / 'progress.json', row)
        if update <= 8 or update % 50 == 0:
            print(json.dumps(row, allow_nan=False), flush=True)
        return row

    @torch.no_grad()
    def ae_validation(self):
        before = rng_state()
        self.ae.eval()
        count, rec_total, joint_total, abs_total, squared_total, element_count = 0, 0., 0., 0., 0., 0
        coordinate_codes, per_axis = set(), [set() for _ in range(6)]
        for start in range(0, len(self.validation), self.params['validation_batch']):
            values = np.stack([self.validation[i] for i in range(start, min(len(self.validation), start + self.params['validation_batch']))])
            motion = torch.tensor(values, device=self.device, dtype=torch.float32)
            _, q = self.ae.encode_with_fsq(motion)
            pred = self.ae.decode_from_fsq(q)
            if pred.shape != motion.shape or not torch.isfinite(pred).all() or not torch.isfinite(q).all():
                raise FloatingPointError('KIT AE validation shape/finite failure')
            rec_total += float(F.smooth_l1_loss(pred, motion)) * len(motion)
            joint_total += float(F.smooth_l1_loss(pred[..., 4:64], motion[..., 4:64])) * len(motion)
            error = pred - motion
            abs_total += float(error.abs().sum())
            squared_total += float(error.square().sum())
            element_count += error.numel()
            count += len(motion)
            integers = torch.round(q * self.ae.fsq._half_levels).to(torch.int16).reshape(-1, 6).cpu().numpy()
            for coordinate in np.unique(integers, axis=0):
                coordinate_codes.add(tuple(map(int, coordinate)))
            for axis in range(6):
                per_axis[axis].update(map(int, np.unique(integers[:, axis])))
        if count == 0:
            raise ValueError('Empty KIT AE validation')
        restore_rng(before)
        rec, joint = rec_total / count, joint_total / count
        result = dict(step=self.step, epoch=self.step // self.stream.updates_per_epoch,
            selection_value=rec + self.params['aux_loss_joints'] * joint,
            normalized_smooth_l1=rec, normalized_local_xyz_smooth_l1=joint,
            normalized_l1=abs_total / element_count, normalized_rmse=math.sqrt(squared_total / element_count),
            windows=count, motions=len(self.validation.records), frames_per_window=self.params['window'],
            unique_integer_coordinate_tuples=len(coordinate_codes), observed_axis_integers=[sorted(x) for x in per_axis],
            nominal_levels=[8, 8, 8, 5, 5, 5], nominal_codebook_product=64000,
            usage_note='Exact integer-coordinate tuples; avoids relying on legacy get_indices for even levels',
            drop_last=False, order='all deterministic contiguous windows, inclusive endpoints', utc=utc())
        atomic_json(self.out / 'validation' / ('epoch_' + str(result['epoch']).zfill(3) + '.json'), result)
        return result

    def generation_validation(self, snapshot, sampler='native_cfg'):
        if sampler not in self.validation_samplers:
            raise ValueError('Sampler is not registered for this KIT model')
        directory = self.out / 'validation' / ('step_' + str(self.step).zfill(6)) / sampler
        # A killed trainer may leave its independent evaluator alive. Rejoin it;
        # never launch a second GPU evaluator for the same output directory.
        pid_file = directory / 'pid.json'
        if pid_file.exists() and not (directory / 'COMPLETE').exists() and os.name != 'nt':
            pid = read_json(pid_file)['pid']
            command_path = Path('/proc') / str(pid) / 'cmdline'
            while command_path.exists():
                try:
                    argv = command_path.read_bytes().split(b'\0')
                except (FileNotFoundError, ProcessLookupError):
                    break
                if str(directory).encode() not in argv or str(snapshot).encode() not in argv:
                    break
                atomic_json(self.out / 'status.json', dict(state='rejoining_validation', step=self.step,
                            evaluator_pid=pid, utc=utc(), pid=os.getpid()))
                time.sleep(2)
        if not (directory / 'COMPLETE').exists():
            directory.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, str(HERE / 'evaluate_control.py'), '--config', str(self.out / 'config.json'),
                       '--checkpoint', str(snapshot), '--output', str(directory), '--split', 'val',
                       '--seed', '3407', '--sampler', sampler, '--base-training', '--resume']
            atomic_json(directory / 'launch.json', dict(command=command, utc=utc(), checkpoint_sha256=sha(snapshot)))
            atomic_json(self.out / 'status.json', dict(state='validating', step=self.step,
                sampler=sampler, utc=utc(), pid=os.getpid()))
            with (directory / 'process.log').open('a', encoding='utf-8') as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                atomic_json(directory / 'pid.json', dict(pid=process.pid, command=command, utc=utc()))
                code = process.wait()
            if code != 0 or not (directory / 'COMPLETE').exists():
                raise RuntimeError('KIT validation failed; see ' + str(directory / 'process.log'))
        summary = read_json(directory / 'summary.json')
        completion = read_json(directory / 'COMPLETE')
        if completion['summary_sha256'] != sha(directory / 'summary.json') or completion['metrics_sha256'] != sha(directory / 'metrics.jsonl'):
            raise ValueError('KIT validation results are corrupt')
        if (completion['checkpoint_sha256'] != sha(snapshot) or summary['split'] != 'val'
                or summary['seeds'] != [3407] or summary['sampler'] != sampler
                or summary.get('mode') != 'quality' or summary.get('smoke_only', False)):
            raise ValueError('KIT validation identity or split mismatch')
        fid = summary['fid']['mean'] if isinstance(summary['fid'], dict) else summary['fid']
        if not math.isfinite(fid):
            raise FloatingPointError('KIT generation validation FID is not finite')
        return dict(step=self.step, epoch=self.step / self.stream.updates_per_epoch, sampler=sampler,
                    selection_value=float(fid), summary=str(directory / 'summary.json'),
                    summary_sha256=sha(directory / 'summary.json'), snapshot=str(snapshot),
                    snapshot_sha256=sha(snapshot), utc=utc())

    def validation_selections(self):
        values = [('reconstruction' if self.stage == 'tokenizer' else 'native_cfg', self.selection, 'best')]
        if self.apg_selection is not None:
            values.append(('direct_apg', self.apg_selection, 'best_apg'))
        return values

    def prune_validation_snapshots(self):
        """Bound only this run's temporary EMA copies; retain current best references."""
        directory = (self.out / 'snapshots').resolve()
        if not directory.is_dir() or directory.parent != self.out.resolve():
            return
        protected = set()
        for _sampler, selection, stem in self.validation_selections():
            if selection.get('best_snapshot'):
                protected.add(Path(selection['best_snapshot']).resolve())
            metadata = self.out / (stem + '.json')
            if metadata.exists():
                reference = read_json(metadata).get('validation', {}).get('snapshot')
                if reference:
                    protected.add(Path(reference).resolve())
        # Keep one previous epoch too, so the previous full recovery boundary can replay.
        oldest_retained_step = max(0, self.step - self.stream.updates_per_epoch)
        for candidate in directory.glob('ema_*.pt'):
            digits = candidate.stem.removeprefix('ema_')
            resolved = candidate.resolve()
            if (digits.isdigit() and int(digits) < oldest_retained_step
                    and resolved.parent == directory and not candidate.is_symlink()
                    and resolved not in protected):
                candidate.unlink()

    def validate(self):
        selections = self.validation_selections()
        if all(selection['last_validation_step'] == self.step for _, selection, _ in selections):
            return
        self.verify_frozen()
        snapshot = None
        if self.stage == 'base':
            snapshot = self.out / 'snapshots' / ('ema_' + str(self.step).zfill(6) + '.pt')
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            if not snapshot.exists():
                atomic_save(snapshot, self.inference())
        for sampler, selection, stem in selections:
            if selection['last_validation_step'] == self.step:
                continue
            result = self.ae_validation() if self.stage == 'tokenizer' else self.generation_validation(snapshot, sampler)
            result['sampler'] = sampler
            value = result['selection_value']
            if selection['best_value'] is None or value < selection['best_value']:
                payload = self.inference()
                payload['sampler'] = sampler
                atomic_save(self.out / (stem + '.pt'), payload)
                selection.update(best_value=value, best_step=self.step,
                    best_epoch=self.step / self.stream.updates_per_epoch,
                    best_snapshot=str(snapshot) if snapshot is not None else None)
                atomic_json(self.out / (stem + '.json'), dict(selection, last_validation_step=self.step,
                    checkpoint=str(self.out / (stem + '.pt')), sampler=sampler,
                    checkpoint_sha256=sha(self.out / (stem + '.pt')), validation=result, dataset='kit', stage=self.stage))
            selection['last_validation_step'] = self.step
            self.validation_history.append(result)
            selection_name = 'selection_apg.json' if sampler == 'direct_apg' else 'selection.json'
            atomic_json(self.out / selection_name, dict(selection, sampler=sampler))
            # Commit each sampler separately. An interrupted APG evaluation never redoes a committed CFG.
            self.checkpoint()
        self.prune_validation_snapshots()
        atomic_json(self.out / 'status.json', dict(state='running', step=self.step, utc=utc(), pid=os.getpid()))

    def run(self, stop_after=None, skip_validation=False):
        if skip_validation and not self.smoke:
            raise ValueError('Validation may be skipped only for explicitly labelled short tests')
        limit = min(stop_after, self.budget) if stop_after is not None else self.budget
        if limit < self.step or (limit == self.step and self.step != self.budget):
            raise ValueError('Requested stop boundary is not ahead of current step')
        # A checkpoint saved immediately before an interrupted validation is replay-safe.
        due = self.step > 0 and (self.step == self.budget or self.step % self.stream.updates_per_epoch == 0)
        if due and not skip_validation:
            self.validate()
        while self.step < limit:
            self.train_update()
            epoch_end = self.step % self.stream.updates_per_epoch == 0
            val_due = epoch_end or self.step == self.budget
            save_due = self.step % 500 == 0 or epoch_end or self.step == limit or STOP_REQUESTED
            if save_due or val_due:
                self.checkpoint()
            if val_due and not skip_validation:
                self.validate()
            if STOP_REQUESTED:
                break
        self.verify_frozen()
        self.checkpoint()
        done = self.step == self.budget and not self.smoke
        record = dict(utc=utc(), state='complete' if done else 'stopped_at_safe_boundary', step=self.step,
            total_updates=self.budget, epochs_complete=self.step // self.stream.updates_per_epoch,
            elapsed_seconds=self.elapsed_before + time.monotonic() - self.started,
            selection=self.selection, apg_selection=self.apg_selection,
            gradient_observed=self.gradient_observed, frozen_unchanged=True,
            peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved(),
            reason='fixed_epoch_budget' if done else 'short_test_or_requested_boundary',
            model_changed=module_sha(self.model) != self.initial_sha, smoke=self.smoke)
        if done:
            atomic_save(self.out / 'final.pt', self.inference())
            record.update(final_checkpoint_sha256=sha(self.out / 'final.pt'),
                          best_checkpoint_sha256=sha(self.out / 'best.pt'),
                          best_checkpoint=str(self.out / 'best.pt'), best_step=self.selection['best_step'])
            if self.apg_selection is not None:
                record.update(best_apg_checkpoint_sha256=sha(self.out / 'best_apg.pt'),
                    best_apg_checkpoint=str(self.out / 'best_apg.pt'), best_apg_step=self.apg_selection['best_step'])
            atomic_json(self.out / 'COMPLETE', record)
        else:
            atomic_json(self.out / ('SMOKE_COMPLETE' if self.smoke else 'STOPPED.json'), record)
        atomic_json(self.out / 'status.json', record)
        self.log.close()
        print(json.dumps(record, allow_nan=False), flush=True)
        return record


def request_stop(_signum, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--stage', choices=['tokenizer', 'base'], required=True)
    parser.add_argument('--output', '--run-dir', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after-updates', type=int)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--skip-validation', action='store_true')
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (output / 'run.lock').open('a+')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        trainer = KitTrainer(read_json(args.config), output, args.stage, args.resume, args.smoke)
        trainer.run(args.stop_after_updates, args.skip_validation)
    except Exception as exc:
        atomic_json(output / ('FAILED_' + str(time.time_ns()) + '.json'), dict(utc=utc(),
            error=repr(exc), traceback=traceback.format_exc(), checkpoint_preserved=(output / 'last.pt').exists()))
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    main()
