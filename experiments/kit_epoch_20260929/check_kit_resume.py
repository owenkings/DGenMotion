"""Compare two real KIT training checkpoints, excluding only wall-clock metadata."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_kit import atomic_json, sha, utc


def require_equal(a, b, label):
    if torch.is_tensor(a):
        assert torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape, label + ': tensor metadata'
        assert torch.equal(a, b), label + ': tensor differs'
    elif isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray) and np.array_equal(a, b), label + ': array differs'
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), label + ': keys differ'
        for key in a:
            require_equal(a[key], b[key], label + '.' + str(key))
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b), label + ': sequence metadata'
        for index, (left, right) in enumerate(zip(a, b)):
            require_equal(left, right, label + '[' + str(index) + ']')
    else:
        assert a == b, label + ': scalar differs'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--continuous', required=True)
    parser.add_argument('--resumed', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    left_path, right_path = Path(args.continuous), Path(args.resumed)
    left = torch.load(left_path, map_location='cpu', weights_only=False)
    right = torch.load(right_path, map_location='cpu', weights_only=False)
    keys = ['dataset', 'stage', 'step', 'student', 'optimizer', 'scheduler', 'rng', 'stream',
            'data_sha256', 'frozen', 'gradient_observed', 'selection']
    keys += ['ae'] if left['stage'] == 'tokenizer' else ['ema_mardm']
    for key in keys:
        require_equal(left[key], right[key], key)
    train_a = [json.loads(x) for x in (left_path.parent / 'train.jsonl').read_text().splitlines()]
    train_b = [json.loads(x) for x in (right_path.parent / 'train.jsonl').read_text().splitlines()]
    assert len(train_a) == len(train_b) == left['step']
    trajectory_keys = ['step', 'epoch', 'epoch_update', 'loss', 'gradient', 'mar_gradient', 'head_gradient',
                       'lr', 'next_lr', 'input_sha256', 'data_indices_sha256', 'data_epoch', 'data_cursor',
                       'student_token_forwards', 'motion_exposures']
    for a, b in zip(train_a, train_b):
        for key in trajectory_keys:
            require_equal(a[key], b[key], 'train[' + str(a['step']) + '].' + key)
    result = dict(PASS=True, utc=utc(), stage=left['stage'], step=left['step'],
        equal_checkpoint_fields=keys, equal_training_fields=trajectory_keys,
        continuous_sha256=sha(left_path), resumed_sha256=sha(right_path),
        real_data_identity=left['data_sha256'], gradient_observed=left['gradient_observed'],
        note='Different wall time/file bytes allowed; tensors, optimizer, scheduler, RNG, data cursor and subsequent losses must match exactly')
    atomic_json(args.output, result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
