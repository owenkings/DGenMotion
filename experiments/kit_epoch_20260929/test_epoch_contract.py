"""Independent CPU tests of the real epoch/checkpoint/selection state machine.

Only numerical training and expensive GPU evaluation are replaced.  These tests
do not establish motion quality, GPU correctness, or real-data equivalence.
"""
import contextlib
import copy
import io
import json
import time
import unittest
import uuid
from pathlib import Path
from unittest import mock

import torch

import train_kit as target


class EvaluationInterrupted(RuntimeError):
    pass


class CpuHarness(target.KitTrainer):
    """Use production run, validate, checkpoint, inference, and restore methods."""

    def __init__(self, output, calls, dual=True, interrupt=None):
        target.seed_all(3407)
        self.out = Path(output)
        self.out.mkdir(parents=True, exist_ok=True)
        self.stage = 'base'
        self.smoke = False
        self.model_name = 'FSQ-MARDM-DiT-XL' if dual else 'FSQ-MARDM-SiT-XL'
        self.config = {
            'model': {'name': self.model_name},
            'paths': {'ae_checkpoint': 'fake-AE-for-state-machine-only'},
            'hashes': {'ae_checkpoint': 'fake-AE-hash'},
            'free_space_floor_bytes': 0,
            'evaluation': {'samplers': ['native_cfg', 'direct_apg'] if dual else ['native_cfg']},
        }
        self.params = {'ema': .9, 'validation_every': 'epoch', 'epochs': 3}
        self.identity = target.canonical_sha(self.config)
        self.data_identity = 'cpu-fixture-not-KIT'
        self.validation_samplers = self.config['evaluation']['samplers']
        self.stream = target.EpochStream(list(range(10)), 3, 3438)
        self.budget = 3 * self.stream.updates_per_epoch
        self.model = torch.nn.Linear(1, 1)
        self.ema = copy.deepcopy(self.model).requires_grad_(False)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=.0001)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 5, gamma=.5)
        self.step = 0
        self.elapsed_before = 0.
        self.started = time.monotonic()
        self.initial_sha = target.module_sha(self.model)
        self.frozen = {}
        self.source = {'test-source': 'fixed'}
        self.gradient_observed = {'mar': True, 'head': True, 'ae': False}
        self.selection = dict(best_value=None, best_step=None, best_epoch=None,
                              last_validation_step=-1, rule='minimum_fid_earlier_tie')
        self.apg_selection = copy.deepcopy(self.selection) if dual else None
        self.validation_history = []
        self.calls = calls
        self.interrupt = interrupt
        self.log = (self.out / 'train.jsonl').open('a', encoding='utf8')

    def verify_frozen(self):
        return True

    def train_update(self):
        # Refuse to enter the next epoch until all prior evaluations committed.
        if self.step and self.step % self.stream.updates_per_epoch == 0:
            assert self.selection['last_validation_step'] == self.step
            if self.apg_selection is not None:
                assert self.apg_selection['last_validation_step'] == self.step
        batch, indices = self.stream.next()
        self.optimizer.zero_grad()
        x = batch.float().reshape(-1, 1)
        loss = (self.model(x) - torch.randn_like(x)).square().mean()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        target.ema_update(self.model, self.ema, self.params['ema'])
        self.step += 1
        self.calls.append(('train', self.step, tuple(indices)))

    def generation_validation(self, snapshot, sampler='native_cfg'):
        self.calls.append(('validate', self.step, sampler))
        if self.interrupt == (self.step, sampler):
            self.interrupt = None
            raise EvaluationInterrupted('second sampler interrupted')
        # Independent optima; a late tie must not replace the earlier best.
        scores = {'native_cfg': {3: 3., 6: 4., 9: 3.},
                  'direct_apg': {3: 9., 6: 5., 9: 5.}}
        return dict(step=self.step, epoch=self.step / 3, sampler=sampler,
                    selection_value=scores[sampler][self.step],
                    snapshot=str(snapshot), snapshot_sha256=target.sha(snapshot),
                    summary='cpu-fixture-only', summary_sha256='not-real-metrics', utc=target.utc())


class EpochContract(unittest.TestCase):
    def setUp(self):
        # Python 3.13's Windows mode-0700 temporary directories can deny the
        # sandbox token. Keep small, explicitly labelled CPU fixtures locally.
        self.root = Path(__file__).resolve().parent / 'checks' / ('epoch_contract_' + uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch.object(target.torch.cuda, 'max_memory_allocated', return_value=0))
        self.stack.enter_context(mock.patch.object(target.torch.cuda, 'max_memory_reserved', return_value=0))
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        target.STOP_REQUESTED = False

    def harness(self, name, calls, **kwargs):
        h = CpuHarness(self.root / name, calls, **kwargs)
        self.addCleanup(lambda: h.log.close() if not h.log.closed else None)
        return h

    def test_stream_resume_on_last_batch_and_next_epoch(self):
        for cut in (2, 3, 4, 6):
            stream = target.EpochStream(list(range(10)), 3, 71)
            for _ in range(cut):
                stream.next()
            state = stream.state_dict()
            restored = target.EpochStream(list(range(10)), 3, 999)
            restored.load_state_dict(state)
            for _ in range(7):
                _, expected = stream.next()
                _, actual = restored.next()
                self.assertEqual(expected, actual)
                self.assertEqual(stream.epoch, restored.epoch)
                self.assertEqual(stream.cursor, restored.cursor)

    def test_every_epoch_and_independent_sampler_best(self):
        calls = []
        h = self.harness('complete', calls)
        h.run()
        self.assertEqual([x for x in calls if x[0] == 'validate'],
                         [('validate', s, k) for s in (3, 6, 9)
                          for k in ('native_cfg', 'direct_apg')])
        self.assertEqual(h.selection['best_step'], 3)
        self.assertEqual(h.apg_selection['best_step'], 6)
        self.assertEqual(h.selection['last_validation_step'], 9)
        self.assertEqual(h.apg_selection['last_validation_step'], 9)
        self.assertEqual(len(h.validation_history), 6)
        self.assertTrue((h.out / 'COMPLETE').is_file())
        for filename, expected in [('last.pt', 9), ('final.pt', 9), ('best.pt', 3), ('best_apg.pt', 6)]:
            payload = torch.load(h.out / filename, weights_only=False)
            self.assertEqual(payload['step'], expected)

    def test_resume_after_cfg_before_apg_does_not_skip_or_repeat_cfg(self):
        baseline_calls = []
        baseline = self.harness('baseline', baseline_calls)
        baseline.run()
        calls = []
        interrupted = self.harness('interrupted', calls, interrupt=(3, 'direct_apg'))
        with self.assertRaises(EvaluationInterrupted):
            interrupted.run()
        self.assertEqual(interrupted.step, 3)
        self.assertFalse((interrupted.out / 'COMPLETE').exists())
        interrupted.log.close()
        saved = torch.load(interrupted.out / 'last.pt', weights_only=False)
        self.assertEqual(saved['selection']['last_validation_step'], 3)
        self.assertLess(saved['apg_selection']['last_validation_step'], 3)
        resumed = self.harness('interrupted', calls)
        resumed.restore(resumed.out / 'last.pt')
        resumed.run()
        self.assertEqual(calls.count(('validate', 3, 'native_cfg')), 1)
        self.assertEqual(calls.count(('validate', 3, 'direct_apg')), 2)
        self.assertEqual([x for x in calls if x[0] == 'train'],
                         [x for x in baseline_calls if x[0] == 'train'])
        for current, expected in ((resumed.selection, baseline.selection),
                                  (resumed.apg_selection, baseline.apg_selection)):
            self.assertEqual({k: v for k, v in current.items() if k != 'best_snapshot'},
                             {k: v for k, v in expected.items() if k != 'best_snapshot'})
            self.assertEqual(Path(current['best_snapshot']).name, Path(expected['best_snapshot']).name)
        self.assertEqual(target.module_sha(resumed.model), target.module_sha(baseline.model))
        self.assertEqual(target.module_sha(resumed.ema), target.module_sha(baseline.ema))
        self.assertEqual(len(resumed.validation_history), 6)

    def test_single_sampler_branch_has_no_apg_artifact(self):
        calls = []
        h = self.harness('sit', calls, dual=False)
        h.run()
        self.assertEqual([x for x in calls if x[0] == 'validate'],
                         [('validate', s, 'native_cfg') for s in (3, 6, 9)])
        self.assertIsNone(h.apg_selection)
        self.assertFalse((h.out / 'best_apg.pt').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
