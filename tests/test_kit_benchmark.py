"""CPU regression checks for published KIT benchmarking; no GPU/data access."""
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import sys
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / 'experiments' / 'kit_epoch_20260929'
sys.path.insert(0, str(EXPERIMENT))
import benchmark_control as target


class ProcFixture:
    def __init__(self, pid, argv, cwd):
        self.name, self.argv, self.cwd = str(pid), argv, cwd

    def __truediv__(self, name):
        if name == 'cmdline':
            return types.SimpleNamespace(read_bytes=lambda: b'\0'.join(os.fsencode(arg) for arg in self.argv) + b'\0')
        if name == 'cwd':
            return types.SimpleNamespace(resolve=lambda strict=False: self.cwd)
        raise AssertionError(name)


class WorkerDetection(unittest.TestCase):
    def test_actual_entrypoints_across_checkout_and_code_snapshot(self):
        snapshot = ROOT.parent / 'isolated source with spaces'
        current = os.getpid()
        processes = [
            ProcFixture(current, ['python', str(EXPERIMENT / 'benchmark_control.py')], ROOT),
            ProcFixture(current + 1, ['python3.10', '-u', str(EXPERIMENT / 'train_kit.py'), '--config', 'cfg.json'], ROOT),
            ProcFixture(current + 2, ['python', '-W', 'ignore', '-X', 'dev', 'train_MARDM.py'], ROOT),
            ProcFixture(current + 3, ['python', '-m', 'experiments.kit_epoch_20260929.evaluate_control'], ROOT),
            ProcFixture(current + 4, ['python', 'train_AE.py'], snapshot),
            # An unrelated repository with the same script basename is not ours.
            ProcFixture(current + 5, ['python', 'train_kit.py'], ROOT.parent / 'other-project'),
            # Do not interpret command strings or arbitrary later arguments as entrypoints.
            ProcFixture(current + 6, ['python', '-c', 'print(1)', str(EXPERIMENT / 'train_kit.py')], ROOT),
            ProcFixture(current + 7, ['python', 'inspect.py', '--output', str(EXPERIMENT / 'train_kit.py')], ROOT),
            ProcFixture(current + 8, ['cat', str(EXPERIMENT / 'train_kit.py')], ROOT),
        ]
        proc_root = types.SimpleNamespace(glob=lambda pattern: processes)
        workers = target.own_gpu_workers(snapshot, proc_root)
        self.assertEqual({row['pid'] for row in workers}, {current + n for n in (1, 2, 3, 4)})
        self.assertEqual(workers[0]['argv'], processes[1].argv)
        self.assertEqual(Path(workers[1]['entrypoint']), ROOT / 'train_MARDM.py')

    def test_process_exit_and_incomplete_python_options_are_harmless(self):
        dead = mock.MagicMock()
        dead.name = str(os.getpid() + 10)
        dead.__truediv__.side_effect = FileNotFoundError
        workers = target.own_gpu_workers(proc_root=types.SimpleNamespace(glob=lambda _: [dead]))
        self.assertEqual(workers, [])
        for argv in (['python', '-W'], ['python', '-m'], ['python', '-cprint(1)', 'train_kit.py']):
            self.assertIsNone(target.python_entrypoint(argv, ROOT))


class NfeReporting(unittest.TestCase):
    @staticmethod
    def row(nfe, latency, token_forwards):
        return dict(batch_size=1, requested_frames=60, actual_returned_frames=60,
                    latency_seconds=latency, peak_allocated_bytes=1, peak_reserved_bytes=2,
                    mar_calls=36, mar_token_forwards=540,
                    net_calls=nfe, net_token_forwards=token_forwards)

    def summarize(self, rows):
        manifest = dict(inputs=dict(batch_sizes=[1], lengths=[60]), dataset='kit',
                        sampler='native_cfg', checkpoint_sha256='fixture', warmups=3,
                        smoke_only=False, timing_boundary={})
        buffer = io.StringIO()
        with mock.patch.object(target, 'write_json') as write, mock.patch.object(
                Path, 'open', return_value=contextlib.nullcontext(buffer)):
            target.write_summaries(Path('unused'), rows, manifest)
        result = write.call_args.args[1]['configurations'][0]
        return result, list(csv.DictReader(io.StringIO(buffer.getvalue())))[0]

    def test_variable_adaptive_work_preserves_distribution_in_json_and_csv(self):
        report, csv_row = self.summarize([
            self.row(120, 1., 240), self.row(240, 3., 960), self.row(180, 2., 540)])
        self.assertEqual(report['head_nfe_values'], [120, 240, 180])
        self.assertIsNone(report['head_nfe'])
        self.assertEqual((report['head_nfe_min'], report['head_nfe_max']), (120, 240))
        self.assertEqual(report['head_nfe_mean'], 180.)
        self.assertEqual(report['head_nfe_median'], 180.)
        self.assertAlmostEqual(report['head_nfe_p95'], 234.)
        self.assertIsNone(report['head_token_forwards'])
        self.assertEqual(report['head_token_forwards_values'], [240, 960, 540])
        self.assertEqual(report['median_seconds'], 2.)
        self.assertEqual(json.loads(csv_row['head_nfe_values']), [120, 240, 180])
        self.assertEqual(csv_row['head_nfe'], '')

    def test_fixed_work_keeps_backward_compatible_scalar(self):
        report, _ = self.summarize([self.row(1800, 1., 3600), self.row(1800, 2., 3600)])
        self.assertEqual(report['head_nfe'], 1800)
        self.assertEqual(report['head_nfe_values'], [1800, 1800])
        self.assertEqual(report['head_token_forwards'], 3600)


class AlternateTiming(unittest.TestCase):
    def test_requires_explicit_fixed_work_not_just_matching_architecture(self):
        for config in ({'adaptive_nfe': True}, {}):
            controller = types.SimpleNamespace(config=config)
            with self.assertRaisesRegex(ValueError, 'fixed-step'):
                target.require_fixed_alternate(controller, Path('alternate.pt'))
            target.require_fixed_alternate(controller, None)
        target.require_fixed_alternate(types.SimpleNamespace(config={'adaptive_nfe': False}), Path('alternate.pt'))

    def test_cli_rejects_adaptive_alternate_before_any_timing(self):
        # Exercise main itself so a disconnected guard cannot satisfy the test.
        args = types.SimpleNamespace(config=Path('config.json'), output=Path('unused'),
            checkpoint=None, alternate_checkpoint=Path('alternate.pt'), smoke=False,
            sampler='native_cfg', resume=False)
        controller = mock.Mock(config={'adaptive_nfe': True})
        fake_fcntl = types.SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=mock.Mock())
        fake_lock = mock.Mock()
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.dict(sys.modules, {'fcntl': fake_fcntl}))
            stack.enter_context(mock.patch.object(Path, 'mkdir'))
            stack.enter_context(mock.patch.object(Path, 'open', return_value=fake_lock))
            stack.enter_context(mock.patch.object(target, 'read_config', return_value=({'paths': {'code_root': str(ROOT)}}, {})))
            stack.enter_context(mock.patch.object(target, 'own_gpu_workers', return_value=[]))
            stack.enter_context(mock.patch.object(target, 'configure_precision'))
            stack.enter_context(mock.patch.object(target, 'load_stats', return_value=({}, {})))
            stack.enter_context(mock.patch.object(target, 'build_dataset', return_value=([], {})))
            stack.enter_context(mock.patch.object(target, 'benchmark_inputs', return_value={}))
            stack.enter_context(mock.patch.object(target, 'load_models', return_value=(None, None, {}, {}, controller)))
            timed = stack.enter_context(mock.patch.object(target, 'timed_call'))
            writes = stack.enter_context(mock.patch.object(target, 'write_json'))
            with self.assertRaisesRegex(ValueError, 'fixed-step'):
                target.main(args)
        timed.assert_not_called()
        controller.remove.assert_called_once()
        fake_lock.close.assert_called_once()
        self.assertTrue(any(Path(call.args[0]).name.startswith('FAILED_') for call in writes.call_args_list))


if __name__ == '__main__':
    unittest.main()
