"""Regression tests for stopped workers, config mistakes and resumable budgets."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import test_lightning as fixtures
FakeWorker = fixtures.FakeWorker
import lightning_run as lr


class ReadyTests(unittest.TestCase):
    def config(self):
        return fixtures.LightningTests().config()

    def invoke(self, root, worker, config=None, extra=()):
        path = root / 'launch.json'
        path.write_text(json.dumps(config or self.config()))
        args = ['lightning_run.py', '--config', str(path), '--state', str(root / 'state.json'), *extra]
        def git(command, **kwargs):
            return 'abc123' if 'rev-parse' in command else ''
        with patch.object(sys, 'argv', args), patch('lightning_sdk.Studio', return_value=worker) as studio, \
             patch('lightning_run.subprocess.check_output', side_effect=git):
            lr.main()
        return studio

    def test_already_stopped_cleanup_is_idempotent(self):
        worker = FakeWorker()
        lr.stop_worker(worker)
        self.assertEqual(worker.stopped, 0)

    def test_shutdown_race_with_remote_guard(self):
        worker = FakeWorker()
        worker.status = 'Running'
        def raced():
            worker.status = 'Stopped'
            raise RuntimeError('Already stopped by guard')
        worker.stop = raced
        with patch('lightning_run.time.sleep'):
            lr.stop_worker(worker)

    def test_wait_for_shutdown_before_next_start(self):
        class Worker:
            calls = 0
            @property
            def status(self):
                self.calls += 1
                return 'Running' if self.calls == 1 else 'Stopping' if self.calls == 2 else 'Stopped'
            def stop(self):
                pass
        worker = Worker()
        with patch('lightning_run.time.sleep'):
            lr.stop_worker(worker)
        self.assertEqual(worker.calls, 3)

    def test_gpu_configuration_rejected_before_any_studio_access(self):
        with tempfile.TemporaryDirectory() as d:
            c = self.config()
            c['embedding_machine'] = 'A100_40GB'
            with patch('lightning_sdk.Studio') as studio:
                with self.assertRaisesRegex(ValueError, 'exactly 2'):
                    self.invoke(Path(d), FakeWorker(), c, ['--run'])
                studio.assert_not_called()
        c = self.config()
        c['gpu_machine'] = 'TYPO_GPU'
        with self.assertRaisesRegex(ValueError, 'Unknown SDK machine'):
            lr.resolve_machines(c)

    def test_plan_and_status_do_not_access_cloud(self):
        with tempfile.TemporaryDirectory() as d:
            for flag in ('--plan', '--status'):
                studio = self.invoke(Path(d), FakeWorker(), extra=[flag])
                studio.assert_not_called()

    def test_start_failure_does_not_mask_original_error(self):
        worker = FakeWorker()
        worker.start = lambda **kw: (_ for _ in ()).throw(RuntimeError('No GPU capacity'))
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with self.assertRaisesRegex(RuntimeError, 'No GPU capacity'):
                self.invoke(root, worker, extra=['--run'])
            state = json.loads((root / 'state.json').read_text())
            self.assertIsNone(state['active'])
            self.assertEqual(state['completed'], [])
            self.assertEqual(worker.stopped, 0)

    def test_worker_source_edits_are_rejected(self):
        worker = FakeWorker()
        original = worker.run
        worker.run = lambda command: 'pipeline.py\n' if 'diff --name-only' in command else original(command)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, 'uncommitted source'):
                self.invoke(Path(d), worker, extra=['--run'])
            self.assertEqual(worker.stopped, 1)

    def test_shutdown_failure_keeps_active_state_for_recovery(self):
        worker = FakeWorker()
        worker.stop = lambda: (_ for _ in ()).throw(RuntimeError('API unavailable'))
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with self.assertRaisesRegex(RuntimeError, 'API unavailable'):
                self.invoke(root, worker, extra=['--run'])
            state = json.loads((root / 'state.json').read_text())
            self.assertEqual(state['active']['phase'], 'prepare')
            self.assertEqual(state['completed'], ['prepare'])
            resumed = FakeWorker()
            self.invoke(root, resumed, extra=['--run'])
            self.assertEqual(resumed.started, ['A100_40GB', 'A100_40GB_X_2', 'DATA_PREP'])
            self.assertIsNone(json.loads((root / 'state.json').read_text())['active'])

    def test_increased_limits_resume_same_run_without_erasing_charges(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with self.assertRaises(RuntimeError):
                self.invoke(root, FakeWorker(fail=True), extra=['--run'])
            state = json.loads((root / 'state.json').read_text())
            state['estimated_compute_usd'] = 35.
            (root / 'state.json').write_text(json.dumps(state))
            c = self.config()
            c.update(max_compute_usd=50., max_hours=24.)
            c['phase_hours']['prepare'] = 6.
            self.invoke(root, FakeWorker(), c, ['--run'])
            final = json.loads((root / 'state.json').read_text())
            self.assertGreaterEqual(final['estimated_compute_usd'], 35.)
            self.assertEqual(final['completed'], lr.PHASES)

    def test_offline_time_does_not_consume_inactive_run_budget(self):
        state = dict(estimated_compute_usd=0., elapsed_worker_seconds=100.,
                     started_at=time.time() - 86400, phase_seconds={'prepare': 50.})
        self.assertGreater(lr.allowance(self.config(), state, 1.48, 3, 'prepare'), 0)
        state['phase_seconds']['prepare'] = 3 * 3600
        self.assertLess(lr.allowance(self.config(), state, 1.48, 3, 'prepare'), 0)

    def test_self_controller_refused(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, LIGHTNING_CLOUD_SPACE_ID='separate-worker'):
            with self.assertRaisesRegex(ValueError, 'OUTSIDE'):
                self.invoke(Path(d), FakeWorker(), extra=['--run'])

    def test_lock_released_after_exception(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'controller.lock'
            with self.assertRaises(RuntimeError):
                with lr.controller_lock(path):
                    with lr.controller_lock(path):
                        pass
            with lr.controller_lock(path):
                pass

    def test_configuration_generator_small_production_profile(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'local_sample.json'
            command = [sys.executable, str(ROOT / 'scripts/configure_lightning.py'), '--teamspace', 'owner/team',
                       '--worker', 'worker', '--input', '/teamspace/er_sample.zip', '--sample', '--output', str(out)]
            subprocess.run(command, check=True, capture_output=True, text=True)
            c = json.loads(out.read_text())
            m = json.loads((out.parent / c['model_config']).read_text())
            lr.validate(c)
            self.assertTrue(c['sample'])
            self.assertEqual(c['embedding_gpus'], 2)
            self.assertEqual(m['encoder']['device'], 'cuda')
            self.assertEqual(m['encoder']['encoder_mode'], 'lora')
            self.assertEqual(m['matcher']['device_type'], 'cpu')
            self.assertEqual(m['preparation']['workers'], 2)
            self.assertIn('paraphrase-multilingual', m['encoder']['model_id'])
            again = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(again.returncode, 0)
            self.assertIn('already exists', again.stderr)

    def test_invalid_numeric_and_transient_path(self):
        for key, value in [('max_compute_usd', float('nan')), ('embedding_gpus', True), ('input', '/tmp/data.zip'),
                           ('work', '/teamspace/../tmp/work')]:
            c = self.config()
            c[key] = value
            with self.assertRaises(ValueError):
                lr.validate(c)


if __name__ == '__main__':
    unittest.main()
