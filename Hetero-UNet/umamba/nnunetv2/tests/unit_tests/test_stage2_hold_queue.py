"""Queue retirement safeguards. Process signals are mocked in these unit tests."""
import importlib.util
from pathlib import Path
import signal
import unittest
from unittest.mock import patch

MODULE = Path(__file__).resolve().parents[4] / 'scripts/hold_stage2_queue.py'
SPEC = importlib.util.spec_from_file_location('hold_stage2_tests', MODULE)
hold = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hold)


class TestHoldQueue(unittest.TestCase):
    def setUp(self):
        # Windows has no SIGSTOP/SIGKILL; unit tests mock Linux constants/signals.
        for name, value in (('SIGSTOP', 19), ('SIGKILL', 9)):
            mocked = patch.object(signal, name, value, create=True)
            mocked.start()
            self.addCleanup(mocked.stop)
        self.queue = {'pid': 50, 'ppid': 40, 'start_ticks': 100, 'state': 'T',
                      'argv': ['bash', '/queue.sh']}
        self.trainer = {'pid': 51, 'ppid': 50, 'start_ticks': 200, 'state': 'R',
                        'argv': ['python', 'scripts/run_stage2.py', '--model', 'F']}
        self.plan = {'hostname': 'test', 'queue_process': self.queue,
                     'trainer_process': self.trainer, 'original_queue_script': '/queue.sh',
                     'F_manifest': '/F.json', 'F_only_queue_file': '/F_queue.json', 'training_commit': 'fixed'}
        self.manifest = {'status': 'completed', 'arguments': {'model': 'F', 'smoke': False},
                         'protocol': {'num_epochs': 1000}, 'git': {'commit': 'fixed'}}

    def test_hold_signals_only_queue_and_not_F(self):
        with patch.object(hold, 'identity', return_value=self.trainer), \
                patch.object(hold, 'signal_queue') as send:
            hold.hold(self.plan)
            send.assert_called_once_with(self.plan, signal.SIGSTOP)

    def test_wrong_trainer_is_not_held(self):
        actual = {**self.trainer, 'argv': ['python', 'scripts/run_stage2.py', '--model', 'C']}
        self.plan['trainer_process'] = actual
        with patch.object(hold, 'identity', return_value=actual), patch.object(hold, 'signal_queue') as send:
            with self.assertRaisesRegex(RuntimeError, 'not F'):
                hold.hold(self.plan)
            send.assert_not_called()

    def test_completed_manifest_still_waits_for_F_process(self):
        with patch.object(hold.socket, 'gethostname', return_value='test'), \
                patch.object(hold.Path, 'read_text', return_value=hold.json.dumps(self.manifest)), \
                patch.object(hold, 'identity', return_value=self.trainer), \
                patch.object(hold, 'write_json'), patch.object(hold, 'signal_queue') as send:
            self.assertEqual(hold.step(self.plan), 'waiting_for_F')
            send.assert_not_called()

    def test_only_retire_held_queue_after_F_exits(self):
        with patch.object(hold.socket, 'gethostname', return_value='test'), \
                patch.object(hold.Path, 'read_text', return_value=hold.json.dumps(self.manifest)), \
                patch.object(hold, 'identity', side_effect=[None, self.queue]), \
                patch.object(hold, 'signal_queue') as send:
            self.assertEqual(hold.step(self.plan), 'retiring_completed_queue')
            send.assert_called_once_with(self.plan, signal.SIGKILL, require_stopped=True)

    def test_PID_reuse_blocks_retirement(self):
        wrong = {**self.queue, 'start_ticks': 999}
        with patch.object(hold.socket, 'gethostname', return_value='test'), \
                patch.object(hold.Path, 'read_text', return_value=hold.json.dumps(self.manifest)), \
                patch.object(hold, 'identity', side_effect=[None, wrong]), \
                patch.object(hold, 'signal_queue') as send:
            with self.assertRaisesRegex(RuntimeError, 'reused'):
                hold.step(self.plan)
            send.assert_not_called()

    def test_reused_or_running_queue_never_receives_kill(self):
        for actual in ({**self.queue, 'start_ticks': 999}, {**self.queue, 'state': 'S'}):
            with patch.object(hold.socket, 'gethostname', return_value='test'), \
                    patch.object(hold.os, 'pidfd_open', return_value=123, create=True), \
                    patch.object(hold.os, 'close'), patch.object(hold, 'identity', return_value=actual), \
                    patch.object(hold.signal, 'pidfd_send_signal', create=True) as send:
                with self.assertRaises(RuntimeError):
                    hold.signal_queue(self.plan, signal.SIGKILL, require_stopped=True)
                send.assert_not_called()


if __name__ == '__main__':
    unittest.main()
