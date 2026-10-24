"""Regression checks: completed results and verified backups must precede shutdown."""
import csv
import importlib.util
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[4] / 'scripts/finalize_stage2_evaluation_20261024.py'
spec = importlib.util.spec_from_file_location('evaluation_finalizer', SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class EvaluationShutdownTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name)
        self.names = [f'case_{i:04}.nii.gz' for i in range(50)]
        protocol = {'cases': [{'id': n[:-7]} for n in self.names]}
        module.write(self.work / 'protocol.json', protocol)
        state = {'sequence': ['F', 'C'], 'status': 'completed_backup_pending', 'current_model': None,
                 'models': {m: {'evaluation': 'completed'} for m in ['F', 'C']}}
        module.write(self.work / 'state.json', state)
        (self.work / 'baseline_A').mkdir()
        (self.work / 'metrics_code').mkdir()
        (self.work / 'export_equivalence.log').write_text('fixture passed')
        (self.work / 'runtime/nnunetv2/inference').mkdir(parents=True)
        (self.work / 'runtime/nnunetv2/inference/export_prediction.py').write_text('# fixture\n')
        for model, value in [('A', 0.8), ('F', 0.81), ('C', 0.82)]:
            folder = self.work / ('baseline_A' if model == 'A' else model)
            folder.mkdir(exist_ok=True)
            for metric in ['DSC', 'NSD']:
                path = folder / (metric + '.csv' if model == 'A' else model + '_' + metric + '.csv')
                with path.open('w', newline='') as stream:
                    writer = csv.writer(stream)
                    writer.writerow(['Name'] + [o + '_' + metric for o in module.ORGANS])
                    writer.writerows([[name] + [value] * 13 for name in self.names])
            if model == 'A':
                continue
            module.write(folder / 'summary.json', {'DSC_percent': value * 100, 'NSD_percent': value * 100})
            predictions = folder / 'predictions'
            predictions.mkdir()
            rows = []
            for name in self.names:
                p = predictions / name
                p.write_bytes(b'hashed prediction fixture')
                rows.append({'case': name[:-7], 'bytes': p.stat().st_size,
                             'sha256': module.sha(p), 'geometry_matches_gt': True})
            module.write(folder / 'prediction_validation.json', {'status': 'passed', 'cases': rows})
        executable = self.work / 'shutdown_fixture'
        executable.write_text('never execute this fixture')
        self.config_path = self.work / 'evaluation_shutdown_config.json'
        self.config = {'work_dir': str(self.work), 'hostname': socket.gethostname(),
                       'shutdown_authorized': True, 'protocol_sha256': module.sha(self.work / 'protocol.json'),
                       'shutdown_executable': str(executable), 'shutdown_sha256': module.sha(executable),
                       'oss_prefix': 'oss://test/', 'user_authorization': 'test authorization'}
        module.write(self.config_path, self.config)
        self.helper = module.Finalizer(self.config_path, armed=True)

    def verified_upload(self, path):
        proof = {'sha256': module.sha(path), 'roundtrip_passed': True}
        self.helper.state['verified'][path.name] = proof
        return proof

    def test_running_or_failed_queue_is_not_ready(self):
        state = module.read(self.work / 'state.json')
        for status in ['predicting', 'evaluating', 'failed', 'backing_up']:
            state['status'] = status
            self.assertFalse(module.ready(state))

    def test_missing_model_is_not_ready(self):
        state = module.read(self.work / 'state.json')
        del state['models']['C']
        self.assertFalse(module.ready(state))

    def test_incomplete_or_nonfinite_table_rejected(self):
        path = self.work / 'F/F_DSC.csv'
        original = path.read_text()
        path.write_text('\n'.join(original.splitlines()[:-1]) + '\n')
        with self.assertRaises(RuntimeError):
            module.table(path, 'DSC', self.names)
        path.write_text(original.replace('0.81', 'nan', 1))
        with self.assertRaises(RuntimeError):
            module.table(path, 'DSC', self.names)

    def test_cancelled_or_changed_protocol_blocks_shutdown(self):
        (self.work / 'DISABLE_EVALUATION_SHUTDOWN').touch()
        with self.assertRaises(RuntimeError):
            self.helper.guard()
        (self.work / 'DISABLE_EVALUATION_SHUTDOWN').unlink()
        (self.work / 'protocol.json').write_text('{}')
        with self.assertRaises(RuntimeError):
            self.helper.guard()

    def test_explicit_authorization_required(self):
        self.config['shutdown_authorized'] = False
        module.write(self.config_path, self.config)
        with self.assertRaises(RuntimeError):
            module.Finalizer(self.config_path, armed=True)

    def test_prediction_tampering_blocks_shutdown(self):
        (self.work / 'C/predictions' / self.names[0]).write_bytes(b'tampered')
        with patch.object(module.subprocess, 'check_output', return_value=''), \
                patch.object(module.subprocess, 'run') as invocation:
            with self.assertRaises(RuntimeError):
                self.helper.finish()
            invocation.assert_not_called()

    def test_backup_failure_blocks_shutdown(self):
        with patch.object(module.subprocess, 'check_output', return_value=''), \
                patch.object(module.subprocess, 'run') as invocation, \
                patch.object(self.helper, 'upload_verify', side_effect=RuntimeError('OSS failed')):
            with self.assertRaises(RuntimeError):
                self.helper.finish()
            invocation.assert_not_called()

    def test_success_summarizes_and_verifies_four_archives_before_shutdown(self):
        with patch.object(module.subprocess, 'check_output', return_value=''), \
                patch.object(module.subprocess, 'run') as invocation, \
                patch.object(self.helper, 'upload_verify', side_effect=self.verified_upload) as upload:
            self.helper.finish()
            self.assertEqual(upload.call_count, 4)
            invocation.assert_called_once()
            self.assertEqual(invocation.call_args.args[0], [self.config['shutdown_executable']])
        result = module.read(self.work / 'final_comparison.json')
        self.assertAlmostEqual(result['DSC']['C_minus_F_pp'], 1.0)
        self.assertTrue((self.work / 'EVALUATION_RESULTS.md').is_file())

    def test_changed_shutdown_executable_blocks_invocation(self):
        Path(self.config['shutdown_executable']).write_text('changed')
        with patch.object(module.subprocess, 'check_output', return_value=''), \
                patch.object(module.subprocess, 'run') as invocation, \
                patch.object(self.helper, 'upload_verify', side_effect=self.verified_upload):
            with self.assertRaises(RuntimeError):
                self.helper.finish()
            invocation.assert_not_called()

    def test_unverified_receipt_blocks_invocation(self):
        with patch.object(module.subprocess, 'check_output', return_value=''), \
                patch.object(module.subprocess, 'run') as invocation, \
                patch.object(self.helper, 'upload_verify', return_value={}):
            with self.assertRaises(RuntimeError):
                self.helper.finish()
            invocation.assert_not_called()


if __name__ == '__main__':
    unittest.main()
