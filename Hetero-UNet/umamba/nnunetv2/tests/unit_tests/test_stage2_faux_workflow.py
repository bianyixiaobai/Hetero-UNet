"""Exercise shutdown gates and corrupt OSS delivery without touching a live job."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

if sys.platform != 'linux':
    raise unittest.SkipTest('The detached server workflow uses Linux file locks')

source = (Path(os.environ['FAUX_WORKFLOW_SOURCE']) if 'FAUX_WORKFLOW_SOURCE' in os.environ else
          Path(__file__).resolve().parents[4] / 'scripts/run_stage2_faux_20261029.py')
spec = importlib.util.spec_from_file_location('faux_workflow', source)
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)


class FauxDeliveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='faux_delivery_test_')
        self.work = Path(self.temp.name)
        self.replace_work = patch.object(workflow, 'WORK', self.work)
        self.replace_work.start()

    def tearDown(self):
        self.replace_work.stop()
        self.temp.cleanup()

    def config(self, authorized=True):
        workflow.write('protocol.json', {'shutdown_authorized':authorized})

    def test_no_shutdown_without_explicit_arm(self):
        self.config()
        with patch.object(workflow,'idle'), patch.object(workflow,'run') as command:
            with self.assertRaises(AssertionError):
                workflow.complete_backup_and_shutdown(False)
            command.assert_not_called()

    def test_config_authorization_is_also_required(self):
        self.config(False)
        with patch.object(workflow,'idle'), patch.object(workflow,'run') as command:
            with self.assertRaises(AssertionError):
                workflow.complete_backup_and_shutdown(True)
            command.assert_not_called()

    def test_active_gpu_blocks_shutdown(self):
        with patch.object(workflow.socket,'gethostname',return_value=workflow.HOST), \
             patch.object(workflow.subprocess,'check_output',return_value='12345\n'):
            with self.assertRaisesRegex(AssertionError,'GPU work'):
                workflow.idle()

    def test_changed_checkpoint_blocks_packaging(self):
        self.config()
        result=self.work/'model'
        (result/'fold_all').mkdir(parents=True)
        (result/'fold_all/checkpoint_best.pth').write_bytes(b'changed')
        workflow.write('training_audit.json', {'checkpoints':[{'file':'checkpoint_best.pth','sha256':'wrong'}]})
        with patch.object(workflow,'idle'), patch.object(workflow,'RESULT',result), \
             patch.object(workflow,'package') as package, patch.object(workflow,'run') as command:
            with self.assertRaises(AssertionError):
                workflow.complete_backup_and_shutdown(True)
            package.assert_not_called()
            command.assert_not_called()

    def test_corrupt_roundtrip_is_retried_before_receipt(self):
        payload=self.work/'sample.tar'
        payload.write_bytes(b'authoritative-content')
        attempts=[]
        def transport(args,*unused,**kwargs):
            if args[:2]==['oss','cp'] and str(args[2]).startswith('oss://'):
                attempts.append(1)
                Path(args[3]).write_bytes(b'corrupt' if len(attempts)==1 else payload.read_bytes())
        with patch.object(workflow,'run',side_effect=transport), patch.object(workflow.time,'sleep') as sleep:
            receipt=workflow.upload_verified(payload)
        self.assertEqual(len(attempts),2)
        sleep.assert_called_once_with(300)
        self.assertEqual(receipt['sha256'],workflow.sha(payload))
        self.assertTrue(receipt['roundtrip_passed'])
        self.assertEqual(json.loads((self.work/'backup_receipt.json').read_text())['sample.tar'],receipt)
        self.assertFalse((self.work/'verify_sample.tar').exists())


if __name__=='__main__':
    unittest.main()
