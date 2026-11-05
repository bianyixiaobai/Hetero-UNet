"""Shutdown gates in isolation: never run an actual platform shutdown."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


class TestStage3Shutdown(unittest.TestCase):
    def gate(self, cancelled=False, receipts=None, sha='expected'):
        source=Path(__file__).resolve().parents[4]/'scripts/run_stage3_de_20261103.py'
        node=next(n for n in ast.parse(source.read_text(encoding='utf-8')).body
                  if isinstance(n,ast.FunctionDef) and n.name=='assert_shutdown_ready')
        path=Mock();path.exists.return_value=cancelled
        work=Mock();work.__truediv__=Mock(return_value=path)
        protocol={'shutdown_authorized':True,'hostname':'test-host',
                  'shutdown_executable':'fake-shutdown','shutdown_sha256':'expected'}
        configs=iter([protocol, receipts or {}])
        fake_path=Mock();fake_path.return_value.glob.return_value=[]
        ns={'WORK':work,'read':lambda p:next(configs),'sha':lambda p:sha,
            'socket':SimpleNamespace(gethostname=lambda:'test-host'),
            'subprocess':SimpleNamespace(check_output=lambda *a,**k:''),
            'Path':fake_path,'TRAINERS':{'D':'trainerD','E':'trainerE'}}
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
        return ns['assert_shutdown_ready']

    def test_cancellation_blocks_shutdown(self):
        with self.assertRaises(AssertionError): self.gate(cancelled=True)()

    def test_modified_executable_blocks_shutdown(self):
        with self.assertRaises(AssertionError): self.gate(sha='changed')()

    def test_missing_receipts_blocks_shutdown(self):
        with self.assertRaises(AssertionError): self.gate()()

    def test_failed_roundtrip_blocks_shutdown(self):
        names=['D_training.tar','E_training.tar','D_evaluation.tar','E_evaluation.tar','records.tar','delivery.tar']
        receipts={n:{'roundtrip_passed':True} for n in names}
        receipts['E_evaluation.tar']['roundtrip_passed']=False
        with self.assertRaises(AssertionError): self.gate(receipts=receipts)()


class TestDRecovery(unittest.TestCase):
    def setUp(self):
        source=Path(__file__).resolve().parents[4]/'scripts/resume_stage3_after_D_20261105.py'
        node=next(n for n in ast.parse(source.read_text(encoding='utf-8')).body
                  if isinstance(n,ast.FunctionDef) and n.name=='validate_resume')
        namespace={}
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
        self.validate=namespace['validate_resume']
        self.manifest={'status':'completed','arguments':{'model':'D'},'git':{'commit':'frozen'},'protocol':{'num_epochs':1000}}
        self.protocol={'commit':'frozen'}
        self.audit={'status':'passed','checkpoints':[{'name':n} for n in ['checkpoint_best.pth','checkpoint_final.pth']]}
        self.receipts={'D_training.tar':{'roundtrip_passed':True}}

    def test_completed_backed_up_D_is_eligible(self):
        self.validate(self.manifest,self.protocol,self.audit,self.receipts)

    def test_unfinished_D_is_rejected(self):
        self.manifest['status']='training'
        with self.assertRaises(AssertionError):self.validate(self.manifest,self.protocol,self.audit,self.receipts)

    def test_wrong_source_is_rejected(self):
        self.protocol['commit']='different'
        with self.assertRaises(AssertionError):self.validate(self.manifest,self.protocol,self.audit,self.receipts)

    def test_unverified_backup_is_rejected(self):
        self.receipts['D_training.tar']['roundtrip_passed']=False
        with self.assertRaises(AssertionError):self.validate(self.manifest,self.protocol,self.audit,self.receipts)


if __name__=='__main__': unittest.main()
