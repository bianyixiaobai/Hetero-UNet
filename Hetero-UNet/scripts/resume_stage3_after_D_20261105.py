"""Resume the frozen D/E queue after completed D training; never retrain D.

Deploy outside the frozen checkout. Uses its original audited operations.
"""
import argparse
import fcntl
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import traceback


def validate_resume(manifest, protocol, audit, receipts):
    assert manifest['status']=='completed' and manifest['arguments']['model']=='D'
    assert manifest['git']['commit']==protocol['commit']
    assert manifest['protocol']['num_epochs']==1000
    assert audit['status']=='passed'
    assert {x['name'] for x in audit['checkpoints']}=={'checkpoint_best.pth','checkpoint_final.pth'}
    assert receipts['D_training.tar']['roundtrip_passed']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm-shutdown',action='store_true')
    args=parser.parse_args()
    assert args.arm_shutdown
    root=Path('/hy-tmp/Hetero-UNet-stage3/Hetero-UNet')
    spec=importlib.util.spec_from_file_location('frozen_queue',root/'scripts/run_stage3_de_20261103.py')
    q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
    recovery=q.WORK/'recovery_20261105';recovery.mkdir(exist_ok=True)
    with (q.WORK/'queue.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            protocol=q.read(q.WORK/'protocol.json')
            manifest=q.read(q.result('D')/'fold_all/stage3_run.json')
            audit=q.read(q.WORK/'D_training_audit.json')
            receipts=q.read(q.WORK/'backup_receipt.json')
            validate_resume(manifest,protocol,audit,receipts)
            assert not q.result('E').exists(), 'E already exists; inspect before recovery'
            assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
            assert subprocess.check_output(['git','-C',q.REPO,'rev-parse','HEAD'],text=True).strip()==protocol['commit']
            # Recheck every named source and descriptor; metadata repair must
            # never alter the trained model or preprocessing implementation.
            for name,expected in manifest['fingerprints']['source_sha256'].items():
                assert q.sha(q.PROJECT/name)==expected,name
            for item in audit['checkpoints']:
                assert q.sha(q.result('D')/'fold_all'/item['name'])==item['sha256']
            for name,expected in manifest['fingerprints']['descriptor_sha256'].items():
                p=q.DATA/'nnUNet_preprocessed/Dataset701_AbdomenCT'/name
                assert (q.sha(p) if p.exists() else None)==expected,name
            for row in protocol['evaluation']['cases']:
                assert q.sha(row['image'])==row['image_sha256']
                assert q.sha(row['gt'])==row['gt_sha256']
            q.run(['nnUNetv2_predict_from_modelfolder','--help'],'recovery_20261105/entrypoint_check.log',q.env(True),120)
            q.write('recovery_20261105/verified.json',{'status':'passed','training_commit':protocol['commit'],
                    'D_retrained':False,'D_checkpoint_hashes_unchanged':True,'source_fingerprints_match':True,
                    'entrypoint_metadata_repaired':True,'recovery_script_sha256':q.sha(__file__)})
            q.state('evaluating',model='D',recovered_from='missing_nnunetv2_package_metadata')
            q.evaluate('D')
            q.legacy.upload_verified(q.legacy.package('D_evaluation.tar',[(q.WORK/'D','D'),
                (q.WORK/'metrics_code','metrics_code'),(q.WORK/'protocol.json','protocol.json')]))
            q.state('training',model='E')
            q.run([sys.executable,q.PROJECT/'scripts/run_stage3.py','--model','E','--results-root',q.RESULTS,
                   '--skip-final-validation'],'E_train.log',q.env())
            q.audit('E')
            q.legacy.upload_verified(q.legacy.package('E_training.tar',[(q.result('E'),q.result('E').name),
                (q.WORK/'E_training_audit.json','training_audit.json')]))
            q.state('evaluating',model='E');q.evaluate('E')
            q.legacy.upload_verified(q.legacy.package('E_evaluation.tar',[(q.WORK/'E','E'),
                (q.WORK/'metrics_code','metrics_code'),(q.WORK/'protocol.json','protocol.json')]))
            import pandas as pd
            helper=q.module('summary_helpers',q.PROJECT/'scripts/run_stage2_evaluation_20261024.py')
            cases=protocol['evaluation']['cases'];comparison={}
            for metric in ['DSC','NSD']:
                frames={'A':helper.metrics_table(q.WORK/'baseline_A'/(metric+'.csv'),metric,cases)}
                frames.update({m:helper.metrics_table(q.WORK/m/(m+'_'+metric+'.csv'),metric,cases) for m in ['F','C','D','E']})
                pd.DataFrame({m:f.mean()*100 for m,f in frames.items()}).to_csv(q.WORK/('per_organ_'+metric+'.csv'))
                paired=pd.DataFrame({m:f.mean(axis=1)*100 for m,f in frames.items()})
                for a,b in [('D','A'),('E','A'),('E','C'),('E','D')]:paired[a+'_minus_'+b+'_pp']=paired[a]-paired[b]
                paired.to_csv(q.WORK/('paired_'+metric+'.csv'))
                comparison[metric]={m:float(f.to_numpy().mean()*100) for m,f in frames.items()}
            q.write('comparison.json',comparison)
            records=[(p,p.name) for p in q.WORK.iterdir() if p.is_file() and p.suffix not in {'.tar','.lock'}]
            q.legacy.upload_verified(q.legacy.package('records.tar',records+[(q.WORK/'source.tar','source.tar'),
                (q.WORK/'runtime','runtime'),(recovery,'recovery_20261105')]))
            q.write('final_delivery.json',{'status':'verified','receipts':q.read(q.WORK/'backup_receipt.json'),'comparison':comparison})
            q.legacy.upload_verified(q.legacy.package('delivery.tar',[(q.WORK/'final_delivery.json','final_delivery.json'),
                (q.WORK/'protocol.json','protocol.json')]))
            q.assert_shutdown_ready();q.state('shutdown_requested',all_results_verified=True)
            q.run([protocol['shutdown_executable']],'shutdown.log',q.env(),120)
        except BaseException:
            q.state('failed_requires_review',error=traceback.format_exc(),automatic_shutdown_blocked=True)
            raise


if __name__=='__main__': main()
