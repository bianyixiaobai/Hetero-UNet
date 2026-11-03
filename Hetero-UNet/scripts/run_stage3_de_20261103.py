"""Authorized D/E queue: restore verified data, CUDA checks, train, evaluate,
round-trip OSS archives, then shut down. No old experimental outputs overwritten.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import traceback
import zipfile

WORK = Path('/hy-tmp/stage3_20261103')
REPO = Path('/hy-tmp/Hetero-UNet-stage3')
PROJECT = REPO / 'Hetero-UNet'
DATA = Path('/hy-tmp/Hetero-UNet/data')
RESULTS = Path('/hy-tmp/stage3_results')
PREFIX = 'oss://backup/stage3_20261103/D_E/'
TRAINERS = {'D': 'nnUNetTrainerMambaUNetMultiScaleSkip',
            'E': 'nnUNetTrainerMambaUNetBGMultiScaleSkip'}


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


legacy = module('verified_backup_helpers', PROJECT / 'scripts/run_stage2_faux_20261029.py')
legacy.WORK, legacy.REPO, legacy.PROJECT, legacy.DATA = WORK, REPO, PROJECT, DATA
legacy.PREFIX = PREFIX
sha, read, write, state, run = legacy.sha, legacy.read, legacy.write, legacy.state, legacy.run


def env(inference=False):
    value = legacy.environment(inference)
    value['nnUNet_results'] = str(RESULTS)
    return value


def safe_tar(path, destination):
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path) as archive:
        for member in archive.getmembers():
            (destination / member.name).resolve().relative_to(destination.resolve())
            assert member.isfile() or member.isdir(), member.name
        archive.extractall(destination)


def restore():
    records = WORK / 'previous_records.tar'
    assert sha(records) == '096e11c3c0201930bd7bbd4eb88616d97cffed62846223394a4b11c89a4a316c'
    previous = WORK / 'previous'
    safe_tar(records, previous)
    for row in read(WORK / 'restore_objects.json'):
        target = WORK / Path(row['remote']).name
        state('restoring_data', object=row['remote'])
        run(['oss', 'cp', row['remote'], target, '-f'], 'restore.log', timeout=7200)
        assert target.stat().st_size == row['bytes'] and sha(target) == row['sha256']
        if target.suffix == '.tar':
            safe_tar(target, DATA / 'nnUNet_preprocessed')
        else:
            with zipfile.ZipFile(target) as archive:
                for member in archive.infolist():
                    parts = Path(member.filename).parts
                    assert not Path(member.filename).is_absolute() and '..' not in parts
                    if 'Dataset701_AbdomenCT' not in parts:
                        continue
                    relative = Path(*parts[parts.index('Dataset701_AbdomenCT'):])
                    output = DATA / 'nnUNet_raw' / relative
                    output.resolve().relative_to((DATA / 'nnUNet_raw').resolve())
                    if member.is_dir():
                        output.mkdir(parents=True, exist_ok=True)
                    else:
                        output.parent.mkdir(parents=True, exist_ok=True)
                        with archive.open(member) as source, output.open('wb') as dest:
                            shutil.copyfileobj(source, dest)
        target.unlink()  # Verified archive remains on OSS; extracted data is retained.
    original = read(previous / 'protocol.json')
    for name, expected in original['training_fingerprints']['descriptor_sha256'].items():
        path = DATA / 'nnUNet_preprocessed/Dataset701_AbdomenCT' / name
        assert (sha(path) if path.exists() else None) == expected, name
    for row in original['evaluation']['cases']:
        assert sha(row['image']) == row['image_sha256'], row['id']
        assert sha(row['gt']) == row['gt_sha256'], row['id']
    # Reuse archived compressed preprocessed arrays, not recomputed preprocessing.
    preprocessed = DATA / 'nnUNet_preprocessed/Dataset701_AbdomenCT/nnUNetPlans_3d_fullres'
    assert len(list(preprocessed.glob('*.npz'))) == len(list(preprocessed.glob('*.pkl'))) == 50
    shutil.copytree(PROJECT / 'umamba/nnunetv2', WORK / 'runtime/nnunetv2',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(previous / 'runtime/nnunetv2/inference/export_prediction.py',
                 WORK / 'runtime/nnunetv2/inference/export_prediction.py')
    assert sha(WORK / 'runtime/nnunetv2/inference/export_prediction.py') == original['evaluation']['low_memory_export_sha256']
    shutil.copytree(previous / 'metrics_code', WORK / 'metrics_code')
    for name, expected in original['evaluation']['metrics_sha256'].items():
        assert sha(WORK / 'metrics_code' / name) == expected
    for name in ['baseline_A', 'F', 'C']:
        shutil.copytree(previous / name, WORK / name)
    with (WORK / 'source.tar').open('wb') as stream:
        subprocess.run(['git', '-C', REPO, 'archive', 'HEAD'], stdout=stream, check=True)
    protocol = {'date': '2026-11-03', 'hostname': socket.gethostname(),
                'commit': subprocess.check_output(['git', '-C', REPO, 'rev-parse', 'HEAD'], text=True).strip(),
                'models': TRAINERS, 'seed':1234, 'epochs':1000, 'physical_batch':2,
                'patch':[40,224,192], 'mha_batch_first':False, 'from_scratch':True,
                'evaluation':original['evaluation'], 'oss_prefix':PREFIX,
                'shutdown_executable':'/usr/local/bin/shutdown',
                'shutdown_sha256':sha('/usr/local/bin/shutdown'),
                'shutdown_authorized':True,
                'user_authorization':'2026-11-03: run module-two D/E experiments, record and archive all data, then shut down.'}
    assert protocol['shutdown_sha256'] == '30b9ea6f23173078ccfe57d9b15ffe8caf5ba1442477f32c2c170ea0e97771d7'
    write('protocol.json', protocol)
    run([sys.executable, '-m', 'pip', 'freeze'], 'environment.txt', env())
    run([sys.executable, PROJECT/'scripts/check_stage3_cuda.py', '--plans',
         DATA/'nnUNet_preprocessed/Dataset701_AbdomenCT/nnUNetPlans.json', '--dataset-json',
         DATA/'nnUNet_preprocessed/Dataset701_AbdomenCT/dataset.json', '--output', WORK/'cuda_check'],
        'cuda_check.log', env(), 1200)
    assert len(read(WORK/'cuda_check/cuda_check.json')['runs']) == 2
    write('preflight.json', {'status':'passed','commit':protocol['commit'],'external_cases_verified':50,
                             'compressed_preprocessing_reused':True,'disk_free':shutil.disk_usage('/hy-tmp').free})
    # Verify an OSS write AND read before committing to long GPU work.
    legacy.upload_verified(legacy.package('startup.tar', [(WORK/'protocol.json','protocol.json'),
        (WORK/'preflight.json','preflight.json'),(WORK/'source.tar','source.tar'),
        (Path(__file__),'run_stage3_de_20261103.py')]))


def result(model):
    return RESULTS / f'stage3_{model}_seed1234/Dataset701_AbdomenCT' / (TRAINERS[model]+'__nnUNetPlans__3d_fullres')


def audit(model):
    import torch
    fold = result(model) / 'fold_all'
    manifest = read(fold/'stage3_run.json')
    assert manifest['status']=='completed' and manifest['git']['commit']==read(WORK/'protocol.json')['commit']
    assert manifest['resume_checkpoint'] is None
    for key, expected in {'num_epochs':1000,'batch_size':2,'patch_size':[40,224,192],
                          'seed':1234,'initial_lr':.01,'mha_batch_first':False}.items():
        assert manifest['protocol'][key] == expected, key
    text='\n'.join(p.read_text() for p in fold.glob('training_log_*.txt'))
    for kind in ['train_loss','val_loss']:
        values=[float(v) for v in re.findall(r': '+kind+r' ([\-0-9.eE]+)', text)]
        assert len(values)==1000 and all(__import__('math').isfinite(v) for v in values)
    rows=[]
    for name in ['checkpoint_best.pth','checkpoint_final.pth']:
        path=fold/name;ck=torch.load(path,map_location='cpu')
        assert ck['trainer_name']==TRAINERS[model]
        assert 1<=ck['current_epoch']<=1000
        assert name!='checkpoint_final.pth' or ck['current_epoch']==1000
        assert ck.get('optimizer_state') is not None
        assert all(torch.isfinite(v).all() for v in ck['network_weights'].values() if torch.is_floating_point(v))
        rows.append({'name':name,'epoch':ck['current_epoch'],'sha256':sha(path),'bytes':path.stat().st_size})
        del ck
    write(model+'_training_audit.json',{'status':'passed','checkpoints':rows})


def evaluate(model):
    helper=module('evaluation_helpers',PROJECT/'scripts/run_stage2_evaluation_20261024.py')
    cases=read(WORK/'protocol.json')['evaluation']['cases']
    dest=WORK/model;dest.mkdir(exist_ok=True)
    run(['nnUNetv2_predict_from_modelfolder','-i',Path(cases[0]['image']).parent,
         '-o',dest/'predictions','-m',result(model),'-f','all','-chk','checkpoint_best.pth',
         '--disable_tta','-step_size','0.5','-npp','1','-nps','1','--disable_progress_bar'],model+'_predict.log',env(True))
    rows=helper.inspect_predictions(dest/'predictions',cases)
    write(model+'/prediction_validation.json',{'status':'passed','cases':rows})
    summary={'model':model,'cases':50,'best_sha256':sha(result(model)/'fold_all/checkpoint_best.pth')}
    for metric,script in [('DSC','abdomen_DSC_Eval.py'),('NSD','nsd_parallel.py')]:
        metric_env=env(True);metric_env['PYTHONPATH']=str(WORK/'metrics_code')+':'+metric_env['PYTHONPATH']
        csv=dest/(model+'_'+metric+'.csv')
        args=[sys.executable,WORK/'metrics_code'/script,'--gt_path',Path(cases[0]['gt']).parent,
              '--seg_path',dest/'predictions','--save_path',csv]
        if metric=='NSD':args+=['--workers','2']
        run(args,model+'_'+metric+'.log',metric_env)
        frame=helper.metrics_table(csv,metric,cases)
        summary[metric+'_percent']=float(frame.to_numpy().mean()*100)
    write(model+'/summary.json',summary)


def assert_shutdown_ready():
    assert not (WORK/'DISABLE_SHUTDOWN').exists(), 'Shutdown cancelled'
    protocol=read(WORK/'protocol.json');assert protocol['shutdown_authorized']
    assert socket.gethostname()==protocol['hostname']
    assert sha(protocol['shutdown_executable'])==protocol['shutdown_sha256']
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:args=p.read_bytes().split(b'\0')
        except OSError:continue
        assert not any(Path(a.decode(errors='replace')).name in
            {'run_stage2.py','run_stage3.py','nnUNetv2_train','nnUNetv2_predict_from_modelfolder'} for a in args)
    receipts=read(WORK/'backup_receipt.json')
    required={m+s for m in TRAINERS for s in ['_training.tar','_evaluation.tar']}|{'records.tar','delivery.tar'}
    assert required.issubset(receipts) and all(receipts[n]['roundtrip_passed'] for n in required)
    for m in TRAINERS:
        assert read(WORK/(m+'_training_audit.json'))['status']=='passed'
        assert len(read(WORK/m/'prediction_validation.json')['cases'])==50


def pipeline(armed):
    assert armed, 'Explicit shutdown authorization flag required'
    with (WORK/'queue.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (WORK/'state.json').exists(), 'Existing queue: inspect before recovery'
        try:
            state('preflight_and_restore');restore()
            for model in TRAINERS:
                state('real_data_smoke',model=model)
                run([sys.executable,PROJECT/'scripts/run_stage3.py','--model',model,'--results-root',RESULTS,'--smoke'],model+'_smoke.log',env())
            for model in TRAINERS:
                state('training',model=model)
                run([sys.executable,PROJECT/'scripts/run_stage3.py','--model',model,'--results-root',RESULTS,'--skip-final-validation'],model+'_train.log',env())
                audit(model)
                legacy.upload_verified(legacy.package(model+'_training.tar',[(result(model),result(model).name),
                    (WORK/(model+'_training_audit.json'),'training_audit.json')]))
                state('evaluating',model=model);evaluate(model)
                legacy.upload_verified(legacy.package(model+'_evaluation.tar',[(WORK/model,model),
                    (WORK/'metrics_code','metrics_code'),(WORK/'protocol.json','protocol.json')]))
            import pandas as pd
            helper=module('summary_helpers',PROJECT/'scripts/run_stage2_evaluation_20261024.py')
            cases=read(WORK/'protocol.json')['evaluation']['cases'];comparison={}
            for metric in ['DSC','NSD']:
                frames={'A':helper.metrics_table(WORK/'baseline_A'/(metric+'.csv'),metric,cases)}
                frames.update({m:helper.metrics_table(WORK/m/(m+'_'+metric+'.csv'),metric,cases) for m in ['F','C','D','E']})
                pd.DataFrame({m:f.mean()*100 for m,f in frames.items()}).to_csv(WORK/('per_organ_'+metric+'.csv'))
                paired=pd.DataFrame({m:f.mean(axis=1)*100 for m,f in frames.items()})
                for a,b in [('D','A'),('E','A'),('E','C'),('E','D')]:paired[a+'_minus_'+b+'_pp']=paired[a]-paired[b]
                paired.to_csv(WORK/('paired_'+metric+'.csv'))
                comparison[metric]={m:float(f.to_numpy().mean()*100) for m,f in frames.items()}
            write('comparison.json',comparison)
            records=[(p,p.name) for p in WORK.iterdir() if p.is_file() and p.suffix not in {'.tar','.lock'}]
            legacy.upload_verified(legacy.package('records.tar',records+[(WORK/'source.tar','source.tar'),(WORK/'runtime','runtime')]))
            write('final_delivery.json',{'status':'verified','receipts':read(WORK/'backup_receipt.json'),'comparison':comparison})
            legacy.upload_verified(legacy.package('delivery.tar',[(WORK/'final_delivery.json','final_delivery.json'),(WORK/'protocol.json','protocol.json')]))
            assert_shutdown_ready();state('shutdown_requested',all_results_verified=True)
            run([read(WORK/'protocol.json')['shutdown_executable']],'shutdown.log',env(),120)
        except BaseException:
            state('failed_requires_review',error=traceback.format_exc(),automatic_shutdown_blocked=True)
            raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true');parser.add_argument('--arm-shutdown',action='store_true')
    args=parser.parse_args()
    assert args.run
    WORK.mkdir(exist_ok=True)
    pipeline(args.arm_shutdown)
