"""Frozen F-aux: preflight, fresh training, external evaluation, verified backup, shutdown.

Deployment is outside the frozen training checkout. No model source is modified.
The explicit --arm-shutdown flag and verified final delivery are both required.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import time
import traceback

WORK = Path('/hy-tmp/stage2_Faux_20261029')
REPO = Path('/hy-tmp/Hetero-UNet-stage2')
PROJECT = REPO / 'Hetero-UNet'
OLD = Path('/hy-tmp/stage2_eval_20261024')
DATA = Path('/hy-tmp/Hetero-UNet/data')
COMMIT = '9fe70ac570a0e14ebce30daa4a16a22e49b4f3fd'
HOST = 'I2bde78031f00701d9e'
TRAINER = 'nnUNetTrainerMambaUNetBlockAttnResBoundarySupervision'
RESULT = Path('/hy-tmp/stage2_results/stage2_F-aux_seed1234/Dataset701_AbdomenCT') / (TRAINER + '__nnUNetPlans__3d_fullres')
PREFIX = 'oss://backup/stage2_20261029/F_aux/'


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(name, data):
    path = WORK / name
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def state(status, **extra):
    data = read(WORK / 'state.json') if (WORK / 'state.json').exists() else {'created_utc': now()}
    data.update(status=status, updated_utc=now(), **extra)
    write('state.json', data)
    print('FAUX_STATE', json.dumps(data), flush=True)


def environment(inference=False):
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               nnUNet_compile='false', WANDB_MODE='disabled', CUDA_VISIBLE_DEVICES='0',
               nnUNet_n_proc_DA='12', PYTHONUNBUFFERED='1',
               nnUNet_raw=str(DATA / 'nnUNet_raw'), nnUNet_preprocessed=str(DATA / 'nnUNet_preprocessed'),
               nnUNet_results='/hy-tmp/stage2_results',
               PYTHONPATH=str(WORK / 'runtime') if inference else str(PROJECT / 'umamba'))
    return env


def run(args, log, env=None, timeout=None):
    print('COMMAND', json.dumps([str(x) for x in args]), flush=True)
    with (WORK / log).open('ab', buffering=0) as stream:
        subprocess.run([str(x) for x in args], cwd=PROJECT, env=env or environment(), check=True,
                       stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def idle():
    assert socket.gethostname() == HOST, 'Wrong instance'
    assert not (WORK / 'DISABLE_SHUTDOWN').exists(), 'Shutdown disabled by sentinel'
    gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    assert not gpu.strip(), 'GPU work is still active'
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = path.read_bytes().split(b'\0')
        except OSError:
            continue
        assert not any(Path(a.decode(errors='replace')).name in
                       {'run_stage2.py', 'nnUNetv2_train', 'nnUNetv2_predict_from_modelfolder'} for a in args), 'Other training/inference active'


def smoke():
    """A separate process: original full-plan F-aux forward/backward, no saved weights."""
    import numpy as np
    import torch
    from torch.nn import functional as F
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMambaUNetAttnRes import (
        nnUNetTrainerMambaUNetBlockAttnResBoundarySupervision as Aux,
        nnUNetTrainerMambaUNetBGBlockAttnRes as C,
    )
    for key in ['group_size', 'auxiliary_boundary', 'boundary_hidden_channels', 'boundary_loss_weight']:
        assert getattr(Aux, key) == getattr(C, key)
    assert not Aux.boundary_guidance and C.boundary_guidance
    torch.set_num_threads(1)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    root = DATA / 'nnUNet_preprocessed/Dataset701_AbdomenCT'
    trainer = Aux(read(root / 'nnUNetPlans.json'), '3d_fullres', 'all', read(root / 'dataset.json'), device=torch.device('cuda'))
    # This process has its own results root; formal weights are never written here.
    trainer.initialize()
    assert Path(trainer.output_folder).is_relative_to(WORK / 'smoke_results')
    net = trainer.network
    assert net.boundary_head is not None and net.auxiliary_boundary and not net.boundary_guidance
    assert all(not a.boundary_conditioned and a.boundary_query is None for a in net.mamba_mlp.aggregators)
    assert all(not layer.mha.batch_first for layer in net.mamba_mlp.layers)
    head_calls = []
    handles = [net.boundary_head.register_forward_hook(lambda *args: head_calls.append(1))]
    calls = []
    def no_injection(mod, args):
        assert len(args) == 2 and args[1] is None
        calls.append(1)
    handles += [a.register_forward_pre_hook(no_injection) for a in net.mamba_mlp.aggregators]
    patch = [40, 224, 192]
    image = torch.randn(2, 1, *patch)
    target = torch.randint(0, 14, (2, 1, *patch))
    targets = [F.interpolate(target.float(), size=[int(round(n*s)) for n,s in zip(patch, scales)], mode='nearest').long()
               for scales in trainer._get_deep_supervision_scales()]
    net.train()
    values = trainer.train_step({'data': image, 'target': targets})
    assert all(np.isfinite(values[k]) for k in ['loss', 'segmentation_loss', 'boundary_loss'])
    assert float(values['boundary_loss']) > 0 and abs(float(values['loss']) - float(values['segmentation_loss']) - .1*float(values['boundary_loss'])) < 1e-5
    gradients = [p.grad for p in net.boundary_head.parameters() if p.grad is not None]
    assert gradients and all(bool(torch.isfinite(g).all()) for g in gradients) and sum(float(g.abs().sum()) for g in gradients) > 0
    assert len(head_calls) == 1 and len(calls) == 5 and net.boundary_logits is None
    net.eval()
    with torch.no_grad():
        val = trainer.validation_step({'data': image, 'target': targets})
    assert np.isfinite(val['loss']) and len(head_calls) == 1 and len(calls) == 10
    for h in handles:
        h.remove()
    write('smoke_check.json', {'status': 'passed', 'boundary_head_train_gradient_nonzero': True,
          'all_aggregators_received_no_guidance': True, 'head_skipped_during_inference': True,
          'MHA_batch_first': False, 'physical_batch': 2, 'patch': patch,
          'train_loss': float(values['loss']), 'boundary_loss': float(values['boundary_loss']),
          'validation_loss': float(val['loss']), 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
          'scope': 'synthetic full-plan CUDA/AMP; not efficacy data'})


def prepare():
    idle()
    assert not (WORK / 'protocol.json').exists(), 'Already prepared'
    assert not RESULT.exists(), 'Formal F-aux output already exists'
    assert shutil.disk_usage('/hy-tmp').free > 12 * 1024**3
    assert subprocess.check_output(['git', '-C', REPO, 'rev-parse', 'HEAD'], text=True).strip() == COMMIT
    assert not subprocess.check_output(['git', '-C', REPO, 'status', '--porcelain'], text=True).strip()
    previous = read(next(Path('/hy-tmp/stage2_results/stage2_C_seed1234').rglob('stage2_run.json')))
    assert previous['status'] == 'completed' and previous['protocol']['num_epochs'] == 1000
    for name, expected in previous['fingerprints']['source_sha256'].items():
        assert sha(PROJECT / name) == expected, name
    base = DATA / 'nnUNet_preprocessed/Dataset701_AbdomenCT'
    for name, expected in previous['fingerprints']['descriptor_sha256'].items():
        assert (sha(base / name) if (base / name).exists() else None) == expected, name
    manifest = Path('/hy-tmp/full_preservation_20261022/complete_manifest.json')
    assert sha(manifest) == '6f9c28d062be07f7f38f2bf76a02ff0b4d3e4c690c2963ee401c44c93fe53c48'
    rows = [r for r in read(manifest)['entries'] if r['kind'] == 'file' and r['path'].startswith('Hetero-UNet/data/nnUNet_preprocessed/')]
    for index, row in enumerate(rows):
        path = Path('/hy-tmp') / row['path']
        assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], row['path']
        if index % 50 == 0:
            print('PREPROCESSED_VERIFIED', index, '/', len(rows), flush=True)
    old = read(OLD / 'protocol.json')
    assert len(old['cases']) == 50 and old['tta'] is False and old['patch_batch'] == 1 and old['step_size'] == .5
    for row in old['cases']:
        assert sha(row['image']) == row['image_sha256'] and sha(row['gt']) == row['gt_sha256']
    shutil.copytree(OLD / 'runtime', WORK / 'runtime', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copytree(OLD / 'metrics_code', WORK / 'metrics_code', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copytree(OLD / 'baseline_A', WORK / 'baseline_A')
    for name, expected in previous['fingerprints']['source_sha256'].items():
        if name.startswith('umamba/'):
            assert sha(WORK / 'runtime' / name[len('umamba/'):]) == expected, name
    assert sha(WORK / 'runtime/nnunetv2/inference/export_prediction.py') == old['low_memory_export_sha256']
    for name, expected in old['metrics_sha256'].items():
        assert sha(WORK / 'metrics_code' / name) == expected
    for model in ['F', 'C']:
        (WORK / model).mkdir()
        for name in [model+'_DSC.csv', model+'_NSD.csv', 'summary.json']:
            shutil.copy2(OLD / model / name, WORK / model / name)
    for name in ['export_equivalence.log', 'run_stage2_evaluation_20261024.py']:
        shutil.copy2(OLD / name, WORK / name)
    with (WORK / 'training_source_9fe70ac.tar').open('wb') as stream:
        subprocess.run(['git', '-C', REPO, 'archive', 'HEAD'], stdout=stream, check=True)
    run([sys.executable, '-m', 'pip', 'freeze'], 'environment_pip_freeze.txt')
    helper = module('old_finalizer', '/hy-tmp/stage2_C_audit_2026-10-22/finalize_stage2.py')
    shutdown = '/usr/local/bin/shutdown'
    assert sha(shutdown) == '30b9ea6f23173078ccfe57d9b15ffe8caf5ba1442477f32c2c170ea0e97771d7'
    protocol = {'created_utc': now(), 'training_commit': COMMIT, 'hostname': HOST,
                'model': 'F-aux', 'trainer': TRAINER, 'seed': 1234, 'epochs': 1000,
                'physical_batch': 2, 'patch': [40,224,192], 'lr': .01, 'group_size': 3,
                'boundary_loss_weight': .1, 'boundary_hidden_channels': 8,
                'auxiliary_boundary': True, 'boundary_guidance': False, 'MHA_batch_first': False,
                'training_fingerprints': previous['fingerprints'], 'preprocessed_files_verified': len(rows),
                'evaluation': old, 'checkpoint': 'checkpoint_best.pth', 'from_scratch': True,
                'shutdown_executable': shutdown, 'shutdown_sha256': sha(shutdown),
                'oss_expiry_utc': helper.auth_expiry('/root/.hycloud_ossutil_config'),
                'oss_prefix': PREFIX, 'shutdown_authorized': True,
                'user_authorization': '2026-10-29: train F-aux, infer/evaluate, archive/upload/verify, then shut down.'}
    write('protocol.json', protocol)
    env = environment()
    env['nnUNet_results'] = str(WORK / 'smoke_results')
    run([sys.executable, Path(__file__), '--smoke'], 'smoke.log', env, 300)
    assert read(WORK / 'smoke_check.json')['status'] == 'passed'
    write('preflight.json', {'status': 'passed', 'checked_utc': now(), 'protocol_sha256': sha(WORK / 'protocol.json'),
          'preprocessed_files_verified': len(rows), 'source_files_verified': len(previous['fingerprints']['source_sha256']),
          'external_cases_verified': len(old['cases']), 'disk_free_bytes': shutil.disk_usage('/hy-tmp').free})


def audit_training():
    import torch
    import numpy as np
    fold = RESULT / 'fold_all'
    manifest = read(fold / 'stage2_run.json')
    proto = read(WORK / 'protocol.json')
    assert manifest['status'] == 'completed' and manifest['git']['commit'] == COMMIT
    assert manifest['fingerprints'] == proto['training_fingerprints'] and manifest['resume_checkpoint'] is None
    assert manifest['arguments']['model'] == 'F-aux' and not manifest['arguments']['smoke']
    for key, value in {'seed':1234, 'num_epochs':1000, 'train_iterations_per_epoch':250,
                       'validation_iterations_per_epoch':50, 'batch_size':2, 'patch_size':[40,224,192],
                       'initial_lr':.01, 'mha_batch_first':False}.items():
        assert manifest['protocol'][key] == value, key
    config = read(fold / 'stage2_config.json')
    assert config['auxiliary_boundary'] and not config['boundary_guidance'] and config['boundary_loss_weight'] == .1
    checkpoints = []
    for name in ['checkpoint_best.pth', 'checkpoint_final.pth']:
        path = fold / name
        ck = torch.load(path, map_location='cpu')
        assert ck['trainer_name'] == TRAINER and 1 <= ck['current_epoch'] <= 1000
        assert name != 'checkpoint_final.pth' or ck['current_epoch'] == 1000
        assert ck['network_weights'] and all(bool(torch.isfinite(v).all()) for v in ck['network_weights'].values() if torch.is_floating_point(v))
        assert ck.get('optimizer_state') is not None, 'Resume optimizer missing'
        checkpoints.append({'file': name, 'epoch':ck['current_epoch'], 'bytes':path.stat().st_size, 'sha256':sha(path)})
        del ck
    text = '\n'.join(p.read_text() for p in sorted(fold.glob('training_log_*.txt')))
    train = [float(v) for v in re.findall(r': train_loss ([\-0-9.eE]+)', text)]
    val = [float(v) for v in re.findall(r': val_loss ([\-0-9.eE]+)', text)]
    parts = [tuple(map(float, row)) for row in re.findall(r'Stage2 loss components: segmentation= ([\-0-9.eE]+) boundary_unweighted= ([\-0-9.eE]+) boundary_weight= ([\-0-9.eE]+)', text)]
    assert len(train) == len(val) == len(parts) == 1000
    assert np.isfinite(train).all() and np.isfinite(val).all() and np.isfinite(parts).all()
    assert all(p[1] >= 0 and p[2] == .1 for p in parts)
    assert max(abs(t - s - b*w) for t,(s,b,w) in zip(train,parts)) < 1e-3
    write('training_audit.json', {'status':'passed', 'checked_utc':now(), 'epochs':1000, 'checkpoints':checkpoints,
          'train_loss_count':len(train), 'validation_loss_count':len(val), 'boundary_loss_count':len(parts),
          'finite_weights_and_losses':True, 'source_and_descriptors_match_F_C':True})


def evaluate():
    import pandas as pd
    helper = module('evaluation_helpers', WORK / 'run_stage2_evaluation_20261024.py')
    protocol = read(WORK / 'protocol.json')['evaluation']
    dest = WORK / 'F-aux'
    dest.mkdir(exist_ok=True)
    for row in protocol['cases']:
        assert sha(row['image']) == row['image_sha256'] and sha(row['gt']) == row['gt_sha256']
    env = environment(True)
    run(['nnUNetv2_predict_from_modelfolder', '-i', Path(protocol['cases'][0]['image']).parent,
         '-o', dest / 'predictions', '-m', RESULT, '-f', 'all', '-chk', 'checkpoint_best.pth',
         '--disable_tta', '-step_size', '0.5', '-npp', '1', '-nps', '1', '--disable_progress_bar'], 'predict.log', env)
    rows = helper.inspect_predictions(dest / 'predictions', protocol['cases'])
    write('F-aux/prediction_validation.json', {'status':'passed','cases':rows})
    summary = {'model':'F-aux', 'cases':50, 'checkpoint_sha256':sha(RESULT / 'fold_all/checkpoint_best.pth')}
    comparison = {}
    for metric, script in [('DSC','abdomen_DSC_Eval.py'),('NSD','nsd_parallel.py')]:
        metric_env = env.copy()
        metric_env['PYTHONPATH'] = str(WORK / 'metrics_code') + ':' + env['PYTHONPATH']
        csv = dest / ('F-aux_' + metric + '.csv')
        args = [sys.executable, WORK/'metrics_code'/script, '--gt_path', Path(protocol['cases'][0]['gt']).parent,
                '--seg_path', dest/'predictions', '--save_path', csv]
        if metric == 'NSD':
            args += ['--workers','2']
        run(args, metric+'.log', metric_env)
        frame = helper.metrics_table(csv, metric, protocol['cases'])
        frames = {'A':helper.metrics_table(WORK/'baseline_A'/(metric+'.csv'),metric,protocol['cases']),
                  **{m:helper.metrics_table(WORK/m/(m+'_'+metric+'.csv'),metric,protocol['cases']) for m in ['F','C']}, 'F-aux':frame}
        means = {m:float(f.to_numpy().mean()*100) for m,f in frames.items()}
        summary[metric+'_percent'] = means['F-aux']
        summary[metric+'_per_organ_percent'] = {k:float(v*100) for k,v in frame.mean().items()}
        paired = pd.DataFrame({m:f.mean(axis=1)*100 for m,f in frames.items()})
        paired['F-aux_minus_F_pp'] = paired['F-aux'] - paired['F']
        paired['C_minus_F-aux_pp'] = paired['C'] - paired['F-aux']
        paired.to_csv(WORK/('paired_'+metric+'.csv'))
        pd.DataFrame({m:f.mean()*100 for m,f in frames.items()}).to_csv(WORK/('per_organ_'+metric+'.csv'))
        comparison[metric] = {'macro_percent':means, 'F_aux_minus_F_pp':means['F-aux']-means['F'],
                              'C_minus_F_aux_pp':means['C']-means['F-aux']}
    write('F-aux/summary.json', summary)
    write('comparison.json', comparison)
    lines = ['# F-aux训练与外部50例结果', '', '同一50例、13器官等权；原best规则、无TTA、滑窗0.5及相同评价脚本。', '',
             '| 指标 | A % | F % | F-aux % | C % | F-aux−F pp | C−F-aux pp |','|---|---:|---:|---:|---:|---:|---:|']
    for metric,row in comparison.items():
        m=row['macro_percent']
        lines.append(f"| {metric} | {m['A']:.4f} | {m['F']:.4f} | {m['F-aux']:.4f} | {m['C']:.4f} | {row['F_aux_minus_F_pp']:+.4f} | {row['C_minus_F_aux_pp']:+.4f} |")
    lines += ['', 'F-aux从头训练1000轮，seed1234，边界辅助监督0.1、无引导注入；模型/MHA及训练评价协议未修改。',
              'F-aux−F用于观察边界辅助监督效应，C−F-aux用于观察在边界监督基础上的注入效应。',
              '首轮单种子探索；F有4090至V100迁移经历，不能将小幅差异直接写成稳定或显著增益。',
              '完整逐病例、逐器官数据及原始预测均归档；外部结果不用于本次选择模型或超参。']
    (WORK/'RESULTS.md').write_text('\n'.join(lines)+'\n')


def package(name, paths):
    path = WORK / name
    with tarfile.open(path, 'w') as archive:
        for source, relative in paths:
            archive.add(source, arcname=relative)
    return path


def upload_verified(path):
    """Retry auth/network failures; never shut down with unverified delivery."""
    expected, size = sha(path), path.stat().st_size
    remote = PREFIX + path.name
    while True:
        try:
            run(['oss','mkdir',PREFIX], 'oss_transfer.log', timeout=120)
            run(['oss','cp',path,remote,'-f'], 'oss_transfer.log', timeout=3600)
            returned = WORK / ('verify_' + path.name)
            run(['oss','cp',remote,returned,'-f'], 'oss_transfer.log', timeout=3600)
            assert returned.stat().st_size == size and sha(returned) == expected, 'OSS SHA mismatch'
            returned.unlink()
            result = {'remote':remote,'bytes':size,'sha256':expected,'roundtrip_passed':True,'verified_utc':now()}
            receipts = read(WORK/'backup_receipt.json') if (WORK/'backup_receipt.json').exists() else {}
            receipts[path.name] = result
            write('backup_receipt.json', receipts)
            return result
        except Exception:
            state('backup_pending_retry', pending_object=path.name, backup_error=traceback.format_exc())
            time.sleep(300)


def complete_backup_and_shutdown(armed):
    idle()
    config = read(WORK / 'protocol.json')
    assert armed and config['shutdown_authorized']
    checks = read(WORK/'training_audit.json')
    for row in checks['checkpoints']:
        assert sha(RESULT/'fold_all'/row['file']) == row['sha256']
    pred = read(WORK/'F-aux/prediction_validation.json')
    assert pred['status']=='passed' and len(pred['cases'])==50
    for row in pred['cases']:
        assert row['geometry_matches_gt'] and sha(WORK/'F-aux/predictions'/(row['case']+'.nii.gz'))==row['sha256']
    weights = package('F_aux_seed1234_training_verified.tar', [(RESULT,RESULT.name),
                      (WORK/'training_audit.json','training_audit.json')])
    upload_verified(weights)
    evaluation = package('F_aux_external50_verified.tar', [(WORK/'F-aux','F-aux'),
                         (WORK/'metrics_code','metrics_code'),(WORK/'protocol.json','protocol.json')])
    upload_verified(evaluation)
    allow = [p for p in WORK.iterdir() if p.is_file() and p.suffix not in {'.tar','.lock'} and not p.name.startswith('verify_')]
    records = package('F_aux_final_records.tar', [(p,p.name) for p in allow] +
                      [(WORK/'training_source_9fe70ac.tar','training_source_9fe70ac.tar'),
                       (WORK/'runtime','runtime'),(WORK/'baseline_A','baseline_A'),(WORK/'F','F'),(WORK/'C','C')])
    upload_verified(records)
    write('final_delivery.json', {'status':'all_results_verified','created_utc':now(),
          'objects':read(WORK/'backup_receipt.json'), 'user_authorization':config['user_authorization'],
          'shutdown_only_after_receipt_roundtrip':True})
    receipt = package('F_aux_final_delivery.tar', [(WORK/'final_delivery.json','final_delivery.json'),
                      (WORK/'protocol.json','protocol.json')])
    upload_verified(receipt)
    expected = {weights.name,evaluation.name,records.name,receipt.name}
    assert expected.issubset(read(WORK/'backup_receipt.json'))
    idle()
    assert sha(config['shutdown_executable']) == config['shutdown_sha256']
    state('shutdown_requested', all_results_verified=True, shutdown_authorized=True)
    run([config['shutdown_executable']], 'shutdown.log', timeout=120)
    state('shutdown_command_returned', note='Platform power state is not independently confirmed by command return')


def pipeline(armed):
    assert armed, 'Explicit --arm-shutdown required for this authorized workflow'
    with (WORK/'pipeline.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (WORK/'state.json').exists(), 'Existing execution; inspect before explicit recovery'
        try:
            state('preflight', shutdown_authorized=True)
            prepare()
            state('training', model='F-aux', epochs=1000)
            run([sys.executable,PROJECT/'scripts/run_stage2.py','--model','F-aux','--seed','1234',
                 '--results-root','/hy-tmp/stage2_results','--skip-final-validation'], 'train.log')
            state('auditing_training')
            audit_training()
            state('predicting_and_evaluating')
            evaluate()
            state('archiving_and_uploading')
            complete_backup_and_shutdown(armed)
        except BaseException:
            state('failed_requires_review', error=traceback.format_exc(), automatic_shutdown_blocked=True)
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--arm-shutdown',action='store_true')
    args=parser.parse_args()
    WORK.mkdir(exist_ok=True)
    if args.smoke:
        smoke()
    elif args.run:
        pipeline(args.arm_shutdown)
    else:
        parser.error('Use --run --arm-shutdown; smoke is an internal subprocess')


if __name__=='__main__':
    main()
