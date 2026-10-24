"""Sequential frozen F/C inference, geometry checks, metrics and verified OSS backup.

Run detached in tmux. No training, checkpoint selection, tuning or shutdown.
"""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import traceback

WORK = Path('/hy-tmp/stage2_eval_20261024')
ORGANS = ['Liver', 'RK', 'Spleen', 'Pancreas', 'Aorta', 'IVC', 'RAG', 'LAG',
          'Gallbladder', 'Esophagus', 'Stomach', 'Duodenum', 'LK']


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            value.update(block)
    return value.hexdigest()


def write(name, value):
    target = WORK / name
    temp = target.with_suffix(target.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(target)


def stamp(state, status, **extra):
    state.update(status=status, updated_utc=datetime.now(timezone.utc).isoformat(), **extra)
    write('state.json', state)
    print('EVAL_STATE', json.dumps(state), flush=True)


def run(command, logfile, env, timeout=None):
    print('COMMAND', json.dumps(command), flush=True)
    with Path(logfile).open('ab', buffering=0) as stream:
        subprocess.run(command, env=env, check=True, stdout=stream, stderr=subprocess.STDOUT,
                       stdin=subprocess.DEVNULL, timeout=timeout)


def inspect_predictions(folder, cases):
    import nibabel as nib
    import numpy as np
    expected = {row['id'] + '.nii.gz' for row in cases}
    assert {p.name for p in folder.glob('*.nii.gz')} == expected
    rows = []
    for row in cases:
        prediction = folder / (row['id'] + '.nii.gz')
        seg, gt = nib.load(prediction), nib.load(row['gt'])
        assert seg.shape == gt.shape and np.allclose(seg.affine, gt.affine, rtol=0, atol=1e-4), row['id']
        assert np.allclose(seg.header.get_zooms(), gt.header.get_zooms(), rtol=0, atol=1e-5), row['id']
        values = np.asanyarray(seg.dataobj)
        assert np.issubdtype(values.dtype, np.integer) and values.min() >= 0 and values.max() <= 13
        assert np.any(values > 0), row['id']
        rows.append({'case': row['id'], 'bytes': prediction.stat().st_size,
                     'sha256': sha(prediction), 'shape': list(seg.shape), 'geometry_matches_gt': True})
        del values, seg, gt
    return rows


def metrics_table(path, metric, cases):
    import numpy as np
    import pandas as pd
    frame = pd.read_csv(path).set_index('Name').sort_index()
    expected = sorted(row['id'] + '.nii.gz' for row in cases)
    assert frame.index.is_unique and list(frame.index) == expected
    assert list(frame.columns) == [organ + '_' + metric for organ in ORGANS]
    data = frame.to_numpy()
    assert data.shape == (50, 13) and np.isfinite(data).all() and (data >= 0).all() and (data <= 1).all()
    return frame


def backup(model, env):
    """Archive predictions/CSVs/logs, round-trip object and return a portable receipt."""
    package = WORK / (model + '_external50.tar')
    with tarfile.open(package, 'w') as archive:
        for name in [model, 'protocol.json', 'state.json', 'baseline_A', 'metrics_code',
                     'export_equivalence.log', 'prepare_stage2_evaluation_20261024.py',
                     'run_stage2_evaluation_20261024.py']:
            archive.add(WORK / name, arcname=name)
    remote = 'oss://backup/stage2_20261024/F_C_external50/'
    log = WORK / 'backup.log'
    run(['oss', 'mkdir', remote], log, env, 120)
    run(['oss', 'cp', str(package), remote + package.name, '-f'], log, env, 1800)
    returned = WORK / ('verify_' + package.name)
    run(['oss', 'cp', remote + package.name, str(returned), '-f'], log, env, 1800)
    expected = sha(package)
    assert returned.stat().st_size == package.stat().st_size and sha(returned) == expected
    returned.unlink()
    receipt = {'remote': remote + package.name, 'bytes': package.stat().st_size,
               'sha256': expected, 'roundtrip_passed': True,
               'verified_utc': datetime.now(timezone.utc).isoformat(), 'shutdown_authorized': False}
    write(model + '/backup_receipt.json', receipt)
    run(['oss', 'cp', str(WORK / model / 'backup_receipt.json'),
         remote + model + '_backup_receipt.json', '-f'], log, env, 120)
    return receipt


def main():
    lock = (WORK / 'evaluation.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (WORK / 'state.json').exists():
        raise RuntimeError('Queue already launched; inspect prior state before any explicit recovery')
    protocol = json.loads((WORK / 'protocol.json').read_text())
    assert protocol['status'] == 'ready' and protocol['sequence'] == ['F', 'C']
    assert protocol['shutdown_authorized'] is False
    # User requested local weight backup and work-plan records before evaluation starts.
    local = json.loads((WORK / 'local_backup_check.json').read_text())
    assert local['status'] == 'local_C_archive_verified'
    assert local['sha256'] == '1e11f84070920cc00399d66f9ad3d9991e926bf37713c0964f9acc4b412c6937'
    assert (WORK / 'WORK_PLAN_v1.15_20261024.md').is_file()
    env = os.environ.copy()
    env.update(PYTHONPATH=str(WORK / 'runtime'), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', nnUNet_compile='false', WANDB_MODE='disabled',
               CUDA_VISIBLE_DEVICES='0', nnUNet_raw='/hy-tmp/Hetero-UNet/data/nnUNet_raw',
               nnUNet_preprocessed='/hy-tmp/Hetero-UNet/data/nnUNet_preprocessed',
               nnUNet_results='/hy-tmp/stage2_results', PYTHONUNBUFFERED='1')
    state = {'sequence': ['F', 'C'], 'models': {}, 'shutdown_authorized': False,
             'started_utc': datetime.now(timezone.utc).isoformat()}
    stamp(state, 'preflight')
    try:
        for row in protocol['cases']:
            assert sha(row['image']) == row['image_sha256'] and sha(row['gt']) == row['gt_sha256']
        for name, value in protocol['metrics_sha256'].items():
            assert sha(WORK / 'metrics_code' / name) == value
        assert sha(WORK / 'runtime/nnunetv2/inference/export_prediction.py') == protocol['low_memory_export_sha256']
        for model in protocol['sequence']:
            config = protocol['models'][model]
            assert sha(config['checkpoint']) == config['checkpoint_sha256']
            destination = WORK / model
            destination.mkdir()
            predictions = destination / 'predictions'
            start = time.monotonic()
            stamp(state, 'predicting', current_model=model)
            run(['nnUNetv2_predict_from_modelfolder', '-i', str(Path(protocol['cases'][0]['image']).parent),
                 '-o', str(predictions), '-m', config['model_folder'], '-f', 'all',
                 '-chk', 'checkpoint_best.pth', '--disable_tta', '-step_size', '0.5',
                 '-npp', '1', '-nps', '1', '--disable_progress_bar'], destination / 'predict.log', env)
            rows = inspect_predictions(predictions, protocol['cases'])
            write(model + '/prediction_validation.json', {'status': 'passed', 'cases': rows})
            stamp(state, 'evaluating', current_model=model)
            metric_env = env.copy()
            metric_env['PYTHONPATH'] = str(WORK / 'metrics_code') + ':' + env['PYTHONPATH']
            summary = {'model': model, 'cases': 50, 'checkpoint_sha256': config['checkpoint_sha256'],
                       'inference_and_geometry_seconds': round(time.monotonic() - start, 2),
                       'macro_definition': 'mean over the same 50 cases and 13 equally weighted organs'}
            for metric, script in [('DSC', 'abdomen_DSC_Eval.py'), ('NSD', 'nsd_parallel.py')]:
                csv = destination / (model + '_' + metric + '.csv')
                command = [sys.executable, str(WORK / 'metrics_code' / script),
                           '--gt_path', str(Path(protocol['cases'][0]['gt']).parent),
                           '--seg_path', str(predictions), '--save_path', str(csv)]
                if metric == 'NSD':
                    command += ['--workers', '2']
                run(command, destination / (metric + '.log'), metric_env)
                frame = metrics_table(csv, metric, protocol['cases'])
                summary[metric + '_percent'] = float(frame.to_numpy().mean() * 100)
                summary[metric + '_per_organ_percent'] = {k: float(v * 100) for k, v in frame.mean().items()}
            summary['inference_geometry_and_metrics_seconds'] = round(time.monotonic() - start, 2)
            write(model + '/summary.json', summary)
            state['models'][model] = {'evaluation': 'completed', 'summary': summary}
            stamp(state, 'backing_up', current_model=model)
            try:
                state['models'][model]['backup'] = backup(model, env)
            except Exception as error:
                # Preserve complete local results and continue the other pre-authorized model.
                state['models'][model]['backup_pending'] = str(error)
                print('BACKUP_PENDING', model, repr(error), flush=True)
            stamp(state, 'model_completed', current_model=model)
        import pandas as pd
        comparisons = {}
        for metric in ['DSC', 'NSD']:
            frames = {'A': metrics_table(WORK / 'baseline_A' / (metric + '.csv'), metric, protocol['cases'])}
            for model in ['F', 'C']:
                frames[model] = metrics_table(WORK / model / (model + '_' + metric + '.csv'), metric, protocol['cases'])
            paired = pd.DataFrame({key: frame.mean(axis=1) * 100 for key, frame in frames.items()})
            for left, right in [('F', 'A'), ('C', 'A'), ('C', 'F')]:
                paired[left + '_minus_' + right + '_pp'] = paired[left] - paired[right]
            paired.to_csv(WORK / ('paired_' + metric + '.csv'))
            comparisons[metric] = {'macro_percent': {k: float(v.to_numpy().mean() * 100) for k, v in frames.items()},
                                   'C_minus_F_pp': float(paired['C_minus_F_pp'].mean()),
                                   'single_seed_exploratory': True}
        write('comparison.json', comparisons)
        all_verified = all('backup' in state['models'][m] for m in ['F', 'C'])
        stamp(state, 'completed' if all_verified else 'completed_backup_pending', current_model=None,
              all_model_archives_verified=all_verified)
        # Upload the compact final comparison/state separately; archives already contain predictions.
        try:
            remote = 'oss://backup/stage2_20261024/F_C_external50/'
            for name in ['comparison.json', 'paired_DSC.csv', 'paired_NSD.csv', 'state.json', 'protocol.json']:
                run(['oss', 'cp', str(WORK / name), remote + name, '-f'], WORK / 'backup.log', env, 120)
        except Exception as error:
            stamp(state, 'completed_backup_pending', final_report_backup_error=str(error))
    except Exception:
        stamp(state, 'failed', error=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
