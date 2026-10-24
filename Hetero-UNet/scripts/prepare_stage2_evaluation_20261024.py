"""Freeze the F/C external evaluation runtime without changing training sources."""
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

WORK = Path('/hy-tmp/stage2_eval_20261024')
REPO = Path('/hy-tmp/Hetero-UNet-stage2')
PROJECT = REPO / 'Hetero-UNet'
RAW = Path('/hy-tmp/Hetero-UNet/data/nnUNet_raw/Dataset701_AbdomenCT')
OLD = Path('/hy-tmp/stage2_audit_2026-10-21/pred50_artifacts/backup50')
TRAINERS = {'F': 'nnUNetTrainerMambaUNetBlockAttnRes',
            'C': 'nnUNetTrainerMambaUNetBGBlockAttnRes'}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            value.update(block)
    return value.hexdigest()


def main():
    WORK.mkdir(exist_ok=True)
    if (WORK / 'protocol.json').exists():
        raise RuntimeError('Evaluation already prepared; refuse to replace frozen runtime')
    assert shutil.disk_usage('/hy-tmp').free > 8 * 1024 ** 3
    original = PROJECT / 'umamba/nnunetv2/inference/export_prediction.py'
    patched = OLD / 'export_prediction_lowmem_patch.py.bak'
    # Verify that this archived baseline patch changes only conversion, not the model/export APIs.
    def other_nodes(path):
        return [ast.dump(n, include_attributes=False) for n in ast.parse(path.read_text()).body
                if not isinstance(n, ast.FunctionDef) or
                n.name != 'convert_predicted_logits_to_segmentation_with_correct_shape']
    assert other_nodes(original) == other_nodes(patched)
    runtime = WORK / 'runtime'
    shutil.copytree(PROJECT / 'umamba/nnunetv2', runtime / 'nnunetv2',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(patched, runtime / 'nnunetv2/inference/export_prediction.py')
    metrics = WORK / 'metrics_code'
    shutil.copytree(OLD / 'eval', metrics)
    for name in ['SurfaceDice.py', 'abdomen_DSC_Eval.py', 'abdomen_NSD_Eval.py']:
        assert sha(metrics / name) == sha(PROJECT / 'evaluation' / name), name
    assert sha(metrics / 'nsd_parallel.py') == sha(REPO / 'evaluation/nsd_parallel.py')
    inputs = sorted((RAW / 'imagesVal').glob('*.nii.gz'))
    gt = sorted((RAW / 'labelsVal').glob('*.nii.gz'))
    ids = [p.name[:-12] for p in inputs]  # remove _0000.nii.gz
    assert len(inputs) == len(gt) == 50 and len(set(ids)) == 50
    assert sorted(ids) == sorted(p.name[:-7] for p in gt)
    assert not set(ids) & {p.name[:-12] for p in (RAW / 'imagesTr').glob('*.nii.gz')}
    cases = [{'id': case, 'image': str(image), 'image_sha256': sha(image),
              'gt': str(RAW / 'labelsVal' / (case + '.nii.gz')),
              'gt_sha256': sha(RAW / 'labelsVal' / (case + '.nii.gz'))}
             for case, image in zip(ids, inputs)]
    models = {}
    previous = None
    for key, trainer in TRAINERS.items():
        folder = Path('/hy-tmp/stage2_results') / ('stage2_' + key + '_seed1234') / \
                 'Dataset701_AbdomenCT' / (trainer + '__nnUNetPlans__3d_fullres')
        manifest = json.loads((folder / 'fold_all/stage2_run.json').read_text())
        assert manifest['status'] == 'completed'
        if previous is not None:
            assert manifest['fingerprints'] == previous['fingerprints']
        previous = manifest
        for name, value in manifest['fingerprints']['source_sha256'].items():
            assert sha(PROJECT / name) == value, name
            if name.startswith('umamba/'):
                assert sha(runtime / name[len('umamba/'):]) == value, name
        checkpoint = folder / 'fold_all/checkpoint_best.pth'
        models[key] = {'trainer': trainer, 'model_folder': str(folder),
                       'checkpoint': str(checkpoint), 'checkpoint_sha256': sha(checkpoint),
                       'run_id': manifest['run_id'], 'source_commit': manifest['git']['commit'],
                       'plans_sha256': sha(folder / 'plans.json'),
                       'dataset_sha256': sha(folder / 'dataset.json')}
    assert models['F']['plans_sha256'] == models['C']['plans_sha256']
    assert models['F']['dataset_sha256'] == models['C']['dataset_sha256']
    (WORK / 'baseline_A').mkdir()
    for metric in ['DSC', 'NSD']:
        shutil.copy2(OLD / ('pred_3d_fullres_' + metric + '.csv'), WORK / 'baseline_A' / (metric + '.csv'))
    # Actual archived converter on tiny resized FP16/FP32 logits; compare segmentation and probabilities.
    check = r'''
import importlib.util,json,numpy as np,torch
from pathlib import Path
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape as patched
spec=importlib.util.spec_from_file_location('original_export',__import__('sys').argv[1])
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
folder=Path(__import__('sys').argv[2]);plans=PlansManager(json.loads((folder/'plans.json').read_text()))
config=plans.get_configuration('3d_fullres');label=plans.get_label_manager(json.loads((folder/'dataset.json').read_text()))
rows=[]
for dtype in [np.float16,np.float32]:
 for spacing in [[1.0,1.0,1.0],[5.0,0.8,0.8]]:
  data=np.random.default_rng(1234).normal(size=(14,5,9,7)).astype(dtype)
  props={'spacing':spacing,'shape_after_cropping_and_before_resampling':[7,13,11],
         'shape_before_cropping':[9,15,13],'bbox_used_for_cropping':[[1,8],[1,14],[1,12]]}
  a=module.convert_predicted_logits_to_segmentation_with_correct_shape(data,plans,config,label,props,True,1)
  b=patched(data,plans,config,label,props,True,1)
  assert np.array_equal(a[0],b[0]) and np.array_equal(a[1],b[1])
  rows.append({'dtype':str(dtype),'spacing':spacing,'segmentation_and_probabilities_identical':True})
print(json.dumps(rows))
'''
    env = os.environ.copy()
    env['PYTHONPATH'] = str(runtime)
    completed = subprocess.run([sys.executable, '-c', check, str(original), models['F']['model_folder']],
                               env=env, check=True, capture_output=True, text=True, timeout=120)
    (WORK / 'export_equivalence.log').write_text(completed.stdout + completed.stderr)
    equivalence = json.loads(completed.stdout.strip().splitlines()[-1])
    protocol = {'status': 'ready', 'created_utc': datetime.now(timezone.utc).isoformat(),
                'sequence': ['F', 'C'], 'models': models, 'cases': cases,
                'external_cases': 50, 'fold': 'all', 'checkpoint': 'checkpoint_best.pth',
                'tta': False, 'step_size': 0.5, 'gaussian': True, 'patch_batch': 1,
                'preprocessing_workers': 1, 'export_workers': 1, 'nsd_workers': 2,
                'postprocessing': 'none', 'runtime': str(runtime),
                'low_memory_export_sha256': sha(patched), 'original_export_sha256': sha(original),
                'export_equivalence_checks': equivalence,
                'metrics_sha256': {p.name: sha(p) for p in metrics.glob('*.py')},
                'baseline_A_csv_sha256': {p.name: sha(p) for p in (WORK / 'baseline_A').glob('*.csv')},
                'MHA': 'official batch_first=False unchanged; one patch per forward',
                'shutdown_authorized': False, 'training_source_unmodified': True,
                'limitations': ['single-seed exploratory comparison',
                               'F resumed across 4090/V100; C trained on V100 throughout',
                               'baseline A exporter used original for 49 cases and equivalent patch for 0050']}
    (WORK / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    print(json.dumps({'status': 'ready', 'cases': len(cases), 'models': models,
                      'export_equivalence_checks': equivalence}, indent=2), flush=True)


if __name__ == '__main__':
    main()
