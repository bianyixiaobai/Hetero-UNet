"""Read-only C startup audit against the completed F and frozen CT arrays."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys

AUDIT = Path('/hy-tmp/stage2_C_audit_2026-10-22')
REPO = Path('/hy-tmp/Hetero-UNet-stage2')
DATA = Path('/hy-tmp/Hetero-UNet/data')
COMMIT = '9fe70ac570a0e14ebce30daa4a16a22e49b4f3fd'


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            value.update(block)
    return value.hexdigest()


def main():
    AUDIT.mkdir(exist_ok=True)
    assert socket.gethostname() == 'I2bde78031f00701d9e'
    assert subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip() == COMMIT
    assert not subprocess.check_output(['git', '-C', str(REPO), 'status', '--porcelain'], text=True).strip()
    target = Path('/hy-tmp/stage2_results/stage2_C_seed1234')
    if target.exists() and list(target.rglob('*.pth')):
        raise RuntimeError('Formal C checkpoints already exist; refuse fresh start')
    if shutil.disk_usage('/hy-tmp').free < 8 * 1024 ** 3:
        raise RuntimeError('Insufficient operating headroom')
    f = next(Path('/hy-tmp/stage2_results/stage2_F_seed1234').rglob('stage2_run.json'))
    previous = json.loads(f.read_text())
    assert previous['status'] == 'completed' and previous['protocol']['num_epochs'] == 1000
    assert previous['protocol']['seed'] == 1234 and previous['protocol']['batch_size'] == 2
    assert previous['protocol']['patch_size'] == [40, 224, 192]
    assert previous['git']['commit'] == COMMIT
    for relative, expected in previous['fingerprints']['source_sha256'].items():
        assert sha(REPO / 'Hetero-UNet' / relative) == expected, relative
    manifest = Path('/hy-tmp/full_preservation_20261022/complete_manifest.json')
    assert sha(manifest) == '6f9c28d062be07f7f38f2bf76a02ff0b4d3e4c690c2963ee401c44c93fe53c48'
    rows = [r for r in json.loads(manifest.read_text())['entries'] if r['kind'] == 'file' and
            r['path'].startswith('Hetero-UNet/data/nnUNet_preprocessed/')]
    for index, row in enumerate(rows):
        path = Path('/hy-tmp') / row['path']
        assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], row['path']
        if index % 50 == 0:
            print('PREPROCESSED_SHA256_CHECK', index, '/', len(rows), flush=True)
    base = DATA / 'nnUNet_preprocessed/Dataset701_AbdomenCT'
    descriptor = base / 'nnUNetPlans.json'
    assert sha(descriptor) == previous['fingerprints']['descriptor_sha256']['nnUNetPlans.json']
    assert sha(base / 'dataset.json') == previous['fingerprints']['descriptor_sha256']['dataset.json']
    plans = json.loads(descriptor.read_text())['configurations']['3d_fullres']
    assert plans['batch_size'] == 2 and plans['patch_size'] == [40, 224, 192]
    arrays = base / 'nnUNetPlans_3d_fullres'
    counts = {ext: len(list(arrays.glob('*.' + ext))) for ext in ['npz', 'pkl', 'npy']}
    assert counts == {'npz': 50, 'pkl': 50, 'npy': 100}
    raw = DATA / 'nnUNet_raw/Dataset701_AbdomenCT'
    raw_counts = {name: len(list((raw / name).glob('*.nii.gz'))) for name in
                  ['imagesTr', 'labelsTr', 'imagesVal', 'labelsVal']}
    assert all(v == 50 for v in raw_counts.values())
    import torch
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMambaUNetAttnRes import nnUNetTrainerMambaUNetBGBlockAttnRes as C
    assert C.boundary_guidance and C.auxiliary_boundary and C.group_size == 3
    assert C.boundary_hidden_channels == 8 and C.boundary_loss_weight == 0.1
    spec = importlib.util.spec_from_file_location(
        'startup_finalizer', AUDIT / 'finalize_stage2.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    report = {'status': 'passed', 'checked_utc': datetime.now(timezone.utc).isoformat(),
              'hostname': socket.gethostname(), 'source_commit': COMMIT, 'source_clean': True,
              'source_fingerprints_matched': len(previous['fingerprints']['source_sha256']),
              'preprocessed_files_SHA_matched': len(rows), 'preprocessed_bytes_SHA_matched': sum(r['bytes'] for r in rows),
              'data_counts': counts, 'raw_counts': raw_counts, 'plans_sha256': sha(descriptor),
              'previous_F_completed': True, 'start_from_scratch': True, 'checkpoint_inherited_from_F': False,
              'python': sys.version, 'torch': torch.__version__, 'CUDA': torch.version.cuda,
              'gpu': torch.cuda.get_device_name(0), 'gpu_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
              'mamba_ssm': importlib.metadata.version('mamba-ssm'),
              'causal_conv1d': importlib.metadata.version('causal-conv1d'),
              'physical_batch': 2, 'patch': [40, 224, 192], 'epochs': 1000, 'seed': 1234,
              'boundary_loss_weight': 0.1, 'group_size': 3, 'boundary_hidden_channels': 8,
              'MHA_batch_first': False, 'disk': shutil.disk_usage('/hy-tmp')._asdict(),
              'OSS_expiry_utc': helper.auth_expiry('/root/.hycloud_ossutil_config'),
              'external_50_evaluation_performed': False}
    (AUDIT / 'preflight.json').write_text(json.dumps(report, indent=2) + '\n')
    subprocess.run(['python', '-m', 'pip', 'freeze'], check=True,
                   stdout=(AUDIT / 'environment_pip_freeze.txt').open('w'))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
