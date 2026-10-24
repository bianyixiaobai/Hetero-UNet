"""Verify completed C checkpoints, all epochs and fresh OSS backups; never shut down."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import tarfile

WORK = Path('/hy-tmp/stage2_C_completed_20261024')
AUDIT = Path('/hy-tmp/stage2_C_audit_2026-10-22')
BACKUP = Path('/hy-tmp/stage2_C_backup_20261022')
REPO = Path('/hy-tmp/Hetero-UNet-stage2')


def digest(stream):
    result = hashlib.sha256()
    for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
        result.update(block)
    return result.hexdigest()


def sha(path):
    with Path(path).open('rb') as stream:
        return digest(stream)


def write(name, value):
    temporary = WORK / (name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(WORK / name)


def transfer(args):
    with (WORK / 'transfers.log').open('ab', buffering=0) as output:
        subprocess.run(args, check=True, stdout=output, stderr=subprocess.STDOUT,
                       stdin=subprocess.DEVNULL, timeout=7200, cwd=WORK)


def main():
    WORK.mkdir(exist_ok=True)
    assert socket.gethostname() == 'I2bde78031f00701d9e'
    os.nice(10)
    state = json.loads((BACKUP / 'state.json').read_text())
    assert state['status'] == 'verified_not_armed' and state['shutdown_authorized'] is False
    queue = json.loads((AUDIT / 'C_queue.json').read_text())
    assert queue['status'] == 'completed' and queue['sequence'] == ['C'] and queue['epochs_each'] == 1000
    model = next(Path('/hy-tmp/stage2_results/stage2_C_seed1234').glob('Dataset*/*/fold_all'))
    manifest = json.loads((model / 'stage2_run.json').read_text())
    previous_F = json.loads(next(Path('/hy-tmp/stage2_results/stage2_F_seed1234').rglob('stage2_run.json')).read_text())
    assert manifest['status'] == 'completed' and manifest['resume_checkpoint'] is None
    assert manifest['fingerprints'] == previous_F['fingerprints']
    assert manifest['git']['commit'] == '9fe70ac570a0e14ebce30daa4a16a22e49b4f3fd'
    for name, expected in manifest['fingerprints']['source_sha256'].items():
        assert sha(REPO / 'Hetero-UNet' / name) == expected, name
    subprocess.run(['git', '-C', str(REPO), 'diff', '--exit-code', 'HEAD'], check=True)
    log = (AUDIT / 'C_queue.log').read_text()
    epochs = list(map(int, re.findall(r': Epoch (\d+)\s*$', log, re.M)))
    train = list(map(float, re.findall(r': train_loss ([\-0-9.eE]+)', log)))
    val = list(map(float, re.findall(r': val_loss ([\-0-9.eE]+)', log)))
    components = [tuple(map(float, p)) for p in re.findall(
        r'Stage2 loss components: segmentation= ([\-0-9.eE]+) boundary_unweighted= ([\-0-9.eE]+) boundary_weight= ([\-0-9.eE]+)', log)]
    times = list(map(float, re.findall(r'Epoch time: ([0-9.]+) s', log)))
    assert epochs == list(range(1000)) and len(train) == len(val) == len(components) == len(times) == 1000
    assert all(math.isfinite(x) for x in train + val + [x for p in components for x in p])
    assert all(w == 0.1 and b >= 0 for _, b, w in components)
    discrepancy = max(abs(t - (s + w * b)) for t, (s, b, w) in zip(train, components))
    assert discrepancy < 0.001
    beijing = timezone(timedelta(hours=8))
    begun = re.search(r'([0-9-]+ [0-9:.]+): Epoch 0\s*$', log, re.M).group(1)
    ended = re.search(r'([0-9-]+ [0-9:.]+): Training done\.', log).group(1)
    begun = datetime.fromisoformat(begun).replace(tzinfo=beijing)
    ended = datetime.fromisoformat(ended).replace(tzinfo=beijing)
    import torch
    torch.set_num_threads(1)
    checkpoints = []
    for name in ['checkpoint_best.pth', 'checkpoint_final.pth']:
        path = model / name
        checkpoint = torch.load(path, map_location='cpu')
        epoch = checkpoint['current_epoch']
        assert 1 <= epoch <= 1000 and (name != 'checkpoint_final.pth' or epoch == 1000)
        assert checkpoint['trainer_name'] == 'nnUNetTrainerMambaUNetBGBlockAttnRes'
        assert checkpoint['network_weights'] and checkpoint['optimizer_state']
        assert all(torch.isfinite(t).all().item() for t in checkpoint['network_weights'].values() if torch.is_floating_point(t))
        checkpoints.append({'checkpoint': name, 'epoch': epoch, 'bytes': path.stat().st_size,
                            'sha256': sha(path), 'network_finite': True, 'optimizer_present': True})
        del checkpoint
    fresh = []
    archive_members = []
    for name, obj in state['verified'].items():
        print('FRESH_OSS_CHECK', name, flush=True)
        downloaded = WORK / ('fresh_' + name)
        transfer(['oss', 'cp', obj['remote'], str(downloaded), '-f'])
        assert downloaded.stat().st_size == obj['bytes'] and sha(downloaded) == obj['sha256']
        if name == 'C_seed1234_final.tar':
            with tarfile.open(downloaded) as archive:
                for member in archive:
                    if not member.isfile():
                        continue
                    rel = Path(member.name)
                    assert not rel.is_absolute() and '..' not in rel.parts and rel.parts[0] == model.parent.name
                    target = model.parent.parent / rel
                    with archive.extractfile(member) as stream:
                        assert member.size == target.stat().st_size and digest(stream) == sha(target)
                    archive_members.append(str(target))
            assert len(archive_members) == len([p for p in model.parent.rglob('*') if p.is_file()])
        fresh.append({**obj, 'fresh_verified_utc': datetime.now(timezone.utc).isoformat()})
        downloaded.unlink()
    report = {'status': 'completed_C_verified', 'checked_utc': datetime.now(timezone.utc).isoformat(),
              'started_beijing': begun.isoformat(), 'finished_beijing': ended.isoformat(),
              'duration_seconds': (ended - begun).total_seconds(), 'epochs_completed': 1000,
              'run_id': manifest['run_id'], 'source_commit': manifest['git']['commit'],
              'F_C_source_and_descriptor_fingerprints_match': True, 'source_files_matched': 14,
              'checkpoints': checkpoints, 'training_loss_count': len(train), 'validation_loss_count': len(val),
              'boundary_loss_count': len(components), 'all_epoch_losses_finite': True,
              'max_loss_composition_rounding_error': discrepancy, 'mean_epoch_compute_seconds': sum(times) / len(times),
              'last_train_loss': train[-1], 'last_validation_loss': val[-1],
              'fresh_OSS_objects': fresh, 'archive_files_individually_matched': len(archive_members),
              'shutdown_authorized': False, 'backup_state': state['status'], 'instance_kept_running': True,
              'external_50_evaluation_performed': False, 'F_aux_formal_training_started': False,
              'disk': shutil.disk_usage('/hy-tmp')._asdict()}
    write('completion_check.json', report)
    for path, name in [(model / 'stage2_run.json', 'C_completed_manifest.json'),
                       (BACKUP / 'state.json', 'final_backup_state.json'),
                       (BACKUP / 'backup_receipt.json', 'backup_receipt.json')]:
        shutil.copy2(path, WORK / name)
    package = WORK / 'completion_review.tar'
    with tarfile.open(package, 'w') as archive:
        for name in ['completion_check.json', 'C_completed_manifest.json', 'final_backup_state.json', 'backup_receipt.json']:
            archive.add(WORK / name, arcname=name)
        archive.add(Path(__file__), arcname=Path(__file__).name)
    prefix = 'oss://backup/stage2_20261024/C_completed_review'
    transfer(['oss', 'mkdir', prefix + '/'])
    transfer(['oss', 'cp', str(package), prefix + '/' + package.name, '-f'])
    returned = WORK / 'verify_completion_review.tar'
    transfer(['oss', 'cp', prefix + '/' + package.name, str(returned), '-f'])
    assert sha(returned) == sha(package)
    returned.unlink()
    report['completion_review_backup'] = {'remote': prefix + '/' + package.name,
                                         'sha256': sha(package), 'roundtrip_passed': True}
    write('completion_check.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
