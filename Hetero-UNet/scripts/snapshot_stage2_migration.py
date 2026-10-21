"""Seal this paused F experiment and exact compressed preprocessing into OSS.

No passwords/config credentials are included. Source code and frozen protocol
remain unchanged; only already-paused experiment files are read.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile

AUDIT = Path('/hy-tmp/stage2_audit_2026-10-21')
WORK = Path('/hy-tmp/stage2_migration_20261022')
PREFIX = 'oss://backup/stage2_20261022/migration_epoch500'
REPO = Path('/hy-tmp/Hetero-UNet-stage2')
DATA = Path('/hy-tmp/Hetero-UNet/data/nnUNet_preprocessed')


def checksum(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def command(args):
    subprocess.run(args, check=True, stdin=subprocess.DEVNULL)


def save(value):
    (WORK / 'migration_objects.json').write_text(json.dumps(value, indent=2) + '\n')


def main():
    WORK.mkdir(exist_ok=True)
    os.nice(10)
    pause = json.loads((AUDIT / 'migration_pause.json').read_text())
    assert pause['status'] == 'paused_for_migration' and pause['checkpoints'][0]['epoch'] == 500
    pid = pause['trainer_process']['pid']
    stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    assert stat[0] == 'T' and int(stat[19]) == pause['trainer_process']['start_ticks']
    assert subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip() == \
        '9fe70ac570a0e14ebce30daa4a16a22e49b4f3fd'
    state = {'status': 'packaging', 'resume_epoch': 500, 'target_total_epochs': 1000,
             'training_commit': '9fe70ac570a0e14ebce30daa4a16a22e49b4f3fd',
             'created_utc': datetime.now(timezone.utc).isoformat(), 'objects': []}
    save(state)
    result = WORK / 'F_epoch500_results.tar'
    command(['tar', '-cf', str(result), '-C', '/hy-tmp/stage2_results', 'stage2_F_seed1234'])
    data = WORK / 'CT701_preprocessed_exact.tar'
    command(['tar', '--exclude=*.npy', '-cf', str(data), '-C', str(DATA), 'Dataset701_AbdomenCT'])
    bundle = WORK / 'training_source.bundle'
    command(['git', '-C', str(REPO), 'bundle', 'create', str(bundle), 'HEAD'])
    audit = WORK / 'migration_audit.tar'
    with tarfile.open(audit, 'w') as archive:
        archive.add(bundle, arcname='training_source.bundle')
        archive.add(Path(__file__), arcname='snapshot_stage2_migration.py')
        for path in sorted(AUDIT.iterdir()):
            if (path.is_file() and not path.is_symlink() and path.suffix in
                    {'.json', '.log', '.txt', '.py', '.sh', '.md'} and path.stat().st_size < 100 * 1024 ** 2):
                archive.add(path, arcname='audit_previous_instance/' + path.name)
        for path in sorted((AUDIT / 'cuda_full_patch').glob('*.json')):
            archive.add(path, arcname='audit_previous_instance/cuda_full_patch/' + path.name)
    command(['oss', 'mkdir', PREFIX + '/'])
    for archive in (result, data, audit):
        row = {'name': archive.name, 'bytes': archive.stat().st_size,
               'sha256': checksum(archive), 'remote': PREFIX + '/' + archive.name}
        state['objects'].append(row)
        save(state)
        command(['oss', 'cp', str(archive), row['remote'], '-f'])
        row['upload_completed_utc'] = datetime.now(timezone.utc).isoformat()
        save(state)
    state['status'] = 'uploaded_pending_destination_verification'
    save(state)
    receipt = WORK / 'migration_receipt.tar'
    with tarfile.open(receipt, 'w') as archive:
        archive.add(WORK / 'migration_objects.json', arcname='migration_objects.json')
    command(['oss', 'cp', str(receipt), PREFIX + '/' + receipt.name, '-f'])
    print('MIGRATION_BACKUP_UPLOADED', json.dumps(state), flush=True)


if __name__ == '__main__':
    main()
