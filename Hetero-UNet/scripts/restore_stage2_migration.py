"""Download and verify the paused epoch500 experiment on the empty new instance."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tarfile
import time

WORK = Path('/hy-tmp/stage2_migration_20261022')
AUDIT = Path('/hy-tmp/stage2_audit_2026-10-22')
PREFIX = 'oss://backup/stage2_20261022/migration_epoch500'


def checksum(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            value.update(block)
    return value.hexdigest()


def command(args, required=True):
    with (AUDIT / 'migration_transfers.log').open('ab', buffering=0) as log:
        result = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, timeout=7200)
    if required and result.returncode:
        raise RuntimeError('Command failed: ' + args[0] + '; inspect migration_transfers.log')
    return result.returncode


def state(status, **details):
    report = {'status': status, 'updated_utc': datetime.now(timezone.utc).isoformat(), **details}
    (AUDIT / 'migration_restore.json').write_text(json.dumps(report, indent=2) + '\n')
    print(status, flush=True)


def extract_checked(archive, destination, expected_roots):
    destination.mkdir(parents=True, exist_ok=True)
    base = destination.resolve()
    with tarfile.open(archive) as stream:
        for member in stream.getmembers():
            relative = Path(member.name)
            if (relative.is_absolute() or '..' in relative.parts or
                    relative.parts[0] not in expected_roots or
                    not (member.isfile() or member.isdir())):
                raise RuntimeError('Unsafe/unexpected member: ' + member.name)
            (base / relative).resolve().relative_to(base)
        stream.extractall(base)


def verify_plan_descriptor(descriptor, manifest):
    expected = manifest['fingerprints']['descriptor_sha256']['nnUNetPlans.json']
    if not re.fullmatch(r'[0-9a-f]{64}', expected):
        raise RuntimeError('Original training manifest has an invalid plans SHA256')
    if checksum(descriptor) != expected:
        raise RuntimeError('Restored plans differ from the original training manifest')
    return expected


def main():
    WORK.mkdir(exist_ok=True)
    AUDIT.mkdir(exist_ok=True)
    receipt = WORK / 'migration_receipt.tar'
    state('waiting_for_source_receipt')
    while True:
        command(['oss', 'cp', PREFIX + '/migration_receipt.tar', str(receipt), '-f'], required=False)
        if receipt.is_file():
            try:
                with tarfile.open(receipt) as stream:
                    metadata = json.load(stream.extractfile('migration_objects.json'))
                break
            except (tarfile.TarError, json.JSONDecodeError, KeyError, OSError):
                pass  # An unsuccessful CLI download may leave an incomplete file.
        time.sleep(10)
    expected = {'F_epoch500_results.tar', 'CT701_preprocessed_exact.tar', 'migration_audit.tar'}
    if (metadata['resume_epoch'] != 500 or metadata['target_total_epochs'] != 1000 or
            {row['name'] for row in metadata['objects']} != expected or
            metadata['training_commit'] != '9fe70ac570a0e14ebce30daa4a16a22e49b4f3fd'):
        raise RuntimeError('Receipt does not match the authorized experiment')
    verified = []
    for row in metadata['objects']:
        if row['remote'] != PREFIX + '/' + row['name']:
            raise RuntimeError('Unexpected OSS location')
        path = WORK / row['name']
        state('downloading_' + row['name'])
        command(['oss', 'cp', row['remote'], str(path), '-f'])
        if path.stat().st_size != row['bytes'] or checksum(path) != row['sha256']:
            raise RuntimeError('Download checksum or size mismatch: ' + row['name'])
        verified.append({**row, 'verified_utc': datetime.now(timezone.utc).isoformat()})
    (AUDIT / 'migration_verified_objects.json').write_text(json.dumps(verified, indent=2) + '\n')
    results = Path('/hy-tmp/stage2_results')
    data = Path('/hy-tmp/Hetero-UNet/data/nnUNet_preprocessed')
    for existing in [results / 'stage2_F_seed1234', data / 'Dataset701_AbdomenCT',
                     Path('/hy-tmp/Hetero-UNet-stage2')]:
        if existing.exists():
            raise FileExistsError('Refusing to overwrite existing experiment: ' + str(existing))
    state('extracting_verified_archives')
    extract_checked(WORK / 'F_epoch500_results.tar', results, {'stage2_F_seed1234'})
    extract_checked(WORK / 'CT701_preprocessed_exact.tar', data, {'Dataset701_AbdomenCT'})
    recovered = WORK / 'recovered_audit'
    extract_checked(WORK / 'migration_audit.tar', recovered,
                    {'training_source.bundle', 'snapshot_stage2_migration.py', 'audit_previous_instance'})
    source = Path('/hy-tmp/Hetero-UNet-stage2')
    command(['git', 'init', str(source)])
    command(['git', '-C', str(source), 'fetch', str(recovered / 'training_source.bundle'), 'HEAD'])
    command(['git', '-C', str(source), 'checkout', '--detach', 'FETCH_HEAD'])
    command(['git', '-C', str(source), 'remote', 'add', 'origin', 'https://github.com/bianyixiaobai/Hetero-UNet.git'])
    command(['python', '-m', 'pip', 'install', '-e', str(source / 'Hetero-UNet/umamba'), '--no-deps'])
    preprocessed = data / 'Dataset701_AbdomenCT/nnUNetPlans_3d_fullres'
    assert len(list(preprocessed.glob('*.npz'))) == 50
    assert len(list(preprocessed.glob('*.pkl'))) == 50
    assert not list(preprocessed.glob('*.npy'))
    descriptor = data / 'Dataset701_AbdomenCT/nnUNetPlans.json'
    manifests = list((results / 'stage2_F_seed1234').glob('Dataset*/*/fold_all/stage2_run.json'))
    assert len(manifests) == 1
    verify_plan_descriptor(descriptor, json.loads(manifests[0].read_text()))
    state('restored_verified', resume_epoch=500, target_total_epochs=1000,
          training_cases=50, exact_preprocessing_reused=True,
          source_commit=metadata['training_commit'], verified_objects=verified)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        state('failed_requires_review', error=type(error).__name__ + ': ' + str(error))
        raise
