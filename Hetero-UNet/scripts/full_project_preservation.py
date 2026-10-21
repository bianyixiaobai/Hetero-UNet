"""Preserve every old /hy-tmp file and root project/environment records.

Existing verified OSS archives provide content-addressed backing. Missing bytes
are uploaded in bounded tar chunks. Import never overwrites a differing live
file, and keeps the original F results/code in a separate archival directory.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import socket
import stat
import subprocess
import tarfile
import time
import zipfile

PREFIX = 'oss://backup/stage2_20261022/complete_4090_project'
WORK = Path('/hy-tmp/full_preservation_20261022')
ARCHIVE = Path('/hy-tmp/archive_4090_20261022')
SOURCE_HOST = 'I2bdb493e5700c01f4e'
DESTINATION_HOST = 'I2bde78031f00701d9e'


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(stream):
    value = hashlib.sha256()
    for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
        value.update(block)
    return value.hexdigest()


def checksum(path):
    with path.open('rb') as stream:
        return digest(stream)


def write_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    os.replace(temporary, path)


def command(args, required=True):
    with (WORK / 'transfers.log').open('ab', buffering=0) as output:
        process = subprocess.run(args, stdin=subprocess.DEVNULL, stdout=output,
                                 stderr=subprocess.STDOUT, timeout=7200, cwd=WORK)
    if required and process.returncode:
        raise RuntimeError('Command failed: ' + args[0] + '; inspect transfers.log')
    return process.returncode


def progress(status, **details):
    write_json(WORK / 'status.json', {'status': status, 'updated_utc': now(), **details})
    print(now(), status, flush=True)


def inventory():
    # Freeze private root project/config records outside the live configuration.
    profiles = WORK / 'root_snapshot'
    profiles.mkdir(exist_ok=True)
    names = ['.bashrc', '.profile', '.condarc', '.history_command', '.bash_history',
             '.pip/pip.conf', '.jupyter/jupyter_lab_config.py', '.jupyter/jupyter_notebook_config.py']
    root_files = [Path('/root') / name for name in names]
    for pattern in ('*.py', '*.sh', '*.md', '*.ipynb', '*.yaml', '*.yml', '*.json', '*.txt'):
        root_files.extend(Path('/root').glob(pattern))
    for source in sorted(set(root_files)):
        if source.is_file() and not source.is_symlink():
            relative = source.relative_to('/root')
            target = profiles / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    pip = subprocess.check_output(['python', '-m', 'pip', 'freeze'], text=True)
    (profiles / 'environment_pip_freeze.txt').write_text(pip)
    import torch
    (profiles / 'environment_torch.txt').write_text(
        str(torch.__version__) + '\nCUDA=' + str(torch.version.cuda) + '\n' + torch.__config__.show())
    subprocess.run(['nvidia-smi', '-q'], stdout=(profiles / 'environment_nvidia.txt').open('w'), check=True)
    rows = []
    for base, directories, files in os.walk('/hy-tmp', followlinks=False):
        directories[:] = sorted(d for d in directories if Path(base) / d != WORK)
        for name in sorted(directories + files):
            path = Path(base) / name
            info = path.lstat()
            row = {'path': path.relative_to('/hy-tmp').as_posix(), 'mode': stat.S_IMODE(info.st_mode),
                   'mtime_ns': info.st_mtime_ns}
            if path.is_symlink():
                row.update(kind='symlink', target=os.readlink(path))
            elif path.is_dir():
                row.update(kind='directory')
            elif path.is_file():
                row.update(kind='file', bytes=info.st_size, sha256=checksum(path), source=str(path))
            else:
                raise RuntimeError('Unexpected non-project special file: ' + str(path))
            rows.append(row)
    for path in sorted(profiles.rglob('*')):
        if path.is_file():
            info = path.stat()
            rows.append({'path': '_root_profiles/' + path.relative_to(profiles).as_posix(),
                         'kind': 'file', 'mode': stat.S_IMODE(info.st_mode), 'mtime_ns': info.st_mtime_ns,
                         'bytes': info.st_size, 'sha256': checksum(path), 'source': str(path)})
    return rows


def known_objects():
    result = []
    for name in ('F_epoch500_results.tar', 'CT701_preprocessed_exact.tar', 'migration_audit.tar',
                 'migration_receipt.tar'):
        path = Path('/hy-tmp/stage2_migration_20261022') / name
        result.append((path, 'oss://backup/stage2_20261022/migration_epoch500/' + name))
    for path in sorted(Path('/hy-tmp/stage2_restore_archives').iterdir()):
        if path.suffix not in ('.zip', '.tar'):
            continue
        remote = ('oss://backup/' if path.name == 'pred50_artifacts_20261013.zip' else 'oss://datasets/')
        result.append((path, remote + path.name))
    result.append((Path('/hy-tmp/stage2_completed_archives/session_audit_start.tar'),
                   'oss://backup/stage2_20261021/session_audit_start.tar'))
    return [(p, remote) for p, remote in result if p.is_file()]


def index_archives(rows):
    backing = {}
    objects = []
    by_source = {row.get('source'): row for row in rows if row['kind'] == 'file'}
    for path, remote in known_objects():
        row = by_source[str(path)]
        index = len(objects)
        objects.append({'remote': remote, 'bytes': row['bytes'], 'sha256': row['sha256'],
                        'source_relative_path': row['path'], 'format': path.suffix[1:]})
        backing.setdefault(row['sha256'], {'object': index})
        if path.suffix == '.tar':
            with tarfile.open(path) as archive:
                for member in archive:
                    if member.isfile():
                        with archive.extractfile(member) as stream:
                            value = digest(stream)
                        backing.setdefault(value, {'object': index, 'member': member.name})
        else:
            with zipfile.ZipFile(path) as archive:
                for member in archive.infolist():
                    if not member.is_dir():
                        with archive.open(member) as stream:
                            value = digest(stream)
                        backing.setdefault(value, {'object': index, 'member': member.filename})
    # np.savez stores the exact NPY bytes used by unpack_dataset. Register only
    # hashes that actually match; differing headers automatically become deltas.
    for row in rows:
        if row['kind'] != 'file' or not row['path'].endswith('.npz'):
            continue
        parent = backing.get(row['sha256'])
        if not parent or 'member' not in parent:
            continue
        with zipfile.ZipFile(row['source']) as archive:
            for name in archive.namelist():
                if name.endswith('.npy'):
                    with archive.open(name) as stream:
                        value = digest(stream)
                    backing.setdefault(value, {**parent, 'nested_member': name})
    return objects, backing


def export_all():
    if socket.gethostname() != SOURCE_HOST:
        raise RuntimeError('Export must run on the old 4090 instance')
    if (WORK / 'complete_manifest.json').exists():
        raise FileExistsError('A sealed source manifest already exists')
    progress('inventorying_all_source_files')
    rows = inventory()
    write_json(WORK / 'source_inventory.json', rows)
    progress('indexing_existing_OSS_backups', source_entries=len(rows))
    objects, backing = index_archives(rows)
    pending = [row for row in rows if row['kind'] == 'file' and row['sha256'] not in backing]
    command(['oss', 'mkdir', PREFIX + '/'])
    groups = []
    group, size = [], 0
    for row in pending:
        if group and size + row['bytes'] > 1800 * 1024 ** 2:
            groups.append(group)
            group, size = [], 0
        group.append(row)
        size += row['bytes']
    if group:
        groups.append(group)
    for number, group in enumerate(groups):
        progress('uploading_missing_files', chunk=number + 1, chunks=len(groups),
                 missing_files=len(pending), missing_bytes=sum(r['bytes'] for r in pending))
        path = WORK / f'delta_{number:03d}.tar'
        with tarfile.open(path, 'w') as archive:
            for row in group:
                source = Path(row['source'])
                if source.stat().st_size != row['bytes'] or checksum(source) != row['sha256']:
                    raise RuntimeError('Source changed during preservation: ' + row['path'])
                archive.add(source, arcname=row['path'], recursive=False)
        index = len(objects)
        objects.append({'remote': PREFIX + '/' + path.name, 'bytes': path.stat().st_size,
                        'sha256': checksum(path), 'format': 'tar', 'delta': True})
        command(['oss', 'cp', str(path), objects[-1]['remote'], '-f'])
        for row in group:
            backing[row['sha256']] = {'object': index, 'member': row['path']}
    for row in rows:
        if row['kind'] == 'file':
            row['backing'] = backing[row['sha256']]
        row.pop('source', None)
    manifest = {'version': 1, 'source_hostname': SOURCE_HOST, 'destination_hostname': DESTINATION_HOST,
                'snapshot_utc': now(), 'source_root': '/hy-tmp',
                'excluded_generated_backup_directory': str(WORK),
                'root_scope': 'project scripts, shell/Jupyter/pip configs, history and reproducible environment records; excludes SSH/OSS credentials and OS/package caches',
                'entries': rows, 'objects': objects,
                'regular_files': sum(r['kind'] == 'file' for r in rows),
                'total_regular_bytes': sum(r.get('bytes', 0) for r in rows)}
    write_json(WORK / 'complete_manifest.json', manifest)
    receipt = WORK / 'complete_manifest.tar'
    with tarfile.open(receipt, 'w') as archive:
        archive.add(WORK / 'complete_manifest.json', arcname='complete_manifest.json')
        archive.add(Path(__file__), arcname='full_project_preservation.py')
    command(['oss', 'cp', str(receipt), PREFIX + '/complete_manifest.tar', '-f'])
    progress('source_preserved_pending_destination_verification', files=manifest['regular_files'],
             bytes=manifest['total_regular_bytes'], objects=len(objects), new_delta_chunks=len(groups))


def destination_for(relative):
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise RuntimeError('Unsafe manifest path')
    if (relative == 'Hetero-UNet-stage2' or relative.startswith('Hetero-UNet-stage2/') or
            relative == 'stage2_results/stage2_F_seed1234' or
            relative.startswith('stage2_results/stage2_F_seed1234/') or
            relative.startswith('_root_profiles/')):
        destination = ARCHIVE / path
    else:
        destination = Path('/hy-tmp') / path
    destination.resolve(strict=False).relative_to(Path('/hy-tmp').resolve())
    return destination


def materialize(row, stream, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.restore4090.partial')
    value = hashlib.sha256()
    size = 0
    with temporary.open('wb') as target:
        while True:
            data = stream.read(8 * 1024 ** 2)
            if not data:
                break
            target.write(data)
            value.update(data)
            size += len(data)
    if size != row['bytes'] or value.hexdigest() != row['sha256']:
        raise RuntimeError('Restored file mismatch: ' + row['path'])
    os.chmod(temporary, row['mode'])
    os.replace(temporary, output)


def import_all():
    if socket.gethostname() != DESTINATION_HOST:
        raise RuntimeError('Import must run on the new V100 instance')
    progress('waiting_for_complete_source_manifest')
    receipt = WORK / 'complete_manifest.tar'
    while True:
        command(['oss', 'cp', PREFIX + '/complete_manifest.tar', str(receipt), '-f'], required=False)
        try:
            with tarfile.open(receipt) as archive:
                manifest = json.load(archive.extractfile('complete_manifest.json'))
            break
        except (OSError, tarfile.TarError, KeyError, json.JSONDecodeError):
            time.sleep(10)
    if manifest['source_hostname'] != SOURCE_HOST or manifest['destination_hostname'] != DESTINATION_HOST:
        raise RuntimeError('Manifest instance identities do not match')
    write_json(WORK / 'complete_manifest.json', manifest)
    rows = manifest['entries']
    restored = []
    pending = {}
    for row in rows:
        target = destination_for(row['path'])
        if row['kind'] == 'directory':
            target.mkdir(parents=True, exist_ok=True)
        elif row['kind'] == 'file':
            if target.is_symlink() or (target.exists() and not target.is_file()):
                target = ARCHIVE / row['path']
            if target.is_file() and checksum(target) == row['sha256']:
                restored.append({'source': row['path'], 'destination': str(target), 'sha256': row['sha256'],
                                 'bytes': row['bytes'], 'status': 'already_identical'})
            else:
                if target.exists():
                    target = ARCHIVE / row['path']
                pending.setdefault(row['backing']['object'], []).append((row, target))
    for index, obj in enumerate(manifest['objects']):
        progress('restoring_object', object=index + 1, objects=len(manifest['objects']),
                 files_verified=len(restored), pending_files=sum(len(v) for v in pending.values()))
        path = None
        if obj.get('source_relative_path'):
            candidate = destination_for(obj['source_relative_path'])
            if candidate.is_file() and candidate.stat().st_size == obj['bytes'] and checksum(candidate) == obj['sha256']:
                path = candidate
        temporary_object = path is None
        if path is None:
            path = WORK / ('object_' + str(index) + '.' + obj['format'])
            command(['oss', 'cp', obj['remote'], str(path), '-f'])
            if path.stat().st_size != obj['bytes'] or checksum(path) != obj['sha256']:
                raise RuntimeError('OSS object does not match the source inventory: ' + obj['remote'])
        if index in pending:
            archive = tarfile.open(path) if obj['format'] == 'tar' else zipfile.ZipFile(path)
            try:
                for row, target in pending[index]:
                    ref = row['backing']
                    if 'member' not in ref:
                        with path.open('rb') as source:
                            materialize(row, source, target)
                    else:
                        source = archive.extractfile(ref['member']) if obj['format'] == 'tar' else archive.open(ref['member'])
                        with source:
                            if 'nested_member' in ref:
                                with zipfile.ZipFile(source) as nested, nested.open(ref['nested_member']) as inner:
                                    materialize(row, inner, target)
                            else:
                                materialize(row, source, target)
                    restored.append({'source': row['path'], 'destination': str(target),
                                     'sha256': row['sha256'], 'bytes': row['bytes'], 'status': 'restored_verified'})
            finally:
                archive.close()
        if temporary_object:
            path.unlink()  # Downloaded transport copy only; source files and OSS remain.
        write_json(WORK / 'restored_files.json', restored)
    symlinks = []
    for row in rows:
        if row['kind'] == 'symlink':
            target = destination_for(row['path'])
            link = Path(row['target'])
            resolved = link if link.is_absolute() else target.parent / link
            resolved.resolve(strict=False).relative_to(Path('/hy-tmp').resolve())
            if target.is_symlink():
                if os.readlink(target) != row['target']:
                    raise RuntimeError('Conflicting symbolic link')
            elif target.exists():
                raise RuntimeError('Symbolic link destination already exists')
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(row['target'], target)
            symlinks.append({'source': row['path'], 'destination': str(target), 'target': row['target']})
    if len(restored) != manifest['regular_files']:
        raise RuntimeError('Not all source files were restored')
    # Independently reread every mapped destination, including preexisting caches.
    for row in restored:
        target = Path(row['destination'])
        if target.stat().st_size != row['bytes'] or checksum(target) != row['sha256']:
            raise RuntimeError('Final independent verification failed: ' + row['source'])
    report = {'status': 'complete_verified', 'checked_utc': now(), 'source_hostname': SOURCE_HOST,
              'destination_hostname': DESTINATION_HOST, 'files_verified': len(restored),
              'bytes_verified': manifest['total_regular_bytes'], 'symlinks_verified': symlinks,
              'source_manifest_sha256': checksum(WORK / 'complete_manifest.json'),
              'all_source_files_have_OSS_backing': True, 'missing_files': 0,
              'live_F_results_or_source_overwritten': False,
              'archived_original_F_and_source_root': str(ARCHIVE)}
    write_json(WORK / 'complete_verification.json', report)
    proof = WORK / 'complete_verification.tar'
    with tarfile.open(proof, 'w') as archive:
        for name in ('complete_verification.json', 'complete_manifest.json', 'restored_files.json'):
            archive.add(WORK / name, arcname=name)
    command(['oss', 'cp', str(proof), PREFIX + '/complete_verification.tar', '-f'])
    returned = WORK / 'verification_roundtrip.tar'
    command(['oss', 'cp', PREFIX + '/complete_verification.tar', str(returned), '-f'])
    if checksum(returned) != checksum(proof):
        raise RuntimeError('Preservation verification receipt failed round-trip check')
    returned.unlink()
    report['verification_receipt_roundtrip_passed'] = True
    write_json(WORK / 'complete_verification.json', report)
    progress('complete_verified', files_verified=len(restored), missing_files=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('export', 'import'))
    args = parser.parse_args()
    WORK.mkdir(exist_ok=True)
    os.nice(15)
    if args.mode == 'export':
        export_all()
    else:
        import_all()


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        progress('failed_requires_review', error=type(error).__name__ + ': ' + str(error))
        raise
