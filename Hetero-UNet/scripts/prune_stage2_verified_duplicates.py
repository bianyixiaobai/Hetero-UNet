"""Audit completed F and delete only allowlisted, OSS-verified duplicate files.

The default prints a plan. --apply is the user-authorized cleanup of this
specific instance. It never deletes directories, active checkpoints or data.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tarfile

ROOT = Path('/hy-tmp')
WORK = ROOT / 'stage2_cleanup_20261022'
AUDIT = ROOT / 'stage2_audit_2026-10-22'
PRESERVED = ROOT / 'full_preservation_20261022'
FINAL = ROOT / 'stage2_F_resumed_autoshutdown_20261022'
REPO = ROOT / 'Hetero-UNet-stage2'
PREFIX = 'oss://backup/stage2_20261022/F_completed_cleanup'
MANIFEST_SHA = '6f9c28d062be07f7f38f2bf76a02ff0b4d3e4c690c2963ee401c44c93fe53c48'
SMOKES = {'stage2_F_seed1234_smoke', 'stage2_C_seed1234_smoke', 'stage2_F-aux_seed1234_smoke'}
MIGRATION_TARS = {'F_epoch500_results.tar', 'CT701_preprocessed_exact.tar',
                  'migration_audit.tar', 'migration_receipt.tar'}


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(stream):
    value = hashlib.sha256()
    for chunk in iter(lambda: stream.read(8 * 1024 ** 2), b''):
        value.update(chunk)
    return value.hexdigest()


def sha(path):
    with Path(path).open('rb') as stream:
        return digest(stream)


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    os.replace(temporary, path)


def status(value, **details):
    save(WORK / 'state.json', {'status': value, 'updated_utc': now(), **details})
    print(now(), value, json.dumps(details), flush=True)


def allowed(path):
    """Only explicit duplicate archives and old/smoke checkpoint files qualify."""
    try:
        rel = Path(path).relative_to(ROOT)
    except ValueError:
        return False
    parts = rel.parts
    if '..' in parts:
        return False
    if len(parts) == 2 and parts[0] == 'stage2_restore_archives':
        return parts[1] in {'3d_fullres_weights.tar', 'pred50_artifacts_20261013.zip',
                            *{f'Dataset701_AbdomenCT-20260910T013250Z-1-{n:03d}.zip' for n in range(1, 4)}}
    if len(parts) == 2 and parts[0] == 'stage2_migration_20261022':
        return parts[1] in MIGRATION_TARS
    if rel.as_posix() in {'stage2_completed_archives/session_audit_start.tar',
                          'stage2_F_resumed_autoshutdown_20261022/F_seed1234_final.tar'}:
        return True
    if len(parts) == 6 and parts[0] == 'stage2_results' and parts[1] in SMOKES:
        return parts[-2] == 'fold_all' and parts[-1] in {'checkpoint_best.pth', 'checkpoint_final.pth'}
    if len(parts) == 7 and parts[:3] == ('archive_4090_20261022', 'stage2_results', 'stage2_F_seed1234'):
        return parts[-2] == 'fold_all' and parts[-1] in {'checkpoint_best.pth', 'checkpoint_latest.pth'}
    return False


def delete_verified(row, verified):
    path = Path(row['path'])
    if not allowed(path) or path.is_symlink() or path.resolve() != path.absolute():
        raise RuntimeError('Refusing unsafe cleanup path: ' + str(path))
    if row['object_sha256'] not in verified:
        raise RuntimeError('No verified OSS recovery object: ' + str(path))
    if not path.is_file() or path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
        raise RuntimeError('Candidate changed; refusing deletion: ' + str(path))
    path.unlink()


def run(args):
    with (WORK / 'transfers.log').open('ab', buffering=0) as log:
        subprocess.run(args, check=True, stdin=subprocess.DEVNULL, stdout=log,
                       stderr=subprocess.STDOUT, timeout=7200, cwd=WORK)


def no_training():
    for entry in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = entry.read_bytes().split(b'\0')
        except OSError:
            continue
        if any(Path(arg.decode(errors='replace')).name in
               {'run_stage2.py', 'nnUNetv2_train', 'run_training.py'} for arg in args):
            raise RuntimeError('Training process is active; cleanup blocked')


def build_plan():
    if sha(PRESERVED / 'complete_manifest.json') != MANIFEST_SHA:
        raise RuntimeError('Original preservation manifest changed')
    manifest = read(PRESERVED / 'complete_manifest.json')
    mapping = read(PRESERVED / 'restored_files.json')
    originals = {row['path']: row for row in manifest['entries'] if row['kind'] == 'file'}
    objects = {obj['sha256']: obj for obj in manifest['objects']}
    candidates = []
    for row in mapping:
        path = Path(row['destination'])
        if not allowed(path) or not path.exists():
            continue
        source = originals[row['source']]
        obj = manifest['objects'][source['backing']['object']]
        candidates.append({'path': str(path), 'bytes': source['bytes'], 'sha256': source['sha256'],
                           'object_sha256': obj['sha256'], 'OSS': obj['remote'],
                           'backing': source['backing'], 'reason': 'old duplicate archive/checkpoint'})
    state = read(FINAL / 'state.json')
    for obj in state['verified'].values():
        objects[obj['sha256']] = obj
    live_tar = FINAL / 'F_seed1234_final.tar'
    obj = state['verified']['F_seed1234_final.tar']
    if live_tar.exists():
        candidates.append({'path': str(live_tar), 'bytes': obj['bytes'], 'sha256': obj['sha256'],
                           'object_sha256': obj['sha256'], 'OSS': obj['remote'],
                           'reason': 'redundant packaging copy; live best/final retained'})
    for name, expected in [('release_ready_audit.tar', read(AUDIT / 'release_ready_check.json')['release_ready_bundle_sha256']),
                           ('complete_verification.tar', read(AUDIT / 'release_ready_check.json')['verification_receipt_sha256'])]:
        p = PRESERVED / name
        objects[expected] = {'sha256': expected, 'bytes': p.stat().st_size,
                             'remote': 'oss://backup/stage2_20261022/complete_4090_project/' + name}
    delivery = read(AUDIT / 'preservation_delivery.json')
    objects[delivery['sha256']] = {**delivery, 'remote': delivery['OSS']}
    return manifest, mapping, candidates, objects


def seal(name, names):
    path = WORK / name
    with tarfile.open(path, 'w') as archive:
        for p in names:
            archive.add(p, arcname=Path(p).name)
    remote = PREFIX + '/' + name
    run(['oss', 'cp', str(path), remote, '-f'])
    returned = WORK / ('verify_' + name)
    run(['oss', 'cp', remote, str(returned), '-f'])
    if sha(returned) != sha(path):
        raise RuntimeError('Cleanup record round-trip failed')
    returned.unlink()
    return {'remote': remote, 'bytes': path.stat().st_size, 'sha256': sha(path), 'roundtrip_passed': True}


def main(apply=False):
    WORK.mkdir(exist_ok=True)
    if socket.gethostname() != 'I2bde78031f00701d9e':
        raise RuntimeError('Wrong instance')
    if (WORK / 'cleanup_result.json').exists():
        raise RuntimeError('Cleanup already completed; inspect its report instead of rerunning')
    no_training()
    manifest, mapping, candidates, objects = build_plan()
    plan = {'created_utc': now(), 'files': candidates, 'count': len(candidates),
            'bytes': sum(row['bytes'] for row in candidates), 'disk_before': shutil.disk_usage(ROOT)._asdict(),
            'protected': ['current F best/final', 'raw CT', 'preprocessing and NPY cache',
                          'A baseline', 'active source', 'all logs/configuration/predictions and private root records']}
    save(WORK / 'cleanup_plan.json', plan)
    status('plan_created', files=plan['count'], bytes=plan['bytes'])
    if not apply:
        return
    os.nice(15)
    queue = read(AUDIT / 'F_resume_queue.json')
    assert queue['status'] == 'completed' and queue['epochs_each'] == 1000
    assert subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip() == \
        '9fe70ac570a0e14ebce30daa4a16a22e49b4f3fd'
    subprocess.run(['git', '-C', str(REPO), 'diff', '--exit-code', 'HEAD'], check=True)
    model_root = next((ROOT / 'stage2_results/stage2_F_seed1234').glob('Dataset*/*/fold_all'))
    current = read(model_root / 'stage2_run.json')
    assert current['status'] == 'completed'
    for relative, expected in current['fingerprints']['source_sha256'].items():
        if sha(REPO / 'Hetero-UNet' / relative) != expected:
            raise RuntimeError('Training source fingerprint changed: ' + relative)
    status('checking_CPU_checkpoints')
    import torch
    torch.set_num_threads(1)
    checkpoints = []
    for name in ['checkpoint_best.pth', 'checkpoint_final.pth']:
        checkpoint = torch.load(model_root / name, map_location='cpu')
        assert checkpoint['current_epoch'] == 1000 and checkpoint['trainer_name'] == 'nnUNetTrainerMambaUNetBlockAttnRes'
        assert checkpoint['network_weights'] and checkpoint['optimizer_state']
        assert all(torch.isfinite(t).all().item() for t in checkpoint['network_weights'].values() if torch.is_floating_point(t))
        checkpoints.append({'name': name, 'epoch': 1000, 'bytes': (model_root / name).stat().st_size,
                            'sha256': sha(model_root / name), 'network_finite': True})
        del checkpoint
    save(WORK / 'F_checkpoint_check.json', checkpoints)
    # Recheck the entire historical restoration before removing any duplicate.
    for index, row in enumerate(mapping):
        if index % 100 == 0:
            status('rechecking_preserved_project', files=index, total=len(mapping))
        p = Path(row['destination'])
        if not p.is_file() or p.stat().st_size != row['bytes'] or sha(p) != row['sha256']:
            raise RuntimeError('Preserved file changed/missing before cleanup: ' + str(p))
    save(WORK / 'precleanup_preservation_check.json', {'checked_utc': now(), 'files': len(mapping),
          'bytes': manifest['total_regular_bytes'], 'all_matched': True, 'source_manifest_sha256': MANIFEST_SHA})
    verified = {}
    for index, (key, obj) in enumerate(objects.items()):
        status('fresh_OSS_roundtrip', object=index + 1, objects=len(objects), remote=obj['remote'])
        downloaded = WORK / 'object_verification.tmp'
        run(['oss', 'cp', obj['remote'], str(downloaded), '-f'])
        if downloaded.stat().st_size != obj['bytes'] or sha(downloaded) != key:
            raise RuntimeError('OSS object no longer matches verified inventory: ' + obj['remote'])
        if obj['remote'].endswith('/F_seed1234_final.tar'):
            checked = []
            with tarfile.open(downloaded) as archive:
                for member in archive:
                    if not member.isfile():
                        continue
                    rel = Path(member.name)
                    if rel.is_absolute() or '..' in rel.parts or rel.parts[0] != model_root.parent.name:
                        raise RuntimeError('Unexpected final F archive member')
                    target = model_root.parent.parent / rel
                    with archive.extractfile(member) as stream:
                        if target.stat().st_size != member.size or sha(target) != digest(stream):
                            raise RuntimeError('Final OSS archive differs from live F results')
                    checked.append(str(target))
            assert len(checked) == len([p for p in model_root.parent.rglob('*') if p.is_file()])
            save(WORK / 'F_archive_content_check.json', {'files': checked, 'all_matched': True})
        verified[key] = {**obj, 'fresh_download_verified_utc': now()}
        downloaded.unlink()
        save(WORK / 'verified_OSS_objects.json', verified)
    run(['oss', 'mkdir', PREFIX + '/'])
    evidence = seal('before_deletion_evidence.tar', [WORK / n for n in ['cleanup_plan.json', 'F_checkpoint_check.json',
                    'precleanup_preservation_check.json', 'verified_OSS_objects.json', 'F_archive_content_check.json']] + [Path(__file__)])
    no_training()
    deleted = []
    for row in candidates:
        delete_verified(row, verified)
        deleted.append(row)
        save(WORK / 'deleted_files.json', deleted)
    result = {'status': 'completed_verified_cleanup', 'completed_utc': now(), 'files_deleted': len(deleted),
              'bytes_deleted': sum(r['bytes'] for r in deleted), 'disk_before': plan['disk_before'],
              'disk_after': shutil.disk_usage(ROOT)._asdict(), 'OSS_objects_freshly_verified': len(verified),
              'historical_files_verified_before_cleanup': len(mapping), 'F_epoch': 1000,
              'live_F_checkpoints_retained': checkpoints, 'protected': plan['protected'],
              'predelete_evidence': evidence, 'external_50_evaluation_performed': False,
              'C_training_started': False, 'shutdown_called': False}
    save(WORK / 'cleanup_result.json', result)
    proof = seal('cleanup_result.tar', [WORK / n for n in ['cleanup_result.json', 'deleted_files.json', 'cleanup_plan.json',
                  'F_checkpoint_check.json', 'precleanup_preservation_check.json', 'verified_OSS_objects.json', 'F_archive_content_check.json']])
    result['cleanup_receipt'] = proof
    save(WORK / 'cleanup_result.json', result)
    status('completed_verified_cleanup', files_deleted=len(deleted), bytes_deleted=result['bytes_deleted'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    try:
        main(parser.parse_args().apply)
    except Exception as error:
        if WORK.exists():
            status('failed_requires_review', error=type(error).__name__ + ': ' + str(error))
        raise
