"""Recover this project's archived files without replanning its frozen CT setup."""
import hashlib
import json
from pathlib import Path
import shutil
import stat
import tarfile
import zipfile

ARCHIVES = Path('/hy-tmp/stage2_restore_archives')
DATA = Path('/hy-tmp/Hetero-UNet/data')
AUDIT = Path('/hy-tmp/stage2_audit_2026-10-21')
RAW = DATA / 'nnUNet_raw'
PREPROCESSED = DATA / 'nnUNet_preprocessed/Dataset701_AbdomenCT'


def safe_target(root, name):
    destination = (root / name).resolve()
    if not destination.is_relative_to(root.resolve()):
        raise ValueError(f'Archive path escapes requested output: {name}')
    return destination


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def extract_zip(archive, root, manifest):
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        for entry in zipped.infolist():
            parts = Path(entry.filename).parts
            if '__MACOSX' in parts or Path(entry.filename).name.startswith('._') or Path(entry.filename).name == '.DS_Store':
                print('SKIP_ARCHIVE_METADATA', entry.filename, flush=True)
                continue
            target = safe_target(root, entry.filename)
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if stat.S_ISLNK(entry.external_attr >> 16):
                raise ValueError('Symlinks are not expected in the archived dataset')
            target.parent.mkdir(parents=True, exist_ok=True)
            h = hashlib.sha256()
            if target.exists():
                with zipped.open(entry) as source:
                    for chunk in iter(lambda: source.read(1024 * 1024), b''):
                        h.update(chunk)
                if sha(target) != h.hexdigest():
                    raise ValueError(f'Conflicting duplicate archive member: {entry.filename}')
            else:
                with zipped.open(entry) as source, target.open('xb') as output:
                    for chunk in iter(lambda: source.read(1024 * 1024), b''):
                        h.update(chunk)
                        output.write(chunk)
            # Reading to EOF also checks each zip member's CRC.
            manifest[str(target.relative_to(root))] = {
                'bytes': entry.file_size, 'sha256': h.hexdigest(),
            }
    print('EXTRACTED', archive.name, 'files', len(manifest), flush=True)


def main():
    manifest = {}
    for archive in sorted(ARCHIVES.glob('Dataset701_AbdomenCT-*.zip')):
        extract_zip(archive, RAW, manifest)
    if len(list(ARCHIVES.glob('Dataset701_AbdomenCT-*.zip'))) != 3:
        raise ValueError('Expected exactly three complete raw-data archives')
    dataset = RAW / 'Dataset701_AbdomenCT'
    import nibabel as nib
    import numpy as np
    case_ids = {}
    counts = {}
    for split in ('Tr', 'Val'):
        images = {p.name[:-12]: p for p in (dataset / ('images' + split)).glob('*_0000.nii.gz')}
        labels = {p.name[:-7]: p for p in (dataset / ('labels' + split)).glob('*.nii.gz')}
        if len(images) != 50 or images.keys() != labels.keys():
            raise ValueError(f'{split}: images/labels mismatch, {len(images)}/{len(labels)}')
        for name in sorted(images):
            image = nib.load(images[name])
            label = nib.load(labels[name])
            if image.shape != label.shape or not np.allclose(image.affine, label.affine, atol=1e-4):
                raise ValueError(f'{name}: image/label geometry differs')
        case_ids[split] = sorted(images)
        counts[split] = len(images)
    if set(case_ids['Tr']) & set(case_ids['Val']):
        raise ValueError('Training and external case IDs overlap')

    baseline_root = DATA / 'nnUNet_results/Dataset701_AbdomenCT'
    baseline_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(ARCHIVES / '3d_fullres_weights.tar') as archive:
        for member in archive.getmembers():
            target = safe_target(baseline_root, member.name)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError('Unexpected non-regular baseline archive member')
            if target.exists():
                raise FileExistsError(f'Preserve existing baseline file: {target}')
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, target.open('xb') as output:
                shutil.copyfileobj(source, output, 1024 * 1024)
    trainer_base = baseline_root / 'nnUNetTrainerMambaUNet__nnUNetPlans__3d_fullres'
    PREPROCESSED.mkdir(parents=True, exist_ok=True)
    for source, destination in (
        ('plans.json', 'nnUNetPlans.json'),
        ('dataset.json', 'dataset.json'),
        ('dataset_fingerprint.json', 'dataset_fingerprint.json'),
    ):
        target = PREPROCESSED / destination
        if target.exists():
            raise FileExistsError(f'Do not overwrite pre-existing descriptor: {target}')
        shutil.copy2(trainer_base / source, target)
    plans = json.loads((PREPROCESSED / 'nnUNetPlans.json').read_text())
    config = plans['configurations']['3d_fullres']
    assert config['batch_size'] == 2 and config['patch_size'] == [40, 224, 192]
    assert sha(PREPROCESSED / 'nnUNetPlans.json') == sha(AUDIT / 'baseline_plans.json')
    archived_dataset = json.loads((PREPROCESSED / 'dataset.json').read_text())
    raw_dataset = json.loads((dataset / 'dataset.json').read_text())
    assert archived_dataset['labels'] == raw_dataset['labels']
    assert archived_dataset['channel_names'] == raw_dataset['channel_names']
    artifacts = {}
    extract_zip(ARCHIVES / 'pred50_artifacts_20261013.zip', AUDIT / 'pred50_artifacts', artifacts)
    report = {
        'status': 'restored_and_header_validated', 'raw_root': str(RAW),
        'counts': counts, 'case_ids': case_ids, 'raw_files': manifest,
        'baseline_best_sha256': sha(trainer_base / 'fold_all/checkpoint_best.pth'),
        'frozen_plans_sha256': sha(PREPROCESSED / 'nnUNetPlans.json'),
        'fingerprint_sha256': sha(PREPROCESSED / 'dataset_fingerprint.json'),
        'artifacts': artifacts,
        'scope': 'zip CRC and compressed file SHA256, paired NIfTI shape/affine and distinct IDs; not duplicate-patient audit',
    }
    (AUDIT / 'data_restore.json').write_text(json.dumps(report, indent=2) + '\n')
    print('RESTORE_VALIDATED', counts, 'plans_sha256', report['frozen_plans_sha256'], flush=True)


if __name__ == '__main__':
    main()
