#!/usr/bin/env python
"""Read-only C archive/protocol audit and CPU profiling of the added block.

Uses the existing local C tar; does not extract or modify clinical arrays.
"""
import argparse
import hashlib
import json
from math import prod
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace

from run_stage3 import sha256_file, validate_spatial_reference, parse_args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    workspace = Path(args.workspace).resolve()
    project = workspace / 'Hetero-UNet'
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    archive = workspace / 'output/stage2_C_completed_2026-10-24/C_seed1234_final.tar'
    receipt = json.loads((archive.parent / 'local_backup_check.json').read_text(encoding='utf-8'))
    digest = sha256_file(archive)
    assert digest == receipt['sha256'], 'C archive no longer matches its verified receipt'
    with tarfile.open(archive) as tar:
        def read_json(suffix):
            member = next(m for m in tar.getmembers() if m.name.endswith('/' + suffix))
            return json.load(tar.extractfile(member))
        plans = read_json('plans.json')
        dataset = read_json('dataset.json')
        run = read_json('fold_all/stage2_run.json')
        config = read_json('fold_all/stage2_config.json')
        checkpoints = []
        for member in tar.getmembers():
            if member.name.endswith(('/checkpoint_best.pth', '/checkpoint_final.pth')):
                sha = hashlib.sha256()
                stream = tar.extractfile(member)
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    sha.update(chunk)
                checkpoints.append({'member': member.name, 'bytes': member.size, 'sha256': sha.hexdigest()})
    contract_trainer = SimpleNamespace(
        plans_manager=SimpleNamespace(plans=plans), dataset_json=dataset,
        num_epochs=1000, num_iterations_per_epoch=250, num_val_iterations_per_epoch=50,
        initial_lr=.01, group_size=3, boundary_hidden_channels=8, boundary_loss_weight=.1,
        boundary_guidance=True, auxiliary_boundary=True)
    launches = {}
    for model in ('C-ND', 'C-R0'):
        launch = parse_args(['--model', model, '--results-root', str(out / 'future_results'), '--dry-run'])
        launches[model] = vars(launch)
        validate_spatial_reference(launch, contract_trainer, project)
    for key in ('num_epochs', 'initial_lr'):
        old = getattr(contract_trainer, key)
        setattr(contract_trainer, key, old + 1)
        try:
            validate_spatial_reference(launch, contract_trainer, project)
        except ValueError:
            pass
        else:
            raise AssertionError('guard did not detect changed ' + key)
        setattr(contract_trainer, key, old)
    counts = {}
    for name in ('imagesTr', 'labelsTr', 'imagesVal', 'labelsVal'):
        files = sorted((workspace / 'data/Dataset701_AbdomenCT' / name).glob('*.nii.gz'))
        assert len(files) == 50
        counts[name] = [p.name for p in files]
    tr_ids = {p.replace('_0000.nii.gz', '').replace('.nii.gz', '') for p in counts['imagesTr']}
    val_ids = {p.replace('_0000.nii.gz', '').replace('.nii.gz', '') for p in counts['imagesVal']}
    assert not tr_ids & val_ids
    assert tr_ids == {p[:-7] for p in counts['labelsTr']}
    assert val_ids == {p[:-7] for p in counts['labelsVal']}

    sys.path.insert(0, str(project / 'umamba'))
    import torch
    from nnunetv2.nets.spatial_receptive_block import SpatialReceptiveBlock3D
    torch.set_num_threads(1)
    ct = plans['configurations']['3d_fullres']
    channels = min(ct['UNet_base_num_features'] * 2 ** (len(ct['pool_op_kernel_sizes']) - 2),
                   ct['unet_max_num_features'])
    shape = list(ct['patch_size'])
    for stride in ct['pool_op_kernel_sizes'][:-1]:
        shape = [i // s for i, s in zip(shape, stride)]
    profiles = []
    for variant in ('ND', 'R0'):
        torch.manual_seed(1234)
        block = SpatialReceptiveBlock3D(channels, variant)
        x = torch.randn(2, channels, *shape, requires_grad=True)
        counts_macs = []
        def count(module, inputs, output):
            counts_macs.append(output.numel() * (module.in_channels // module.groups) * prod(module.kernel_size))
        hooks = [m.register_forward_hook(count) for m in block.modules() if isinstance(m, torch.nn.Conv3d)]
        y = block(x)
        y.square().mean().backward()
        for h in hooks:
            h.remove()
        assert sum(counts_macs) == block.conv_macs(shape, 2)
        assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()
        profiles.append({'variant': variant, 'input_shape': list(x.shape),
                         'parameters': sum(p.numel() for p in block.parameters()),
                         'conv_MACs_batch2': sum(counts_macs), 'conv_FLOPs_batch2': 2 * sum(counts_macs),
                         'scope': 'added block only on CPU; not full network/GPU memory'})
    report = {'date': '2026-11-16', 'C_archive_sha256': digest, 'archive_receipt_match': True,
              'checkpoints': checkpoints, 'C_protocol': run['protocol'], 'C_module_config': config,
              'C_reference_guard_positive_and_negative_checks': 'passed', 'launches_dry_only': launches,
              'data_files': counts, 'block_profiles': profiles,
              'data_audit_scope': 'case names/counts and label pairing, not new voxel-by-voxel hashing',
              'clinical_evaluation': 'none; existing C results retained',
              'torch': torch.__version__, 'cuda_available': torch.cuda.is_available()}
    (out / 'local_audit.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'archive_receipt_match': True, 'profiles': profiles, 'status': 'passed'}, indent=2))


if __name__ == '__main__':
    main()
