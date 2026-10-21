#!/usr/bin/env python
"""Real CUDA/AMP full-plan diagnostic with synthetic batches, not efficacy data."""
import argparse
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plans', required=True)
    parser.add_argument('--dataset-json', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    folder = Path(args.output).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    for name in ('raw', 'preprocessed', 'results'):
        path = folder / name
        path.mkdir(exist_ok=True)
        os.environ['nnUNet_' + name] = str(path)
    os.environ['nnUNet_compile'] = 'false'
    os.environ['WANDB_MODE'] = 'disabled'
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'umamba'))
    import numpy as np
    import torch
    from torch.nn import functional as functional
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMambaUNetAttnRes import (
        nnUNetTrainerMambaUNetBlockAttnRes,
        nnUNetTrainerMambaUNetBGBlockAttnRes,
        nnUNetTrainerMambaUNetBlockAttnResBoundarySupervision,
    )
    if not torch.cuda.is_available():
        raise RuntimeError('This diagnostic requires real CUDA and real Mamba; no CPU fallback')
    torch.set_num_threads(1)
    plans = json.loads(Path(args.plans).read_text())
    dataset = json.loads(Path(args.dataset_json).read_text())
    report = {
        'scope': 'real CUDA/AMP and original CT plan, synthetic batches only',
        'not_tested': ['real data loader', 'training convergence', 'DSC/NSD'],
        'torch': torch.__version__, 'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0),
        'gpu_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
        'runs': [],
    }
    report_path = folder / 'cuda_check.json'
    for name, trainer_class in (
        ('F', nnUNetTrainerMambaUNetBlockAttnRes),
        ('C', nnUNetTrainerMambaUNetBGBlockAttnRes),
        ('F-aux', nnUNetTrainerMambaUNetBlockAttnResBoundarySupervision),
    ):
        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
        torch.cuda.reset_peak_memory_stats()
        trainer = trainer_class(deepcopy(plans), '3d_fullres', 'all',
                                deepcopy(dataset), device=torch.device('cuda'))
        trainer.initialize()
        patch = tuple(trainer.configuration_manager.patch_size)
        if trainer.batch_size != 2 or patch != (40, 224, 192):
            raise ValueError(f'Frozen CT footprint mismatch: batch={trainer.batch_size}, patch={patch}')
        image = torch.randn(2, 1, *patch)
        target = torch.randint(0, 14, (2, 1, *patch))
        targets = [functional.interpolate(target.float(),
                    size=[int(round(n * scale)) for n, scale in zip(patch, scales)],
                    mode='nearest').long()
                   for scales in trainer._get_deep_supervision_scales()]
        trainer.network.train()
        torch.cuda.synchronize()
        started = time.perf_counter()
        values = trainer.train_step({'data': image, 'target': targets})
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        if not np.isfinite(values['loss']):
            raise AssertionError('Nonfinite training loss')
        gradients = [p.grad for p in trainer.network.parameters() if p.grad is not None]
        if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
            raise AssertionError('Missing or nonfinite gradients')
        assert trainer.network.boundary_logits is None
        assert all(not layer.mha.batch_first for layer in trainer.network.mamba_mlp.layers)
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        trainer.network.eval()
        with torch.no_grad():
            validation = trainer.validation_step({'data': image, 'target': targets})
        if not np.isfinite(validation['loss']):
            raise AssertionError('Nonfinite validation loss')
        row = {'model': name, 'status': 'passed', 'physical_batch': 2,
               'patch': list(patch), 'train_loss': float(values['loss']),
               'validation_loss': float(validation['loss']),
               'train_step_seconds': seconds,
               'peak_train_allocated_bytes': peak_allocated,
               'peak_train_reserved_bytes': peak_reserved,
               'parameters': sum(p.numel() for p in trainer.network.parameters()),
               'amp': trainer.grad_scaler is not None,
               'mamba': 'real_mamba_ssm', 'seed': 1234}
        if trainer.auxiliary_boundary:
            row['boundary_loss'] = float(values['boundary_loss'])
        report['runs'].append(row)
        report_path.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(row), flush=True)
        del gradients, trainer, image, target, targets, values, validation
        gc.collect()
        torch.cuda.empty_cache()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
