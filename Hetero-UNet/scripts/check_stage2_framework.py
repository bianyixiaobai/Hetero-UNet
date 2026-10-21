#!/usr/bin/env python
"""Synthetic CPU check of actual nnU-Net wiring with an explicit Mamba stub.

This checks the real residual encoder/decoder, original MHA/MLP, nnU-Net loss,
optimizer/validation and checkpoint selection. The Mamba kernel and W&B are
test doubles; no clinical data or GPU is used. It is NOT the deployment smoke
test, which must use run_stage2.py --smoke in the actual CUDA environment.
"""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import types


def synthetic_plans(dim):
    patch = [8, 16, 16] if dim == 3 else [16, 16]
    name = '3d_fullres' if dim == 3 else '2d'
    config = {
        'data_identifier': 'synthetic_' + name, 'preprocessor_name': 'DefaultPreprocessor',
        'batch_size': 2, 'patch_size': patch, 'median_image_size_in_voxels': patch,
        'spacing': [1.] * dim, 'normalization_schemes': ['ZScoreNormalization'],
        'use_mask_for_norm': [False], 'UNet_class_name': 'PlainConvUNet',
        'UNet_base_num_features': 8, 'unet_max_num_features': 32,
        'n_conv_per_stage_encoder': [1, 1, 1], 'n_conv_per_stage_decoder': [1, 1],
        'num_pool_per_axis': [2] * dim,
        'pool_op_kernel_sizes': [[1] * dim, [2] * dim, [2] * dim],
        'conv_kernel_sizes': [[3] * dim] * 3,
        'batch_dice': True,
        'resampling_fn_data': 'resample_data_or_seg_to_shape',
        'resampling_fn_data_kwargs': {'is_seg': False, 'order': 3, 'order_z': 0},
        'resampling_fn_seg': 'resample_data_or_seg_to_shape',
        'resampling_fn_seg_kwargs': {'is_seg': True, 'order': 1, 'order_z': 0},
        'resampling_fn_probabilities': 'resample_data_or_seg_to_shape',
        'resampling_fn_probabilities_kwargs': {'is_seg': False, 'order': 1, 'order_z': 0},
    }
    return {
        'dataset_name': 'Dataset999_Stage2Synthetic', 'plans_name': 'SyntheticPlans',
        'image_reader_writer': 'SimpleITKIO', 'transpose_forward': list(range(dim)),
        'transpose_backward': list(range(dim)), 'configurations': {name: config},
        'original_median_spacing_after_transp': [1.] * dim,
        'original_median_shape_after_transp': patch,
    }, name


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='Isolated directory for synthetic artifacts')
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for key, sub in [('nnUNet_raw', 'raw'), ('nnUNet_preprocessed', 'preprocessed'),
                     ('nnUNet_results', 'results')]:
        os.environ[key] = str(output / sub)
        (output / sub).mkdir(exist_ok=True)
    os.environ['nnUNet_compile'] = 'false'
    os.environ['WANDB_MODE'] = 'disabled'
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'umamba'))
    import torch
    from torch import nn
    from torch.nn import functional as functional
    torch.set_num_threads(1)

    # Stubs are installed only by this explicit diagnostic process. Production
    # training imports the real mamba_ssm and cannot silently use these stubs.
    class CpuMambaTestDouble(nn.Module):
        def __init__(self, d_model, **kwargs):
            super().__init__()
            self.projection = nn.Linear(d_model, d_model)

        def forward(self, tokens):
            return self.projection(tokens).tanh()

    stub = types.ModuleType('mamba_ssm')
    stub.Mamba = CpuMambaTestDouble
    sys.modules['mamba_ssm'] = stub
    tracking_stub = types.ModuleType('wandb')
    tracking_stub.log = lambda *a, **kw: None
    sys.modules['wandb'] = tracking_stub

    from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMambaUNetAttnRes import (
        nnUNetTrainerMambaUNetBlockAttnRes,
        nnUNetTrainerMambaUNetBGBlockAttnRes,
        nnUNetTrainerMambaUNetBlockAttnResBoundarySupervision,
    )
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    dataset = {'channel_names': {'0': 'CT'},
               'labels': {'background': 0, 'organ_a': 1, 'organ_b': 2},
               'numTraining': 2, 'file_ending': '.nii.gz'}
    reports = []
    for dim in (2, 3):
        plans, configuration = synthetic_plans(dim)
        for trainer_class in (nnUNetTrainerMambaUNetBlockAttnRes,
                              nnUNetTrainerMambaUNetBGBlockAttnRes,
                              nnUNetTrainerMambaUNetBlockAttnResBoundarySupervision):
            torch.manual_seed(1234)
            trainer = trainer_class(deepcopy(plans), configuration, 'all',
                                    deepcopy(dataset), device=torch.device('cpu'))
            trainer.initialize()
            assert Path(trainer.output_folder).resolve().is_relative_to(output / 'results')
            trainer.network.train()
            shape = trainer.configuration_manager.patch_size
            image = torch.randn(2, 1, *shape)
            target = torch.randint(0, 3, (2, 1, *shape))
            down = functional.interpolate(target.float(), scale_factor=0.5, mode='nearest').long()
            train = trainer.train_step({'data': image, 'target': [target, down]})
            assert torch.isfinite(torch.as_tensor(train['loss'])), train
            assert all(not layer.mha.batch_first for layer in trainer.network.mamba_mlp.layers)
            assert trainer.network.boundary_logits is None
            trainer.network.eval()
            with torch.no_grad():
                val = trainer.validation_step({'data': image, 'target': [target.clone(), down.clone()]})
            assert torch.isfinite(torch.as_tensor(val['loss'])), val
            assert val['tp_hard'].shape == (2,)
            base = Path(trainer.output_folder_base)
            (base / 'plans.json').write_text(json.dumps(plans), encoding='utf-8')
            (base / 'dataset.json').write_text(json.dumps(dataset), encoding='utf-8')
            # Store distinct real checkpoint files; the loader must respect -chk.
            checkpoint_best = Path(trainer.output_folder) / 'checkpoint_best.pth'
            trainer.save_checkpoint(str(checkpoint_best))
            checkpoint_final = Path(trainer.output_folder) / 'checkpoint_final.pth'
            trainer.save_checkpoint(str(checkpoint_final))
            predictor = nnUNetPredictor(device=torch.device('cpu'), perform_everything_on_device=False)
            predictor.initialize_from_trained_model_folder(str(base), ['all'], 'checkpoint_best.pth')
            predictor.network.load_state_dict(predictor.list_of_parameters[0])
            predictor.network.eval()
            trainer.set_deep_supervision_enabled(False)
            with torch.no_grad():
                expected = trainer.network(image[:1])
                actual = predictor.network(image[:1])
            torch.testing.assert_close(actual, expected)
            assert actual.shape == (1, 3, *shape)
            # A missing requested file must fail instead of silently loading best.
            try:
                predictor.initialize_from_trained_model_folder(str(base), ['all'], 'missing.pth')
            except FileNotFoundError:
                pass
            else:
                raise AssertionError('checkpoint argument was ignored')
            assert nnUNetPredictor.auto_detect_available_folds(str(base), 'checkpoint_best.pth') == ['all']
            reports.append({'dim': dim, 'trainer': trainer_class.__name__,
                            'train_loss': float(train['loss']), 'validation_loss': float(val['loss']),
                            'roundtrip_shape': list(actual.shape),
                            'status': 'passed', 'mamba': 'explicit_cpu_test_double'})
    report = {'scope': 'real nnU-Net encoder/decoder/trainer with synthetic batches',
              'not_tested': ['real selective-scan CUDA kernel', 'full CT patch GPU memory', 'DSC/NSD efficacy'],
              'torch': torch.__version__, 'runs': reports}
    (output / 'framework_check.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
