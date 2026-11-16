#!/usr/bin/env python
"""Real nnU-Net CPU wiring check; explicit Mamba test double, synthetic data.

Never installed by the production runner. No CUDA/AMP/clinical efficacy claim.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import types

from check_stage3_framework import synthetic_plans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for key, sub in [('nnUNet_raw', 'raw'), ('nnUNet_preprocessed', 'preprocessed'), ('nnUNet_results', 'results')]:
        os.environ[key] = str(out / sub)
        (out / sub).mkdir(exist_ok=True)
    os.environ['nnUNet_compile'] = 'false'
    # Only self-generated trusted checkpoints are loaded in this process.
    # nnU-Net 2.1.1 predates PyTorch's new weights_only default.
    os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'umamba'))
    import numpy as np
    import torch
    from torch import nn
    from torch.nn import functional as F
    torch.set_num_threads(1)

    class ExplicitCpuMambaDouble(nn.Module):
        def __init__(self, d_model, **kwargs):
            super().__init__()
            self.projection = nn.Linear(d_model, d_model)

        def forward(self, x):
            return self.projection(x).tanh()

    stub = types.ModuleType('mamba_ssm')
    stub.Mamba = ExplicitCpuMambaDouble
    sys.modules['mamba_ssm'] = stub
    wandb = types.ModuleType('wandb')
    wandb.log = lambda *a, **kw: None
    sys.modules['wandb'] = wandb
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMambaUNetAttnRes import nnUNetTrainerMambaUNetBGBlockAttnRes as C
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMambaUNetSpatialRefinement import (
        nnUNetTrainerMambaUNetBGSpatialND as ND, nnUNetTrainerMambaUNetBGSpatialR0 as R0)
    from nnunetv2.nets.MambaUNet import UNetResDecoder
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    plans, configuration = synthetic_plans(3)
    plans['foreground_intensity_properties_per_channel'] = {'0': {}}
    dataset = {'channel_names': {'0': 'CT'}, 'labels': {'background': 0, 'organ_a': 1, 'organ_b': 2},
               'numTraining': 2, 'file_ending': '.nii.gz'}
    def make(cls):
        torch.manual_seed(1234)
        trainer = cls(deepcopy(plans), configuration, 'all', deepcopy(dataset), device=torch.device('cpu'))
        trainer.initialize()
        return trainer

    baseline = make(C)
    baseline_state = {k: v.detach().clone() for k, v in baseline.network.state_dict().items()}
    initial_nd = None
    reports = []
    for cls in (ND, R0):
        trainer = make(cls)
        net = trainer.network
        for key, value in baseline_state.items():
            torch.testing.assert_close(net.state_dict()[key], value, rtol=0, atol=0)
        if initial_nd is None:
            initial_nd = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            for key, value in initial_nd.items():
                torch.testing.assert_close(net.state_dict()[key], value, rtol=0, atol=0)
        optimizer_ids = {id(p) for group in trainer.optimizer.param_groups for p in group['params']}
        assert optimizer_ids == {id(p) for p in net.parameters() if p.requires_grad}
        assert all(not layer.mha.batch_first for layer in net.mamba_mlp.layers)
        assert trainer.boundary_loss_weight == .1 and net.boundary_guidance and net.auxiliary_boundary
        shape = trainer.configuration_manager.patch_size
        image = torch.randn(2, 1, *shape)
        target = torch.randint(0, 3, (2, 1, *shape))
        batch = {'data': image, 'target': [target, F.interpolate(target.float(), scale_factor=.5, mode='nearest').long()]}
        # With the extra block replaced by identity, decoder exactly matches C.
        block = net.decoder.spatial_block
        net.eval()
        with torch.no_grad():
            skips = net.encoder(image)
            expected = UNetResDecoder.forward(net.decoder, skips)
            net.decoder.spatial_block = nn.Identity()
            actual = net.decoder(skips)
            net.decoder.spatial_block = block
            for a, b in zip(expected, actual):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            # Verify both the coarsest head and downstream upsampling see the block output.
            observed = {}
            handles = [block.register_forward_hook(lambda m, i, o: observed.update(block=o.detach().clone())),
                       net.decoder.seg_layers[0].register_forward_pre_hook(lambda m, i: observed.update(head=i[0].detach().clone())),
                       net.decoder.transpconvs[1].register_forward_pre_hook(lambda m, i: observed.update(next=i[0].detach().clone()))]
            try:
                net(image)
            finally:
                for handle in handles:
                    handle.remove()
            assert torch.equal(observed['block'], observed['head'])
            assert torch.equal(observed['block'], observed['next'])
        net.train()
        before = [p.detach().clone() for p in block.parameters()]
        loss = trainer.train_step(batch)
        assert np.isfinite(loss['loss']) and np.isfinite(loss['boundary_loss'])
        for name, p in block.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert any(not torch.equal(a, b) for a, b in zip(before, block.parameters()))
        assert net.boundary_logits is None
        net.eval()
        with torch.no_grad():
            val = trainer.validation_step(batch)
        assert np.isfinite(val['loss'])

        folder = Path(trainer.output_folder)
        base = Path(trainer.output_folder_base)
        (base / 'plans.json').write_text(json.dumps(plans), encoding='utf-8')
        (base / 'dataset.json').write_text(json.dumps(dataset), encoding='utf-8')
        trainer.current_epoch = 4
        trainer.lr_scheduler.step(4)
        checkpoint = folder / 'checkpoint_best.pth'
        trainer.save_checkpoint(str(checkpoint))
        resumed = make(cls)
        resumed.load_checkpoint(str(checkpoint))
        assert resumed.current_epoch == 5
        for p, q in zip(net.parameters(), resumed.network.parameters()):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
            if 'momentum_buffer' in trainer.optimizer.state[p]:
                torch.testing.assert_close(trainer.optimizer.state[p]['momentum_buffer'],
                                           resumed.optimizer.state[q]['momentum_buffer'], rtol=0, atol=0)
        # The inherited scheduler resumes from the saved epoch, not a stored scheduler dict.
        trainer.lr_scheduler.step(5)
        resumed.lr_scheduler.step(resumed.current_epoch)
        assert trainer.optimizer.param_groups[0]['lr'] == resumed.optimizer.param_groups[0]['lr']
        trainer.network.train()
        resumed.network.train()
        trainer.train_step(batch)
        resumed.train_step(batch)
        for p, q in zip(trainer.network.parameters(), resumed.network.parameters()):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
        trainer.save_checkpoint(str(checkpoint))

        predictor = nnUNetPredictor(tile_step_size=.5, use_gaussian=True, use_mirroring=False,
                                   device=torch.device('cpu'), perform_everything_on_device=False,
                                   allow_tqdm=False)
        predictor.initialize_from_trained_model_folder(str(base), ['all'], 'checkpoint_best.pth')
        predictor.network.load_state_dict(predictor.list_of_parameters[0])
        predictor.network.eval()
        trainer.set_deep_supervision_enabled(False)
        trainer.network.eval()
        with torch.no_grad():
            expected = trainer.network(image[:1])
            actual = predictor.network(image[:1])
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
            # Overlap + padding exercised with a larger synthetic volume and a smaller one.
            for volume_shape in [(11, 19, 21), (5, 9, 11)]:
                volume = torch.randn(1, *volume_shape)
                logits = predictor.predict_sliding_window_return_logits(volume)
                assert tuple(logits.shape) == (3, *volume_shape) and torch.isfinite(logits).all()
            segmentation = predictor.predict_single_npy_array(
                np.random.default_rng(1234).normal(size=(1, 11, 19, 21)).astype('float32'),
                {'spacing': [1., 1., 1.]}, None, None, False)
            assert segmentation.shape == (11, 19, 21)
            assert set(np.unique(segmentation)).issubset({0, 1, 2})
            # A constant-logit test isolates coverage/Gaussian normalization
            # from legitimate context-dependent differences between windows.
            saved_network = predictor.network
            class ConstantLogits(nn.Module):
                def forward(self, x):
                    return torch.full((x.shape[0], 3, *x.shape[2:]), 2., device=x.device)
            predictor.network = ConstantLogits()
            try:
                for volume_shape in [(11, 19, 21), (5, 9, 11)]:
                    logits = predictor.predict_sliding_window_return_logits(torch.zeros(1, *volume_shape))
                    torch.testing.assert_close(logits, torch.full_like(logits, 2.), rtol=1e-3, atol=1e-3)
            finally:
                predictor.network = saved_network
        reports.append({'trainer': cls.__name__, 'variant': cls.spatial_variant,
                        'added_parameters': sum(p.numel() for p in block.parameters()),
                        'inherited_C_initial_weights_identical': True,
                        'ND_R0_initial_weights_identical': True,
                        'optimizer_coverage': True, 'decoder_location_verified': True,
                        'train_loss': float(loss['loss']), 'boundary_loss': float(loss['boundary_loss']),
                        'validation_loss': float(val['loss']),
                        'checkpoint_eval_exact': True, 'momentum_and_next_update_exact': True,
                        'resume_lr': resumed.optimizer.param_groups[0]['lr'],
                        'sliding_overlap_padding_and_raw_array_pipeline': 'passed',
                        'constant_logits_gaussian_coverage': 'passed',
                        'mamba': 'explicit_CPU_test_double', 'status': 'passed'})
    report = {'scope': 'synthetic CPU integration with actual nnU-Net; explicit Mamba/W&B doubles',
              'torch': torch.__version__, 'cuda_available': torch.cuda.is_available(), 'runs': reports,
              'not_tested': ['real Mamba CUDA', 'AMP GradScaler restoration', 'full-patch GPU memory',
                             'clinical CT file export and 50-case evaluation', 'worker RNG exact resume', 'efficacy']}
    (out / 'framework_check.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
