"""Distinct discoverable Trainer names for D and E checkpoint reconstruction."""
import json
from pathlib import Path

from nnunetv2.nets.MambaUNetSkipFusion import get_skip_fusion_mamba_from_plans
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMambaUNet import nnUNetTrainerMambaUNet
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMambaUNetAttnRes import nnUNetTrainerMambaUNetBGBlockAttnRes


class SkipFusionMixin:
    fusion_width = 16
    with_bg = False

    @classmethod
    def build_network_architecture(cls, plans_manager, dataset_json, configuration_manager,
                                   num_input_channels, enable_deep_supervision=True):
        return get_skip_fusion_mamba_from_plans(
            plans_manager, dataset_json, configuration_manager, num_input_channels,
            enable_deep_supervision, cls.with_bg, cls.fusion_width)

    def _do_i_compile(self):
        return False

    def initialize(self):
        if self.is_ddp:
            raise RuntimeError('Frozen MHA protocol requires single GPU, physical batch=2')
        super().initialize()
        config = {
            'trainer': type(self).__name__, 'model': 'E' if self.with_bg else 'D',
            'base': 'C' if self.with_bg else 'A', 'fusion_width_cap': self.fusion_width,
            'sources': 'target_and_adjacent_non_bottleneck_encoder_scales',
            'weights': 'decoder_conditioned_voxelwise_softmax_over_sources',
            'alignment': 'average_shrink_project_linear_upsample_align_corners_false',
            'initialization': 'zero_output_projection_identity_skip',
            'mha_batch_first': False, 'torch_compile': False,
            'batch_size': self.batch_size, 'patch_size': list(self.configuration_manager.patch_size),
            'epochs': self.num_epochs,
        }
        Path(self.output_folder, 'stage3_config.json').write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        self.print_to_log_file('Stage3 configuration:', config)


class nnUNetTrainerMambaUNetMultiScaleSkip(SkipFusionMixin, nnUNetTrainerMambaUNet):
    """D: A + cross-scale fusion; no auxiliary boundary objective."""


class nnUNetTrainerMambaUNetBGMultiScaleSkip(SkipFusionMixin, nnUNetTrainerMambaUNetBGBlockAttnRes):
    """E: C + the identical cross-scale fusion and original C boundary loss."""
    with_bg = True
