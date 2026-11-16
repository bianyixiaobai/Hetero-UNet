"""Isolated discoverable Trainers inheriting C's loss and training protocol."""
import hashlib
import json
from pathlib import Path

from nnunetv2.nets.MambaUNetSpatialRefinement import get_spatial_refinement_mamba_from_plans
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMambaUNetAttnRes import nnUNetTrainerMambaUNetBGBlockAttnRes


class SpatialRefinementMixin:
    spatial_variant = 'R0'

    @classmethod
    def build_network_architecture(cls, plans_manager, dataset_json, configuration_manager,
                                   num_input_channels, enable_deep_supervision=True):
        return get_spatial_refinement_mamba_from_plans(
            plans_manager, dataset_json, configuration_manager, num_input_channels,
            deep_supervision=enable_deep_supervision, variant=cls.spatial_variant,
            block_size=cls.group_size, boundary_hidden_channels=cls.boundary_hidden_channels)

    def initialize(self):
        super().initialize()
        block = self.network.decoder.spatial_block
        config = {
            'schema_version': 1, 'model': 'C+' + self.spatial_variant,
            'trainer': type(self).__name__, 'base': 'C',
            'location': 'after_decoder_stage_0_before_segmentation_head_and_next_upsample',
            'channels': block.channels, 'kernels': [1, 3, 3],
            'dilations': [1, 1, 2 if self.spatial_variant == 'R0' else 1],
            'branch_groups': block.channels, 'norm_groups': block.channels,
            'norm_channels': 3 * block.channels, 'residual': True,
            'initialization': 'pytorch_defaults_after_original_C_initialization',
            'added_parameters': sum(p.numel() for p in block.parameters()),
            'group_size': self.group_size, 'boundary_guidance': self.boundary_guidance,
            'auxiliary_boundary': self.auxiliary_boundary,
            'boundary_hidden_channels': self.boundary_hidden_channels,
            'boundary_loss_weight': self.boundary_loss_weight,
            'mha_batch_first': False, 'torch_compile': False,
            'batch_size': self.batch_size, 'patch_size': list(self.configuration_manager.patch_size),
            'epochs': self.num_epochs,
        }
        digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode('utf-8')).hexdigest()
        config['config_sha256'] = digest
        Path(self.output_folder, 'spatial_refinement_config.json').write_text(
            json.dumps(config, indent=2) + '\n', encoding='utf-8')
        self.print_to_log_file('Spatial refinement configuration:', config)


class nnUNetTrainerMambaUNetBGSpatialND(SpatialRefinementMixin, nnUNetTrainerMambaUNetBGBlockAttnRes):
    spatial_variant = 'ND'


class nnUNetTrainerMambaUNetBGSpatialR0(SpatialRefinementMixin, nnUNetTrainerMambaUNetBGBlockAttnRes):
    spatial_variant = 'R0'
