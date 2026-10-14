"""Separate nnU-Net Trainer names keep A/F/C checkpoints in separate folders."""

import json
from pathlib import Path

import numpy as np

from nnunetv2.nets.MambaUNetAttnRes import get_attnres_mamba_from_plans
from nnunetv2.training.loss.segmentation_boundary import SegmentationBoundaryLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMambaUNet import nnUNetTrainerMambaUNet


class nnUNetTrainerMambaUNetBlockAttnRes(nnUNetTrainerMambaUNet):
    """F: 12 unchanged TransMamba layers in four groups of three, no boundary."""

    group_size = 3
    boundary_guidance = False
    auxiliary_boundary = False
    boundary_hidden_channels = 8
    boundary_loss_weight = 0.1

    @classmethod
    def build_network_architecture(cls, plans_manager, dataset_json,
                                   configuration_manager, num_input_channels,
                                   enable_deep_supervision: bool = True):
        return get_attnres_mamba_from_plans(
            plans_manager, dataset_json, configuration_manager, num_input_channels,
            deep_supervision=enable_deep_supervision, block_size=cls.group_size,
            boundary_guidance=cls.boundary_guidance,
            auxiliary_boundary=cls.auxiliary_boundary,
            boundary_hidden_channels=cls.boundary_hidden_channels,
        )

    def _do_i_compile(self):
        # The initial research implementation uses a transient auxiliary cache.
        # Avoid compiling a stateful cache; F/C both follow this policy.
        return False

    def initialize(self):
        if self.is_ddp:
            raise RuntimeError(
                "Stage-two protocol requires one GPU and physical batch=2; "
                "DDP changes the inherited MHA batch interaction."
            )
        if self.label_manager.has_regions:
            raise NotImplementedError("Stage-two boundary supervision currently supports semantic class labels, not regions")
        super().initialize()
        config = {
            'trainer': self.__class__.__name__, 'group_size': self.group_size,
            'grouping_unit': 'complete_TransMamba_layer',
            'boundary_guidance': self.boundary_guidance,
            'auxiliary_boundary': self.auxiliary_boundary,
            'boundary_hidden_channels': self.boundary_hidden_channels,
            'boundary_loss_weight': self.boundary_loss_weight if self.auxiliary_boundary else 0,
            'boundary_target': 'two_sided_4_or_6_connected_semantic_label_interfaces',
            'mha_batch_first': False, 'epochs': self.num_epochs,
            'batch_size': self.batch_size,
            'patch_size': list(self.configuration_manager.patch_size),
            'torch_compile': False,
        }
        folder = Path(self.output_folder)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'stage2_config.json').write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8',
        )
        self.print_to_log_file('Stage2 configuration:', config)

    def _build_loss(self):
        segmentation_loss = super()._build_loss()
        if self.auxiliary_boundary:
            return SegmentationBoundaryLoss(
                segmentation_loss, self.network,
                self.boundary_loss_weight, self.label_manager.ignore_label,
            )
        return segmentation_loss

    def train_step(self, batch):
        output = super().train_step(batch)
        if self.auxiliary_boundary:
            output['segmentation_loss'] = self.loss.last_segmentation_loss.cpu().numpy()
            output['boundary_loss'] = self.loss.last_boundary_loss.cpu().numpy()
        return output

    def on_train_epoch_end(self, train_outputs):
        super().on_train_epoch_end(train_outputs)
        if self.auxiliary_boundary:
            self.print_to_log_file(
                'Stage2 loss components: segmentation=',
                float(np.mean([o['segmentation_loss'] for o in train_outputs])),
                'boundary_unweighted=',
                float(np.mean([o['boundary_loss'] for o in train_outputs])),
                'boundary_weight=', self.boundary_loss_weight,
            )


class nnUNetTrainerMambaUNetBGBlockAttnRes(nnUNetTrainerMambaUNetBlockAttnRes):
    """C: source-dependent boundary-guided aggregation + supervised head."""

    boundary_guidance = True
    auxiliary_boundary = True


class nnUNetTrainerMambaUNetBlockAttnResBoundarySupervision(nnUNetTrainerMambaUNetBlockAttnRes):
    """F+aux: same supervised boundary head as C, but no boundary injection."""

    boundary_guidance = False
    auxiliary_boundary = True
