"""C+ND/C+R0: retain C and refine the first decoder stage before its head."""
from math import prod

import torch
from torch import nn

from nnunetv2.nets.MambaUNetAttnRes import get_attnres_mamba_from_plans
from nnunetv2.nets.spatial_receptive_block import SpatialReceptiveBlock3D


class SpatialRefinementDecoder(nn.Module):
    """Preserve the original decoder state-dict names; register one extra block.

    The forward is the original UNetResDecoder path plus spatial_block after
    stages[0]. No hooks or train/eval-specific bypass are used.
    """
    def __init__(self, decoder, variant):
        super().__init__()
        if decoder.encoder.conv_op is not nn.Conv3d:
            raise ValueError('Spatial refinement experiments support 3D CT only')
        if len(decoder.stages) < 1:
            raise ValueError('at least one decoder stage is required')
        self.deep_supervision = decoder.deep_supervision
        self.encoder = decoder.encoder
        self.num_classes = decoder.num_classes
        self.stages = decoder.stages
        self.transpconvs = decoder.transpconvs
        self.seg_layers = decoder.seg_layers
        self.spatial_block = SpatialReceptiveBlock3D(self.encoder.output_channels[-2], variant)

    def forward(self, skips):
        low = skips[-1]
        outputs = []
        for index, stage in enumerate(self.stages):
            x = self.transpconvs[index](low)
            x = stage(torch.cat((x, skips[-(index + 2)]), dim=1))
            if index == 0:
                x = self.spatial_block(x)
            if self.deep_supervision:
                outputs.append(self.seg_layers[index](x))
            elif index == len(self.stages) - 1:
                outputs.append(self.seg_layers[index](x))
            low = x
        return outputs[::-1] if self.deep_supervision else outputs[0]

    def compute_conv_feature_map_size(self, input_size):
        # Same baseline estimate plus convolution outputs of the extra block.
        # Not a peak-memory estimator; it excludes attention and temporary ops.
        sizes = []
        for stride in self.encoder.strides[:-1]:
            input_size = [int(i) // int(s) for i, s in zip(input_size, stride)]
            sizes.append(input_size)
        total = 0
        for index, stage in enumerate(self.stages):
            shape = sizes[-(index + 1)]
            total += stage.compute_conv_feature_map_size(shape)
            total += self.encoder.output_channels[-(index + 2)] * prod(shape)
            if self.deep_supervision or index == len(self.stages) - 1:
                total += self.num_classes * prod(shape)
        total += 4 * self.spatial_block.channels * prod(sizes[-1])
        return int(total)


def get_spatial_refinement_mamba_from_plans(
        plans_manager, dataset_json, configuration_manager, num_input_channels,
        deep_supervision=True, variant='R0', block_size=3, boundary_hidden_channels=8):
    if len(configuration_manager.patch_size) != 3:
        raise ValueError('C+ND/C+R0 require a 3D configuration')
    # Build/initialize ALL original C modules first, then initialize the addition.
    # With the same RNG state, ND and R0 have identical corresponding tensors.
    model = get_attnres_mamba_from_plans(
        plans_manager, dataset_json, configuration_manager, num_input_channels,
        deep_supervision=deep_supervision, block_size=block_size,
        boundary_guidance=True, auxiliary_boundary=True,
        boundary_hidden_channels=boundary_hidden_channels)
    model.decoder = SpatialRefinementDecoder(model.decoder, variant)
    return model
