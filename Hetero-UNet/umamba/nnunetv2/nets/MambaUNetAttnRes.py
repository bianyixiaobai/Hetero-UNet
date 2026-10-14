"""Stage-two F/C models; the original MambaUNet remains unchanged.

The group-level adaptation stores residual *deltas*, not cumulative hidden
states. Original TransMamba layers and their MHA layout are used verbatim.
Only the factory needs nnU-Net/dynamic-network-architectures/Mamba; the wrapper
can be checked with small torch networks on a CPU.
"""

from typing import Optional

import torch
from torch import Tensor, nn

from nnunetv2.nets.block_attention_residuals import DepthAttentionAggregator
from nnunetv2.nets.boundary_prior import BoundaryPriorPredictor, pool_boundary_prior


class BlockAttnResMambaEncoder(nn.Module):
    """Preserve standard residuals within groups; attend between groups.

    H initially contains the bottleneck embedding. For group g, a_g=Attn_g(H),
    z_g=original_group_g(a_g), and H.append(z_g-a_g). Return
    last_norm(Attn_final(H)). Group size counts full TransMamba layers, not the
    MHA/Mamba/MLP sublayers. This is a group-level vision adaptation of Block
    AttnRes, not a bit-for-bit reproduction of its LLM sublayer pseudocode.
    """

    def __init__(self, layers: nn.ModuleList, last_norm: nn.Module,
                 hidden_size: int, block_size: int = 3,
                 boundary_guidance: bool = False) -> None:
        super().__init__()
        if not isinstance(block_size, int) or isinstance(block_size, bool) or block_size < 1:
            raise ValueError("block_size must be a positive integer")
        if len(layers) < 1:
            raise ValueError("at least one TransMamba layer is required")
        self.layers = layers
        self.last_norm = last_norm
        self.block_size = block_size
        self.hidden_size = hidden_size
        self.boundary_guidance = boundary_guidance
        group_count = (len(layers) + block_size - 1) // block_size
        self.aggregators = nn.ModuleList([
            DepthAttentionAggregator(hidden_size, boundary_conditioned=boundary_guidance)
            for _ in range(group_count + 1)
        ])

    def forward(self, tokens: Tensor, boundary: Optional[Tensor] = None) -> Tensor:
        history = [tokens]
        for group_index, start in enumerate(range(0, len(self.layers), self.block_size)):
            group_input = self.aggregators[group_index](history, boundary)
            group_output = group_input
            for layer in self.layers[start:start + self.block_size]:
                group_output = layer(group_output)
            history.append(group_output - group_input)
        return self.last_norm(self.aggregators[-1](history, boundary))


class AttnResMambaUNet(nn.Module):
    """Wrap a freshly constructed baseline with F, C, or F+boundary supervision.

    Decoder outputs keep the nnU-Net tensor/deep-supervision-list contract.
    Auxiliary logits are transient, training-only and consumed by the loss;
    they are never stored in a checkpoint or returned as segmentation classes.
    """

    def __init__(self, base_model: nn.Module, hidden_size: int, spatial_dims: int,
                 block_size: int = 3, boundary_guidance: bool = False,
                 auxiliary_boundary: bool = False,
                 boundary_hidden_channels: int = 8) -> None:
        super().__init__()
        if spatial_dims not in (2, 3):
            raise ValueError("only 2D and 3D inputs are supported")
        self.encoder = base_model.encoder
        self.decoder = base_model.decoder
        self.spatial_dims = spatial_dims
        self.boundary_guidance = boundary_guidance
        self.auxiliary_boundary = auxiliary_boundary
        self.mamba_mlp = BlockAttnResMambaEncoder(
            base_model.mamba_mlp.layers, base_model.mamba_mlp.last_norm,
            hidden_size, block_size, boundary_guidance,
        )
        if boundary_guidance or auxiliary_boundary:
            self.boundary_head = BoundaryPriorPredictor(
                self.encoder.output_channels[0], spatial_dims, boundary_hidden_channels,
            )
        else:
            self.boundary_head = None
        self._boundary_logits: Optional[Tensor] = None

    @property
    def boundary_logits(self) -> Optional[Tensor]:
        return self._boundary_logits

    def consume_boundary_logits(self) -> Optional[Tensor]:
        logits = self._boundary_logits
        self._boundary_logits = None
        return logits

    def forward(self, x: Tensor):
        self._boundary_logits = None
        if x.ndim != self.spatial_dims + 2:
            raise ValueError("input must be [B, C, *spatial] for the configured dimensionality")
        skips = list(self.encoder(x))
        middle = skips[-1]
        tokens = middle.flatten(2).transpose(1, 2)
        boundary = None
        # The F+aux control does not need its head at inference.
        if self.boundary_head is not None and (self.boundary_guidance or self.training):
            logits = self.boundary_head(skips[0])
            if self.training and self.auxiliary_boundary:
                self._boundary_logits = logits
            if self.boundary_guidance:
                prior = pool_boundary_prior(logits, middle.shape[2:])
                boundary = prior.flatten(2).transpose(1, 2)
        encoded = self.mamba_mlp(tokens, boundary)
        skips[-1] = encoded.transpose(1, 2).reshape_as(middle)
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        # Keep the baseline planner interface. It excludes attention/Mamba and
        # the added head, so real GPU memory still needs a deployment check.
        return (self.encoder.compute_conv_feature_map_size(input_size)
                + self.decoder.compute_conv_feature_map_size(input_size))


def get_attnres_mamba_from_plans(plans_manager, dataset_json, configuration_manager,
                               num_input_channels: int, deep_supervision: bool = True,
                               block_size: int = 3, boundary_guidance: bool = False,
                               auxiliary_boundary: bool = False,
                               boundary_hidden_channels: int = 8):
    # The baseline factory initializes its original modules exactly as before.
    # Added queries remain zero initialized; do not reinitialize the whole model.
    from nnunetv2.nets.MambaUNet import get_umamba_bot_from_plans

    base = get_umamba_bot_from_plans(
        plans_manager, dataset_json, configuration_manager,
        num_input_channels, deep_supervision=deep_supervision,
    )
    hidden_size = base.mamba_mlp.last_norm.normalized_shape[0]
    spatial_dims = len(configuration_manager.conv_kernel_sizes[0])
    return AttnResMambaUNet(
        base, hidden_size, spatial_dims, block_size,
        boundary_guidance, auxiliary_boundary, boundary_hidden_channels,
    )
