"""DMRFB reference adaptation and a parameter-matched non-dilated control.

Reference: Rahman et al., EfficientMedNeXt, MICCAI 2025, section 2.1.
https://github.com/SLDGroup/EfficientMedNeXt/blob/main/networks/MedNeXt/mednextv1/efficient_mednext_blocks.py
This is an explicit same-channel, stride-one implementation, not a new method.
ND retains all branches/parameters and changes only branch 3 dilation/padding.
"""
from math import prod

import torch
from torch import nn


class SpatialReceptiveBlock3D(nn.Module):
    def __init__(self, channels: int, variant: str):
        super().__init__()
        if not isinstance(channels, int) or isinstance(channels, bool) or channels < 1:
            raise ValueError('channels must be a positive integer')
        if variant not in ('ND', 'R0'):
            raise ValueError('variant must be ND or R0')
        self.channels = channels
        self.variant = variant
        dilation = 2 if variant == 'R0' else 1
        self.branches = nn.ModuleList([
            nn.Conv3d(channels, channels, 1, groups=channels, bias=False),
            nn.Conv3d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv3d(channels, channels, 3, padding=dilation, dilation=dilation,
                      groups=channels, bias=False),
        ])
        # Match the reference's group count and concatenation order exactly.
        # These groups are not equivalent to a separate norm on each branch.
        self.norm = nn.GroupNorm(channels, 3 * channels)
        self.act = nn.GELU()
        self.projection = nn.Conv3d(3 * channels, channels, 1, bias=True)

    def forward(self, x):
        if x.ndim != 5 or x.shape[1] != self.channels:
            raise ValueError('expected [B, channels, D, H, W]')
        features = torch.cat([branch(x) for branch in self.branches], dim=1)
        return x + self.projection(self.act(self.norm(features)))

    def conv_macs(self, spatial_shape, batch_size=1):
        """Dense convolution MACs only; excludes bias, norm, GELU and residual.

        One multiply-accumulate is one MAC (two FLOPs in the report).
        Padding positions are counted, as in standard dense-convolution counts.
        """
        return batch_size * prod(spatial_shape) * (55 * self.channels + 3 * self.channels ** 2)

    def extra_repr(self):
        return f'channels={self.channels}, variant={self.variant}'
