"""Lightweight image-feature boundary prediction for BG-Block-AttnRes.

The full-resolution logits are used for auxiliary supervision. Only a pooled
copy is passed to the bottleneck aggregator; inference never needs labels.
"""

import math
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class BoundaryPriorPredictor(nn.Module):
    """Predict one boundary logit per shallow encoder location in 2D or 3D.

    GroupNorm keeps the predictor independent of the physical batch size. The
    shallow features should have the original segmentation target resolution,
    so small boundaries are supervised before any bottleneck downsampling.
    """

    def __init__(self, in_channels: int, spatial_dims: int = 3,
                 hidden_channels: int = 8):
        super().__init__()
        if spatial_dims not in (2, 3):
            raise ValueError("spatial_dims must be 2 or 3")
        if in_channels < 1 or hidden_channels < 2:
            raise ValueError("in_channels must be positive and hidden_channels >= 2")
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        conv = nn.Conv2d if spatial_dims == 2 else nn.Conv3d
        # At least two channels per group for the default 8/16-channel heads,
        # including very small spatial inputs used for debugging.
        groups = math.gcd(hidden_channels, 4)
        if groups == hidden_channels:
            groups = 1
        self.features = nn.Sequential(
            conv(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden_channels),
            nn.GELU(),
        )
        self.output = conv(hidden_channels, 1, kernel_size=1, bias=True)

    def forward(self, shallow_features: torch.Tensor) -> torch.Tensor:
        if shallow_features.ndim != self.spatial_dims + 2:
            raise ValueError(
                f"expected [B, C, *spatial] with {self.spatial_dims} spatial dimensions, "
                f"got {tuple(shallow_features.shape)}"
            )
        if shallow_features.shape[1] != self.in_channels:
            raise ValueError(f"expected {self.in_channels} channels")
        if any(size < 1 for size in shallow_features.shape[2:]):
            raise ValueError("spatial dimensions must be non-empty")
        return self.output(self.features(shallow_features))


def pool_boundary_prior(logits: torch.Tensor, output_size: Sequence[int]) -> torch.Tensor:
    """Align full-resolution logits to a smaller bottleneck and apply sigmoid.

    Max pooling preserves a confident boundary within each pooling cell. Because
    sigmoid is monotone, sigmoid(max(logits)) equals max(sigmoid(logits)). This
    is feature alignment, not physical-distance dilation of a target boundary.
    The returned probability retains gradients into the shallow predictor.
    """
    if logits.ndim not in (4, 5) or logits.shape[1] != 1:
        raise ValueError("boundary logits must have shape [B, 1, H, W] or [B, 1, D, H, W]")
    output_size = tuple(output_size)
    if len(output_size) != logits.ndim - 2:
        raise ValueError("output_size must match the number of spatial dimensions")
    if any(not isinstance(size, int) or isinstance(size, bool) or size < 1
           for size in output_size):
        raise ValueError("output_size entries must be positive integers")
    if any(out > inp for out, inp in zip(output_size, logits.shape[2:])):
        raise ValueError("pool_boundary_prior only downsamples; output_size exceeds logits size")
    pool = F.adaptive_max_pool2d if logits.ndim == 4 else F.adaptive_max_pool3d
    return pool(logits, output_size).sigmoid()
