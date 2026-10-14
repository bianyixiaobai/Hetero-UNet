"""Auxiliary boundary supervision from ordinary semantic segmentation labels.

Boundaries use 4-connected (2D) or 6-connected (3D) voxel neighbors. Both sides
of an organ/background or organ/organ transition are marked. No padding is
compared, so a homogeneous foreground patch has no artificial edge boundary.
This definition is in voxels, not a spacing-aware millimeter dilation, and is
independent of the NSD evaluation tolerance.
"""

import math
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


@torch.no_grad()
def semantic_boundary_target(target: torch.Tensor, ignore_label: Optional[int] = None
                             ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return float32 boundary labels and a bool auxiliary supervision mask.

    ``target`` is a semantic label tensor [B, 1, H, W] or [B, 1, D, H, W],
    rather than one-hot regions. Ignored voxels never participate in a label
    comparison. They and their in-volume immediate neighbors are excluded from
    supervision, because the true boundary at that interface is unknown.

    Derive this target at the finest label resolution. Deriving boundaries from
    downsampled semantic labels first can remove narrow organs and transitions.
    """
    if target.ndim not in (4, 5) or target.shape[1] != 1:
        raise ValueError("semantic target must have shape [B, 1, H, W] or [B, 1, D, H, W]")
    if any(size < 1 for size in target.shape[2:]):
        raise ValueError("target spatial dimensions must be non-empty")
    valid = torch.ones_like(target, dtype=torch.bool)
    if ignore_label is not None:
        valid = target != ignore_label
    supervision_mask = valid.clone()
    boundary = torch.zeros_like(target, dtype=torch.bool)
    for axis in range(2, target.ndim):
        left = [slice(None)] * target.ndim
        right = [slice(None)] * target.ndim
        left[axis] = slice(None, -1)
        right[axis] = slice(1, None)
        left, right = tuple(left), tuple(right)
        pair_valid = valid[left] & valid[right]
        different = (target[left] != target[right]) & pair_valid
        boundary[left] |= different
        boundary[right] |= different
        # Keep the original valid mask for comparisons; exclusion must not
        # recursively spread through the patch beyond one voxel.
        supervision_mask[left] &= valid[right]
        supervision_mask[right] &= valid[left]
    boundary &= supervision_mask
    return boundary.to(dtype=torch.float32), supervision_mask


class BoundaryAuxiliaryLoss(nn.Module):
    """Masked BCE-with-logits plus optional per-sample binary soft Dice loss.

    The predictor logits must be at the finest segmentation target resolution.
    Ignored samples do not dilute either component. An entirely ignored batch
    returns a finite differentiable zero with zero gradients to the logits.
    Compute in float32 so this loss remains stable under mixed precision.
    """

    def __init__(self, ignore_label: Optional[int] = None, bce_weight: float = 1.,
                 dice_weight: float = 1., smooth: float = 1e-5):
        super().__init__()
        if not all(math.isfinite(weight) and weight >= 0
                   for weight in (bce_weight, dice_weight)):
            raise ValueError("BCE and Dice weights must be finite and non-negative")
        if not math.isfinite(smooth) or smooth <= 0:
            raise ValueError("smooth must be finite and positive")
        self.ignore_label = ignore_label
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.shape != target.shape:
            raise ValueError(
                "boundary logits and finest semantic target must have the same shape; "
                f"got {tuple(logits.shape)} and {tuple(target.shape)}"
            )
        boundary, mask = semantic_boundary_target(target, self.ignore_label)
        prediction = logits.float()
        boundary = boundary.to(device=prediction.device)
        mask_float = mask.to(device=prediction.device, dtype=prediction.dtype)
        # Initialize with a graph-connected zero, including the case in which
        # both optional components have zero weight.
        result = prediction.sum() * 0.
        if self.bce_weight:
            bce = F.binary_cross_entropy_with_logits(prediction, boundary, reduction="none")
            bce = (bce * mask_float).sum() / mask_float.sum().clamp_min(1.)
            result = result + self.bce_weight * bce
        if self.dice_weight:
            axes = tuple(range(1, prediction.ndim))
            probability = prediction.sigmoid() * mask_float
            reference = boundary * mask_float
            intersection = (probability * reference).sum(dim=axes)
            denominator = probability.sum(dim=axes) + reference.sum(dim=axes)
            dice = (2. * intersection + self.smooth) / (denominator + self.smooth)
            sample_valid = (mask_float.sum(dim=axes) > 0).to(dtype=prediction.dtype)
            dice_loss = ((1. - dice) * sample_valid).sum() / sample_valid.sum().clamp_min(1.)
            result = result + self.dice_weight * dice_loss
        return result
