"""Add boundary supervision without changing the segmentation output contract."""

import math
import weakref

import torch
from torch import nn

from nnunetv2.training.loss.boundary import BoundaryAuxiliaryLoss


class SegmentationBoundaryLoss(nn.Module):
    def __init__(self, segmentation_loss: nn.Module, network: nn.Module,
                 boundary_weight: float = 0.1, ignore_label=None) -> None:
        super().__init__()
        if not math.isfinite(boundary_weight) or boundary_weight < 0:
            raise ValueError("boundary_weight must be finite and nonnegative")
        self.segmentation_loss = segmentation_loss
        self.boundary_loss = BoundaryAuxiliaryLoss(ignore_label=ignore_label)
        self.boundary_weight = boundary_weight
        # Do not register the entire model again under the loss module.
        self._network_ref = weakref.ref(network)
        self.last_segmentation_loss = None
        self.last_boundary_loss = None

    def forward(self, output, target):
        network = self._network_ref()
        if network is None:
            raise RuntimeError("boundary-supervised network no longer exists")
        logits = network.consume_boundary_logits()
        segmentation = self.segmentation_loss(output, target)
        boundary = segmentation.new_zeros(())
        if logits is not None and network.training:
            finest_target = target[0] if isinstance(target, (tuple, list)) else target
            boundary = self.boundary_loss(logits, finest_target)
        elif network.training:
            raise RuntimeError("training boundary logits missing; forward must precede the auxiliary loss")
        # Eval-mode online validation uses segmentation loss only; checkpoint
        # selection still uses the original online Dice, not the auxiliary term.
        self.last_segmentation_loss = segmentation.detach()
        self.last_boundary_loss = boundary.detach()
        return segmentation + self.boundary_weight * boundary
