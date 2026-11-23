"""C with a single-scale, training-only adaptation of pixel-anchor contrast."""

import json

import numpy as np
import torch

from nnunetv2.training.loss.organ_contrast import organ_contrast
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMambaUNetAttnRes import nnUNetTrainerMambaUNetBGBlockAttnRes


def organ_step_coverage(counts, labels):
    """Count once per optimizer step regardless of the number of sites sampled."""
    steps = len(counts)
    present_steps = (counts[:, :, 0] > 0).sum(axis=0)
    active_steps = (counts[:, :, 2] > 0).sum(axis=0)
    hard_steps = (counts[:, :, 4] > 0).sum(axis=0)
    organs = {}
    for name, label in labels.items():
        if not 1 <= label <= counts.shape[1]:
            continue
        i = label - 1
        present, active, hard = int(present_steps[i]), int(active_steps[i]), int(hard_steps[i])
        organs[name] = dict(gt_steps=present, active_steps=active, hard_used_steps=hard,
                            gt_fraction=present / steps, active_fraction=active / steps,
                            hard_fraction_of_active=hard / active if active else None)
    return dict(total_steps=steps, organs=organs)


class _TrainingLoss(torch.nn.Module):
    def __init__(self, base, ignore_label, weight):
        super().__init__()
        self.base = base
        self.ignore_label = ignore_label
        self.weight = weight
        self.feature = None

    @property
    def last_boundary_loss(self):
        return self.base.last_boundary_loss

    @property
    def last_segmentation_loss(self):
        return self.base.last_segmentation_loss

    def capture(self, module, inputs, output):
        self.feature = output

    def forward(self, output, target):
        base = self.base(output, target)
        if not self.base._network_ref().training:
            return base
        if self.feature is None:
            raise RuntimeError("Middle encoder feature missing in contrast training step")
        raw, counts = organ_contrast(self.feature, output[0], target[0], self.ignore_label)
        self.feature = None
        self.raw = raw.detach()
        self.counts = counts
        return base + self.weight * raw


class nnUNetTrainerMambaUNetCContextrast(nnUNetTrainerMambaUNetBGBlockAttnRes):
    contrast_weight = 0.1
    feature_stage = 2

    def _build_loss(self):
        strides = self.configuration_manager.pool_op_kernel_sizes
        if tuple(np.prod(strides[:self.feature_stage + 1], axis=0)) != (2, 4, 4):
            raise ValueError("C contrast expects the 3D encoder grid at stride (2, 4, 4)")
        return _TrainingLoss(super()._build_loss(), self.label_manager.ignore_label, self.contrast_weight)

    def train_step(self, batch):
        stage = self.network.encoder.stages[self.feature_stage]
        with stage.register_forward_hook(self.loss.capture):
            try:
                result = super().train_step(batch)
            finally:
                self.loss.feature = None
        result["contrast_raw"] = self.loss.raw.cpu().numpy()
        result["organ_counts"] = self.loss.counts.cpu().numpy()
        return result

    def on_train_epoch_end(self, train_outputs):
        super().on_train_epoch_end(train_outputs)
        counts = np.stack([item["organ_counts"] for item in train_outputs])
        coverage = organ_step_coverage(counts, self.dataset_json["labels"])
        self.print_to_log_file("Organ contrast step coverage:", json.dumps(
            dict(epoch=self.current_epoch, **coverage), allow_nan=False))
        self.print_to_log_file("Training contrast weight/raw mean:", self.contrast_weight,
                               float(np.mean([item["contrast_raw"] for item in train_outputs])),
                               "active steps per organ:", (counts[:, :, 2] > 0).sum(0).tolist(),
                               "totals per organ [GT, correct, pos_used, hard, hard_used, neg_used]:",
                               counts.sum(0).tolist())
