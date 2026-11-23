"""Single-scale pixel-to-organ contrast on the native 3D encoder features."""

import torch
from torch.nn import functional as F


def error_surface(labels, prediction, valid):
    """Six-neighbour inner shell of each misclassified GT region, within the patch."""
    errors = torch.where(valid & (prediction != labels), labels, -1)
    edge = torch.zeros_like(valid)
    for axis in (1, 2, 3):
        left = [slice(None)] * 4
        right = left.copy()
        left[axis], right[axis] = slice(None, -1), slice(1, None)
        left, right = tuple(left), tuple(right)
        change = (errors[left] != errors[right]) & valid[left] & valid[right]
        edge[left] |= change
        edge[right] |= change
    return edge & (errors >= 0)


def _sample(indices, limit):
    if indices.numel() <= limit:
        return indices
    order = torch.randperm(indices.numel(), device=indices.device)[:limit]
    return indices[order]


def organ_contrast(features, logits, target, ignore_label=None, cap=128, temperature=0.1):
    """Return a class-balanced loss and counts in ascending foreground label order.

    The six counts per row are GT sites, correct sites, sampled positives,
    eligible hard negatives, sampled hard negatives and sampled total negatives.
    """
    grid = features.shape[2:]
    with torch.no_grad():
        labels = F.interpolate(target.float(), size=grid, mode="nearest")[:, 0].long()
        pred = F.interpolate(logits.detach(), size=grid, mode="nearest").argmax(1)
        valid = (labels >= 0) & (labels < logits.shape[1])
        if ignore_label is not None:
            valid &= labels != ignore_label
        shell = error_surface(labels, pred, valid).flatten()
        labels, pred, valid = labels.flatten(), pred.flatten(), valid.flatten()
    vectors = features.flatten(2).transpose(1, 2)
    sites = vectors.shape[1]

    def gather(indices):
        return F.normalize(vectors[indices // sites, indices % sites].float(), dim=1)

    rows, correct_per_organ, losses = [], [], []
    with torch.autocast(device_type=features.device.type, enabled=False):
        for organ in range(1, logits.shape[1]):
            belongs = valid & (labels == organ)
            others = valid & (labels != organ)
            positives = belongs.nonzero().flatten()
            correct = (belongs & (pred == organ)).sum()
            hard_mask = others & shell & (pred == organ)
            hard = hard_mask.nonzero().flatten()
            chosen = _sample(positives, cap)
            negatives_used = hard_used = 0
            if chosen.numel():
                priority = _sample(hard, cap)
                remaining = cap - priority.numel()
                extra = _sample((others & ~hard_mask).nonzero().flatten(), remaining) if remaining else priority[:0]
                negatives = torch.cat((priority, extra))
                hard_used, negatives_used = priority.numel(), negatives.numel()
                if negatives_used:
                    pos = gather(chosen)
                    anchor = F.normalize(pos.mean(0), dim=0)
                    pos_score = (pos @ anchor) / temperature
                    neg_score = (gather(negatives) @ anchor) / temperature
                    losses.append(F.softplus(neg_score.logsumexp(0) - pos_score).mean())
            correct_per_organ.append(correct)
            rows.append([positives.numel(), 0, chosen.numel() if negatives_used else 0,
                         hard.numel(), hard_used, negatives_used])
    counts = torch.tensor(rows, device=features.device)
    counts[:, 1] = torch.stack(correct_per_organ)
    loss = torch.stack(losses).mean() if losses else features[0, 0, 0, 0, 0].float() * 0
    return loss, counts
