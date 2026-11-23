"""CPU engineering check on two training CTs and the archived C checkpoint."""

import argparse
import gc
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tarfile
import types
from uuid import uuid4

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class CpuMambaRecurrence(nn.Module):
    """Diagnostic-only float32 recurrence for Mamba v1, initialized from archived weights."""

    def __init__(self, d_model, d_state, d_conv, expand):
        super().__init__()
        inner = d_model * expand
        self.rank = math.ceil(d_model / 16)
        self.state_size = d_state
        self.in_proj = nn.Linear(d_model, 2 * inner, bias=False)
        self.conv1d = nn.Conv1d(inner, inner, d_conv, groups=inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(inner, self.rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.rank, inner)
        self.A_log = nn.Parameter(torch.zeros(inner, d_state))
        self.D = nn.Parameter(torch.ones(inner))
        self.out_proj = nn.Linear(inner, d_model, bias=False)

    def forward(self, tokens):
        x, gate = self.in_proj(tokens).chunk(2, dim=-1)
        x = F.silu(self.conv1d(x.transpose(1, 2))[:, :, :tokens.shape[1]].transpose(1, 2))
        dt, b, c = self.x_proj(x).split([self.rank, self.state_size, self.state_size], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        a = -self.A_log.exp()
        state = x.new_zeros(x.shape[0], x.shape[2], self.state_size)
        outputs = []
        for t in range(tokens.shape[1]):
            dt_t = dt[:, t, :, None]
            state = (dt_t * a).exp() * state + dt_t * x[:, t, :, None] * b[:, t, None, :]
            outputs.append((state * c[:, t, None, :]).sum(-1) + self.D * x[:, t])
        return self.out_proj(torch.stack(outputs, dim=1) * F.silu(gate))


def real_batch(root, manager, config, dataset):
    from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
    sizes = [(16, 64, 64), (32, 128, 128)]
    crops = {size: [] for size in sizes}
    raw = root / "data/Dataset701_AbdomenCT"
    for key, organ in [("FLARE22_Tr_0001", 8), ("FLARE22_Tr_0002", 9)]:
        image, seg, _ = DefaultPreprocessor(verbose=False).run_case(
            [str(raw / "imagesTr" / (key + "_0000.nii.gz"))], str(raw / "labelsTr" / (key + ".nii.gz")),
            manager, config, dataset)
        coords = np.argwhere(seg[0] == organ)
        assert len(coords), (key, organ)
        for size in sizes:
            shape = np.array(size)
            start = np.clip(coords.mean(0).astype(int) - shape // 2, 0, np.array(seg.shape[1:]) - shape)
            region = (slice(None), *(slice(a, a + n) for a, n in zip(start, shape)))
            metadata = dict(case=key, center_class=organ, start=start.tolist(), shape=list(size))
            crops[size].append((image[region].copy(), np.maximum(seg[region], 0), metadata))
    batches = []
    for size in sizes:
        images, targets, metadata = zip(*crops[size])
        batches.append((torch.from_numpy(np.stack(images)), torch.from_numpy(np.stack(targets)).long(), metadata))
    return batches


def loss_edges():
    from nnunetv2.training.loss.organ_contrast import organ_contrast, error_surface
    t = torch.zeros(2, 1, 3, 3, 3, dtype=torch.long)
    t[0, 0, 1, 1, 1] = 1
    f = torch.randn(2, 8, 3, 3, 3, requires_grad=True)
    logits = F.one_hot(t[:, 0], 3).movedim(-1, 1).float()
    loss, counts = organ_contrast(f, logits, t)
    assert counts[0].tolist() == [1, 1, 1, 0, 0, 53] and not counts[1].any()
    assert torch.isfinite(torch.autograd.grad(loss, f)[0]).all() and loss > 0
    logits[1, 1, 1, 1, 1] = 2
    _, counts = organ_contrast(f, logits, t)
    assert counts[0, 4] == 1
    valid = torch.ones_like(t[:, 0], dtype=torch.bool)
    shell = error_surface(t[:, 0], logits.argmax(1), valid)
    assert shell.sum() == 1 and shell[1, 1, 1, 1]
    for value in (0, 1, 99, -1):
        degenerate, _ = organ_contrast(f, logits, torch.full_like(t, value), ignore_label=99)
        assert degenerate == 0 and torch.autograd.grad(degenerate, f)[0].count_nonzero() == 0
    return ["single positive in one patch", "absent class", "perfect prediction with ordinary negatives",
            "six-neighbour hard error", "all background", "one class without negatives", "ignored or padded"]


def nominal_loss_check(root):
    """Exercise the auxiliary loss at the actual planned sizes independently of the backbone."""
    sys.path.insert(0, str(root / "Hetero-UNet/umamba"))
    from nnunetv2.training.loss.organ_contrast import organ_contrast
    torch.manual_seed(1234)
    torch.set_num_threads(3)
    f = torch.randn(2, 128, 20, 56, 48, dtype=torch.bfloat16, requires_grad=True)
    logits = torch.randn(2, 14, 40, 224, 192, dtype=torch.bfloat16)
    target = torch.randint(0, 14, (2, 1, 40, 224, 192), dtype=torch.uint8)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss, counts = organ_contrast(f, logits, target)
    loss.backward()
    assert loss.dtype == torch.float32 and torch.isfinite(f.grad).all() and f.grad.norm() > 0
    assert (counts[:, 2] == 128).all() and (counts[:, 4] == 128).all()
    return dict(scope="synthetic auxiliary tensors only", patch=list(target.shape), features=list(f.shape),
                logits=list(logits.shape), feature_dtype=str(f.dtype), loss_dtype=str(loss.dtype),
                loss=loss.item(), grad_norm=f.grad.float().norm().item(), counts=counts.tolist())


def run(root, scratch):
    for name in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        os.environ[name] = str(scratch / name)
    sys.path.insert(0, str(root / "Hetero-UNet/umamba"))
    mamba = types.ModuleType("mamba_ssm")
    mamba.Mamba = CpuMambaRecurrence
    sys.modules["mamba_ssm"] = mamba
    tracking = types.ModuleType("wandb")
    tracking.log = lambda *args, **kwargs: None
    sys.modules["wandb"] = tracking
    from nnunetv2.training.loss.organ_contrast import organ_contrast
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMambaUNetCContextrast import nnUNetTrainerMambaUNetCContextrast
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMambaUNetAttnRes import nnUNetTrainerMambaUNetBGBlockAttnRes
    torch.set_num_threads(3)
    torch.manual_seed(1234)
    edge_checks = loss_edges()
    configs = root / "output/module2_spatial_20261116"
    plans = json.loads((configs / "C_plans.json").read_text(encoding="utf-8-sig"))
    dataset = json.loads((configs / "C_dataset.json").read_text(encoding="utf-8-sig"))
    trainer = nnUNetTrainerMambaUNetCContextrast(plans, "3d_fullres", "all", dataset, device=torch.device("cpu"))
    trainer.initialize()
    print("Preparing two real training CT crops", flush=True)
    (image, target, patches), (wide_image, wide_target, wide_patches) = real_batch(
        root, trainer.plans_manager, trainer.configuration_manager, dataset)
    archive = root / "output/stage2_C_completed_2026-10-24/C_seed1234_final.tar"
    local_checkpoint = scratch / "baseline.pth"
    with tarfile.open(archive) as tar:
        name, = [name for name in tar.getnames() if name.endswith("/checkpoint_best.pth")]
        with tar.extractfile(name) as source, local_checkpoint.open("wb") as dest:
            shutil.copyfileobj(source, dest)
    weights = torch.load(local_checkpoint, mmap=True, weights_only=False, map_location="cpu")["network_weights"]
    trainer.network.load_state_dict(weights, strict=True)
    base = nnUNetTrainerMambaUNetBGBlockAttnRes.build_network_architecture(
        trainer.plans_manager, dataset, trainer.configuration_manager, 1, False)
    base.load_state_dict(weights, strict=True)
    assert set(weights) == set(base.state_dict()) == set(trainer.network.state_dict())
    del weights
    gc.collect()
    trainer.network.eval()
    trainer.set_deep_supervision_enabled(False)
    with torch.no_grad():
        expected = base.eval()(image[:1])
        actual = trainer.network(image[:1])
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    equal_diff = float((actual - expected).abs().max())
    del base, actual, expected
    gc.collect()
    trainer.set_deep_supervision_enabled(True)
    trainer.network.train()
    stage = trainer.network.encoder.stages[trainer.feature_stage]
    with torch.no_grad(), stage.register_forward_hook(trainer.loss.capture):
        wide_logits = trainer.network(wide_image)[0]
    wide_feature = trainer.loss.feature.detach().requires_grad_()
    wide_loss, wide_counts = organ_contrast(wide_feature, wide_logits, wide_target)
    wide_grad = torch.autograd.grad(trainer.contrast_weight * wide_loss, wide_feature)[0]
    assert torch.isfinite(wide_grad).all() and wide_grad.norm() > 0
    wide_grad_norm = wide_grad.norm().item()
    # Compare label definitions descriptively on the same wider crops. A 50/50
    # tie can support two labels here; the prototype training path uses nearest.
    occupancy = []
    for idx in range(1, wide_logits.shape[1]):
        nearest = wide_target[:, :, ::2, ::4, ::4] == idx
        majority = F.avg_pool3d((wide_target == idx).float(), (2, 4, 4)) >= 0.5
        occupancy.append([int(nearest.sum()), int(majority.sum()), int((nearest & majority).sum())])
    trainer.loss.feature = None
    trainer.network.consume_boundary_logits()
    del wide_logits, wide_image, wide_target, wide_feature, wide_loss, wide_grad
    print("Hard negatives used on wider crops:", wide_counts[:, 4].tolist(), flush=True)
    with stage.register_forward_hook(trainer.loss.capture):
        outputs = trainer.network(image)
    targets = [F.interpolate(target.float(), size=o.shape[2:], mode="nearest").long() for o in outputs]
    feature = trainer.loss.feature
    base_loss = trainer.loss.base(outputs, targets)
    raw_loss, counts = organ_contrast(feature, outputs[0], targets[0])
    base_grad = torch.autograd.grad(base_loss, feature, retain_graph=True)[0].flatten()
    aux_grad = torch.autograd.grad(trainer.contrast_weight * raw_loss, feature)[0].flatten()
    assert torch.isfinite(base_grad).all() and torch.isfinite(aux_grad).all()
    print("Gradient norms (C, contrast), per-class counts:", base_grad.norm().item(),
          aux_grad.norm().item(), counts.tolist(), flush=True)
    assert base_grad.norm() > 0, "No C gradient on shared features"
    assert aux_grad.norm() > 0, "No contrast gradient on shared features"
    assert (counts[:, 4] + wide_counts[:, 4]).sum() > 0, "No hard samples used on either crop size"
    report = dict(torch=torch.__version__, seed=1234, patches=patches, edge_checks=edge_checks,
                  checkpoint=name, strict_keys=len(trainer.network.state_dict()), same_weight_max_diff=equal_diff,
                  feature_shape=list(feature.shape), base_loss=base_loss.item(), raw_contrast=raw_loss.item(),
                  weighted_contrast=trainer.contrast_weight * raw_loss.item(), base_grad_norm=base_grad.norm().item(),
                  weighted_contrast_grad_norm=aux_grad.norm().item(), grad_cosine=F.cosine_similarity(base_grad, aux_grad, dim=0).item(),
                  per_organ_counts=counts.tolist(), wider_patches=wide_patches, wider_counts=wide_counts.tolist(),
                  wider_weighted_feature_grad=wide_grad_norm, wider_gt_nearest_half_overlap=occupancy,
                  deep_supervision_shapes=[list(o.shape) for o in outputs])
    trainer.loss.feature = None
    del feature, outputs, base_grad, aux_grad, base_loss, raw_loss
    gc.collect()
    print("Checking actual Trainer optimizer and validation entry points", flush=True)
    batch = {"data": image, "target": targets}
    param = next(stage.parameters())
    original = param.detach().clone()
    step = trainer.train_step(batch)
    assert np.isfinite(step["loss"]) and np.isfinite(step["contrast_raw"])
    trainer.on_train_epoch_end([step])
    marker = "Organ contrast step coverage:"
    records = Path(trainer.log_file).read_text(encoding="utf-8").splitlines()
    coverage = json.loads(next(line.split(marker, 1)[1] for line in reversed(records) if marker in line))
    assert coverage["total_steps"] == 1 and len(coverage["organs"]) == 13
    for organ, entry in coverage["organs"].items():
        row = step["organ_counts"][dataset["labels"][organ] - 1]
        assert entry["gt_steps"] == int(row[0] > 0)
        assert entry["active_steps"] == int(row[2] > 0)
        assert entry["hard_used_steps"] == int(row[4] > 0)
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMambaUNetCContextrast import organ_step_coverage
    cases = np.zeros((4, 2, 6), dtype=int)
    cases[:3, 0, 0] = 10
    cases[:2, 0, 2] = 3
    cases[0, 0, 4] = 5
    cases[3, 1, 3] = 8  # candidates for an absent organ, none used
    known = organ_step_coverage(cases, {"a": 1, "b": 2})["organs"]
    assert known["a"] == dict(gt_steps=3, active_steps=2, hard_used_steps=1,
                              gt_fraction=0.75, active_fraction=0.5, hard_fraction_of_active=0.5)
    assert known["b"]["hard_used_steps"] == 0 and known["b"]["hard_fraction_of_active"] is None
    report["train_step_coverage"] = coverage
    delta = (param.detach() - original).abs().max().item()
    assert delta > 0 and trainer.loss.feature is None and not stage._forward_hooks
    trainer.network.eval()
    with torch.no_grad():
        validation = trainer.validation_step(batch)
        expected_val_loss = trainer.loss.base.segmentation_loss(trainer.network(image), targets)
    assert np.isclose(float(validation["loss"]), expected_val_loss.item(), atol=1e-7, rtol=0)
    assert np.isfinite(validation["loss"]) and len(validation["tp_hard"]) == 13
    trainer.on_validation_epoch_end([validation])
    trainer._best_ema = trainer.logger.my_fantastic_logging["ema_fg_dice"][-1]
    expected_ema = float(trainer._best_ema)
    trainer.set_deep_supervision_enabled(False)
    with torch.no_grad():
        after = trainer.network(image[:1])
    resume_file = scratch / "roundtrip.pth"
    trainer.save_checkpoint(str(resume_file))
    momentum = trainer.optimizer.state[param]["momentum_buffer"].detach().clone()
    del batch, param, stage, trainer
    gc.collect()
    restored = nnUNetTrainerMambaUNetCContextrast(plans, "3d_fullres", "all", dataset, device=torch.device("cpu"))
    restored.load_checkpoint(str(resume_file))
    assert restored._best_ema == expected_ema
    restored.network.eval()
    restored.set_deep_supervision_enabled(False)
    with torch.no_grad():
        replay = restored.network(image[:1])
    torch.testing.assert_close(after, replay, rtol=0, atol=0)
    restored_param = next(restored.network.encoder.stages[2].parameters())
    torch.testing.assert_close(momentum, restored.optimizer.state[restored_param]["momentum_buffer"], rtol=0, atol=0)
    restored.network.train()
    restored.set_deep_supervision_enabled(True)
    resumed_step = restored.train_step({"data": image, "target": targets})
    assert np.isfinite(resumed_step["loss"])
    report.update(train_loss=float(step["loss"]), validation_loss=float(validation["loss"]),
                  optimizer_max_weight_change=delta, checkpoint_max_diff=(after - replay).abs().max().item(),
                  momentum_restored=True, resumed_train_loss=float(resumed_step["loss"]))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--nominal-loss-only", action="store_true")
    args = parser.parse_args()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    scratch = args.report.parent.resolve() / ("contrast_check_" + uuid4().hex)
    scratch.mkdir()
    try:
        root = Path(__file__).resolve().parents[2]
        result = nominal_loss_check(root) if args.nominal_loss_only else run(root, scratch)
    finally:
        if scratch.resolve().parent == args.report.parent.resolve() and scratch.name.startswith("contrast_check_"):
            shutil.rmtree(scratch)
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("All local assertions passed:", args.report)
