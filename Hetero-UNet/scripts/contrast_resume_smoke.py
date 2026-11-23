"""Instrumentation for the two-process CUDA smoke in run_stage2.py."""

import json
import math
import os
from pathlib import Path
from time import perf_counter

import torch


class CheckpointReady(Exception):
    """Exit the first process at a completed checkpoint boundary."""


def instrument_amp_smoke(trainer, resuming, manifest):
    manifest["smoke_pid"] = os.getpid()
    scaler_receipt = Path(trainer.output_folder) / "smoke_checkpoint_scaler.json"
    expected_epoch = int(resuming)
    if trainer.current_epoch != expected_epoch:
        raise RuntimeError(f"Expected epoch {expected_epoch}, loaded {trainer.current_epoch}")
    if trainer.grad_scaler is None or not trainer.grad_scaler.is_enabled():
        raise RuntimeError("CUDA AMP scaler must be enabled")
    if resuming and not any("momentum_buffer" in state for state in trainer.optimizer.state.values()):
        raise RuntimeError("SGD momentum missing after native resume")
    if resuming and trainer.grad_scaler.state_dict() != json.loads(scaler_receipt.read_text(encoding="utf-8")):
        raise RuntimeError("AMP scaler state differs from the first process's saved state")
    trainer.save_every = 1
    manifest["amp_steps"] = []
    train_step = trainer.train_step
    epoch_end = trainer.on_epoch_end

    def checked_step(batch):
        param = next(trainer.network.encoder.stages[trainer.feature_stage].parameters())
        before = param.detach().clone()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = perf_counter()
        result = train_step(batch)
        torch.cuda.synchronize()
        seconds = perf_counter() - start
        peak = torch.cuda.max_memory_allocated()
        norms = [p.grad.detach().norm() for p in trainer.network.parameters() if p.grad is not None]
        grad_norm = torch.stack(norms).norm().item()
        change = (param.detach() - before).abs().max().item()
        weighted = trainer.contrast_weight * float(result["contrast_raw"])
        total = float(result["loss"])
        step = dict(epoch=trainer.current_epoch, seconds=seconds, peak_allocated_bytes=peak,
                    base_loss=total - weighted, weighted_contrast=weighted, loss=total,
                    clipped_grad_norm=grad_norm, max_parameter_change=change,
                    amp_scale=trainer.grad_scaler.get_scale())
        manifest["amp_steps"].append(step)
        trainer.print_to_log_file("CUDA AMP smoke:", step)
        if not all(math.isfinite(v) for v in step.values()) or grad_norm <= 0 or change <= 0:
            raise RuntimeError("AMP smoke requires finite losses and gradients and an actual parameter update")
        return result

    def checked_epoch_end():
        epoch_end()
        if not resuming:
            latest = Path(trainer.output_folder) / "checkpoint_latest.pth"
            if not latest.is_file() or trainer.current_epoch != 1 or not manifest["amp_steps"]:
                raise RuntimeError("First AMP epoch did not produce the expected checkpoint")
            scaler_receipt.write_text(json.dumps(trainer.grad_scaler.state_dict()), encoding="utf-8")
            raise CheckpointReady

    trainer.train_step = checked_step
    trainer.on_epoch_end = checked_epoch_end
