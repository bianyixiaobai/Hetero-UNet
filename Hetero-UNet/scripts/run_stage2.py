#!/usr/bin/env python
"""Launch isolated, seeded stage-two nnU-Net experiments on one GPU.

All heavyweight imports are deferred until after nnUNet_results is configured.
The upstream run_training function is intentionally bypassed because it resets
every requested seed to 1234. --dry-run and --help need only the Python stdlib.
"""

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback
from uuid import uuid4


TRAINERS = {
    "F": "nnUNetTrainerMambaUNetBlockAttnRes",
    "C": "nnUNetTrainerMambaUNetBGBlockAttnRes",
    "F-aux": "nnUNetTrainerMambaUNetBlockAttnResBoundarySupervision",
}

FROZEN_CT_PATCH = (40, 224, 192)


def is_ct701_dataset(dataset):
    value = str(dataset)
    if value.isdigit():
        return int(value) == 701
    return value.startswith("Dataset701_")


def dataset_argument(value):
    if value.isdigit() and int(value) > 0:
        return value
    if re.fullmatch(r"Dataset\d{3}_.+", value):
        return value
    raise argparse.ArgumentTypeError("dataset must be a positive ID or DatasetXXX_Name")


def fold_argument(value):
    if value == "all":
        return value
    if value.isdigit() and 0 <= int(value) <= 4:
        return int(value)
    raise argparse.ArgumentTypeError("fold must be all or an integer from 0 to 4")


def seed_argument(value):
    try:
        seed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("seed must be an integer") from error
    if not 0 <= seed < 2 ** 32:
        raise argparse.ArgumentTypeError("seed must be in [0, 2**32 - 1]")
    return seed


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(TRAINERS), required=True)
    parser.add_argument("--dataset", type=dataset_argument, default="701")
    parser.add_argument("--configuration", default="3d_fullres")
    parser.add_argument("--fold", type=fold_argument, default="all")
    parser.add_argument("--development", action="store_true",
                        help="Explicitly mark a development run, allowing numeric folds "
                             "and separate development plans. Output gets a _dev_foldN tag "
                             "and must not be used as a frozen main-experiment result.")
    parser.add_argument("--plans", default="nnUNetPlans")
    parser.add_argument("--seed", type=seed_argument, default=1234)
    parser.add_argument("--results-root", default=None,
                        help="Parent results directory (defaults to nnUNet_results). "
                             "A model/seed/smoke-specific subdirectory is added.")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda",
                        help="One GPU by default; cpu is diagnostic-only and "
                             "does not make the real CUDA Mamba kernel CPU-compatible.")
    parser.add_argument("--smoke", action="store_true",
                        help="2 epochs, 2 training iterations and 1 epoch-validation "
                             "iteration per epoch; skip final full-volume validation.")
    parser.add_argument("--continue", dest="continue_training", action="store_true",
                        help="Resume an existing checkpoint in this exact experiment; "
                             "fail if none exists.")
    parser.add_argument("--skip-final-validation", action="store_true",
                        help="Skip the post-training full-volume validation export. "
                             "Epoch validation and best-checkpoint selection remain active. "
                             "With fold=all, this export uses training cases, not the external 50.")
    parser.add_argument("--use-compressed", action="store_true",
                        help="Read compressed preprocessed data instead of unpacking it.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the resolved launch settings without importing torch, "
                             "creating files or starting training.")
    return parser


def parse_args(argv=None, environ=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.fold != "all" and not args.development:
        parser.error("numeric folds require --development; formal experiments use --fold all")
    if is_ct701_dataset(args.dataset) and not args.development and args.configuration != "3d_fullres":
        parser.error("frozen CT701 experiments require --configuration 3d_fullres; "
                     "use --development for a separately labeled development run")
    environment = os.environ if environ is None else environ
    root = args.results_root or environment.get("nnUNet_results")
    if root is None or not str(root).strip():
        parser.error("provide --results-root or set nnUNet_results")
    args.results_root = str(Path(os.path.expandvars(root)).expanduser().resolve())
    args.effective_results_root = str(
        Path(args.results_root) / (
            f"stage2_{args.model}_seed{args.seed}"
            + (f"_dev_fold{args.fold}" if args.development else "")
            + ("_smoke" if args.smoke else "")
        )
    )
    args.trainer = TRAINERS[args.model]
    args.final_validation = not (args.smoke or args.skip_final_validation)
    return args


def validate_frozen_ct_protocol(args, trainer):
    """Check resolved physical plans before constructing/running the network.

    Formal CT701 runs, including full-patch smoke tests, must match the archived
    baseline footprint. Development runs are explicitly labeled and isolated;
    MR/other datasets keep their own plans instead of inheriting the CT patch.
    """
    if args.development:
        return {"scope": "development", "frozen_ct_enforced": False,
                "eligible_for_frozen_CT701_comparison": False}
    if not is_ct701_dataset(args.dataset):
        return {"scope": "dataset_specific", "frozen_ct_enforced": False,
                "eligible_for_frozen_CT701_comparison": False}
    violations = []
    if args.configuration != "3d_fullres":
        violations.append(f"configuration={args.configuration!r}; expected '3d_fullres'")
    if args.fold != "all":
        violations.append(f"fold={args.fold!r}; expected 'all'")
    if trainer.batch_size != 2:
        violations.append(f"physical batch={trainer.batch_size}; expected 2")
    patch = tuple(trainer.configuration_manager.patch_size)
    if patch != FROZEN_CT_PATCH:
        violations.append(f"patch={list(patch)}; expected {list(FROZEN_CT_PATCH)}")
    if violations:
        raise ValueError(
            "Frozen CT701 protocol mismatch: " + "; ".join(violations)
            + ". Restore the archived CT plans; development runs require --development "
              "and must remain separate from the main comparison."
        )
    return {"scope": "frozen_CT701", "frozen_ct_enforced": True,
            "eligible_for_frozen_CT701_comparison": not args.smoke,
            "required_physical_batch_size": 2, "required_patch_size": list(FROZEN_CT_PATCH)}


def sha256_file(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def experiment_fingerprints(args, trainer, source_root):
    """Fingerprint named source files and descriptors, never credentials/images."""
    code = {"scripts/run_stage2.py": sha256_file(Path(__file__).resolve())}
    names = (
        "nets/MambaUNet.py", "nets/MambaUNetAttnRes.py",
        "nets/block_attention_residuals.py", "nets/boundary_prior.py",
        "training/loss/boundary.py",
        "training/loss/segmentation_boundary.py",
        "training/lr_scheduler/polylr.py",
        "training/nnUNetTrainer/nnUNetTrainerMambaUNet.py",
        "training/nnUNetTrainer/nnUNetTrainerMambaUNetAttnRes.py",
        "inference/predict_from_raw_data.py",
    )
    for name in names:
        file = source_root / "nnunetv2" / name
        code["umamba/nnunetv2/" + name] = sha256_file(file)
    descriptors = {}
    base = Path(trainer.preprocessed_dataset_folder_base)
    for name in (args.plans + ".json", "dataset.json", "splits_final.json"):
        file = base / name
        descriptors[name] = sha256_file(file) if file.is_file() else None
    return {"source_sha256": code, "descriptor_sha256": descriptors,
            "scope": "named code and dataset descriptors; excludes image-array content"}


def checkpoint_for_resume(output_folder):
    """Mirror upstream resume precedence, but never silently start from scratch."""
    for name in ("checkpoint_final.pth", "checkpoint_latest.pth", "checkpoint_best.pth"):
        candidate = Path(output_folder) / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"--continue requested, but no final/latest/best checkpoint exists in {output_folder}"
    )


def git_metadata(repository):
    result = {"commit": None, "dirty": None, "status_porcelain": None}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repository,
            text=True, capture_output=True, check=True, timeout=15,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=repository,
            text=True, capture_output=True, check=True, timeout=15,
        ).stdout
        result.update(commit=commit, dirty=bool(status.strip()), status_porcelain=status)
    except (OSError, subprocess.SubprocessError) as error:
        result["error"] = str(error)
    return result


class TeeStream:
    def __init__(self, terminal, logfile):
        self.terminal = terminal
        self.logfile = logfile

    def write(self, value):
        self.terminal.write(value)
        self.logfile.write(value)
        self.logfile.flush()
        return len(value)

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

    def isatty(self):
        return False


def write_manifest(manifest, history_path, latest_path):
    value = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    history_path.write_text(value, encoding="utf-8")
    latest_path.write_text(value, encoding="utf-8")


def run_experiment(args):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("This runner requires one process and one GPU; do not use torchrun/DDP")
    # nnunetv2.paths captures environment values on import. Configure it first.
    os.environ["nnUNet_results"] = args.effective_results_root
    os.environ["nnUNet_compile"] = "false"
    os.environ["WANDB_MODE"] = "disabled"
    source_root = Path(__file__).resolve().parents[1] / "umamba"
    sys.path.insert(0, str(source_root))

    import random
    import numpy as np
    import torch
    import wandb
    from nnunetv2.run.run_training import get_trainer_from_args, maybe_load_checkpoint

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; stage-two training needs the CUDA Mamba environment")
    if args.device == "cpu":
        print("CPU mode is diagnostic-only; the real Mamba implementation may reject CPU tensors.")
    else:
        torch.set_num_threads(1)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    # Seed before constructing the Trainer/network. Kernel and worker behavior
    # can still be nondeterministic; these are recorded seeds, not a bitwise promise.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    trainer = get_trainer_from_args(
        args.dataset, args.configuration, args.fold, args.trainer,
        args.plans, args.use_compressed, device=torch.device(args.device),
    )
    frozen_protocol = validate_frozen_ct_protocol(args, trainer)
    folder = Path(trainer.output_folder)
    folder.mkdir(parents=True, exist_ok=True)
    if args.continue_training:
        resume_checkpoint = checkpoint_for_resume(folder)
    else:
        resume_checkpoint = None
        if any((folder / name).is_file() for name in
               ("checkpoint_final.pth", "checkpoint_latest.pth", "checkpoint_best.pth")):
            raise FileExistsError(
                f"Checkpoints already exist in {folder}; use --continue to resume, "
                "or choose a different results root. Existing results will not be overwritten."
            )
    if args.smoke:
        trainer.num_epochs = 2
        trainer.num_iterations_per_epoch = 2
        trainer.num_val_iterations_per_epoch = 1

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_id = f"{stamp}_{uuid4().hex[:8]}"
    history_folder = folder / "stage2_run_manifests"
    history_folder.mkdir(exist_ok=True)
    history_path = history_folder / f"{run_id}.json"
    latest_path = folder / "stage2_run.json"
    console_path = folder / f"stage2_console_{run_id}.log"
    manifest = {
        "run_id": run_id, "status": "starting", "started_utc": stamp,
        "arguments": vars(args), "argv": sys.argv,
        "git": git_metadata(Path(__file__).resolve().parents[2]),
        "fingerprints": experiment_fingerprints(args, trainer, source_root),
        "environment": {key: os.environ.get(key) for key in
                        ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results",
                         "CUDA_VISIBLE_DEVICES", "nnUNet_compile", "WANDB_MODE")},
        "runtime": {"python": sys.version, "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "cudnn_benchmark": torch.backends.cudnn.benchmark,
                    "cudnn_deterministic": torch.backends.cudnn.deterministic},
        "protocol": {
            **frozen_protocol,
            "trainer": args.trainer, "seed": args.seed,
            "num_epochs": trainer.num_epochs,
            "train_iterations_per_epoch": trainer.num_iterations_per_epoch,
            "validation_iterations_per_epoch": trainer.num_val_iterations_per_epoch,
            "batch_size": trainer.batch_size,
            "patch_size": [int(value) for value in trainer.configuration_manager.patch_size],
            "initial_lr": trainer.initial_lr,
            "mha_batch_first": False, "physical_batch_must_match_across_models": True,
            "checkpoint_selection_for_external_evaluation": "checkpoint_best.pth",
            "full_volume_validation_requested": args.final_validation,
            "epoch_validation_remains_enabled": True,
            "external_50_evaluation": "Separate inference/DSC/NSD step; not run by this launcher",
        },
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        "resume_rng_note": "Upstream checkpoints do not restore Python/NumPy worker RNG streams; "
                           "resume is not claimed to be bitwise identical to uninterrupted training",
        "trainer_output_folder": str(folder), "console_log": str(console_path),
    }
    write_manifest(manifest, history_path, latest_path)
    wandb_run = None
    with console_path.open("a", encoding="utf-8", buffering=1) as console:
        with redirect_stdout(TeeStream(sys.stdout, console)), redirect_stderr(TeeStream(sys.stderr, console)):
            try:
                print(json.dumps(manifest, ensure_ascii=False, indent=2))
                # The inherited Trainer calls wandb.log; disabled init satisfies
                # its lifecycle without contacting any external logging service.
                wandb_run = wandb.init(mode="disabled", project="hetero-unet-stage2",
                                       config=vars(args))
                if args.continue_training:
                    maybe_load_checkpoint(trainer, True, False)
                manifest["status"] = "training"
                write_manifest(manifest, history_path, latest_path)
                trainer.run_training()
                if args.final_validation:
                    best = folder / "checkpoint_best.pth"
                    if not best.is_file():
                        raise FileNotFoundError(f"Best checkpoint missing after training: {best}")
                    trainer.load_checkpoint(str(best))
                    manifest["status"] = "full_volume_validation"
                    write_manifest(manifest, history_path, latest_path)
                    trainer.perform_actual_validation(save_probabilities=False)
                else:
                    print("Skipped post-training full-volume export; epoch validation was retained. "
                          "The external 50-case evaluation is a separate step.")
                manifest["status"] = "completed"
            except BaseException as error:
                manifest["status"] = "failed"
                manifest["error"] = f"{type(error).__name__}: {error}"
                traceback.print_exc()
                raise
            finally:
                manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
                write_manifest(manifest, history_path, latest_path)
                if wandb_run is not None:
                    wandb_run.finish()


def main(argv=None):
    args = parse_args(argv)
    if args.dry_run:
        print(json.dumps(vars(args), ensure_ascii=False, indent=2))
        return 0
    run_experiment(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
