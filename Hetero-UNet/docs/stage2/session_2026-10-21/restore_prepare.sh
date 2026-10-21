#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export nnUNet_raw=/hy-tmp/Hetero-UNet/data/nnUNet_raw
export nnUNet_preprocessed=/hy-tmp/Hetero-UNet/data/nnUNet_preprocessed
export nnUNet_results=/hy-tmp/stage2_results
export nnUNet_compile=false
export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=0
while tmux has-session -t stage2_restore_20261021 2>/dev/null; do sleep 5; done
grep -q DOWNLOADS_COMPLETE /hy-tmp/stage2_audit_2026-10-21/download.log
printf '\nRESTORE_START %s\n' "$(date -u +%FT%TZ)"
python -u /hy-tmp/stage2_audit_2026-10-21/restore_validate.py
printf '\nPREPROCESS_START %s\n' "$(date -u +%FT%TZ)"
nnUNetv2_preprocess -d 701 -c 3d_fullres -np 4
printf '\nPREPROCESS_COMPLETE %s\n' "$(date -u +%FT%TZ)"
