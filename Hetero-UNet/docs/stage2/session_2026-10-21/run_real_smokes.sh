#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export nnUNet_raw=/hy-tmp/Hetero-UNet/data/nnUNet_raw
export nnUNet_preprocessed=/hy-tmp/Hetero-UNet/data/nnUNet_preprocessed
export nnUNet_results=/hy-tmp/stage2_results
export nnUNet_compile=false WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0
export nnUNet_n_proc_DA=12
cd /hy-tmp/Hetero-UNet-stage2/Hetero-UNet
for model in F C F-aux; do
  printf '\nREAL_SMOKE_START model=%s UTC=%s\n' "$model" "$(date -u +%FT%TZ)"
  python -u scripts/run_stage2.py --model "$model" --seed 1234 --results-root /hy-tmp/stage2_results --smoke
  printf '\nREAL_SMOKE_COMPLETE model=%s UTC=%s\n' "$model" "$(date -u +%FT%TZ)"
done
printf '\nALL_REAL_SMOKES_COMPLETE UTC=%s\n' "$(date -u +%FT%TZ)"
