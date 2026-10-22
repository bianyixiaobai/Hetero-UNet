#!/usr/bin/env bash
set -euo pipefail
AUDIT=/hy-tmp/stage2_C_audit_2026-10-22
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export nnUNet_raw=/hy-tmp/Hetero-UNet/data/nnUNet_raw
export nnUNet_preprocessed=/hy-tmp/Hetero-UNet/data/nnUNet_preprocessed
export nnUNet_results=/hy-tmp/stage2_results
export nnUNet_compile=false WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0
export nnUNet_n_proc_DA=12
cd "$AUDIT"
python -u "$AUDIT/prepare_stage2_C_20261022.py"
python -u /hy-tmp/Hetero-UNet-stage2/Hetero-UNet/scripts/check_stage2_cuda.py \
  --plans /hy-tmp/Hetero-UNet/data/nnUNet_preprocessed/Dataset701_AbdomenCT/nnUNetPlans.json \
  --dataset-json /hy-tmp/Hetero-UNet/data/nnUNet_preprocessed/Dataset701_AbdomenCT/dataset.json \
  --output "$AUDIT/cuda_check" > "$AUDIT/cuda_check.log" 2>&1
tmux new-session -d -s stage2_C_20261022 \
  'exec bash /hy-tmp/stage2_C_audit_2026-10-22/run_C.sh >> /hy-tmp/stage2_C_audit_2026-10-22/C_queue.log 2>&1'
while [ ! -f "$AUDIT/C_queue.json" ]; do sleep 1; done
tmux new-session -d -s stage2_finalize_C_20261022 \
  'exec python -u /hy-tmp/stage2_C_audit_2026-10-22/finalize_stage2.py --config /hy-tmp/stage2_C_audit_2026-10-22/autoshutdown_C_config.json --watch --arm-shutdown >> /hy-tmp/stage2_C_audit_2026-10-22/finalize_C.log 2>&1'
echo C_AND_FINALIZER_STARTED
