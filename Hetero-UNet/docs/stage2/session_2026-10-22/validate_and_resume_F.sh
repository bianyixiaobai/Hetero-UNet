#!/usr/bin/env bash
set -euo pipefail
AUDIT=/hy-tmp/stage2_audit_2026-10-22
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
while true; do
  code=0
  python - <<'PY' || code=$?
from pathlib import Path
import json
p=Path('/hy-tmp/stage2_audit_2026-10-22/migration_restore.json')
status=json.loads(p.read_text())['status'] if p.exists() else 'waiting'
if status=='failed_requires_review':raise SystemExit(2)
raise SystemExit(0 if status=='restored_verified' else 1)
PY
  if [ "$code" = 0 ]; then break; fi
  if [ "$code" = 2 ]; then exit 2; fi
  sleep 5
done
RECOVERED=/hy-tmp/stage2_migration_20261022/recovered_audit/audit_previous_instance
for name in migration_pause.json frozen_experiment_protocol.json baseline_plans.json dataset.json; do
  cp "$RECOVERED/$name" "$AUDIT/$name"
done
cd /hy-tmp/Hetero-UNet-stage2/Hetero-UNet
python scripts/check_stage2_cuda.py \
  --plans /hy-tmp/Hetero-UNet/data/nnUNet_preprocessed/Dataset701_AbdomenCT/nnUNetPlans.json \
  --dataset-json /hy-tmp/Hetero-UNet/data/nnUNet_preprocessed/Dataset701_AbdomenCT/dataset.json \
  --output "$AUDIT/cuda_full_patch" > "$AUDIT/cuda_full_patch.log" 2>&1
tmux new-session -d -s stage2_F_resume_20261022 \
  'exec bash /hy-tmp/stage2_audit_2026-10-22/run_F_resume.sh >> /hy-tmp/stage2_audit_2026-10-22/F_resume_queue.log 2>&1'
while [ ! -f "$AUDIT/F_resume_queue.json" ]; do sleep 1; done
tmux new-session -d -s stage2_finalize_F_resume_20261022 \
  'exec python -u /hy-tmp/stage2_audit_2026-10-22/finalize_stage2.py --config /hy-tmp/stage2_audit_2026-10-22/autoshutdown_F_resume_config.json --watch --arm-shutdown >> /hy-tmp/stage2_audit_2026-10-22/finalize_F_resume.log 2>&1'
echo RESUMED_F_AND_FINALIZER_STARTED
