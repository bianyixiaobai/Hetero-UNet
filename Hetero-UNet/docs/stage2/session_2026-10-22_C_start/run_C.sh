#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export nnUNet_raw=/hy-tmp/Hetero-UNet/data/nnUNet_raw
export nnUNet_preprocessed=/hy-tmp/Hetero-UNet/data/nnUNet_preprocessed
export nnUNet_results=/hy-tmp/stage2_results
export nnUNet_compile=false WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0
export nnUNet_n_proc_DA=12
AUDIT=/hy-tmp/stage2_C_audit_2026-10-22
state() {
  python - "$1" <<'PY'
from pathlib import Path
import datetime,json,sys
p=Path('/hy-tmp/stage2_C_audit_2026-10-22/C_queue.json')
status=sys.argv[1]
d={'sequence':['C'],'seed':1234,'epochs_each':1000,'status':status,
   'current_model':'none' if status=='completed' else 'C','models':{'C':status},
   'start_from_scratch':True,'deferred_models':['F-aux'],
   'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
print('C_QUEUE',json.dumps(d),flush=True)
PY
}
trap 'state failed' ERR
cd "$AUDIT"
python - <<'PY'
from pathlib import Path
import json
a=Path('/hy-tmp/stage2_C_audit_2026-10-22')
assert json.loads((a/'preflight.json').read_text())['status']=='passed'
checks=json.loads((a/'cuda_check/cuda_check.json').read_text())['runs']
assert len(checks)==3 and all(r['status']=='passed' for r in checks)
assert not list(Path('/hy-tmp/stage2_results/stage2_C_seed1234').rglob('*.pth'))
PY
state training
python -u /hy-tmp/Hetero-UNet-stage2/Hetero-UNet/scripts/run_stage2.py \
  --model C --seed 1234 --results-root /hy-tmp/stage2_results --skip-final-validation
state completed
