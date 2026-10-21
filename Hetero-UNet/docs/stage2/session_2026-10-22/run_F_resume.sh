#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export nnUNet_raw=/hy-tmp/Hetero-UNet/data/nnUNet_raw
export nnUNet_preprocessed=/hy-tmp/Hetero-UNet/data/nnUNet_preprocessed
export nnUNet_results=/hy-tmp/stage2_results
export nnUNet_compile=false WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0
export nnUNet_n_proc_DA=12
AUDIT=/hy-tmp/stage2_audit_2026-10-22
state() {
  python - "$1" <<'PY'
import datetime,json,sys
from pathlib import Path
p=Path('/hy-tmp/stage2_audit_2026-10-22/F_resume_queue.json')
d={'sequence':['F'],'seed':1234,'epochs_each':1000,'deferred_models':['C'],
   'resumed_from_epoch':500,'status':sys.argv[1],
   'current_model':'none' if sys.argv[1]=='completed' else 'F',
   'models':{'F':sys.argv[1]},'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)
print('F_RESUME_QUEUE',json.dumps(d),flush=True)
PY
}
trap 'state failed' ERR
cd /hy-tmp/Hetero-UNet-stage2/Hetero-UNet
python - <<'PY'
import json,torch
from pathlib import Path
a=Path('/hy-tmp/stage2_audit_2026-10-22')
assert json.loads((a/'migration_restore.json').read_text())['status']=='restored_verified'
checks=json.loads((a/'cuda_full_patch/cuda_check.json').read_text())['runs']
assert len(checks)==3 and all(x['status']=='passed' for x in checks)
p=Path('/hy-tmp/stage2_results/stage2_F_seed1234/Dataset701_AbdomenCT/nnUNetTrainerMambaUNetBlockAttnRes__nnUNetPlans__3d_fullres/fold_all')
assert not (p/'checkpoint_final.pth').exists()
c=torch.load(p/'checkpoint_latest.pth',map_location='cpu')
assert c['current_epoch']==500 and c['trainer_name']=='nnUNetTrainerMambaUNetBlockAttnRes'
assert c['optimizer_state'] and c['network_weights']
print('VERIFIED_RESUME_FROM_EPOCH',c['current_epoch'])
PY
state training
python -u scripts/run_stage2.py --model F --seed 1234 --continue \
  --results-root /hy-tmp/stage2_results --skip-final-validation
state completed
