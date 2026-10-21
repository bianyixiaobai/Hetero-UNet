#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export nnUNet_raw=/hy-tmp/Hetero-UNet/data/nnUNet_raw
export nnUNet_preprocessed=/hy-tmp/Hetero-UNet/data/nnUNet_preprocessed
export nnUNet_results=/hy-tmp/stage2_results
export nnUNet_compile=false WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0
export nnUNet_n_proc_DA=12
AUDIT=/hy-tmp/stage2_audit_2026-10-21
cd /hy-tmp/Hetero-UNet-stage2/Hetero-UNet
state() {
  python - "$1" "${2:-none}" <<'PY'
from pathlib import Path
import datetime,json,sys
p=Path('/hy-tmp/stage2_audit_2026-10-21/main_queue.json')
d=json.loads(p.read_text()) if p.exists() else {'sequence':['F','C'],'seed':1234,'epochs_each':1000,'models':{}}
d.update(status=sys.argv[1],current_model=sys.argv[2],updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
if sys.argv[2] != 'none': d['models'][sys.argv[2]]=sys.argv[1]
p.write_text(json.dumps(d,indent=2)+'\n')
print('QUEUE_STATE',d,flush=True)
PY
}
model=none
trap 'state failed "$model"' ERR
python - <<'PY'
from pathlib import Path
import json
a=Path('/hy-tmp/stage2_audit_2026-10-21')
rows=json.loads((a/'prediction_smokes.json').read_text())
assert len(rows)==3 and all(row['status']=='passed' for row in rows)
for name in ['F','C','F-aux']:
 root=Path('/hy-tmp/stage2_results')/f'stage2_{name}_seed1234_smoke'
 files=list(root.glob('Dataset701_*/*/fold_all/stage2_run.json'))
 assert len(files)==1 and json.loads(files[0].read_text())['status']=='completed'
assert json.loads((a/'preprocess_complete.json').read_text())['training_cases']==50
PY
for model in F C; do
  state training "$model"
  python -u scripts/run_stage2.py --model "$model" --seed 1234 --results-root /hy-tmp/stage2_results --skip-final-validation
  state trained "$model"
  mkdir -p /hy-tmp/stage2_completed_archives
  archive="/hy-tmp/stage2_completed_archives/${model}_seed1234_completed.tar"
  tar -cf "$archive" -C "/hy-tmp/stage2_results/stage2_${model}_seed1234" Dataset701_AbdomenCT
  sha256sum "$archive" > "${archive}.sha256"
  if oss cp "$archive" "oss://backup/stage2_20261021/${model}_seed1234_completed.tar"; then
    state backed_up "$model"
  else
    printf '\nOSS_BACKUP_FAILED: retain local archive %s and continue the next model\n' "$archive"
    state trained_backup_pending "$model"
  fi
done
state completed none
