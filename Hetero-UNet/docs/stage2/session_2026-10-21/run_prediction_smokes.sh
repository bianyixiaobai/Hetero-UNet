#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export nnUNet_raw=/hy-tmp/Hetero-UNet/data/nnUNet_raw
export nnUNet_preprocessed=/hy-tmp/Hetero-UNet/data/nnUNet_preprocessed
export nnUNet_compile=false WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0
AUDIT=/hy-tmp/stage2_audit_2026-10-21
mkdir -p "$AUDIT/smoke_input"
if [ ! -e "$AUDIT/smoke_input/FLARE22_Tr_0001_0000.nii.gz" ]; then
  ln -s "$nnUNet_raw/Dataset701_AbdomenCT/imagesTr/FLARE22_Tr_0001_0000.nii.gz" "$AUDIT/smoke_input/FLARE22_Tr_0001_0000.nii.gz"
fi
for model in F C F-aux; do
  case "$model" in
    F) trainer=nnUNetTrainerMambaUNetBlockAttnRes ;;
    C) trainer=nnUNetTrainerMambaUNetBGBlockAttnRes ;;
    F-aux) trainer=nnUNetTrainerMambaUNetBlockAttnResBoundarySupervision ;;
  esac
  export nnUNet_results="/hy-tmp/stage2_results/stage2_${model}_seed1234_smoke"
  nnUNetv2_predict -i "$AUDIT/smoke_input" -o "$AUDIT/smoke_predictions/$model" -d 701 -c 3d_fullres -f all -tr "$trainer" -chk checkpoint_best.pth --disable_tta -npp 1 -nps 1
done
python - <<'PY'
from pathlib import Path
import nibabel as nib
import numpy as np
import json
base=Path('/hy-tmp/stage2_audit_2026-10-21')
image=nib.load(base/'smoke_input/FLARE22_Tr_0001_0000.nii.gz')
rows=[]
for name in ['F','C','F-aux']:
 p=base/'smoke_predictions'/name/'FLARE22_Tr_0001.nii.gz'
 pred=nib.load(p)
 assert pred.shape==image.shape and np.allclose(pred.affine,image.affine,atol=1e-4)
 values=np.unique(np.asanyarray(pred.dataobj))
 assert np.isfinite(values).all() and set(values.tolist()) <= set(range(14))
 rows.append({'model':name,'status':'passed','shape':list(pred.shape),'labels':values.tolist()})
(base/'prediction_smokes.json').write_text(json.dumps(rows,indent=2)+'\n')
print('PREDICTION_SMOKES_COMPLETE',rows,flush=True)
PY
