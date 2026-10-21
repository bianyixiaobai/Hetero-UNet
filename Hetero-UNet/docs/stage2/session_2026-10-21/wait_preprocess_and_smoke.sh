#!/usr/bin/env bash
set -euo pipefail
while tmux has-session -t stage2_preprocess_20261021 2>/dev/null; do sleep 5; done
python - <<'PY'
from pathlib import Path
import hashlib,json
base=Path('/hy-tmp/Hetero-UNet/data/nnUNet_preprocessed/Dataset701_AbdomenCT')
audit=Path('/hy-tmp/stage2_audit_2026-10-21')
if 'Traceback (most recent call last)' in (audit/'preprocess.log').read_text():
    raise RuntimeError('Preprocessing failed; preserve logs and do not start training')
plans=json.loads((base/'nnUNetPlans.json').read_text())
folder=base/plans['configurations']['3d_fullres']['data_identifier']
expected=set(json.loads((audit/'data_restore.json').read_text())['case_ids']['Tr'])
assert {p.stem for p in folder.glob('*.npz')} == expected
assert {p.stem for p in folder.glob('*.pkl')} == expected
value={'status':'completed','training_cases':50,'external_cases_used':0,
       'folder':str(folder),'plans_sha256':hashlib.sha256((base/'nnUNetPlans.json').read_bytes()).hexdigest()}
(audit/'preprocess_complete.json').write_text(json.dumps(value,indent=2)+'\n')
print('PREPROCESS_VALIDATED',value,flush=True)
PY
bash /hy-tmp/stage2_audit_2026-10-21/run_real_smokes.sh
