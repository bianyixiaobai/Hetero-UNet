"""Describe all F/C errors without changing the frozen evaluation protocol."""
from pathlib import Path
import csv, hashlib, json, time
import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion

WORK=Path('/hy-tmp/stage2_diagnostics_20261029')
EVAL=Path('/hy-tmp/stage2_eval_20261024')
ORGANS=['Liver','RK','Spleen','Pancreas','Aorta','IVC','RAG','LAG','Gallbladder','Esophagus','Stomach','Duodenum','LK']

def main():
    WORK.mkdir(exist_ok=True)
    protocol=json.loads((EVAL/'protocol.json').read_text())
    metrics={}
    for model in ['F','C']:
        metrics[model]={}
        for metric in ['DSC','NSD']:
            with (EVAL/model/(model+'_'+metric+'.csv')).open() as f:
                metrics[model][metric]={r['Name']:r for r in csv.DictReader(f)}
    rows=[]
    selections={'FLARETs_0031':[8,9],'FLARETs_0032':[9,13,11],'FLARETs_0048':[9,12],'FLARETs_0001':[13],'FLARETs_0010':[4]}
    for index,case in enumerate(protocol['cases']):
        name=case['id']+'.nii.gz';gt_path=Path(case['gt'])
        assert hashlib.sha256(gt_path.read_bytes()).hexdigest()==case['gt_sha256']
        reference=nib.load(gt_path);gt=np.asanyarray(reference.dataobj)
        counts=np.bincount(gt.ravel().astype(np.int64),minlength=14)
        segs={}
        for model in ['F','C']:
            image=nib.load(EVAL/model/'predictions'/name)
            assert image.shape==reference.shape and np.allclose(image.affine,reference.affine,atol=1e-4)
            seg=np.asanyarray(image.dataobj);segs[model]=seg
            pred_count=np.bincount(seg.ravel().astype(np.int64),minlength=14)
            intersection=np.bincount(gt[seg==gt].ravel().astype(np.int64),minlength=14)
            for label,organ in enumerate(ORGANS,1):
                g,p,tp=int(counts[label]),int(pred_count[label]),int(intersection[label]);fp=p-tp;fn=g-tp
                category=('empty_correct' if p==0 else 'empty_false_positive') if g==0 else ('complete_miss' if p==0 else 'present_with_prediction')
                ratio=p/g if g else None
                rows.append(dict(case=case['id'],model=model,organ=organ,label=label,GT_voxels=g,pred_voxels=p,TP=tp,FP=fp,FN=fn,
                                 recall=tp/g if g else None,precision=tp/p if p else None,volume_ratio=ratio,
                                 FN_fraction=fn/g if g else None,FP_fraction=fp/p if p else None,category=category,
                                 severe_under_volume=bool(g and ratio<0.5),severe_over_volume=bool(g and ratio>1.5),
                                 DSC=float(metrics[model]['DSC'][name][organ+'_DSC']),NSD=float(metrics[model]['NSD'][name][organ+'_NSD'])))
        if case['id'] in selections:
            ct=np.asanyarray(nib.load(case['image']).dataobj)
            for label in selections[case['id']]:
                masks=[gt==label,segs['F']==label,segs['C']==label]
                z=int(np.argmax(sum(m.sum(axis=(0,1)) for m in masks)))
                fig,axes=plt.subplots(1,3,figsize=(12,4))
                for ax,mask,title in zip(axes,masks,['GT','F','C']):
                    ax.imshow(ct[:,:,z].T,cmap='gray',vmin=-150,vmax=250,origin='lower')
                    view=mask[:,:,z].T;edge=view & ~binary_erosion(view)
                    overlay=np.zeros((*view.shape,4));overlay[edge]=[1,0.2,0.1,1]
                    ax.imshow(overlay,origin='lower');ax.set_title(title+' voxels='+str(int(mask.sum())));ax.axis('off')
                fig.suptitle(case['id']+' '+ORGANS[label-1]+' native z='+str(z));fig.tight_layout()
                fig.savefig(WORK/(case['id']+'_'+ORGANS[label-1]+'_overlay.png'),dpi=140);plt.close(fig)
            del ct
        print('ERROR_AUDIT',index+1,'/50',case['id'],flush=True)
        del gt,segs
    with (WORK/'error_types.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={}
    for model in ['F','C']:
        selected=[r for r in rows if r['model']==model];empty=[r for r in selected if r['GT_voxels']==0];present=[r for r in selected if r['GT_voxels']>0]
        summary[model]={'pairs':len(selected),'GT_empty_pairs':len(empty),'empty_false_positive_pairs':sum(r['category']=='empty_false_positive' for r in empty),
                        'empty_FP_voxels':sum(r['FP'] for r in empty),'GT_present_pairs':len(present),'complete_misses':sum(r['category']=='complete_miss' for r in present),
                        'under_volume_ratio_below_0_5':sum(r['severe_under_volume'] for r in present),'over_volume_ratio_above_1_5':sum(r['severe_over_volume'] for r in present),
                        'present_only_DSC':float(np.mean([r['DSC'] for r in present])),'present_only_NSD':float(np.mean([r['NSD'] for r in present])),
                        'empty_rule_macro_DSC_contribution':sum(r['DSC'] for r in empty)/650,
                        'present_macro_DSC_contribution':sum(r['DSC'] for r in present)/650}
    (WORK/'error_summary.json').write_text(json.dumps({'status':'completed','summary':summary,'rows':len(rows),
        'notes':['GT-empty does not imply anatomically absent','Volume thresholds are descriptive flags, not tuning or a new scoring rule',
                 'NSD retains original tolerances and z-cropping; voxel errors use the whole native grid',
                 'Under/over volume is not a definitive spatial error diagnosis; inspect TP/FP/FN and overlays']},indent=2)+'\n')

if __name__=='__main__':main()
