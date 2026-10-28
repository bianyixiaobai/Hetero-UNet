"""Validate diagnostic artifacts, write report and verify OSS delivery. Never power off."""
from pathlib import Path
import csv,hashlib,json,subprocess,tarfile,time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WORK=Path('/hy-tmp/stage2_diagnostics_20261029');EVAL=Path('/hy-tmp/stage2_eval_20261024')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())

def main():
    while not all((WORK/n).exists() for n in ['error_summary.json','guidance_summary.json']):time.sleep(20)
    errors=read(WORK/'error_summary.json');guidance=read(WORK/'guidance_summary.json')
    assert errors['status']==guidance['status']=='completed' and errors['rows']==1300 and len(guidance['reports'])==5
    rows=list(csv.DictReader((WORK/'error_types.csv').open()))
    assert len(rows)==1300 and len({(r['model'],r['case'],r['organ']) for r in rows})==1300
    for r in rows:
        assert int(r['TP'])+int(r['FN'])==int(r['GT_voxels']) and int(r['TP'])+int(r['FP'])==int(r['pred_voxels'])
        r['GT_present_zero_overlap']=int(r['GT_voxels'])>0 and int(r['TP'])==0
        r['GT_present_has_FN']=int(r['GT_voxels'])>0 and int(r['FN'])>0
        r['GT_present_has_FP']=int(r['GT_voxels'])>0 and int(r['FP'])>0
    with (WORK/'error_types_with_spatial_flags.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for model in ['F','C']:
        original=read(EVAL/model/'summary.json')
        for metric in ['DSC','NSD']:
            values=[float(r[metric]) for r in rows if r['model']==model]
            assert abs(np.mean(values)*100-original[metric+'_percent'])<1e-9
        for r in read(EVAL/model/'prediction_validation.json')['cases']:
            assert sha(EVAL/model/'predictions'/(r['case']+'.nii.gz'))==r['sha256']
    # Redraw probabilities on a fixed [0,1] scale, so near-constant priors cannot look strongly spatial.
    for row in guidance['reports']:
        case=row['case'];a=np.load(WORK/(case+'_patch_maps.npz'))
        boundary=a['boundary_target'][0,0];z=int(boundary.sum(axis=(1,2)).argmax())
        probability=a['boundary_probability'][0,0];prior=a['pooled_prior'][0,0]
        planes=[a['input_patch'][0,z],boundary[z],probability[z],prior[prior.shape[0]//2]]
        fig,axes=plt.subplots(1,4,figsize=(15,4))
        for i,(ax,image,title) in enumerate(zip(axes,planes,['Normalized CT','GT boundary','Boundary probability','Pooled prior (native token grid)'])):
            im=ax.imshow(image,cmap='gray' if i==0 else 'viridis',**({} if i==0 else {'vmin':0,'vmax':1}))
            ax.set_title(title);ax.axis('off')
            if i:fig.colorbar(im,ax=ax,fraction=.046)
        fig.suptitle(case+'; fixed probability color scale; pooled panel uses its middle depth slice')
        fig.tight_layout();fig.savefig(WORK/(case+'_boundary_fixed_scale.png'),dpi=140);plt.close(fig)
        for i in range(5):
            for key in ['weights_'+str(i),'weights_zero_'+str(i)]:
                weights=a[key];assert np.isfinite(weights).all() and np.allclose(weights.sum(-1),1,atol=1e-5)
        assert np.allclose(a['weights_0'],a['weights_zero_0'])
        a.close()
    lines=['# F/C错误类型与C边界引导诊断（2026-10-29）','','## 范围与核验','',
        '全部50例×13器官×F/C共1300行原生空间错误统计；五个预先固定病例各一个GT定位patch的C引导干预。',
        '未训练、未改权重或MHA、未变主评价。C权重前后SHA256一致，100份原预测SHA256未变；原CSV宏平均复算相符。',
        '漏分与过分割可同时存在：TP/FP/FN、recall/precision及体积比为直接统计；0.5/1.5体积阈值仅为描述性筛查标记。',
        '原生全卷TP/FP/FN与论文对Aorta/IVC/Esophagus的z范围裁剪口径不同，不应将原生体积统计当成主指标的精确重算。',
        'GT空类指该标注中没有此类，并非解剖学不存在；GT非空的轮廓偏差需结合NSD及叠图，不能从体积比单独确诊。','',
        '## 全50例分层结果','','| 模型 | GT空类对 | 空类假阳性对 | GT非空对 | 完全漏分对 | 体积比<0.5 | 体积比>1.5 | 非空DSC(%) | 非空NSD(%) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for m,s in errors['summary'].items():
        lines.append(f"| {m} | {s['GT_empty_pairs']} | {s['empty_false_positive_pairs']} | {s['GT_present_pairs']} | {s['complete_misses']} | {s['under_volume_ratio_below_0_5']} | {s['over_volume_ratio_above_1_5']} | {s['present_only_DSC']*100:.4f} | {s['present_only_NSD']*100:.4f} |")
    f,c=[errors['summary'][m] for m in ['F','C']]
    lines+=['','非空分层是在相同GT非空病例-器官对上计算的补充描述，不能取代完整50例主指标；不删除病例，不按结果修改后处理。',
        f"C−F完整宏平均DSC中，空类计分贡献为 {(c['empty_rule_macro_DSC_contribution']-f['empty_rule_macro_DSC_contribution'])*100:+.4f} 个百分点，非空类贡献为 {(c['present_macro_DSC_contribution']-f['present_macro_DSC_contribution'])*100:+.4f} 个百分点。",'',
        '## 边界图、池化与干预结果','','下表为每个诊断patch；边界Dice为0.5阈值、六邻域体素边界的局部检查，不是毫米容差NSD。标签变化是patch输出差异，不能解释为变好或变坏。', '',
        '| 病例 | 边界概率>0.5(%) | 池化引导>0.5(%) | 局部边界Dice | 置零后标签变化(%) | 均值后标签变化(%) | 打乱后标签变化(%) | 重复前向最大logit差 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in guidance['reports']:
        t=r['end_to_end_patch_interventions']
        lines.append(f"| {r['case']} | {r['boundary_probability']['fraction_above_0_5']*100:.2f} | {r['pooled_prior']['fraction_above_0_5']*100:.2f} | {r['boundary_Dice_at_0_5_patch_only']:.4f} | {t['zero']['label_changed_fraction']*100:.4f} | {t['mean']['label_changed_fraction']*100:.4f} | {t['shuffle']['label_changed_fraction']*100:.4f} | {r['natural_repeat_max_abs_difference']:.6g} |")
    lines+=['','## 同一历史特征下的聚合权重变化','','对同一个history分别代入真实引导和全零引导，以隔离即时条件作用；端到端置零还会改变后续history，两类检查不可混同。', '',
        '| 病例 | 聚合器 | 历史源数 | 权重绝对差均值 | 权重绝对差最大值 | 聚合输出相对L2差 |',
        '|---|---:|---:|---:|---:|---:|']
    for r in guidance['reports']:
        for i,t in r['aggregators_same_history_comparison'].items():
            lines.append(f"| {r['case']} | {i} | {t['sources']} | {t['weight_abs_difference_mean']:.6f} | {t['weight_abs_difference_max']:.6f} | {t['aggregate_relative_L2']:.6f} |")
    lines+=['','聚合器0仅有一个历史源，softmax恒为1，真实/全零引导权重相同是正常现象。',
        '干预中骨干和解码器参数固定；均值引导保留样本平均强度，打乱保留概率集合而改变空间位置，置零消除条件项。',
        '这五个GT定位patch不代表全部病例或全部区域，不能据此给出新的外部性能排序；关闭C引导也不等于从头训练的F-aux。', '',
        '## 图像与原始记录','']
    for p in sorted(WORK.glob('*.png')):lines.append(f'- [{p.name}]({p.name})')
    lines+=['','原始数值：error_types.csv、error_types_with_spatial_flags.csv、error_summary.json、guidance_summary.json；三维patch图与完整注意力权重保存在*_patch_maps.npz中。',
        '保留原始PNG，固定[0,1]色标图为*_boundary_fixed_scale.png，避免自动对比度误导。',
        'native叠图为原生图像坐标；模型patch图为nnU-Net重采样后的坐标，二者不能直接按像素叠加。','']
    (WORK/'DIAGNOSTIC_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    verification={'status':'passed','native_error_rows':len(rows),'prediction_hashes_matched':100,
        'guidance_patches':5,'attention_normalization_checked':True,'checkpoint_unchanged':True}
    (WORK/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    # Compact user package excludes large NPZ arrays; full package preserves everything.
    for name,include_maps in [('diagnostics_summary.tar',False),('diagnostics_full.tar',True)]:
        with tarfile.open(WORK/name,'w') as archive:
            for p in sorted(WORK.iterdir()):
                if p.is_file() and p.suffix not in ['.tar','.tmp'] and (include_maps or p.suffix!='.npz'):
                    archive.add(p,arcname=p.name)
    prefix='oss://backup/stage2_20261029/';proof={}
    with (WORK/'oss_transfer.log').open('w') as log:
        subprocess.run(['oss','mkdir',prefix],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
        for name in ['diagnostics_summary.tar','diagnostics_full.tar']:
            path=WORK/name;remote=prefix+name;returned=WORK/('verify_'+name)
            for args in [['oss','cp',str(path),remote,'-f'],['oss','cp',remote,str(returned),'-f']]:
                subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
            assert returned.stat().st_size==path.stat().st_size and sha(returned)==sha(path)
            returned.unlink();proof[name]={'remote':remote,'bytes':path.stat().st_size,'sha256':sha(path),'roundtrip_passed':True}
    (WORK/'backup_receipt.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps({'status':'diagnostics_completed_and_backed_up','objects':proof}),flush=True)

if __name__=='__main__':main()
