"""Rebuild the frozen four-model tables, scientific figures and stage-two report."""
from pathlib import Path
import csv, hashlib, json, math, statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'output/stage2_Faux_20261029/extracted'
OUT=ROOT/'output/stage2_summary_20261030'
FIG=ROOT/'Hetero-UNet/docs/stage2/figures/stage2_20261030'
REPORT=ROOT/'Hetero-UNet_阶段二实验总结与四组消融对比.md'
MODELS=['A','F','C','F-aux']
ORG=['Liver','RK','Spleen','Pancreas','Aorta','IVC','RAG','LAG','Gallbladder','Esophagus','Stomach','Duodenum','LK']
CN=['肝脏','右肾','脾脏','胰腺','主动脉','下腔静脉','右肾上腺','左肾上腺','胆囊','食管','胃','十二指肠','左肾']
PAIRS=[('F','A'),('C','A'),('F-aux','A'),('F-aux','F'),('C','F-aux'),('C','F')]
COLORS=['#67788a','#157f87','#d47135','#8158a3']
OUT.mkdir(exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
font=Path('C:/Windows/Fonts/msyh.ttc')
if font.exists():font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white','svg.fonttype':'none'})

def export(name,headers,rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(headers);w.writerows(rows)

def table(headers,rows):
    return ['| '+' | '.join(headers)+' |','|'+'|'.join(['---']+['---:']*(len(headers)-1))+'|']+['| '+' | '.join(str(v) for v in row)+' |' for row in rows]

def save(fig,name):
    fig.savefig(FIG/(name+'.png'),dpi=190,bbox_inches='tight')
    svg=FIG/(name+'.svg')
    fig.savefig(svg,bbox_inches='tight')
    plt.close(fig)
    content='\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n'
    temporary=svg.with_suffix('.svg.tmp')
    temporary.write_text(content,encoding='utf-8')
    temporary.replace(svg)

def embed(name,caption):
    return ['',f'![{caption}](Hetero-UNet/docs/stage2/figures/stage2_20261030/{name}.png)',f'*{caption}*','']

data={};sources=[];names=None
for metric in ['DSC','NSD']:
    data[metric]={}
    for m in MODELS:
        p=SRC/'baseline_A'/(metric+'.csv') if m=='A' else SRC/m/(m+'_'+metric+'.csv')
        with p.open(newline='') as f:
            reader=csv.DictReader(f);assert reader.fieldnames==['Name']+[o+'_'+metric for o in ORG];rows=list(reader)
        rows=sorted(rows,key=lambda r:r['Name']);ids=[r['Name'] for r in rows]
        assert len(ids)==len(set(ids))==50
        if names is None:names=ids
        assert names==ids
        a=np.array([[float(r[o+'_'+metric]) for o in ORG] for r in rows])
        assert a.shape==(50,13) and np.isfinite(a).all() and ((a>=0)&(a<=1)).all()
        data[metric][m]=a*100
        sources.append({'path':p.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
means={k:{m:float(a.mean()) for m,a in d.items()} for k,d in data.items()}
case={k:{m:a.mean(axis=1) for m,a in d.items()} for k,d in data.items()}
organ={k:{m:a.mean(axis=0) for m,a in d.items()} for k,d in data.items()}
ref=json.loads((ROOT/'output/stage2_Faux_20261029/independent_comparison.json').read_text())
assert all(abs(means[k][m]-ref[k]['macro_percent'][m])<1e-9 for k in means for m in MODELS)
gt={}
gtfile=ROOT/'output/stage2_diagnostics_20261029/error_types_with_spatial_flags.csv'
with gtfile.open(newline='') as f:
    for r in csv.DictReader(f):
        if r['model']=='F':gt[(r['case']+'.nii.gz',r['organ'])]=int(r['GT_voxels'])
empty=np.array([[gt[(n,o)]==0 for o in ORG] for n in names]);assert empty.sum()==19
strata={k:{m:{'empty':float(a[empty].mean()),'present':float(a[~empty].mean()),'empty_correct':int(np.count_nonzero(a[empty]==100))} for m,a in d.items()} for k,d in data.items()}
summary=[];paired=[];details=[]
for m in MODELS:
    summary.append([m,means['DSC'][m],means['NSD'][m],float(np.std(case['DSC'][m],ddof=1)),float(np.std(case['NSD'][m],ddof=1))])
for k in data:
    for l,r in PAIRS:
        v=case[k][l]-case[k][r];assert abs(v.mean()-(means[k][l]-means[k][r]))<1e-9
        paired.append([k,l+'−'+r,float(v.mean()),float(np.median(v)),int((v>1e-9).sum()),int((v< -1e-9).sum()),int((abs(v)<=1e-9).sum())])
for i,n in enumerate(names):
    for j,o in enumerate(ORG):
        for m in MODELS:details.append([n,m,o,CN[j],bool(empty[i,j]),data['DSC'][m][i,j]/100,data['NSD'][m][i,j]/100])
export('overall.csv',['Model','DSC_percent','NSD_percent','Case_DSC_SD_pp','Case_NSD_SD_pp'],summary)
export('paired_comparisons.csv',['Metric','Comparison','Mean_delta_pp','Median_case_delta_pp','Higher_cases','Lower_cases','Equal_cases'],paired)
export('all_2600_case_organ_model_records.csv',['Case','Model','Organ','Organ_CN','GT_empty','DSC_fraction','NSD_fraction'],details)
for k in data:
    export('organs_'+k+'.csv',['Organ','Organ_CN']+MODELS,[[o,CN[j]]+[organ[k][m][j] for m in MODELS] for j,o in enumerate(ORG)])
    export('cases_'+k+'.csv',['Case']+MODELS+['%s_minus_%s_pp'%(l,r) for l,r in PAIRS],[[n]+[case[k][m][i] for m in MODELS]+[case[k][l][i]-case[k][r][i] for l,r in PAIRS] for i,n in enumerate(names)])
export('GT_strata.csv',['Metric','Model','GT_empty_pairs','GT_empty_correct_pairs','GT_empty_mean_percent','GT_present_pairs','GT_present_mean_percent'],[[k,m,19,strata[k][m]['empty_correct'],strata[k][m]['empty'],631,strata[k][m]['present']] for k in data for m in MODELS])

# Scientific charts use explicit axes and common color scales; no seed error bars.
fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
for ax,k,xlim,paper in zip(axes,['DSC','NSD'],[(86,88),(90,92)],[86.7,90.5]):
    for i,m in enumerate(MODELS):
        ax.scatter(means[k][m],i,color=COLORS[i],s=95,zorder=3)
        ax.annotate(f'{means[k][m]:.4f}%',(means[k][m],i),xytext=(10,0),textcoords='offset points',va='center')
    ax.axvline(paper,color='#9a9a9a',ls='--',lw=1,label='论文参考（一位小数）')
    ax.set(yticks=range(4),yticklabels=MODELS,xlim=xlim,ylim=(3.5,-.5),xlabel=k+'（%）',title=k+'：50例×13器官等权宏平均');ax.grid(axis='x',alpha=.2);ax.legend(loc='lower right',fontsize=8)
fig.suptitle('四组首轮结果；横轴为局部范围，无跨种子误差条',fontsize=13)
save(fig,'01_overall')
fig,axes=plt.subplots(1,2,figsize=(10,8),layout='constrained')
for ax,k in zip(axes,['DSC','NSD']):
    arr=np.column_stack([organ[k][m] for m in MODELS]);im=ax.imshow(arr,cmap='YlGnBu',vmin=65,vmax=100,aspect='auto')
    ax.set(xticks=range(4),xticklabels=MODELS,yticks=range(13),yticklabels=CN,title=k+'（%）')
    for i in range(13):
        for j in range(4):ax.text(j,i,f'{arr[i,j]:.2f}',ha='center',va='center',color='white' if arr[i,j]>87 else '#172536',fontsize=9)
fig.colorbar(im,ax=axes,shrink=.75,label='绝对得分（%，两图同一色标）');save(fig,'02_organs_absolute')
delta={k:np.column_stack([organ[k][m]-organ[k]['A'] for m in MODELS[1:]]) for k in data}
limit=math.ceil(max(abs(a).max() for a in delta.values()))
fig,axes=plt.subplots(1,2,figsize=(10,8),layout='constrained')
for ax,k in zip(axes,['DSC','NSD']):
    arr=delta[k];im=ax.imshow(arr,cmap='RdBu',vmin=-limit,vmax=limit,aspect='auto')
    ax.set(xticks=range(3),xticklabels=['F−A','C−A','F-aux−A'],yticks=range(13),yticklabels=CN,title=k+'相对A（百分点）')
    for i in range(13):
        for j in range(3):ax.text(j,i,f'{arr[i,j]:+.2f}',ha='center',va='center',color='white' if abs(arr[i,j])>.65*limit else '#172536',fontsize=9)
fig.colorbar(im,ax=axes,shrink=.75,label='相对A的百分点差，两图同一色标');save(fig,'03_organs_delta')
fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
paired_limit=math.ceil(max(abs(case[k][l]-case[k][r]).max() for k in data for l,r in [('C','F'),('F-aux','F')]))
for row,(l,r) in enumerate([('C','F'),('F-aux','F')]):
    for col,k in enumerate(['DSC','NSD']):
        ax=axes[row,col];v=case[k][l]-case[k][r]
        ax.axhline(0,color='#444',lw=.8);ax.scatter(range(1,51),v,s=22,color='#d47135' if l=='C' else '#8158a3')
        ax.axhline(v.mean(),color='#157f87',ls='--',label=f'均值 {v.mean():+.4f} pp')
        ax.set(xlim=(0,51),ylim=(-paired_limit,paired_limit),xticks=[1,10,20,30,40,50],xlabel='病例编号 FLARETs_0001–0050',ylabel='病例均值差（百分点）',title=f'{k}：{l}−{r}');ax.legend(fontsize=8);ax.grid(alpha=.15)
save(fig,'04_paired_cases')
fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
contributions={}
for ax,k in zip(axes,['DSC','NSD']):
    rows=[]
    for l,r in [('F-aux','F'),('C','F')]:
        a=data[k][l]-data[k][r];ev=float(a[empty].sum()/650);pv=float(a[~empty].sum()/650)
        assert abs(ev+pv-(means[k][l]-means[k][r]))<1e-9;rows.append((ev,pv));contributions[k+' '+l+'−'+r]={'GT_empty_pp':ev,'GT_present_pp':pv}
    for i,(ev,pv) in enumerate(rows):
        ax.barh(i,ev,color='#dba05a',height=.45,label='GT空类贡献' if i==0 else None)
        ax.barh(i,pv,left=ev if ev*pv>=0 else 0,color='#447e94',height=.45,label='GT非空贡献' if i==0 else None)
        ax.text(ev+pv+(.03 if ev+pv>=0 else -.03),i,f'{ev+pv:+.4f}',va='center',ha='left' if ev+pv>=0 else 'right')
    ax.axvline(0,color='#444',lw=.8);ax.set(yticks=[0,1],yticklabels=['F-aux−F','C−F'],xlim=(-1.6,.45),xlabel='对完整宏平均差的贡献（百分点）',title=k+'：空类/非空类分解');ax.legend(loc='upper left',fontsize=8)
save(fig,'05_GT_contributions')

result={'means_percent':means,'paired_comparisons':paired,'GT_strata':strata,'GT_contributions':contributions,'sources':sources,'source_GT_mask':{'path':gtfile.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(gtfile.read_bytes()).hexdigest()},'dimensions':{'cases':50,'organs':13,'models':4,'observations':2600,'scores':5200},'seed_uncertainty':'not estimated; single run per configuration','numpy':np.__version__,'matplotlib':matplotlib.__version__,'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(OUT/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

L=['# 阶段二实验总结与A/F/C/F-aux四组消融对比','','报告日期：2026-10-31；实验数据截至2026-10-30。研究任务：模块一BG-Block-AttnRes实现与验证。数据：Dataset701_AbdomenCT，相同外部50例、13器官。','',
'**阶段二已完成实现、真实数据验收、F/C/F-aux各一轮完整训练与统一外部评价，形成了可追溯的首轮消融结果。当前最明确的现象是F、C高于已有A，F的DSC最高、C的NSD最高；C对F的附加收益尚不明确，F-aux本轮下降。稳定性与机制归因尚未完成，当前仍在阶段二2.5。**','','## 1. 如何阅读这次对比','','| 配置 | 深度聚合 | 边界辅助监督 | 边界引导注入 | 推理边界头 |','|---|---|---|---|---|','| A | 原基线 | 无 | 无 | 无 |','| F | 有 | 无 | 无 | 无 |','| C | 有 | 有，0.1 | 有 | 执行 |','| F-aux | 有，与F相同 | 有，与C相同 | 无 | 跳过 |','',
'F−A观察基础深度聚合；F-aux−F观察辅助监督；C−F-aux观察在辅助监督基础上的注入增量；C−F是监督与注入的组合差异。F-aux从头训练，不能用训练好的C在推理时关闭引导替代。所有组保留原MHA实现规则；这不意味着MHA权重或数值输出在各组相同。','','## 2. 评价口径与可比性','','原始8份CSV均保留，按病例ID和13器官列对齐，未删低分病例。主指标为650个病例–器官对等权平均；单位为%，相减的单位是百分点（pp）。原CSV已四舍五入至四位小数，因此这里是对已保存指标复算。','',
'F/C/F-aux均1000轮、seed1234、fold_all/3d_fullres、物理batch2、patch[40,224,192]、初始lr0.01，组大小3，边界头隐藏通道8。统一best选择规则，F及F-aux取第1000轮best，C取第994轮best；不按外部50例重选权重。无TTA、滑窗0.5、Gaussian、单patch前向，预处理/导出各1进程，NSD两进程，无新增后处理。','',
'A为实验室自行训练后迁移评估的复现基线；F从4090第500轮迁至V100续训，C/F-aux在V100从头训练。不能宣称硬件和增强轨迹完全一致；A能否计入匹配种子重复实验还需核对其记录。fold_all的在线验证复用训练病例，不是独立开发集，其Dice/损失不能代替外部评价或泛化验证。','',
'A导出历史为49例原导出、0050低内存补丁；F/C/F-aux统一使用独立runtime中的等价低内存补丁，保留已有验证范围。DSC/NSD沿用同一空类规则、NSD逐器官容差及Aorta/IVC/Esophagus的GT z范围裁剪。论文86.7%/90.5%仅作为原表参考，不构成同次受控实验。','','## 3. 总体数据对比','']
L+=table(['来源/组别','DSC %','NSD %'],[['论文Hetero-UNet参考','86.7','90.5']]+[[m,f'{means["DSC"][m]:.4f}',f'{means["NSD"][m]:.4f}'] for m in MODELS])
L+=embed('01_overall','图1：四组宏平均。点图使用明确的局部横轴；没有跨种子误差条，因为目前未完成重复种子实验。')
L+=table(['比较','DSC差 pp','NSD差 pp'],[[l+'−'+r,f'{means["DSC"][l]-means["DSC"][r]:+.4f}',f'{means["NSD"][l]-means["NSD"][r]:+.4f}'] for l,r in PAIRS])
L+=['','## 4. 全部13器官对比','']
for k in data:
    L+=['### '+k+'（%）','']+table(['器官']+MODELS,[[CN[i]+' '+o]+[f'{organ[k][m][i]:.4f}' for m in MODELS] for i,o in enumerate(ORG)])+['']
L+=embed('02_organs_absolute','图2：逐器官绝对得分，DSC与NSD使用同一65–100色标。')
L+=embed('03_organs_delta','图3：逐器官相对A差值。两图使用同一对称色标，正值为改善。')
L+=['### C相对F的器官差异','']
for k in data:
    v=organ[k]['C']-organ[k]['F'];order=np.argsort(v)
    L.append(k+'：'+str(int((v>0).sum()))+'个器官均值升、'+str(int((v<0).sum()))+'个降。改善最多：'+'、'.join(f'{CN[i]} {v[i]:+.4f} pp' for i in order[-3:][::-1])+'；下降最多：'+'、'.join(f'{CN[i]} {v[i]:+.4f} pp' for i in order[:3])+'。')
L+=['','这些是组均值层面的器官方向，不代表每个病例均改善。边界引导的作用不应只由总体均值判断，也不能由少数器官改善推断普遍收益。','','## 5. 全部50例配对分析','']
L+=table(['指标/比较','均值差 pp','中位数差 pp','升/降/平病例数'],[[k+' '+c,f'{mean:+.4f}',f'{med:+.4f}',f'{up}/{down}/{tie}'] for k,c,mean,med,up,down,tie in paired])
L+=embed('04_paired_cases','图4：每个点为同一病例13器官均值的配对差；正值表示左侧组更好。病例不是独立训练重复，不由该图推断跨种子稳定性。')
L+=['C−F的DSC为27例升/23例降，中位数差为+0.0373 pp，但均值为−0.0465 pp；NSD为28例升/22例降，中位数差+0.3577 pp。均值接近并不等于所有病例都没有变化，少数幅度较大的病例可以影响总体方向。F-aux−F的下降则出现在多数病例。','']
L+=['病例差反映效应不均匀。F-aux−F下降最多的五例为：'+ '、'.join(names[i].replace('.nii.gz','')+f' {(case["DSC"]["F-aux"]-case["DSC"]["F"])[i]:+.4f} pp' for i in np.argsort(case['DSC']['F-aux']-case['DSC']['F'])[:5])+'。这些病例用于诊断时应说明选择依据，不因低分而从主结果中删除。','','## 6. GT空类与非空类分层','']
L+=table(['组别','空类正确/错误对（共19）','非空DSC %（631对）','非空NSD %（631对）'],[[m,f'{strata["DSC"][m]["empty_correct"]}/{19-strata["DSC"][m]["empty_correct"]}',f'{strata["DSC"][m]["present"]:.4f}',f'{strata["NSD"][m]["present"]:.4f}'] for m in MODELS])
L+=['','F-aux比F/C多2个空类错误计分对，对完整宏平均的净贡献为−0.3077 pp；F-aux−F其余DSC下降约−0.9768 pp、NSD约−0.6541 pp来自GT非空对。F与C空类得分合计相同，因此C−F净差完全来自GT非空对。','',
'与A相比，F-aux在GT非空对上DSC略高约0.0207 pp、NSD高约0.1797 pp，但空类错误计分对更多，完整宏平均仍低于A。这个分解解释计分构成，不意味着F-aux已达到稳定改善。GT为空不等同器官解剖学不存在；分层仅作补充，不能替换完整50例主指标。']
L+=embed('05_GT_contributions','图5：完整宏平均差值的空类/非空类加权分解；不是把两个分层均值直接相加。')
L+=['## 7. 阶段二实验总结与消融结论','','| 任务 | 已取得证据 | 当前判断 |','|---|---|---|','| 2.1基础深度聚合 | 代码/梯度检查、真实数据验收及F完整训练评价 | 工程实现完成；F相对A有首轮提升，待重复确认 |','| 2.2边界头与引导 | C完整训练、边界图/池化引导/权重干预 | 分支参与计算，但是否带来稳定净增益仍未确认 |','| 2.3组装与运行 | 完整batch/patch CUDA/AMP、独立Trainer、权重加载和导出 | 运行链路完成，未改MHA |','| 2.4首轮统一评价 | A/F/C/F-aux同50例DSC/NSD、逐器官和逐病例 | 首轮数据齐全；不是多种子稳定性验收 |','| 2.5归因与确认 | F-aux从头训练，监督/注入控制对照已补齐 | 首轮消融完成；重复实验与机制解释仍待完成 |','',
'**可以表述的总体结果**：在当前冻结基座和本轮设置下，新增深度聚合的F及完整C均高于已有A；C对F总体表现为DSC略低、NSD略高。相同0.1辅助监督但无注入的F-aux低于F；C相对F-aux更好，提示本轮辅助监督与条件注入的组合行为需要分开研究。','',
'**尚不能表述的结论**：不能说边界监督必然有害、边界注入已证明稳定有效、模块一已全面胜出，或将C−F-aux直接当作可推广的因果效应。F-aux差于F不等于边界标签、损失或池化一定存在代码错误；同样，C与F接近也不证明引导完全无效。','',
'### 与已完成诊断怎样对应','','10月29日的五个C诊断patch中，重复前向一致；置零引导会改变深度权重及部分标签，说明引导路径在这些输入上生效。变化主要集中在最后聚合器。高分辨率边界概率>0.5约占1.82%–6.08%，最大池化后对应高引导token约占74.76%–93.81%，提示空间选择性可能减弱，但尚不是性能原因的证明。五例GT定位patch不是无偏测试集或F-aux训练替代。','',
'F/C全50例错误分层已完成，F-aux尚未做新的全卷逐体素错误分型。训练/验证loss不能直接跨F与含辅助损失的组比较；先拆分segmentation与boundary项，且记住fold_all在线验证复用训练病例。','','## 8. 后续建议：先补解释与稳定性，再决定改动','','当前建议是保留四组原始结果，先做低成本离线检查，再推进受控的内部开发与关键重复。服务器已执行关机，本报告不启动新训练。','',
'| 优先级 | 工作 | 具体产出/完成条件 |','|---|---|---|','| P0：离线复核 | 拆分F/C/F-aux训练曲线，核对学习率、segmentation/boundary量级和best选择；复核0009/0044/0048等退化例并包含改善对照例 | 统一曲线图、病例/器官错误清单；区分空类FP、GT非空错位/漏分、覆盖过大和轮廓偏差 |','| P1：内部开发诊断 | 固定训练集内部划分；检查边界目标与预测对齐、池化空间选择性、浅层共享特征上的分割/辅助梯度 | 只记录可测的现象，如梯度范数/夹角与引导空间变化，不由宏平均猜原因 |','| P2：关键重复 | 优先补F/C相同种子集合；若论文要主张监督/注入独立贡献，F-aux也纳入匹配重复；A按主计划补匹配基线 | 目标至少3个预先记录种子，报告跨种子均值±标准差；预算不足则明确为探索结果 |','| P3：必要的单因素改进 | 如内部诊断支持，再分别测试较小辅助权重或更保留空间差异的池化；每次只改变一个因素 | 在内部开发集决定新版本，冻结后统一训练相关对照；不把多处修正合算成某个模块收益 |','| P4：模块二与联合模型 | 模块一版本及结论边界记录清楚后，继续原多尺度自适应跳跃融合D，再做联合E | D对A独立验证，E对A/C/D分离联合效应；不把模块二收益倒算给模块一 |','',
'F/C差异较小，所以有限算力优先用于判断它是否可重复。辅助权重0.1及组大小3是首轮固定值，不是已验证最优；若做新超参试验，候选值和内部选择规则应在开跑前登记。现有A可否计入重复集合取决于种子和协议可追溯情况。','',
'外部50例已用于评价和错误诊断，后续应如实说明其使用历史。它不再适合充当反复选择新池化/辅助权重的开发集。条件允许时增加未参与开发的新测试数据或按预设流程交叉验证。保留MHA的决定不变；修正轴布局是独立补充协议，不是当前消融的前置步骤。','','## 9. 后续计划与阶段推进','',
'| 顺序 | 工作阶段 | 是否需要新训练 | 进入下一步的依据 |','|---|---|---|---|','| 1 | 完成离线日志和代表病例复核 | 不需要 | 有可追溯的现象清单，明确是否存在实际实现错误 |','| 2 | 固定内部开发划分、登记重复种子与资源预算 | 仅必要的短程/开发实验 | 实验版本、唯一改动、评价口径和选择规则已记录 |','| 3 | 关键组配对重复与必要的监督/注入确认 | 需要，按统一协议 | 判断F/C差异是否稳定；形成模块一可报告版本 |','| 4 | 推进阶段三模块二D及阶段四联合E | 后续独立训练 | 不改A–H定义；模块一负结果和局限一并保留 |','',
'当前阶段是“阶段二首轮实验完成，2.5结果确认未完全完成”。不返回阶段一重做已经核验的全部流程，也不要求在进入模块二前完成所有横向方法和MR实验。完整训练耗时应依据各版本GPU实测重新估计；本轮C约40.57小时、F-aux约39.15小时，不能把它们当作所有后续配置的固定工期。','','## 10. 数据表、图和证据位置','',
'主报告为本文件。可复用表格在`output/stage2_summary_20261030/`：`overall.csv`、`paired_comparisons.csv`、`organs_DSC.csv`、`organs_NSD.csv`、`cases_DSC.csv`、`cases_NSD.csv`、`GT_strata.csv`及`all_2600_case_organ_model_records.csv`。CSV百分数和fraction字段已在列名标注；病例间SD不是跨种子SD。','',
'五张图均在`Hetero-UNet/docs/stage2/figures/stage2_20261030/`提供PNG及可编辑SVG。数值及源文件SHA256见`output/stage2_summary_20261030/analysis.json`。原始8份CSV来自本机已核验的F-aux最终记录包，未覆盖原文件。','',
'[F-aux完成核验](Hetero-UNet/docs/stage2/FAUX_COMPLETED_2026-10-30.md)；[F/C错误与引导诊断](Hetero-UNet/docs/stage2/DIAGNOSTICS_2026-10-29.md)；[工作计划](Mamba-Attention医学图像分割_工作计划.md)。原权重、预测及最终记录在本机与OSS的对应四份归档中保留，本报告仅派生汇总。','','## 附录：全部50例病例均值','']
for k in data:
    L+=['### '+k+'（每例13器官均值，%）','']+table(['病例']+MODELS,[[n.replace('.nii.gz','')]+[f'{case[k][m][i]:.4f}' for m in MODELS] for i,n in enumerate(names)])+['']
REPORT.write_text('\n'.join(L).rstrip()+'\n',encoding='utf-8')
assert len(details)==2600 and len(paired)==12
assert all(hashlib.sha256((ROOT/s['path']).read_bytes()).hexdigest()==s['sha256'] for s in sources)
(OUT/'validation.json').write_text(json.dumps({'status':'passed','input_files':8,'cases_each':50,'organs_each':13,'models':4,'raw_records_exported':2600,'scores':5200,'existing_summary_match':True,'paired_macro_equals_mean_case_difference':True,'GT_contribution_sums_match':True,'figures':5,'formats':['png','svg'],'raw_inputs_preserved':True},indent=2)+'\n')
print(json.dumps({'report':str(REPORT),'means':means,'figures':5,'tables':8},ensure_ascii=True))
