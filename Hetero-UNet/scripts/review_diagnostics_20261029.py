"""Local, fixed-scale attention plots and interpretation from verified diagnostics."""
from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    root=Path(__file__).resolve().parents[2];work=root/'output/stage2_diagnostics_20261029'
    reports=json.loads((work/'guidance_summary.json').read_text())['reports']
    global_max=0.
    for r in reports:
        with np.load(work/(r['case']+'_patch_maps.npz')) as a:
            global_max=max(global_max,max(float(np.abs(a['weights_'+str(i)]-a['weights_zero_'+str(i)]).max()) for i in range(5)))
    for r in reports:
        with np.load(work/(r['case']+'_patch_maps.npz')) as a:
            fig,axes=plt.subplots(2,5,figsize=(17,6),layout='constrained')
            for i in range(5):
                w=a['weights_'+str(i)];w0=a['weights_zero_'+str(i)]
                normal=axes[0,i].imshow(w[0].T,aspect='auto',vmin=0,vmax=1)
                delta=axes[1,i].imshow(np.abs(w-w0)[0].T,aspect='auto',vmin=0,vmax=global_max)
                axes[0,i].set_title('Aggregator '+str(i));axes[1,i].set_xlabel('Spatial token index')
            axes[0,0].set_ylabel('Depth source');axes[1,0].set_ylabel('Depth source')
            fig.colorbar(normal,ax=axes[0,:].tolist(),label='Natural attention probability')
            fig.colorbar(delta,ax=axes[1,:].tolist(),label='Absolute natural-zero difference (same history)')
            fig.suptitle(r['case']+'; one common difference scale across all five cases')
            fig.savefig(work/(r['case']+'_depth_weights_fixed_scale.png'),dpi=140);plt.close(fig)
    text='''# F/C错误类型与边界引导：诊断结论

日期：2026-10-29。全50例、F/C各650个病例-器官对；另对C的五个GT定位patch做功能干预。未训练、未改权重/MHA、未改主指标。

## 1. 空标签影响局部，但不是总体差异的来源

共有19个GT空类对，F/C各有8个假阳性对，空类计分对总体均值的贡献相同。因此此前0031/0032/0048中的0与1切换确实存在，但全部50例汇总后的净贡献抵消。

在相同631个GT非空对上，F/C的DSC为88.5015%/88.4536%，NSD为92.2313%/92.3630%。C仍表现为DSC略低、NSD略高。此分层为补充描述，不替代原完整50例主结果。

## 2. 漏分、错位和过大预测需要分开看

- F在0025胆囊预测为空；C虽无预测为空的GT非空类，却在0002胆囊预测10391体素、GT1433体素，重叠TP为0。两组各有一个与GT完全零重叠的器官对，不能将C“预测不为空”误写成消除完全漏分。
- 原生全卷体积比<0.5的对：F为27、C为28；>1.5的对：F为6、C为14。这些只是描述性标记，体积大不一定覆盖到了正确位置。
- Aorta/IVC/Esophagus的原指标有GT标注z范围裁剪，原生全卷FP/FN不等同该评分范围；其他器官也需结合空间重叠和叠图判断。
- 0048十二指肠叠图显示两组均有明显覆盖不足，C在图示切片上的预测轮廓更小；全卷GT/F/C体素为73269/30065/16309。图示只是一个切片，完整结论结合CSV和全卷核查。

## 3. 引导分支有作用，但作用不均匀

五个patch重复原前向的logit差均为0。置零引导后，标签变化约0.0190%–0.6603%；均值替代约0.0045%–0.0695%，打乱位置约0.0058%–0.1496%。说明这五个输入上的引导不是未生效；变化比例不代表性能提升比例。

同一history下，聚合器1–3的权重变化较小，主要变化集中于最后聚合器4，平均绝对差约0.0337–0.0420。聚合器0仅有一个源，权重恒为1，无变化是正确行为。

高分辨率边界概率>0.5的体素约占1.82%–6.08%，最大池化后，引导>0.5的token约占74.76%–93.81%。边界图确实变成了大范围高响应的粗粒度引导，未必还保持精细定位；这是值得验证的解释线索，不是池化实现错误或性能下降原因的证明。GT薄边界的patch Dice与NSD不是同一指标。

## 4. 下一步建议

保持当前F/C结果冻结，先结合错位和覆盖不足病例做空间检查；下一次有必要的功能检查应针对池化后的空间选择性和末端聚合，而不是直接认定整个边界头无效。若仍要验证边界注入的独立贡献，F-aux有价值，但这五个patch的干预不能替代它。任何新方案的选择都应在内部开发划分进行，不在这50例外部结果上反复调参。

目前仍不能把小幅C/F差异唯一归因于边界注入，也未测试改池化后的性能；研究方向、模块二计划和MHA冻结决定保持不变。

## 5. 查看材料

- [完整统计与原始数值报告](DIAGNOSTIC_REPORT.md)
- [1300行错误统计](error_types_with_spatial_flags.csv)
- [边界与干预原始记录](guidance_summary.json)
- [两份归档的OSS回读收据](backup_receipt.json)
- [0048十二指肠叠图](FLARETs_0048_Duodenum_overlay.png)

以下图均使用明确的固定色标。权重差图在五例间共用同一尺度，避免把很小的早期权重差显示得与末端差异一样强。
'''
    for r in reports:
        case=r['case'];text+=f'\n### {case}\n\n![边界与池化]({case}_boundary_fixed_scale.png)\n\n![深度注意力权重]({case}_depth_weights_fixed_scale.png)\n'
    (work/'诊断结论与下一步.md').write_text(text,encoding='utf-8')
    p=work/'DIAGNOSTIC_REPORT.md';raw=p.read_text(encoding='utf-8')
    raw=raw.replace('完全漏分对','预测完全为空对')
    raw+='\n\n补充：C在0002胆囊的预测不为空，但与GT零重叠；F在0025胆囊预测为空。两组各有一个GT非空而TP=0的器官对。请结合[诊断结论](诊断结论与下一步.md)，不要仅根据预测是否为空判断完全漏分。\n'
    p.write_text(raw,encoding='utf-8')

if __name__=='__main__':main()
