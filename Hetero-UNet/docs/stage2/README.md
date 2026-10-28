# 阶段二实现说明：BG-Block-AttnRes

更新日期：2026-10-29。项目入口见[文档导航](../../../项目文档导航.md)，总体阶段和任务以[根目录工作计划](../../../Mamba-Attention医学图像分割_工作计划.md)为准。

**已核验进度**：2.1–2.3实现与真实CUDA/真实数据短程验收完成；F/C首轮1000轮及同协议外部50例评价完成，权重与结果归档已在OSS及本机校验；全50例错误分层和五例patch边界引导诊断完成。当前推进阶段二2.5的贡献确认，尚未完成模块一的稳定增益验收。

| 外部50例、13器官等权宏平均 | DSC % | NSD % |
|---|---:|---:|
| A，自行训练的复现基线 | 86.7638 | 90.5529 |
| F，无边界深度聚合 | 87.6068 | 91.2276 |
| C，完整模块一 | 87.5603 | 91.3554 |

本表为已归档CSV的首轮结果；F跨4090/V100续训，C在V100从头训练。C−F为DSC−0.0465、NSD+0.1278个百分点，不能据此声称边界注入带来稳定改善。论文Hetero-UNet参考值为86.7%/90.5%，不与本项目实测数据混写。

**当前执行**：[F-aux从头1000轮→同协议50例推理评价→四组对比→OSS归档回读→关机](FAUX_TRAINING_2026-10-29.md)。10月29日03:13的留存快照为已完成3轮、进入第4轮；不是实时进度或完训证明。本轮保留原MHA、边界辅助权重0.1，关闭引导注入；已有F/C与边界算法冻结。历史文件里的“只跑F”“C暂缓”“保持开机”不适用于本轮。

| 配置 | Trainer | 聚合 | 边界监督 | 边界注入 |
|---|---|---|---|---|
| A | `nnUNetTrainerMambaUNet` | 原基线 | 无 | 无 |
| F | `nnUNetTrainerMambaUNetBlockAttnRes` | 组间深度注意力 | 无 | 无 |
| C | `nnUNetTrainerMambaUNetBGBlockAttnRes` | 组间深度注意力 | 有 | 有 |
| F-aux | `nnUNetTrainerMambaUNetBlockAttnResBoundarySupervision` | 与F相同 | 与C相同 | 无 |

F-aux是附加控制，不改变原A–H编号：C对F包含“新增边界监督+注入”的总差异，C对F-aux进一步检验注入是否有贡献。F不必优于A才允许开发C。所有配置保留原MHA，不改`batch_first`及输入轴；A的网络和Trainer文件没有修改。

## 1. 基础聚合：2.1

保留12个原TransMamba层，每3层组成一组，共4组。这里“组大小3”是首轮可运行设计值，并非已经调优的结论。

设瓶颈embedding为`x0[B,L,C]`，历史池初始`H=[x0]`。第g组：

```text
ag = DepthAttention_g(H, boundary)
zg = 原来的3个TransMamba层(ag)  # 组内MHA、Mamba、MLP及标准残差保持原样
dg = zg - ag                    # 保存组残差增量，避免重复累计历史
H.append(dg)
最终输出 = 原last_norm(DepthAttention_final(H, boundary))
```

末组若不足3层也可运行，但正式配置为12/3。实现是完整TransMamba层的**组间适配**；[原Attention Residuals伪代码](https://github.com/MoonshotAI/Attention-Residuals)还在Transformer子层前聚合，不能把本实现写成逐行复刻其LLM版本。

单个聚合器的源s：

```text
ks = hs / sqrt(mean_channels(hs²) + eps)
es = (q·ks + b * qb·ks) / sqrt(C)   # F没有第二项
alpha = softmax(es, dim=深度源S)
output = sum_s(alpha_s * hs)
```

- 权重形状为`[B,L,S]`，只沿历史源S做softmax，不混合batch或空间token。
- 边界项依赖每个源的内容；把同一个边界偏置加到所有源会被softmax抵消，代码没有这样做。
- `q`、`qb`零初始化，初始是等权平均，不是等权求和，也不保证与原A输出相同。无边界F和有边界C使用一致的基础聚合结构。
- FP16/BF16输入使用FP32计算keys、权重和加权累积，再恢复输入dtype；避免复制完整`[S,B,L,C]`历史堆栈。真实激活内存仍需GPU实测。
- 零初始化时，分割损失首先更新边界查询；查询非零后可向边界预测分支传递梯度。C的辅助损失从首步直接监督该分支。

## 2. 边界分支：2.2

从编码器第一尺度特征预测单通道完整分辨率边界logits：`3×3 Conv → GroupNorm → GELU → 1×1 Conv`，默认隐藏通道8。最大池化对齐瓶颈，再sigmoid生成`[B,L,1]`的引导概率。

训练标签边界定义为2D四邻域/3D六邻域中语义标签不同的相邻体素，交界两侧均标记，包括器官与背景、器官与器官；不人为把patch外沿当边界。忽略标签及其直接邻居不参与监督。

边界从最高分辨率target生成，先监督细结构，不先缩小语义标签再求边界。该定义是体素邻域，不是毫米膨胀，与NSD的逐器官毫米容差不同。

```text
L_boundary = masked BCEWithLogits + binary soft Dice
L_C = 原nnU-Net分割/深监督损失 + 0.1 * L_boundary
```

辅助权重0.1是首轮预先固定值，尚未证明最优；已有F/C结果同时改变监督与注入，单独贡献仍待F-aux及重复实验确认。全ignore批次返回可反传零损失。当前支持普通语义类标签，不支持nnU-Net重叠region标签。

推理只从图像特征产生边界，不输入GT。输出仍是原分割tensor或深监督list；训练logits通过一次性缓存交给辅助loss并清除，不混入分割类别。

## 3. 模型组装与训练：2.3

- 新factory先构造原基线模块，再加入聚合器/边界头，保留原encoder、decoder、TransMamba层对象和初始化；不在加入后重新He初始化查询。
- F/C/F-aux拥有不同Trainer名及结果目录；相同seed的基座初始化可对齐，但不承诺新增模块后随机数消费、数据增强patch流逐位相同。
- 训练时缓存边界logits、消费并清理；eval时不保留缓存，在线验证loss仅记录分割项。best选择继续沿用原online Dice，不按辅助loss择优。
- 新Trainer记录`stage2_config.json`，epoch日志包含分割与未加权边界loss。初始版本禁用`torch.compile`；单GPU运行，DDP会改变原MHA的每卡batch交互，因此入口拒绝多进程。
- 推理器已恢复按`fold`和`checkpoint`参数加载，默认参数含义恢复正常。当前协议显式指定`-f all -chk checkpoint_best.pth`；不要依赖旧代码曾经写死best的行为。
- runner绕开原入口内部固定seed=1234，记录明确种子、参数、Git版本/dirty状态和每次运行日志；默认禁用W&B联网。

## 4. 现在如何使用

- [服务器拉取、短程检查与正式运行](SERVER_RUNBOOK.md)
- [逐项修改记录与本机验证](CHANGELOG.md)
- [Git交付记录与恢复工作入口](DELIVERY.md)
- [当前：F-aux训练、推理与归档关机](FAUX_TRAINING_2026-10-29.md)
- [F/C错误类型与边界引导诊断](DIAGNOSTICS_2026-10-29.md)
- [已完成：F/C外部50例评价记录](F_C_EVALUATION_2026-10-24.md)
- [历史：F500轮迁至V100、旧项目完整保存](MIGRATION_2026-10-22.md)
- 本机基础测试（内层`Hetero-UNet/umamba`目录）：`python -m unittest discover -s nnunetv2/tests/unit_tests -p "test_*.py" -v`

CPU单元测试只需PyTorch，不需要Mamba CUDA。部分测试明确使用小型替身验证接口和梯度；不能将其写成真实CUDA kernel或完整医学分割训练验证。

## 5. 已完成与仍待确认的边界

- 已完成：真实Mamba CUDA/AMP、完整patch与物理batch=2的短程训练/验证和检查点导出；F/C各1000轮、同50例DSC/NSD和错误/引导诊断。
- 正在执行：F-aux从头训练与后续外部评价，用于区分边界辅助监督和引导注入效应。
- 待确认：关键组按相同种子集合重复训练、结果稳定性，以及相关参数/效率目标。单次小幅差异与五例功能干预均不能替代这些确认。
- 后续若修改池化或注入设计，在内部开发划分选择方案并单独记录版本；不围绕这50例外部结果反复挑选超参。

模块二多尺度自适应跳跃融合及联合模型仍按原计划进入后续阶段。
