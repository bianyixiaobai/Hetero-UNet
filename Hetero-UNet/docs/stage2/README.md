# 阶段二实现说明：BG-Block-AttnRes

更新日期：2026-10-14。对应工作计划阶段二2.1–2.3的本机实现及验证准备。

**2026-10-21更新**：[新实例恢复与真实CUDA/真实数据验收记录](SERVER_SESSION_2026-10-21.md)。数据、旧预测与补丁已恢复，按原plans预处理完成，F/C/F-aux真实数据短程训练通过；路径配置修复后69项测试通过。首轮F/C按[固定协议](EXPERIMENT_PROTOCOL_2026-10-21.json)顺序长训练，完成状态以运行日志为准。以下2026-10-14的交付范围作为历史记录保留。

**当前交付范围**：F/C网络、边界分支和辅助损失、独立Trainer、检查点选择修正、带种子与运行记录的部署入口及CPU测试。服务器CUDA短程检查、1000 epochs训练、外部50例评价和调参尚未执行，不把代码测试通过当作模块有效性结论。

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

辅助权重0.1是初始开发值，尚无有效性或最优性证据。全ignore批次返回可反传零损失。当前支持普通语义类标签，不支持nnU-Net重叠region标签。

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
- [本轮训练结束后自动备份与关机、OSS续期及取消方式](AUTOSHUTDOWN_2026-10-21.md)
- 本机基础测试（内层`Hetero-UNet/umamba`目录）：`python -m unittest discover -s nnunetv2/tests/unit_tests -p "test_*.py" -v`

CPU单元测试只需PyTorch，不需要Mamba CUDA。部分测试明确使用小型替身验证接口和梯度；不能将其写成真实CUDA kernel或完整医学分割训练验证。

## 5. 尚待实验确认

1. 服务器原依赖下的真实Mamba前向/反向、AMP及batch=2完整patch显存。
2. F/C/F-aux的短程检查，再按开发划分调参并冻结配置。
3. 同协议1000 epochs训练与外部50例DSC/NSD，检查边界图与失败样例。
4. 关键结果的多种子确认及监督/注入贡献消融。

这些属于阶段二2.3的服务器验收和2.4/2.5实验工作，未标记为已完成。
