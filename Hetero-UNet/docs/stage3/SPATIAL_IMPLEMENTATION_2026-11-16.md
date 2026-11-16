# C+ND / C+R0本机实现与验收记录

日期：2026-11-16。**本机实现及CPU验收完成；没有连接服务器、启动正式训练、改变50/50划分或重训C。** 本机验收后的GitHub交付安排见[分支交付与服务器拉取说明](SPATIAL_GITHUB_HANDOFF_2026-11-16.md)；本次文件哈希见本目录校验清单。

## 1. 本次实现

| 项目 | 实现及位置 |
|---|---|
| 空间块 | [spatial_receptive_block.py](../../umamba/nnunetv2/nets/spatial_receptive_block.py)：R0为1³、3³(d=1)、3³(d=2)深度卷积；ND仅将第三分支改为d=1并调整padding |
| 接入 | [MambaUNetSpatialRefinement.py](../../umamba/nnunetv2/nets/MambaUNetSpatialRefinement.py)：C构建完成后包装原解码器，第一解码stage后、该stage分割头和下次上采样之前增加空间块 |
| Trainer | [nnUNetTrainerMambaUNetSpatialRefinement.py](../../umamba/nnunetv2/training/nnUNetTrainer/nnUNetTrainerMambaUNetSpatialRefinement.py)：分别注册BGSpatialND/BGSpatialR0，继承C损失、训练、优化器、checkpoint逻辑 |
| 启动器 | [run_stage3.py](../../scripts/run_stage3.py)：新增`C-ND`、`C-R0`入口，旧D/E入口保留；新增组拒绝development、数值fold、不同seed或plans |
| 参照合同 | [C_reference_contract.json](spatial_validation_2026-11-16/C_reference_contract.json)：从既有C最终归档提取配置、描述符摘要及继承文件哈希 |

两组均拼接三分支，再用GroupNorm（groups=C，channels=3C）、GELU、1³投影和残差。采用与参考块一致的常规PyTorch初始化，不额外进行零残差初始化。所有C模块先按原工厂初始化，再创建新增块；没有对整个网络重新初始化。

ND仍是多分支、多核大小对照，不能称为完全单感受野。R0为DMRFB参考适配，非本项目已成立的原创机制。方法依据及原始代码链接见[选型文档](模块二重新选型_近期方法与实验路线_2026-11-11.md)。

## 2. C参照和原协议核对

- 本机C最终归档`C_seed1234_final.tar`重新计算SHA256，与原备份收据一致；包内best/final权重分别计算哈希并记录，没有修改或重新训练它们。
- 14项继承代码文件与C归档的源代码指纹一致（按LF统一换行；部分Windows原文件为CRLF）。原MHA、C聚合、边界头、损失、训练器和推理代码未修改。
- 原训练图像/标签、评价图像/标签分别50份；病例ID配对且两组ID不重叠。本次是清单核对，没有重新逐体素扫描全部CT。
- 保留原`fold=all`、seed1234、1000 epochs、250/50次训练/训练内验证迭代、batch2、patch `[40,224,192]`、初始学习率0.01，以及C边界监督权重0.1。
- 启动新组时会验证实际plans、dataset的规范化JSON哈希及继承代码指纹，并检查训练默认值；不一致则报错。此检查不能替代服务器原始数据与预处理数组的完整性检查。
- 新组与C归档一样跳过训后训练集全卷导出；每轮训练内验证及best选择仍保留。外部50例推理评价是独立步骤。

完整本机核对结果：[local_audit.json](../../../output/module2_spatial_20261116/local_audit.json)。权重只进行归档及包内文件哈希核验，本轮没有重载既有C的724MB检查点进行真实Mamba前向。

## 3. 验收证据与边界

| 验收 | 结果 | 范围 |
|---|---|---|
| 新模块与启动器单元检查 | 8项通过 | 公式对照、采样支持范围、形状/梯度/更新、参数与初始化匹配、协议拒绝条件、D/E解析保持 |
| 相关回归检查 | 37项通过 | 原AttnRes、深度聚合、边界损失、旧跳跃融合及启动器 |
| 两组完整框架集成 | 均通过 | 实际nnU-Net编码器/解码器、MHA/MLP、loss、SGD、Trainer与Predictor；Mamba和W&B使用显式替身 |
| 接入位置 | 通过 | 测试记录证明该stage分割头和下一次上采样都接收新增块输出；恒等替换后与原解码路径逐元素一致 |
| 参数与初始状态 | 通过 | 优化器覆盖全部可训练参数；两组对应初始张量一致；继承C的初始张量与单独构建C一致 |
| 保存恢复 | 通过 | 自生成检查点eval输出逐元素一致；SGD动量恢复；按保存epoch恢复PolyLR，固定批次的下一步更新逐元素一致 |
| 推理 | 通过 | 实际Predictor类发现/重建、step_size=0.5、Gaussian、无TTA；较大重叠体积、小于patch体积填充、原始NumPy数组预处理到分割输出 |
| Gaussian覆盖 | 通过 | 常量logits诊断验证重叠权重归一化和完整覆盖，容差1e-3；不宣称真实分割没有接缝 |
| 原CT尺寸新增块检查 | 两组通过 | CPU只运行新增块的前向/反向，输入`[2,320,5,14,12]`；不是整网原CT patch检查 |

本机环境为PyTorch `2.12.0+cpu`，CUDA不可用、真实mamba_ssm未安装。合成框架检查的patch为`[8,16,16]`、3类，不是临床训练。测试替身仅由显式诊断脚本注册，生产启动器不会自动退回替身。

首轮框架检查发现合成plans缺少预处理所需的强度统计字段，已补齐合成夹具，生产模型与原数据未改。失败日志保留为`framework_attempt1.log`。本机新PyTorch加载旧格式自生成nnU-Net检查点时，仅诊断进程设置可信检查点兼容选项；没有全局改变生产加载配置。

**未通过本机验证的项目**：真实Mamba selective scan、完整CT patch整网GPU显存与耗时、CUDA AMP/GradScaler恢复、临床NIfTI导出和50例评价、数据加载worker随机流的精确续训、模型效果。原训练器按epoch重建PolyLR，不保存独立scheduler字典；固定批次CPU恢复一致不能推广成所有真实续训逐位一致。

检查脚本与原始证据：

- [单元测试](../../umamba/nnunetv2/tests/unit_tests/test_spatial_receptive_block.py)、[框架检查脚本](../../scripts/check_spatial_framework.py)、[C归档与新增块核对脚本](../../scripts/audit_spatial_local.py)。
- [单元日志](../../../output/module2_spatial_20261116/unit_tests.log)、[回归日志](../../../output/module2_spatial_20261116/regression_tests.log)、[框架日志](../../../output/module2_spatial_20261116/framework.log)。
- [小型框架收据](spatial_validation_2026-11-16/framework_check.json)随代码保存；完整合成检查点位于本机`output/module2_spatial_20261116/framework/`。

## 4. 新增开销

以下是原plans下**新增空间块**的计数，不是整个网络开销或GPU峰值。

| 项目 | C+ND新增块 | C+R0新增块 |
|---|---:|---:|
| 通道数 | 320 | 320 |
| 特征尺寸（D/H/W） | 5×14×12 | 5×14×12 |
| 参数量 | 327,040 | 327,040 |
| batch2卷积MACs | 545,664,000 | 545,664,000 |
| batch2卷积FLOPs（2/MAC） | 1,091,328,000 | 1,091,328,000 |

运算量经实际Conv3d输出hook计数与解析公式相互核对，包括padding对应的标准稠密卷积运算，不含归一化、GELU、偏置加法和残差加法。两组理论量相同，实际GPU速度和显存仍需实测。

## 5. 后续服务器操作准备

用户租好服务器并通知后再部署。代码通过实验分支交付，拉取方式见交付说明；尚未远程运行或关机。不要使用旧D/E自动队列启动这些新组。

服务器项目目录内，先配置原`nnUNet_raw`、`nnUNet_preprocessed`与`nnUNet_results`路径，并核对C归档plans和数据。以下命令是准备好的入口，未在服务器执行：

```bash
# 只解析配置，不创建训练任务
python scripts/run_stage3.py --model C-ND --results-root /hy-tmp/spatial_results --dry-run
python scripts/run_stage3.py --model C-R0 --results-root /hy-tmp/spatial_results --dry-run

# 真实Mamba/CUDA检查；PLANS和DATASET_JSON替换成核对后的原文件路径
python scripts/check_spatial_cuda.py --plans PLANS --dataset-json DATASET_JSON --output /hy-tmp/spatial_cuda_check

# 原完整patch真实数据短程；输出自动带_smoke，与正式实验分开
python scripts/run_stage3.py --model C-ND --results-root /hy-tmp/spatial_results --smoke
python scripts/run_stage3.py --model C-R0 --results-root /hy-tmp/spatial_results --smoke

# GPU及真实数据短程通过后，按ND→R0顺序从头正式训练
python scripts/run_stage3.py --model C-ND --results-root /hy-tmp/spatial_results
python scripts/run_stage3.py --model C-R0 --results-root /hy-tmp/spatial_results
```

CUDA检查脚本已准备并通过语法检查，尚未运行。正式输出分别位于`stage3_C-ND_seed1234`和`stage3_C-R0_seed1234`；内部还包含dataset/Trainer/plans/configuration/fold目录。新组模型名在报告写作中为C+ND/C+R0，CLI使用C-ND/C-R0。

如需续训，显式加`--continue`，只能恢复同一实验目录的检查点。已有正式权重时不带此标志会拒绝覆盖；不允许将smoke检查点当正式初始化。训练启动器不自动运行外部50例评价、OSS上传或关机；这些流程按原评价规则在部署时接入并独立核验。

## 6. 下一步与留存

- [x] C参照合同、归档哈希及病例清单核对。
- [x] ND/R0、独立wrapper/Trainer、受约束启动入口实现。
- [x] CPU单元、相关回归、合成框架/推理/保存恢复检查。
- [x] 原CT空间块参数和卷积运算量匹配核对。
- [ ] 服务器拉取GitHub交付版本，核对提交、环境与数据（等待用户通知）。
- [ ] 真实GPU/AMP/原patch与真实数据短程验收。
- [ ] 新增两组从头训练1000轮及原外部50例推理评价。
- [ ] 与已有C汇总比较、图表、OSS及本机完整归档。

历史实验、权重、原始数据及评分脚本保留。本次更新前文档备份在`output/document_backups/2026-11-16_spatial_implementation/`。核心研究问题和三组实验定义继续遵循[选型及执行清单](模块二重新选型_近期方法与实验路线_2026-11-11.md)。
