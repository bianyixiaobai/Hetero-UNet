# 阶段二修改与验证记录

维护更新：2026-10-29。分支：`codex/bg-block-attnres-stage2`。

本页主体记录**10月14日初次实现与CPU验证**，其中“待执行”只表示当时进度。后续真实CUDA验收、F/C训练及50例评价均已完成，当前F-aux执行见[阶段二入口](README.md)和[F-aux记录](FAUX_TRAINING_2026-10-29.md)。`output/`目前也包含正式结果和完整权重备份。

## 2026-10-29文档整理

新增[项目文档导航](../../../项目文档导航.md)、[docs说明](../README.md)和本机output说明；统一工作计划与入口的已完成/进行中事项；标记日期记录和重复副本的时间范围；恢复/操作手册区分固定训练源码与后续调度提交。具体清单和备份见[维护记录](../DOCUMENTATION_2026-10-29.md)。本次未改模型、训练脚本、数据、指标或原始收据。

## 2026-10-14初次交付记录

### 当时交付状态

阶段二2.1、2.2本机实现完成；2.3网络组装、CPU接口/梯度与框架诊断通过，真实服务器CUDA检查待执行。2.4完整训练/外部评价、2.5调参与重复实验未完成。没有连接已关闭的服务器，没有启动付费实例或宣称分割性能提升。

研究方向、七阶段规划、模块二多尺度目标及现有A结果保持不变。A的`MambaUNet.py`和`nnUNetTrainerMambaUNet.py`与交付前Git版本一致；MHA轴与默认配置保留。新模型从头训练，不能把A权重直接当成新增聚合结构已经训练好的权重。

## 文件与作用

以下路径相对内层`Hetero-UNet/`目录。

| 文件 | 修改及原因 |
|---|---|
| `umamba/nnunetv2/nets/block_attention_residuals.py` | F/C共用深度源聚合；RMS keys、零初始化查询、沿历史源softmax，半精度输入时FP32累积 |
| `umamba/nnunetv2/nets/boundary_prior.py` | 浅层特征预测边界；最大池化对齐瓶颈，推理只使用图像特征 |
| `umamba/nnunetv2/nets/MambaUNetAttnRes.py` | 12个完整TransMamba层按3层分4组，历史存组残差增量；独立F/C包装，不改A |
| `umamba/nnunetv2/training/loss/boundary.py` | 语义边界标签和masked BCE+Dice；包含器官间边界，处理ignore区域 |
| `umamba/nnunetv2/training/loss/segmentation_boundary.py` | 分割损失加0.1边界辅助损失；最高分辨率target、一次性训练缓存、验证仅分割loss |
| `umamba/nnunetv2/training/nnUNetTrainer/nnUNetTrainerMambaUNetAttnRes.py` | F/C/F-aux三个独立Trainer与输出路径；保留原best选择；记录组件loss，拒绝DDP |
| `umamba/nnunetv2/inference/predict_from_raw_data.py` | 恢复按用户指定fold/checkpoint加载；支持自动检测仅有fold_all的目录，错误路径明确报错 |
| `umamba/nnunetv2/training/lr_scheduler/polylr.py` | 去除新Torch不接受的调度器构造参数verbose；保留原poly学习率公式、指数及预算 |
| `scripts/run_stage2.py` | 隔离模型/种子/开发/smoke目录；绕开旧入口硬编码种子；检查CT物理batch和patch，记录代码/配置哈希、日志、状态，保护已有检查点 |
| `scripts/check_stage2_framework.py` | 显式CPU诊断：真实nnU-Net编码器、解码器、原MHA、Trainer、优化器、验证和检查点往返；仅在诊断进程替换CUDA Mamba |
| `umamba/nnunetv2/tests/unit_tests/test_*.py` | 65项数值、梯度、形状、边界、缓存、协议、运行入口和调度器测试 |
| `README.md`及本目录 | 入口、设计、修改记录和服务器操作说明 |

外层根目录工作计划更新到v1.8，记录上述真实进展。用户CSV、图片、历史备份及其他研究文档不随本次代码发布批量提交。工作计划v1.7备份在本机`output/document_backups/`。

## 已落实的设计选择

- 每组3个完整TransMamba层；组内原残差不改，组间保存`组输出−组输入`，最终额外聚合后接原last_norm。这是视觉分组适配，不能写成原LLM子层实现的逐行复刻。
- 查询初始为零，对历史源等权平均；不保证初始网络与A函数相同。F/C采用相同聚合骨架。
- 边界项为源相关的`b * dot(q_boundary, key_source)`；避免所有源同加一个偏置被softmax抵消。
- 边界头隐藏通道8、辅助权重0.1、组大小3均为首轮开发值。调参使用开发划分，冻结后才进行外部50例最终评价。
- F-aux用于区分额外监督与条件注入：C对F是二者总差异，C对F-aux检验注入增量。它是附加控制，不重命名原A–H实验矩阵。
- 不启用compile/DDP；各组固定相同物理batch=2，梯度累积不能替代此MHA的物理batch交互。
- 相同种子记录不代表数据增强流或CUDA结果逐位一致。断点续训沿用上游状态恢复，不声称恢复所有worker随机数状态。

## 本机验证记录

2026-10-14，Python 3.10.20、PyTorch 2.12.0+cpu。无本机CUDA。基础命令在内层`Hetero-UNet/umamba`运行：

```powershell
& E:/anaconda/envs/mam/python.exe -m unittest discover -s nnunetv2/tests/unit_tests -p 'test_*.py' -q
```

结果：**65项通过**。部分网络测试使用小型替身；没有把替身作为生产代码降级实现。

框架诊断在外层仓库根目录运行：

```powershell
& ./output/stage2_venv/Scripts/python.exe ./Hetero-UNet/scripts/check_stage2_framework.py --output ./output/stage2_framework_check
```

结果：**F/C/F-aux × 2D/3D，共6组通过**。覆盖单步训练、验证、分割输出、真实模块检查点保存/推理加载和输出一致性。诊断明确使用线性+tanh的Mamba替身与W&B替身；真实MHA、nnU-Net卷积编码器/解码器和训练框架参与测试。报告为本机`output/stage2_framework_check/framework_check.json`；诊断loss仅是合成数据的查错信号，不能用于比较方法性能。

隔离诊断环境在`output/stage2_venv/`，没有升级用户原服务器环境。主要依赖：dynamic-network-architectures 0.2、batchgenerators 0.25.1、acvl-utils 0.2、scipy 1.14.1、scikit-image 0.24.0、scikit-learn 1.5.2、nibabel 5.3.2、SimpleITK 2.4.1、matplotlib 3.9.4、wandb 0.18.7。首次框架诊断暴露新版Torch移除verbose参数，已修复调度器兼容性并重新通过测试。该CPU环境不能复制到服务器替代原CUDA环境。

## Git交付与后续边界

本地交付前HEAD为`04180d4`；检查时远端main为`8d7d320e4146fa21bcf91fa17946dc8902c1036b`，双方没有共同祖先，目录布局也不同。本次只发布新分支，不强推或替换main。服务器采用新目录克隆分支，见[操作手册](SERVER_RUNBOOK.md)。提交号由`git log -1 --oneline`及GitHub分支记录获取。

在10月14日交付时，`output/`只包含诊断环境、合成检查点、报告和备份，当时尚无F/C正式结果；真实CUDA及完整实验当时仍待服务器验收。后续这些验收和F/C实验已完成，当前状态以本页顶部导航为准。发生OOM时先检查资源或更换显存；不能只给C单独降低物理batch后与原A作同协议结论。
