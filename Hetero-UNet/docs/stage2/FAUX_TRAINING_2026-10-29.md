# F-aux正式训练、评价与自动归档关机

日期：2026-10-29，北京时间。属于阶段二2.5的贡献归因实验。

用户授权：现在开始F-aux训练；完成后自动推理、评价、汇总、归档上传并校验，全部完成后关闭当前V100服务器。

## 已执行

- 服务器：`i-2.gpushare.com:56465`，实例`I2bde78031f00701d9e`，V100-SXM2-32GB。
- 数据盘150G，启动时约131G可用；GPU原为空闲，没有遗留训练或关机任务。
- 原训练仓库`/hy-tmp/Hetero-UNet-stage2`保持`9fe70ac570a0e14ebce30daa4a16a22e49b4f3fd`且干净，未更新模型源码。
- 与完成的C组核对14项源码指纹、数据描述文件；原253份预处理文件与保存清单逐一核对大小和SHA256，外部50例图像及GT指纹通过。
- 单独进程完成完整patch、物理batch=2的F-aux CUDA/AMP训练步及验证步。边界损失有限且非零、边界头梯度非零；五个聚合器均未接收引导，未注册边界query；验证时边界头跳过；原MHA保持`batch_first=False`。
- 后台队列03:04启动预检，03:05启动正式训练进程，03:06进入日志中的Epoch 0（第1轮）。
- 第1轮于03:08完成：train_loss=1.0059、val_loss=0.6241，边界未加权损失1.1823，权重0.1，轮计算时间145.32秒；best已产生并进入第2轮。初始轮结果仅用于确认运行，不能评价模块有效性。
- 自动收尾的5项隔离检查通过：未显式启用关机、配置未授权、GPU仍忙、权重哈希变化时均阻止收尾；OSS回读内容损坏时拒绝签发成功收据并重试。测试使用临时目录和模拟传输，未操作真实训练或关机。

本机启动核验材料保存在`output/stage2_Faux_20261029/`。正式训练正在进行；本记录不表示训练、推理或最终备份已经完成。

03:10已将启动记录、工作计划v1.19、环境、调度脚本及原训练源码快照打包为`F_aux_startup.tar`，上传本次OSS目录并回读通过：1,351,680字节，SHA256为`94484e145b06e39257a5dd0533d0b2f20417002cf1a2e0978bdcc1be4db5f541e`。这是启动备份，不替代完训后的四份归档。

## 冻结实验定义

| 配置项 | F-aux本次执行 |
|---|---|
| Trainer | `nnUNetTrainerMambaUNetBlockAttnResBoundarySupervision` |
| 初始权重 | 从随机初始化训练，不加载F/C权重 |
| 数据与配置 | Dataset701 / 3d_fullres / fold_all，原50训练例 |
| 种子、轮数 | seed1234，1000轮；每轮250训练步、50在线验证步 |
| 物理batch、patch | 2；[40,224,192] |
| 初始学习率 | 0.01，原优化器和调度器 |
| 深度聚合 | 完整TransMamba层分组，group_size=3 |
| 边界头、辅助损失 | 与C相同，hidden_channels=8，loss_weight=0.1 |
| 边界引导注入 | 关闭；不改变池化或注入算法，不进行新方案调参 |
| MHA | 原官方实现保持不变 |
| best选择 | 原在线验证Dice规则；不按外部50例选择检查点 |

F-aux−F用于观察边界辅助监督效应；C−F-aux用于观察已有边界监督时引导注入的附加效应。此轮为单种子探索，不能直接声称统计显著或稳定贡献；F经历4090→V100迁移的历史限制仍保留。

## 后台流程与位置

tmux会话：`stage2_Faux_20261029`。调度脚本部署在原仓库之外，避免改变训练指纹。

工作目录：`/hy-tmp/stage2_Faux_20261029/`。

正式结果：`/hy-tmp/stage2_results/stage2_F-aux_seed1234/Dataset701_AbdomenCT/nnUNetTrainerMambaUNetBlockAttnResBoundarySupervision__nnUNetPlans__3d_fullres/`。

1. 正式训练1000轮；保留每轮训练、验证及边界损失、best/final、协议和环境记录。
2. 独立核验完训状态、1000轮日志、损失组合、权重有限性、final轮数和源码指纹。
3. 用原best推理相同外部50例；无TTA、滑窗步长0.5、Gaussian、每次一个patch、预处理与导出各1进程。复用原等价低内存导出补丁，不改模型。
4. 核查50份预测的尺寸、affine、spacing、标签范围，运行原DSC/NSD脚本（NSD两进程）；保留原空类及裁剪计分规则。
5. 生成A/F/F-aux/C汇总、逐病例配对与逐器官表；不覆盖A/F/C旧CSV或预测。
6. 归档训练权重及日志、50例预测及评价、代码与运行记录、最终收据；上传OSS，再逐份下载回读比较文件大小与SHA256。
7. 四份归档及最终收据均通过后，确认没有训练或推理/GPU进程，核对平台关机程序指纹，再执行关机。

主要状态：`state.json`；队列日志`pipeline.log`；训练日志`train.log`；后续`predict.log`、`DSC.log`、`NSD.log`；最终结果`RESULTS.md`、`comparison.json`及配对CSV。

可查看进度：

```bash
tail -n 30 /hy-tmp/stage2_Faux_20261029/train.log
cat /hy-tmp/stage2_Faux_20261029/state.json
```

`DISABLE_SHUTDOWN`文件只阻止关机，不用于暂停训练。异常时保留数据并记录错误，不执行成功关机。

## OSS归档与授权有效期

目录：`oss://backup/stage2_20261029/F_aux/`。

预计产生：

- `F_aux_seed1234_training_verified.tar`：best/final、优化器状态、训练日志、plans/dataset及完训核验。
- `F_aux_external50_verified.tar`：50例预测、原始DSC/NSD、几何及哈希检查、评价代码与协议。
- `F_aux_final_records.tar`：A/F/C参考CSV、比较结果、执行脚本、原训练源码快照、实际推理runtime、环境、工作计划与日志。
- `F_aux_final_delivery.tar`：最终上传收据与授权协议；也必须通过OSS回读。

OSS现有授权到期为**2026-10-30 02:27:27北京时间**。此前C组训练耗时40.57小时，本次初估10月30日晚完训并完成约2–3小时评价，实际根据前几轮速度修正。当前授权无法覆盖预计收尾，已向用户说明需在10月30日中午至傍晚再次执行`oss login`。本机SSH密码不能替代平台登录。

若上传时授权失效或网络异常，保留所有结果并每5分钟重试；不会在未验证上传时关机。因此自动关机时间受OSS授权及传输成功情况约束。

脚本：`scripts/run_stage2_faux_20261029.py`。训练源码始终保持原版本，调度、测试与记录另行提交GitHub实验分支。
