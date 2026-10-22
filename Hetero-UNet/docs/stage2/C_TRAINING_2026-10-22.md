# C组首轮正式训练：2026-10-22

用户在F完成及数据盘整理后启动GPU实例，要求按原计划进行C训练并保留代码、数据及日志记录。本轮只运行C，不重跑F，不启动F-aux。外部50例评价仍是后续独立步骤。

**23:36用户更新：完训后上传、校验，保持服务器运行。** 原关机会话已取消，23:37:59启用仅备份程序；本页以下为当前有效安排。最初启动包中的关机配置为历史版本，以新备份配置及本次策略变更凭据为准。

## 实验定义与启动

北京时间2026-10-22 **23:23:09**进入正式C队列，**23:23:17**开始零基epoch0；run_id为`20261022T152315.040085Z_1ad47600`。Trainer为`nnUNetTrainerMambaUNetBGBlockAttnRes`，**从随机初始化开始1000轮，不继承F的权重或优化器，也不使用`--continue`**。

| 项目 | 本次冻结值 |
|---|---|
| 数据与plans | Dataset701_AbdomenCT，50例训练；原nnUNetPlans、3d_fullres、fold_all |
| 总轮数与种子 | 1000 epochs，seed=1234；250训练步/轮，50验证步/轮 |
| 物理batch与patch | 2；[40,224,192] |
| 初始学习率 | 0.01，原优化器/调度器 |
| MHA | 保留官方实现、`batch_first=False`及原输入布局 |
| 模块一 | 12层TransMamba，每3层一组，边界隐藏通道8 |
| C开关 | `boundary_guidance=True`、`auxiliary_boundary=True`、边界损失权重0.1 |
| 精度与选择 | AMP；按内部验证EMA保存best，外部评价用best |
| 最终整卷导出 | 与F一样跳过训练后fold_all整卷导出；每轮内部验证保留；外部50例另行推理 |
| 模型代码 | 9fe70ac570a0e14ebce30daa4a16a22e49b4f3fd，14项源码SHA256与F一致 |
| GPU | Tesla V100-SXM2-32GB，单卡 |
| 数据加载 | 与F续训入口相同，`nnUNet_n_proc_DA=12`；不更改batch或patch |

实际命令保存在[run_C.sh](session_2026-10-22_C_start/run_C.sh)：

```bash
python -u /hy-tmp/Hetero-UNet-stage2/Hetero-UNet/scripts/run_stage2.py \
  --model C --seed 1234 --results-root /hy-tmp/stage2_results --skip-final-validation
```

F此前包含第500轮的跨硬件迁移续训，相关随机状态恢复限制继续保留；本次只保证冻结数据和实验参数一致，不声称两组执行轨迹逐位相同。

## 已核验事项

- 253份预处理文件逐一核对SHA256，沿用原50个npz、50个pkl及100个npy缓存，没有重新预处理。
- 原始数据的imagesTr、labelsTr、imagesVal、labelsVal各50例；训练入口不使用外部50例进行训练或调参。
- plans与dataset描述符及14项源码指纹和F记录一致；新C目录不存在旧检查点。
- 单卡真实CUDA/AMP、原完整patch和batch=2的F/C/F-aux前后向与验证检查全部通过，梯度和loss有限。合成检查不作为模型有效性结论，也不是额外正式F或F-aux训练。
- C正式manifest为非smoke、非continue、source clean，边界注入和辅助监督开关及0.1权重已核实。
- 两份由OSS工具生成的未跟踪诊断日志移至本轮私有审计目录，内容保留；正式模型源码没有修改。运行工作目录设在审计目录，避免诊断日志再次污染源码目录。
- 自动收尾现有18项安全单元测试通过，shell语法与配置JSON检查通过。缺少可选`hiddenlayer`只影响网络结构绘图，不影响训练。

实时推进、首轮训练/边界/验证loss和后续时间估算以本次session的`startup_check.json`及实际日志为准，不把启动检查当作1000轮已经完成。

**首轮已通过**：epoch0训练loss=1.0417，验证loss=0.6631；分割分量0.922976、未加权边界分量1.187571，满足`总loss≈分割loss+0.1×边界loss`。首轮耗时150.46秒，23:27已进入epoch1，GPU约11.6GiB、利用率100%。按首轮加约2秒/轮开销粗估于**10月24日傍晚**结束，后续以稳定实测速度修正，不把约42小时估算当作保证。

## 路径与备份

| 内容 | 位置 |
|---|---|
| C训练会话 | `stage2_C_20261022` |
| 正式C结果 | `/hy-tmp/stage2_results/stage2_C_seed1234/Dataset701_AbdomenCT/nnUNetTrainerMambaUNetBGBlockAttnRes__nnUNetPlans__3d_fullres/fold_all/` |
| 总日志 | `/hy-tmp/stage2_C_audit_2026-10-22/C_queue.log` |
| 队列状态 | `/hy-tmp/stage2_C_audit_2026-10-22/C_queue.json` |
| 审计/环境/启动命令 | `/hy-tmp/stage2_C_audit_2026-10-22/` |
| 模型配置、运行manifest和逐轮日志 | 正式C的fold_all目录；每次运行另存run_id历史manifest与控制台日志 |
| 自动备份会话 | `stage2_backup_C_20261022` |
| 备份状态 | `/hy-tmp/stage2_C_backup_20261022/state.json` |
| 备份配置 | `/hy-tmp/stage2_C_audit_2026-10-22/C_backup_only_config.json`，`shutdown_authorized=false` |
| C的OSS目标 | `oss://backup/stage2_20261022/C_seed1234_verified/` |

最新权重按原Trainer周期保存，best随内部验证更新；结束后保存final。独立备份程序等待正式C累计1000轮并退出，CPU校验best/final后上传权重、源码、审计日志与收据，逐包回读校验通过后进入`verified_not_armed`状态，**保持服务器运行，不调用关机程序**。上传失败会重试，保留本地结果。

备份进程只带`--watch`，没有`--arm-shutdown`，配置也显式禁止关机；即使误加关机参数，程序也拒绝启动。新模式的20项安全测试通过，包括不调用关机、缺少关机程序仍可备份、显式禁止关机时拒绝授权等检查。

旧`stage2_finalize_C_20261022`已退出，原状态为`cancelled_by_sentinel`，旧阻断文件`DISABLE_C_AUTOSHUTDOWN`保留。新备份程序使用独立取消标记`DISABLE_C_FINAL_BACKUP`，因此旧阻断不妨碍本次自动上传。两份旧/新配置均标注关机未授权，不使用旧F收尾配置。

**OSS授权窗口**：启动时授权到北京时间10月23日20:45:33；23:29观察到已刷新至**10月23日23:29:08**，仍早于1000轮预计结束。请在10月23日晚（例如22时）或10月24日上午执行一次`oss login`，使新授权覆盖训练结束和上传窗口。训练不依赖OSS令牌，收尾读取更新后的授权；账户密码不写入日志或Git。最新观测见`handoff_check.json`，完训估计以实测速度为准。

启动审计另存`startup_audit.tar`并上传上述C的OSS目录，回读校验凭据见session的`startup_backup_receipt.json`。它只证明启动时的源码、协议和初始日志已保存，**不是C训练完成权重**。

用户变更后的程序、配置、撤销凭据及最新工作计划另存`backup_only_policy_20261022.tar`，上传回读证明见`backup_only_policy_receipt.json`。恢复本轮运行时优先采用这个最新策略包，不重新使用最初启动包内的旧关机配置。

后续顺序是C完成并归档，再按同一冻结推理/评价协议补齐F/C外部50例DSC/NSD；必要的F-aux和多种子任务另行安排。当前仍为阶段二2.4，不改变A–H定义或模块二方向。
