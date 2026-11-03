# 2026-11-03 D/E服务器执行记录

用户授权：服务器已启动，开展模块二D/E消融，记录权重、数据、日志，全部训练/评价/归档上传校验完成后关机。没有自动开展G/H或其他外部方法训练。

## 环境和恢复

- 实例：`I2bde78031f00701d9e`，SSH端口56465，Tesla V100-SXM2-32GB。
- Python3.11、PyTorch2.2.1+cu121/CUDA12.1，真实Mamba导入通过；tmux和已知平台关机程序存在。
- 初次读取`/hy-tmp`为空、容量50G；用户扩容后再次确认150G。原临时数据盘已清除，从OSS恢复，不能把上次归档完成当作本次文件仍在实例上。
- 训练代码独立放在`/hy-tmp/Hetero-UNet-stage3/Hetero-UNet`，实际队列代码提交为`08218c634736a0123d8bfef4dc43235f9839932c`；不覆盖旧基线结果。
- 复用原压缩预处理归档及3个原始数据ZIP，分别按原大小和SHA256验证；全部50例外部图像/GT再按原协议逐文件核验。训练前保留原plans和数据描述符。
- 首次恢复在评价依赖处停止：`metrics_code`位于单独的外部评价归档，而不在final_records包。已从该包核验SHA256并补回；旧失败状态留作`state_restore_attempt1.json`，部分runtime改名保留。训练尚未启动时重试恢复，无正式权重被覆盖。
- 后续通用恢复脚本已在本地修正为自动补齐该分包，提交`976623b`；GitHub推送曾遇网络连接失败。运行队列仍使用上述固定08218c及已核验的依赖恢复，不在活跃训练中拉取代码。

## 队列与协议

后台tmux会话`stage3_DE_20261103_retry`，主日志`/hy-tmp/stage3_20261103/pipeline.log`、状态`state.json`。执行顺序：

1. 归档数据恢复与校验；D/E完整patch、batch2的真实CUDA/AMP合成前后向。
2. D/E真实数据各2 epochs短程，独立smoke结果目录。
3. D从头1000 epochs → best/final及训练日志核验 → OSS回读 → D外部50例推理评价 → 归档回读。
4. E同协议从头1000 epochs → 权重核验与上传 → 外部50例评价与上传。
5. 汇总A/F/C/D/E逐器官、逐病例和配对数据，归档源码、runtime、环境、日志与最终收据；所有收据通过后检查GPU和进程为空，才执行关机。

seed1234、fold_all、3d_fullres、物理batch2、patch `[40,224,192]`、原MHA、原best规则；推理关闭TTA，滑窗0.5，原低内存导出补丁，DSC/NSD原规则。E保留完整C的分组、边界监督/注入及辅助权重0.1，不加载C权重微调。

## 输出与收尾

- 权重：`/hy-tmp/stage3_results/stage3_D_seed1234/`、`stage3_E_seed1234/`。
- 执行审计、预测、CSV及日志：`/hy-tmp/stage3_20261103/`；每次训练另有stage3配置和manifest。
- OSS：`oss://backup/stage3_20261103/D_E/`，包括startup、D/E training/evaluation、records和delivery归档；每个包上传后下载重算SHA256，返回副本核验后才清理。
- 失败不关机；OSS授权/网络失败保留结果并重试，每组上传校验后才继续下一阶段。取消关机可创建`/hy-tmp/stage3_20261103/DISABLE_SHUTDOWN`。
- 当前OSS授权截至北京时间11月4日15:45；用户已答复会刷新。需覆盖每组完训上传窗口，不能仅凭开始时登录有效就保证最终上传成功。

## 验证边界

4项新增关机拦截测试通过：取消标记、可执行文件哈希改变、缺收据、回读失败均阻断。

北京时间16:07完整CUDA检查通过：D/E均为原patch、物理batch2、真实Mamba、AMP，峰值已分配显存分别11,882,438,656与12,204,496,384字节。随后D/E真实数据短程各2轮通过，日志包含有限train/val loss和Training done；E边界loss分量已记录。startup.tar的OSS上传回读通过，原始凭据另存本机与本目录session子目录。

**16:08:19队列进入D正式1000轮；16:08:27开始零基Epoch 0，学习率0.01，GPU工作正常。** D/E正式训练、外部评价及最终收尾尚未完成；本启动说明不替代最终完成收据，正式DSC/NSD尚未生成。
