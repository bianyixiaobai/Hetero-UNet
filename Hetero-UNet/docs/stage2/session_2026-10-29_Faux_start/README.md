# F-aux启动核验快照：2026-10-29

本目录随代码保存小型启动证据，完整说明见[F-aux执行记录](../FAUX_TRAINING_2026-10-29.md)。这些文件不能作为1000轮完训、外部评价或最终上传完成的证明。

| 文件 | 含义 |
|---|---|
| `preflight.json` | 253份预处理、14项源码、50例外部输入/GT核验通过，绑定当次完整协议哈希 |
| `smoke_check.json` | 完整batch/patch的CUDA/AMP单步训练与验证；边界头梯度存在、无引导注入 |
| `startup_check.json` | 首轮真实训练完成、源码干净及后台队列正常运行的时间点快照 |
| `workflow_tests.log` | 自动收尾保护的5项隔离检查，传输与关机均使用模拟 |
| `startup_backup_receipt.json` | 启动材料TAR已上传OSS并回读匹配SHA256 |
| `local_startup_check.json` | 同一启动包下载本机并核对SHA256 |

完整协议和启动包在本机`output/stage2_Faux_20261029/`。训练版本固定9fe70ac，调度与文档分支可以另有提交；本次原始JSON和日志保持签发时内容。
