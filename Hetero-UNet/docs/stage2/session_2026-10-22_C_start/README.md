# C启动审计：2026-10-22 23时

详情见[C组训练记录](../C_TRAINING_2026-10-22.md)。此处为启动时快照，训练完成以服务器最终manifest及OSS最终收据为准。

| 文件 | 内容 |
|---|---|
| `preflight.json` | 原预处理253文件、14项代码、plans、环境、GPU与C从头初始化检查 |
| `cuda_check.json` | 当前V100的完整patch、batch=2、真实Mamba/AMP/梯度及验证检查 |
| `C_run_start_manifest.json` | 正式C的原始运行manifest快照，包含协议、run_id、指纹与完整参数 |
| `startup_check.json` | 首轮真实训练/边界/验证loss、运行状态、GPU、时间估计及启动备份证据 |
| `handoff_check.json` | 后续真实epoch推进、有限loss和最新OSS授权观测 |
| `startup_backup_receipt.json` | 启动审计上传、下载回读的大小及SHA256凭据；不是最终训练权重 |
| `run_C.sh` | 仅C、seed1234、1000轮，从头训练与队列状态记录 |
| `prepare_and_start_C.sh` | 文件及CUDA检查通过后才启动训练和独立收尾 |
| `autoshutdown_C_config.json` | 只等待C，上传回读校验通过后关机的配置 |

私有OSS诊断日志、完整pip环境清单留在实例和个人OSS中；不把账户配置、密码或大权重提交Git。模型代码固定9fe70ac，后续提交仅记录运维脚本与文档，不在训练过程中切换源码。
