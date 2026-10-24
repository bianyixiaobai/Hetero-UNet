# C完成核验记录

参见[C首轮训练完成记录](../C_COMPLETED_2026-10-24.md)。只确认训练、检查点和备份完成，外部50例评价尚未开展。

| 文件 | 含义 |
|---|---|
| `completion_check.json` | 全1000轮日志、loss组成、两份CPU检查点、F/C指纹及三个OSS对象重新核验 |
| `C_completed_manifest.json` | 原C运行manifest最终状态，保存原run_id与冻结协议 |
| `final_backup_state.json` | 15:59自动备份完成、关机未授权、保持运行 |
| `backup_receipt.json` | 最终权重/审计包大小、SHA256、检查点轮数及无关机授权 |
| `local_backup_check.json` | C完整包本机下载及SHA256核验完成凭据 |

本机大包和原始审计位于`output/stage2_C_completed_2026-10-24/`；本目录只保留轻量记录，不提交权重、账户凭据或私有配置。
