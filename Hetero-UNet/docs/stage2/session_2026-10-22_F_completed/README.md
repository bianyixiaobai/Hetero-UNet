# F首轮完成及数据盘整理凭据

> **历史运行快照。** JSON、日志与本页状态均描述该目录日期的运行，不代表实时进度。当前任务见[阶段二入口](../README.md)；原始证据不回写。

参见[F完成与存储说明](../F_COMPLETED_STORAGE_2026-10-22.md)。记录针对2026-10-22晚用户无卡启动后的复查；不会把训练完成写成外部评价完成。

| 文件 | 含义 |
|---|---|
| `finalizer_state.json` | 21:06完成F权重包、审计包与收据上传、下载回读，之后关机程序返回 |
| `backup_receipt.json` | 自动备份对象的OSS路径、大小、SHA256及两份1000轮检查点 |
| `F_checkpoint_check.json` | 本次CPU重新加载best/final，网络参数有限及哈希 |
| `precleanup_preservation_check.json` | 删除前1,091个旧项目恢复文件再次核验全部匹配 |
| `cleanup_result.json` | 实际释放字节、保留项和OSS删除收据回读证据 |
| `deleted_files.json` | 每个删除文件及其精确OSS恢复对象，旧文件含归档成员映射 |
| `post_cleanup_check.json` | 清理后仍存在的数据、A/F权重、源码及最终磁盘状态 |
| `local_backup_check.json` | F完整权重包及最终审计包已下载本机，SHA256均匹配 |

权重和`final_audit.tar`另存在本机`output/stage2_F_completed_2026-10-22/`；本目录只保存轻量验收记录，不提交私有账户配置或大包。旧`complete_verification.json`为凌晨完整迁移的历史记录，当前哪些重复文件已删除应查本次`deleted_files.json`。
