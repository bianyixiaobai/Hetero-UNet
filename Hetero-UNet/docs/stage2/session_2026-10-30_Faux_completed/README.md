# F-aux完成核验：2026-10-30

参见[完成记录](../FAUX_COMPLETED_2026-10-30.md)。本目录保存小型证据，不存权重或医学预测图像。

| 文件 | 含义 |
|---|---|
| `training_audit.json` | 原程序对1000轮训练、best/final、有限性及指纹的验收 |
| `backup_receipt.json` | 四份最终TAR的OSS大小/SHA256回读证明 |
| `original_final_state.json` | 原程序20:33调用关机后的状态，不代表本次重启后的状态 |
| `independent_comparison.json` | 本机从原始CSV复算的宏平均、器官均值、病例方向与GT分层 |
| `independent_training_check.json` | 本机从完整日志复核1000轮与损失组合 |
| `local_weights_receipt.json` | 完整训练权重包下载本机并核对SHA256 |
| `local_final_backup_check.json` | 本机四份归档及包内best/final、50份预测的汇总核验 |
| `shutdown_review_20261030.json` | 本次补备份收尾后的关机调用记录 |

完整归档、预测、CSV与报告保留在本机`output/stage2_Faux_20261029/`；四份原归档在`oss://backup/stage2_20261029/F_aux/`。新增报告/计划/复核记录另行打包上传，不能替代四份原始归档。
