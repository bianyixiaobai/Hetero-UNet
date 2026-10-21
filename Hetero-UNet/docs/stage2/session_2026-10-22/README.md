# 2026-10-22 迁移与完整保存记录

本轮从4090的F第500轮检查点迁至V100继续累计1000轮，C未启动。用户进一步要求释放旧机前完整保存项目，因此还进行全目录清点和逐文件恢复。详情见[迁移与保存说明](../MIGRATION_2026-10-22.md)。

| 文件 | 含义 |
|---|---|
| `migration_pause.json` | 旧F暂停时间、500轮latest与486轮best的CPU加载结果 |
| `migration_source_objects.json` | 原迁移三包大小、SHA256、OSS地址 |
| `migration_restore.json` | 新机回读、恢复及plans/source校验；含65字符哈希笔误的纠正记录 |
| `migration_handoff_accepted.json` | 新机完成首个续训epoch、三个迁移包核对及审计上传回读证明 |
| `cuda_check.json` | 新V100三组合成批次完整尺寸CUDA/AMP诊断，不能当作C正式训练 |
| `resume_start_check.json` | 固定源码、14项指纹、续训入口、实际协议及7项归档测试 |
| `run_F_resume.sh` | 只从epoch500继续F并写当前队列状态，不启动C |
| `validate_and_resume_F.sh` | 新机恢复后做完整CUDA检查，再启动F及独立收尾 |
| `autoshutdown_F_resume_config.json` | 新机只F收尾、OSS目标、取消标记和官方关机程序校验值 |
| `complete_verification.json` | 1,091文件、33,756,412,733字节逐文件独立SHA256通过，缺失0，核验收据OSS回读通过 |
| `release_ready_check.json` | 完整保存通过后复查当前训练、源码、四组各50例原始数据及释放收据回读 |
| `old_instance_shutdown_check.json` | 完整保存后旧机官方关机返回0，随后旧SSH端口拒绝连接；V100继续训练 |

完整项目清单与逐文件映射保存在V100及OSS完整保存目录，本机另存`output/stage2_migration_2026-10-22/`。02:39完整核验、02:42活动训练复查与释放收据回读均通过，旧4090可以释放。这里不存大权重、压缩包或SSH/OSS凭据。私有root配置仅存在用户个人OSS与V100归档中，不把其内容提交Git。
