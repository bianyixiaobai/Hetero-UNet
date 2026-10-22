# 当前执行安排：F完成后备份关机，C暂缓

> 2026-10-22晚已完成：F累计1000轮，OSS最终备份回读通过，自动关机程序已返回；用户随后无卡启动复核和整理存储。以[F完成与存储记录](F_COMPLETED_STORAGE_2026-10-22.md)为当前状态，下文为历史执行过程。

> 2026-10-22已迁移：F完成500轮后转至V100续训，旧4090项目另做完整保存以便释放。当前会话、备份和恢复位置见[最新记录](MIGRATION_2026-10-22.md)。下文保留原4090上的执行历史。

2026-10-21晚，用户要求先完成F、保存结果并关机，C等待之后更便宜的实例再训练。**本记录取代此前“F→C全部结束后关机”的安排**；不改变F的模型代码、种子、batch、patch、1000 epochs或best选择规则。

## 已实际切换的状态

- 北京时间19:46:59启用F-only控制器与收尾任务。
- F原进程继续运行，19:47已进入epoch240（从0计数）；没有中断、重启或重新加载权重。
- 原队列Bash正在等待F子进程；只对这个Bash的单个PID发送SIGSTOP，让它不能接续启动C。F训练进程与数据加载进程没有被暂停。
- F manifest为completed且F主进程真正退出后，控制器核对原Bash的PID、启动时刻、父PID、完整命令及暂停状态，通过Linux pidfd仅终止这个队列进程，确保不执行下一轮C。PID复用或身份不符时拒绝操作。
- 随后新的F-only状态文件标为completed，收尾任务加载F的best/final，检查final epoch=1000，打包、上传、回读SHA256一致后调用官方关机程序。
- 旧F/C收尾会话已停止，旧取消标记保留；旧`main_queue.json`留作历史记录，不再代表当前控制逻辑。

该方式避免编辑已经解析进入内存的Bash循环，也没有改动当前训练项目目录下的源码。服务器模型Git仍为`9fe70ac570a0e14ebce30daa4a16a22e49b4f3fd`、工作区干净。原队列的训练后打包步骤由新的独立收尾程序接管。

## 当前后台会话与文件

| 名称 | 用途 |
|---|---|
| `stage2_main_20261021` | F原训练所在会话；队列Bash暂停，F继续 |
| `stage2_F_only_control_20261021` | 等F退出、清理已暂停的原队列、写F-only完成状态 |
| `stage2_finalize_F_only_20261021` | 等F正式完成、验收权重、备份校验后关机 |
| `/hy-tmp/stage2_audit_2026-10-21/F_only_control.json` | 原队列/F主进程身份及切换原因，无凭据 |
| `/hy-tmp/stage2_audit_2026-10-21/F_only_queue.json` | 当前权威队列状态，sequence仅为F，C记录为deferred |
| `/hy-tmp/stage2_F_only_autoshutdown_20261021/state.json` | 当前备份、校验与关机状态 |

脚本为`scripts/hold_stage2_queue.py`和`scripts/finalize_stage2.py`。服务器独立部署名分别为`hold_stage2_queue.py`和`finalize_stage2_F_only.py`，均在审计目录；配置为`autoshutdown_F_only_config.json`。旧收尾配置不再使用。

## OSS授权已经更新

用户已重新执行`oss login`。19:48核验：OSS列目录正常，新令牌到期时间为**2026-10-22 19:42:39北京时间**，当前收尾状态也已读到新期限。预计F在10月22日13–15时完成，通常可覆盖本次结束及上传窗口；预估会随实际速度变化，若明显延后仍需核对期限。

不再要求10月23日上午为C刷新登录，因为本轮不跑C。没有保存平台密码或令牌值。授权/网络失败仍会保留服务器结果并重试，校验不通过不关机；异常情况下实例会继续计费。

## 当前备份目标

`oss://backup/stage2_20261021/F_only_verified/`

- `F_seed1234_final.tar`：F完整结果，含best/final、日志、plans、dataset/fingerprint、训练配置与运行清单。
- `final_audit.tar`：训练源码快照、冻结协议、恢复/验证记录、当前与历史控制脚本及配置、检查点验收记录。
- `backup_receipt.tar`：备份对象大小/SHA256/回读时间，明确仅F完成、C暂缓；收据自身也上传回读校验。

服务器不删除F结果或已封存归档，仅清理重复下载的校验副本。原始数据、A权重及旧50例预测的既有OSS归档保留；不重复上传可从OSS恢复的大数据。本轮不执行外部50例评价，不把F训练完成写成阶段二全部完成。

## 查看和取消

```bash
cat /hy-tmp/stage2_audit_2026-10-21/F_only_queue.json
cat /hy-tmp/stage2_F_only_autoshutdown_20261021/state.json
tail -n 30 /hy-tmp/stage2_audit_2026-10-21/F_only_control.log
tail -n 30 /hy-tmp/stage2_audit_2026-10-21/finalize_F_only_watch.log
```

如需只取消后续关机：

```bash
touch /hy-tmp/stage2_audit_2026-10-21/DISABLE_F_ONLY_AUTOSHUTDOWN
```

该取消不影响F训练，也不会恢复C队列。**不要对原队列手工发送SIGCONT**，否则可能恢复旧F→C循环。后续C在新实例单独启动，无需恢复这个队列。

## 之后换服务器继续C

1. 保留/下载本次F备份与收据；确认OSS数据及A基线归档仍在。
2. 新实例拉取实验分支，恢复同一数据、原plans/fingerprint和匹配的环境；如预处理数组未保留，依原plans重建，不能重新规划后直接与F混比。
3. 验收真实Mamba、完整patch、物理batch=2、AMP及短程训练。按实际吞吐和价格选择实例，不能只按小时价格比较成本。
4. 配置好数据/结果环境路径后，只启动`python scripts/run_stage2.py --model C --seed 1234 --results-root /hy-tmp/stage2_results --skip-final-validation`；不重跑F，不为换卡单独降低C的batch/patch。
5. C也用1000 epochs及同一冻结协议；后续再安排F/C外部50例评价与必要控制/多种子。

本轮没有购买新实例、构建新镜像或启动C完整训练。

## 验证

- 本机与服务器均24项安全测试通过：此前上传/关机门槛、F-only不依赖C、F进程退出前不能清理队列、仅暂停/清理队列、PID复用与未暂停队列拒绝清理等。
- 服务器独立Linux模拟队列实际验证：暂停模拟Bash后，模拟F心跳仍推进；模拟F退出后控制器清理队列，模拟C标记从未产生。只向自建测试进程发信号，没有触及真实F。首次探针因缺少初始training manifest失败，补齐测试夹具后通过；生产F manifest始终存在。
- 新OSS目录实际上传并回读40,960字节部署探针，SHA256一致，为`58d20a441244384b73fdbeda24b633f5a198f593694ad72fe026c68fcb8b4cba`；19:50再次确认F已进入epoch242，源码仍干净未变。探针不是最终权重备份。
- 切换后检查原Bash状态T、F主进程状态S（可运行等待，非暂停），后续epoch继续推进，GPU约95%。没有执行真实测试关机。
- 备份校验、官方关机二进制静态验证原则沿用[此前记录](AUTOSHUTDOWN_2026-10-21.md)，其中F/C目标、会话、OSS目录和旧授权期限均已被本记录替代。
