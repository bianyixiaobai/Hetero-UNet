# 本轮F/C训练完成后备份与自动关机

> **已被晚间的新安排取代：只完成F、备份后关机，C暂缓。** 当前会话、备份目标及已刷新的OSS授权见[F-only执行记录](F_ONLY_AUTOSHUTDOWN_2026-10-21.md)。下文保留作为初始部署历史，不应再用旧配置启动收尾任务。

用户已明确授权：保持当前训练不受影响，两组训练完成后自动上传结果，再关机停止实例计费。2026-10-21 15:59（北京时间）已启动独立后台收尾任务，**尚未执行关机**。

## 当前运行方式

- 原训练队列保持不变：`stage2_main_20261021`，F→C，各1000 epochs，seed=1234。
- 收尾会话：`stage2_finalize_20261021`。每60秒检查一次状态，使用独占文件锁避免重复运行。
- 收尾脚本部署在`/hy-tmp/stage2_audit_2026-10-21/finalize_stage2.py`，与正在训练的代码目录分开；训练源提交仍为`9fe70ac570a0e14ebce30daa4a16a22e49b4f3fd`。
- 普通调用只检查就绪状态；只有同时传入`--watch --arm-shutdown`才允许最终调用关机程序。本次已按用户授权启用。
- 等待期间只读小型状态文件，不导入Torch、不占用GPU、不更改训练参数或检查点。打包和传输仅在两组完成后执行，进程降低CPU调度优先级。

## 关机必须满足的条件

1. 实例hostname匹配本次配置；没有取消标记。
2. 队列明确为completed，序列是F/C、seed=1234、各1000 epochs；原训练tmux会话已结束，没有其他训练进程或GPU计算任务。
3. 两个正式`stage2_run.json`均为completed，模型、Trainer、源提交和冻结协议匹配；排除smoke结果。
4. 两组best/final都存在，CPU加载成功、网络参数非空、Trainer正确；final记录epoch=1000，best在1–1000范围内。
5. F、C、审计包及备份收据四个归档均上传成功，并分别从OSS重新下载，逐个核对文件大小与SHA256。
6. 临近关机再次检查队列、GPU/训练进程、取消标记和官方关机程序SHA256。

上传、校验、磁盘空间或训练异常都会阻止关机，保存错误并重试；不会为了停止计费而关闭未完成备份的实例。已通过校验的归档记录保存在状态文件中，后续重试可复用。服务器结果和已打包归档保留，只删除成功校验后的重复下载副本。

## OSS授权：仍需在训练结束前刷新一次

当前令牌公开到期时间是**2026-10-22 11:45:59（北京时间）**，早于本轮预计完成时间。旧令牌实际返回过HTTP 401；不能假定本次授权可一直使用。现有OSS CLI没有提供可用的免密码自动续期方式，也没有保存用户平台密码。

**建议2026-10-23上午重新执行一次`oss login`**，使授权覆盖预计当日下午/晚间的结束及上传窗口。若训练延后，仍以新令牌的实际期限为准。这个登录操作不会中断训练，收尾脚本每轮重新启动OSS命令，自动使用更新后的配置，无需重启训练或收尾会话。

```bash
# 在服务器终端执行；Username填恒源云平台账号，Password填平台密码
oss login
# 确认成功后检查个人数据可访问
oss ls -s oss://backup/stage2_20261021/
```

此密码是恒源云平台密码，与SSH实例密码不同。不要把密码或OSS配置写入项目/聊天记录。无需现在重复登录来延长两天后的窗口。

若届时授权仍失效，状态会显示`blocked_retrying`，错误细节见传输日志；**实例会继续运行并计费，直到上传校验成功后才能自动关机**。因此当前完成的是附带上述登录前提的自动收尾部署，不能承诺在不续期的情况下实现全程无人值守关机。SSH断开或本地电脑关闭不影响两个tmux会话。

## 备份内容与下载

最终备份目录：`oss://backup/stage2_20261021/final_verified/`。

| 对象 | 内容 |
|---|---|
| `F_seed1234_final.tar` | F完整Trainer结果目录，含best/final、日志、plans/dataset/fingerprint及运行清单 |
| `C_seed1234_final.tar` | C对应完整结果目录 |
| `final_audit.tar` | 训练提交的Git源码快照、冻结协议、恢复/检查/队列日志与脚本、收尾脚本和配置、检查点验证记录 |
| `backup_receipt.tar` | 前三个对象的大小、SHA256、回读校验时间及实验范围说明；此收据自身也回读校验 |

原队列自己的`F_seed1234_completed.tar`和`C_seed1234_completed.tar`仍会尝试上传到上一层目录，作为另一份备份；收尾任务不依赖这次尝试一定成功。原数据和A基线备份已在OSS，不再重复上传大数据集。启动探针`autoshutdown_setup_probe.tar`只证明部署时传输正常，不能当成最终训练结果。

本次自动关机的触发范围是**F/C训练及结果备份完成**。外部50例推理、0050大病例导出复核、DSC/NSD、F-aux完整控制及多种子实验均留待后续，不能把自动关机等同于阶段二所有研究任务完成。

关机后可以从本地或新实例执行：

```bash
oss login
oss cp oss://backup/stage2_20261021/final_verified/backup_receipt.tar .
oss cp oss://backup/stage2_20261021/final_verified/F_seed1234_final.tar .
oss cp oss://backup/stage2_20261021/final_verified/C_seed1234_final.tar .
oss cp oss://backup/stage2_20261021/final_verified/final_audit.tar .
# 解开backup_receipt.tar后，对照backup_receipt.json检查其余三包SHA256
```

## 查看状态与取消

```bash
cat /hy-tmp/stage2_autoshutdown_20261021/state.json
tail -n 30 /hy-tmp/stage2_audit_2026-10-21/finalize_watch.log
tail -n 30 /hy-tmp/stage2_autoshutdown_20261021/transfers.log
tmux ls
```

如要取消后续关机，执行下面的命令。它不影响F/C训练；脚本会在下一次检查时退出，临近关机也会复核：

```bash
touch /hy-tmp/stage2_audit_2026-10-21/DISABLE_AUTOSHUTDOWN
```

脚本运行命令归档如下；正常使用不必重复启动。不要删除状态文件来绕过错误，先核对原因。

```bash
python -u /hy-tmp/stage2_audit_2026-10-21/finalize_stage2.py \
  --config /hy-tmp/stage2_audit_2026-10-21/autoshutdown_config.json \
  --watch --arm-shutdown
```

## 验证记录

- 本机与服务器各16项测试通过，覆盖未完成队列、训练/GPU未退出、失败训练、缺少C final、smoke混用、错误hostname、final epoch不足、传输失败、回读损坏、临近关机取消、关机程序被替换、未授权关机模式等。所有测试的关机与OSS均为模拟，不执行真实关机。
- 服务器实际OSS探针：30,720字节，上传再下载SHA256均为`bce9aa7a79af9931c78e10db98b172a6b4b9a96ef11ac0a8237fd37d316c93929`。
- `/usr/local/bin/shutdown`与官方链接下载内容逐字节SHA256一致：`30b9ea6f23173078ccfe57d9b15ffe8caf5ba1442477f32c2c170ea0e97771d7`，2,177,600字节。仅静态核查，没有执行`shutdown`或可能触发关机的参数测试。
- 16:00部署后检查：训练和收尾tmux均存在，收尾状态`waiting_for_F_and_C`；F进入epoch79（从0计数），GPU约98%，正式模型代码和队列脚本未改动。
- 状态`shutdown_requested`仅表示脚本将调用关机命令；`shutdown_command_returned`仅表示命令正常返回，不独立证明平台已停止计费。最终平台电源/计费状态仍以控制台为准。

参考：[恒源云官方上传后关机流程](https://gpushare.com/docs/best_practices/train/#upload)。按用户“不影响当前训练”的要求，本次没有执行文档建议的真实测试关机。
