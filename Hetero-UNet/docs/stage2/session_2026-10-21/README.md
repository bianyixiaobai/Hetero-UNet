# 本次服务器运行归档

这里保存2026-10-21实际使用的恢复、短程检查和后台队列脚本，以及小型验证报告。脚本面向这次新建的空实例，不应在已有正式结果或正在训练的预处理目录上直接重跑；文件存在检查会保护已有权重/描述文件。

| 文件 | 用途 |
|---|---|
| `restore_download.sh` | 使用既有OSS登录下载三份数据ZIP、基线权重与旧预测备份，计算归档SHA256 |
| `restore_validate.py` | 安全解包、忽略macOS元数据、检查50/50图像标签配对与空间信息，恢复原plans/fingerprint及基线 |
| `restore_prepare.sh` | 本次初次恢复与预处理命令；最初因旧paths忽略环境变量而失败，路径修复后单独重新执行预处理成功 |
| `wait_preprocess_and_smoke.sh` | 核验50个训练npz/pkl完成后运行三组短程训练 |
| `run_real_smokes.sh` | F/C/F-aux真实数据各2 epochs×2训练/1验证迭代，独立smoke目录 |
| `run_prediction_smokes.sh` | 用三组smoke的best权重推理训练病例0001，检查输出形状、affine和标签范围 |
| `run_main_queue.sh` | 已启动的F→C正式队列，各1000 epochs；训练后本机归档并尝试OSS备份 |
| `cuda_check.json` | 原plans、合成批次、真实CUDA/AMP诊断；包含资源检查，不作为有效性或速度比较 |
| `prediction_smokes.json` | 三组真实检查点加载及单病例NIfTI导出通过 |
| `preprocess_complete.json` | 50例训练预处理完成，外部病例参与数为0 |
| `archive_sha256.txt` | 下载的五份源归档校验值 |
| `autoshutdown_config.json` | 本实例F/C完成门槛、OSS目标、取消标记与官方关机程序校验值；不含凭据 |
| `finalize_setup_check.json` | 16项服务器测试及真实OSS探针上传/回读SHA256通过的记录；未执行真实关机 |

正式训练代码固定在`9fe70ac570a0e14ebce30daa4a16a22e49b4f3fd`，启动时服务器工作区干净。后续本目录归档与文档提交不改变该次训练使用的模型代码。具体协议见上级`EXPERIMENT_PROTOCOL_2026-10-21.json`。

后台会话：`stage2_main_20261021`。状态文件：`/hy-tmp/stage2_audit_2026-10-21/main_queue.json`；总日志：同目录`main_queue.log`。各组的`stage2_run.json`记录真实训练状态、源文件与plans指纹，`completed`才表示该次请求完成。

本机完整日志/恢复清单保存在外层`output/stage2_server_2026-10-21/`；没有把密码、OSS配置或大权重放入Git。启动时的服务器审计包已上传`oss://backup/stage2_20261021/session_audit_start.tar`，不包含正在训练的最终权重。

自动收尾代码在上级项目`scripts/finalize_stage2.py`；服务器独立部署到审计目录，不变更正在训练的源码。收尾tmux为`stage2_finalize_20261021`，完整说明与OSS必须续期的限制见[自动关机记录](../AUTOSHUTDOWN_2026-10-21.md)。
