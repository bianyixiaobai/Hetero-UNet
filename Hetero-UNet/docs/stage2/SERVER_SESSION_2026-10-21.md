# 2026-10-21 新实例恢复与CUDA验收

## 当前状态

用户从旧实例镜像新建恒源云实例并授权继续阶段二。已连接、部署阶段二代码、完成真实Mamba CUDA与原CT完整尺寸训练步检查。**真实数据尚未恢复，未启动正式数据短程训练或1000 epochs主实验。**

SSH地址、实例密码及OSS凭据不写入本记录。浏览器连接、SSH会话与OSS授权是独立状态。

## 实例与环境

- 系统：Ubuntu 20.04.6；Python 3.11；PyTorch `2.2.1+cu121`；CUDA 12.1；Mamba `2.2.2`。
- GPU报告名称：NVIDIA GeForce RTX 4090，设备总显存`51,002,015,744`字节（约47.5 GiB），计算能力8.9。以实际查询为准，不按常见零售卡规格推断容量。
- 镜像保留了Torch/Mamba/causal-conv1d及nnU-Net所需依赖；原可编辑安装指向的项目目录已不存在，表现为`import nnunetv2`失败。
- `/hy-tmp`为空，新磁盘约90 GB。没有把系统镜像当成数据盘备份，也没有重新安装/升级Torch或Mamba。
- 从GitHub将`codex/bg-block-attnres-stage2`克隆至`/hy-tmp/Hetero-UNet-stage2`，初始提交`2955544`，随后`pip install -e Hetero-UNet/umamba --no-deps`恢复代码导入。

## 已通过的验证

1. 服务器原环境下65项单元测试通过。
2. 真实`mamba_ssm.Mamba`在GPU上完成前向/反向，输出和输入梯度均有限；没有使用CPU替身。
3. 从本机既有`3d_fullres_weights.tar`提取原`plans.json`，使用同一Dataset701标签定义，检查F/C/F-aux。
4. 完整patch=`[40,224,192]`、物理batch=2，真实卷积编码器/解码器、原MHA、Mamba、分割深监督、C/F-aux辅助边界监督、AMP、优化器训练步及验证步通过。检查梯度有限、训练缓存消费和原MHA配置保留。

| 模型 | 参数量 | 峰值训练已分配显存 | 峰值训练保留显存 | 结果 |
|---|---:|---:|---:|---|
| F | 90,378,406 | 约8.79 GiB | 约10.49 GiB | 通过 |
| C | 90,386,943 | 约9.06 GiB | 约10.81 GiB | 通过 |
| F-aux | 90,385,343 | 约9.05 GiB | 约10.77 GiB | 通过 |

使用合成图像与标签，只证明上述尺寸的CUDA/AMP计算链路可运行；未覆盖真实数据加载、长时间收敛、病例推理导出或DSC/NSD有效性。单步耗时包含不同初始化/预热状态，不能当作速度比较。真实训练可能产生额外内存占用。

新增诊断入口为`scripts/check_stage2_cuda.py`，必须提供原plans和dataset.json，禁止CPU替身回退；诊断结果写到独立目录，不与正式结果混用：

```bash
python scripts/check_stage2_cuda.py \
  --plans /hy-tmp/stage2_audit_2026-10-21/baseline_plans.json \
  --dataset-json /hy-tmp/stage2_audit_2026-10-21/dataset.json \
  --output /hy-tmp/stage2_audit_2026-10-21/cuda_full_patch
```

## 数据恢复的实际卡点

镜像与本机的`oss ls`均返回HTTP 401，恒源云首页连通性检查返回HTTP 200，SSH仍正常。两处OSS配置保存的是同一令牌，其公开过期时间为2026-10-13 07:46:20 UTC（北京时间15:46:20）。仅检查了到期元数据，没有打印或归档令牌值。

需要由用户在新实例终端执行`oss login`，登录恒源云平台账号刷新原授权。无需把平台密码发到聊天中，也不需要重新配置SSH。等待此授权期间没有猜测平台密码、修改OSS访问权限或创建持久凭据。

恢复后按以下顺序继续：列出并核验OSS备份 → 恢复训练数据、预处理、原plans及低内存导出补丁 → 核验50例训练/50例外评病例隔离与文件完整性 → F/C/F-aux真实数据smoke → 内部开发验证与配置冻结 → 主实验训练。保持原MHA、patch和物理batch。

## 记录位置

- 服务器审计目录：`/hy-tmp/stage2_audit_2026-10-21/`，含原plans、dataset.json、环境快照、GPU信息、CUDA诊断日志与JSON。
- 本机外层：`output/stage2_server_2026-10-21/`，保存归档plans及下载的诊断JSON/日志/环境快照；该目录被Git忽略。
- 工作计划在本机已有用户新增架构图说明，本次保留其未提交修改，不回退或覆盖。

阶段二2.3的“完整尺寸CUDA计算检查”已完成；真实数据训练/检查点与外评验收仍待数据恢复，不将本记录等同于阶段二整体完成。
