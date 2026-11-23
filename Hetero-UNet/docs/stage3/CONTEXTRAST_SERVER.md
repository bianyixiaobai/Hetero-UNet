# C＋Contextrast 服务器启动说明

使用实验分支 `codex/bg-block-attnres-stage2`。以下命令在包含 `.git` 的仓库根目录执行，复用 C 的 CUDA/Mamba 环境和 Dataset701 预处理训练数据。先确认服务器上的 `nnUNet_preprocessed` 或项目默认数据路径指向原来的 50 例训练集及对应 plans。GitHub 提供源码，数据可从原数据盘或已有备份恢复。

```bash
git branch --show-current
git pull --ff-only origin codex/bg-block-attnres-stage2
git rev-parse HEAD
python Hetero-UNet/scripts/run_stage2.py --model C-Contextrast --results-root /hy-tmp/contrast_results --skip-final-validation --dry-run
```

拉取前应已在上述实验分支。dry-run 核对独立 Trainer、Dataset701、3d_fullres、fold all、seed 1234。启动器在实际加载 plans 后核对物理 batch 2 和 patch 40×224×192。在原 CUDA/Mamba 环境中分别启动以下两个 Python 进程，第一条成功后再运行第二条。

```bash
python Hetero-UNet/scripts/run_stage2.py --model C-Contextrast --results-root /hy-tmp/contrast_results --smoke --check-amp-resume
python Hetero-UNet/scripts/run_stage2.py --model C-Contextrast --results-root /hy-tmp/contrast_results --smoke --check-amp-resume --continue
```

第一进程完成训练步、在线验证与轮末日志，在 nnU-Net 写完 checkpoint_latest.pth 后退出。第二进程用正式启动器的原生恢复路径读取该检查点，核实恢复到下一轮、SGD 动量存在、CUDA AMP 已启用，再进行实际参数更新。两个进程均在 `stage2_run_manifests` 中记录分割与边界合计损失、加权对比损失、裁剪后梯度范数、参数变化、每步耗时与峰值已分配显存。第一份状态应为 `checkpoint_ready_for_fresh_process`，第二份应为 `completed` 且有下一轮的 `amp_steps`。若原生加载或 AMP 更新失败，先核查该运行的日志。两次通过后，在 tmux 中启动正式训练。

```bash
python Hetero-UNet/scripts/run_stage2.py --model C-Contextrast --results-root /hy-tmp/contrast_results --skip-final-validation
```

冒烟和正式训练各自有输出目录。正式训练从第 1 轮使用原 1000 轮学习率日程。前 100–150 轮检查显存、耗时、数值稳定性，并在 `training_log_*.txt` 的 `Organ contrast step coverage` 中查看逐器官的 GT 出现率、对比参与率及参与步中使用困难负样本的比例。正式训练中断后在同一条正式命令后加 `--continue` 恢复。

`--skip-final-validation` 跳过 fold_all 训练病例的整卷导出，逐轮在线验证和 best 权重选择照常进行。完训后取正式目录的 `checkpoint_best.pth`，按 C 的相同协议单独运行外部 50 例推理、DSC/NSD 评价与病例配对。只有这一步能给出本次精度差值。

查看“见到器官后实际参与损失的频率”时，按器官计算 `active_steps / gt_steps`。跨轮时先分别求和。分母为零表示未观察到该器官，报告为缺失。
