# F/C外部50例评价执行记录

北京时间2026-10-24 **17:16:40**启动后台队列，17:16:54进入F推理。顺序固定为**F推理→完整性检查→DSC/NSD→归档，再执行C的相同步骤**。这是阶段二2.4/2.5的评价工作，F/C训练已经完成；本页启动时尚无最终评价结果。

用户明确要求在本机完整权重备份和工作计划记录完成后启动。C完整包1,451,171,840字节已下载并核验SHA256，与服务器及OSS一致，见[C完成记录](C_COMPLETED_2026-10-24.md)及[本机收据](session_2026-10-24_C_completed/local_backup_check.json)。不重训、不改MHA，完成后保持开机。

## 冻结配置与公平比较

| 项目 | 本次执行 |
|---|---|
| 数据 | 同一Dataset701_AbdomenCT的imagesVal/labelsVal，共50例；病例ID与训练集无交集；图片和GT逐文件SHA256冻结 |
| 检查点 | 各自`checkpoint_best.pth`，F第1000轮，C第994轮；权重路径和SHA256写入protocol.json |
| 推理 | 3d_fullres、fold_all、原patch；逐patch前向batch=1，无TTA，滑窗步长0.5，Gaussian加权；不保存全体概率，不加后处理 |
| 内存限制 | 预处理/导出各1进程，NSD为2进程；NSD工作进程数仅影响并行度，不改变算法或容差 |
| 模型源码 | 训练时9fe70ac版本，14项指纹与训练记录一致；原训练仓库不切换、不修改 |
| 独立推理目录 | 复制nnunetv2包至本次runtime，只替换归档的低内存导出函数；F/C共用 |
| 补丁 | SHA256 `89896becba72549deb4a794c159dadffaaccde79972fa30f915ca00a641d0b40`；AST核对其他函数未变；4组FP16/FP32、不同spacing的重采样检查中分割及概率逐元素一致 |
| 评价实现 | 从原A的50例归档恢复，与当前DSC/NSD/SurfaceDice及并行NSD脚本SHA256一致；13器官顺序、毫米容差、空器官与Aorta/IVC/Esophagus切片裁剪规则全部沿用 |
| A参考 | 从同一归档复制A的DSC/NSD CSV；按病例对齐，保留此前49例原导出、0050低内存补丁的执行历史 |

每组推理退出后，先检查恰好50个预测、文件名与GT一一对应、shape/affine/spacing一致、整数标签在0–13且非全背景；通过后才运行指标脚本。两个CSV必须均为50例×13器官、病例唯一、无遗漏或非有限值。内部训练loss不作为外部评价结果。

## 路径、状态与持久记录

服务器工作目录：`/hy-tmp/stage2_eval_20261024/`，tmux会话`stage2_eval_20261024`。

| 路径（相对工作目录） | 用途 |
|---|---|
| `protocol.json` | 冻结病例、数据/权重/指标脚本SHA256、模型来源与推理参数 |
| `state.json`、`queue.log` | 队列当前状态、开始/更新时间、命令与异常；失败停止且保留已有结果 |
| `F/predict.log`、`C/predict.log` | 各组50例推理及导出日志 |
| `F/predictions/`、`C/predictions/` | 完整nii.gz预测及nnU-Net推理参数 |
| `F/prediction_validation.json`等 | 50例预测的SHA256、形状及空间信息核验 |
| `F/F_DSC.csv`、`F/F_NSD.csv`等 | 逐病例、逐器官原始指标 |
| `F/summary.json`、`C/summary.json` | 各组总体与逐器官均值、推理/评价耗时 |
| `paired_DSC.csv`、`paired_NSD.csv` | A/F/C逐病例均值与配对百分点差 |
| `comparison.json` | 同病例同器官宏平均、C−F百分点差；单种子探索结论 |
| `F/backup_receipt.json`等、`backup.log` | 结果包上传及下载回读SHA256核验 |

OSS结果目录：`oss://backup/stage2_20261024/F_C_external50/`。每组单独打包预测、CSV、日志、协议及运行脚本，上传后回读核验；最后上传汇总与配对表。若OSS授权失效，记录`backup_pending`并保留本机结果，另一组评价仍继续，不删除预测、不关机。最终汇总为独立小文件上传，完整预测以每组归档回读校验为准。

查看进度：`cat /hy-tmp/stage2_eval_20261024/state.json`，或`tail -n 30 /hy-tmp/stage2_eval_20261024/F/predict.log`；C启动后将路径F替换为C。检查完整导出数以`predictions/*.nii.gz`为准，日志里的“done”可能仅表示已送入导出队列。

该队列不修改组大小、辅助损失权重或检查点选择规则，不根据这50例结果挑选其他权重。单种子与F跨硬件续训的局限继续报告；F-aux及多种子确认仍按原计划另行安排。
