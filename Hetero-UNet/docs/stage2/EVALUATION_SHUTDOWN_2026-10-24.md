# F/C评价完成后汇总、备份及关机

> **历史执行记录。** 本页“当前/最新/待完成”仅指标题日期对应的运行时点；原事实与策略变更顺序保留。F/C训练评价现已完成，本轮任务见[阶段二入口](README.md)和[F-aux执行记录](FAUX_TRAINING_2026-10-29.md)。

**完成补注（2026-10-29整理）**：F/C最终预测、评价汇总和本页列出的四份TAR已在OSS回读及本机校验通过；服务器后续重新启动用于诊断和F-aux。本页19:48进展与关机配置属于当次执行历史，当前F-aux使用自己的独立收尾条件。

用户于2026-10-24明确更新安排：**C组推理完成后，做好记录与数据归总，然后关闭服务器。** 本次授权取代此前“评价完成后保持开机”的安排；此前的训练备份收据及推理protocol中的`shutdown_authorized=false`保留为历史快照，不回写。

## 当前进展

19:48核查：F的50例预测、空间信息核验及DSC/NSD评价已完成，DSC宏平均87.6068%，NSD宏平均91.2276%；C于19:42接续推理。F原始结果包已在OSS上传、回读，SHA256一致；独立JSON收据上传被平台以“不支持的文件类型”拒绝，不是预测包损坏或登录过期。

新增独立程序`finalize_stage2_evaluation_20261024.py`在`stage2_eval_finalize_20261024` tmux会话等待，不重启或中断F/C原队列，不修改模型、推理和评价算法。收尾以新的`evaluation_shutdown_config.json`及`finalization_state.json`为准。

## 实际关机条件

1. 原队列已退出、F和C都完成50例预测及DSC/NSD，队列锁可取得；任一组失败不关机。
2. 再核对两组各50个预测及原空间信息审计、文件大小/SHA256，两个原始CSV均为相同50例×13器官、数值有限且在[0,1]。
3. 从原始CSV独立汇总A/F/C宏平均，核对F/C汇总与原程序相符，生成总体比较、逐器官及逐病例配对百分点差；保留原始CSV。
4. 统一使用TAR归档，分别上传并下载回读四个对象，全部大小与SHA256一致：F预测/CSV/日志包，C预测/CSV/日志包，最终汇总/工作计划/代码/记录包，最终收据包。
5. 再核对当前实例、冻结protocol哈希、无训练进程及无取消标记；平台关机程序SHA256保持`30b9ea6f23173078ccfe57d9b15ffe8caf5ba1442477f32c2c170ea0e97771d7`，才允许调用。

OSS地址仍为`oss://backup/stage2_20261024/F_C_external50/`，新权威归档为：

- `F_external50_verified.tar`
- `C_external50_verified.tar`
- `F_C_evaluation_final_records.tar`
- `evaluation_final_receipt.tar`

原来的`F_external50.tar`等保留为历史。新收尾程序不再单独上传JSON，解决该平台的文件类型限制；原推理程序可能继续记录JSON上传失败，最终以以上四个TAR的回读收据为准。上传或校验失败会保留结果并每60秒重试，不关机。

## 汇总和记录位置

服务器目录`/hy-tmp/stage2_eval_20261024/`：

| 文件 | 用途 |
|---|---|
| `EVALUATION_RESULTS.md` | 完成后生成的A/F/C结果汇总与实验限制说明 |
| `final_comparison.json` | DSC/NSD总体均值、F−A/C−A/C−F百分点差 |
| `per_organ_comparison.csv` | 13器官×两指标的A/F/C及差值 |
| `final_paired_DSC.csv`、`final_paired_NSD.csv` | 相同50例配对结果 |
| `evaluation_final_receipt.json` | 四份归档中前三份的校验收据及关机授权，随后自身打包并回读 |
| `finalization_state.json` | 独立收尾进度、全部四份归档收据、关机请求状态 |
| `finalization.log`、`finalization_transfers.log` | 运行与上传/回读日志 |

训练权重、原始数据、预处理、F/C预测均不删除。F/C完整训练权重此前已归档OSS及本机。关机后即使`/hy-tmp`后续被清理，也可从OSS四份归档和已有权重包恢复；不会把数据只留在临时盘。

查看状态：`cat /hy-tmp/stage2_eval_20261024/finalization_state.json`。如用户后续撤销关机，创建`/hy-tmp/stage2_eval_20261024/DISABLE_EVALUATION_SHUTDOWN`即可取消；程序在真正关机前再次检查该标记。

10项隔离测试通过，覆盖未完训/缺组、CSV缺例与NaN、取消或协议变化、未授权、预测篡改、OSS失败、关机程序变化、收据不完整的阻断，以及四份归档验证后才调用关机的成功路径。测试中的关机调用全部模拟，没有实际关闭实例。
