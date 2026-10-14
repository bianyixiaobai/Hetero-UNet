# 本次交付与恢复工作入口

交付日期：2026-10-14。

- 仓库：[bianyixiaobai/Hetero-UNet](https://github.com/bianyixiaobai/Hetero-UNet)。
- 发布分支：[codex/bg-block-attnres-stage2](https://github.com/bianyixiaobai/Hetero-UNet/tree/codex/bg-block-attnres-stage2)。
- 实现提交：[3d6d7f629ccc19e7db6a2b7cb24b49732d271288](https://github.com/bianyixiaobai/Hetero-UNet/commit/3d6d7f629ccc19e7db6a2b7cb24b49732d271288)。已成功推送，并通过`git ls-remote`核验远端分支为此提交。本记录随后以文档提交补充，最新HEAD用`git log -1 --oneline`查询。
- main未覆盖；原A网络/Trainer与实现前`04180d4`逐文件diff为空，原MHA保留。
- 最终复测：65项单元测试通过；6组F/C/F-aux的2D/3D CPU框架诊断通过；Python语法与Git空白检查通过。

已实现F、C、F-aux、边界分支/损失、分组深度聚合、独立Trainer、运行入口和追溯记录。服务器已关闭，因此没有开展真实CUDA检查、完整训练或新增50例评价。CPU诊断中的Mamba为显式替身，不作为性能证据。

恢复工作时先阅读[服务器操作手册](SERVER_RUNBOOK.md)：新目录克隆分支 → 恢复数据/plans和OSS导出补丁 → 原GPU环境安装代码 → F/C短程CUDA验收 → 开发划分调参与冻结 → 完整训练与外部评价。当前工作计划为外层根目录v1.8；[变更记录](CHANGELOG.md)解释各文件用途和设计选择。

本机附属产物在外层`output/`：`document_backups/`保存工作计划v1.7备份，`stage2_framework_check/`保存合成检查点/JSON报告/最终日志，`stage2_venv/`是CPU诊断环境。这些被Git忽略；正式代码与文档在分支上，服务器不需要上传整个output目录。用户CSV、图片、旧分析文档和备份未混入本次提交。
