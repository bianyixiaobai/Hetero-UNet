# ND/R0：GitHub交付与服务器拉取

日期：2026-11-16。仓库为`bianyixiaobai/Hetero-UNet`，沿用实验分支`codex/bg-block-attnres-stage2`。本记录随ND/R0源码、测试、C参照合同和验收摘要一起提交。`main`不承载本轮新实验实现。

## 交付内容

包含C+ND/C+R0空间块、独立解码器wrapper与Trainer、启动器入口和协议保护、CPU测试、GPU检查脚本、文档与小型校验摘要。保留已有45项单元/回归及两组CPU集成检查证据。本次只是发布已测试代码，没有追加GPU效果结论。

原始CT、预处理数组、权重、预测、完整运行日志和本机诊断环境均不随本次Git提交上传。服务器必须恢复或保留这些数据；clone代码不能恢复训练数据。数据/权重继续沿用原OSS备份流程。

## 已有项目的服务器

用户租好并通知后，在**包含.git的仓库根目录**核对工作区和分支：

```bash
git status --short
git branch --show-current
git fetch origin codex/bg-block-attnres-stage2
```

若当前已是上述实验分支且工作区干净，再执行：

```bash
git pull --ff-only origin codex/bg-block-attnres-stage2
git rev-parse HEAD
```

将提交号与本机发布收据核对。若服务器有未提交修改或历史分叉，先审查差异和保留本机/远程补丁；不要使用reset --hard或强推覆盖。分支不匹配时先查明，不直接往main合并。

## 新服务器或独立代码目录

若没有可复用的干净checkout，可在尚不存在的代码目录克隆：

```bash
git clone --branch codex/bg-block-attnres-stage2 --single-branch https://github.com/bianyixiaobai/Hetero-UNet.git /hy-tmp/Hetero-UNet-spatial
cd /hy-tmp/Hetero-UNet-spatial
git rev-parse HEAD
cd Hetero-UNet
```

新代码目录不要求重新安装整个环境；先核对现有环境与原Mamba/CUDA依赖，再按实际缺项处理。若环境曾以editable方式安装其他checkout，需确认实际导入路径指向本次代码，避免训练启动和独立推理使用不同版本。

## 拉取之后

先恢复数据路径、核对C参照合同，再执行[本机实现记录第5节](SPATIAL_IMPLEMENTATION_2026-11-16.md#5-后续服务器操作准备)中的dry-run、真实CUDA和真实数据短程。本轮仅执行 C+R0 的检查、正式训练及原50例评价。汇总相对已有 C 的结果后，由用户决定是否安排 C+ND 等消融。

精确本地提交号、推送后的远端分支HEAD及匹配核验记录保存在本机`output/module2_spatial_20261116/github_delivery_receipt.json`。该收据在推送成功后生成，不把本记录自身的提交号硬编码进同一提交。
