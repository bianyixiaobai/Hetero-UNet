# 模块二运行说明

日期：2026-11-03。以下供服务器租好后使用；本次未执行服务器命令。

从包含`umamba/`和`scripts/`的项目目录执行；激活原Mamba环境，设置已恢复数据的`nnUNet_raw`及`nnUNet_preprocessed`路径。代码拉取到独立目录，不覆盖旧实验产物。Python与PyTorch/Mamba/CUDA版本须先核对原环境。

先检查命令解析，不启动训练：

```bash
python scripts/run_stage3.py --model D --seed 1234 --results-root /hy-tmp/stage3_results --dry-run
```

真实CUDA短程（原CT plans、patch、物理batch2；2 epochs，每轮2次训练/1次验证，仅验可运行性）：

```bash
python scripts/run_stage3.py --model D --seed 1234 --results-root /hy-tmp/stage3_results --smoke
python scripts/run_stage3.py --model E --seed 1234 --results-root /hy-tmp/stage3_results --smoke
```

短程通过且显存可承受后，在tmux中先跑D正式1000轮：

```bash
python scripts/run_stage3.py --model D --seed 1234 --results-root /hy-tmp/stage3_results --skip-final-validation
```

`--skip-final-validation`只省略fold_all训练病例的额外全体积导出，保留每轮验证和best选择。外部50例推理及DSC/NSD是单独步骤，使用D的Trainer、`-f all -chk checkpoint_best.pth --disable_tta -npp 1 -nps 1`，沿用既有低内存导出与评价规则。

正式E命令相同，仅将`--model D`换为`--model E`；按计划在D验证后再安排。续训使用`--continue`，禁止把D的检查点放进E目录或反过来使用。

结果父目录自动隔离为`stage3_D_seed1234`或`stage3_E_seed1234`；短程另加`_smoke`。每次运行保留`stage3_run_manifests/`、`stage3_run.json`、独立控制台日志及源码/描述符SHA256，Trainer另写模块配置。

本入口不自动上传OSS、不自动关机、不串行接续E或外部推理。服务器就绪后，再按当次用户安排部署备份、回读校验和收尾流程，避免沿用旧实例的凭据及关机脚本。

本机CPU诊断（明确使用Mamba替身，不是服务器验收）：

```bash
python scripts/check_stage3_framework.py --output /path/to/isolated/stage3_cpu_check
```
