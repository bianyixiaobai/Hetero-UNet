# 阶段二服务器操作手册

更新日期：2026-10-29。本页为恢复和手动操作参考。真实CUDA短程、F/C正式训练及外部50例评价已完成；F-aux也已完成训练评价与OSS归档回读，见[完成核验](FAUX_COMPLETED_2026-10-30.md)。以下命令不表示待立即重跑的任务。

当前V100训练仓库固定为`9fe70ac`；新文档及调度脚本在实验分支另行提交。训练、推理期间保持原源码和环境，手动命令不与现有队列重复启动。总体进度见[阶段二入口](README.md)。

## 1. 在新目录拉取代码

本地代码分支与远端main历史及目录结构不同，不能在原官方代码目录直接合并。保留原`/hy-tmp/Hetero-UNet`数据/基线目录，在其旁边克隆：

```bash
cd /hy-tmp
git clone --branch codex/bg-block-attnres-stage2 --single-branch https://github.com/bianyixiaobai/Hetero-UNet.git Hetero-UNet-stage2
cd /hy-tmp/Hetero-UNet-stage2
git log -1 --oneline
git status --short
export STAGE2_REPO=/hy-tmp/Hetero-UNet-stage2
export STAGE2_PROJECT=$STAGE2_REPO/Hetero-UNet
```

新实例部署时按待恢复实验的记录选择提交；已有实例只有在相关训练/推理停止且确认新版本后，才使用`git pull --ff-only`更新。当前V100固定训练版本不执行此更新。不要将实验分支强行合并到旧main。如果`/hy-tmp`数据已清理，先从OSS恢复数据、旧plans、权重与低内存导出补丁；核对路径和完整性后再跑。Git仓库不提供这些大文件。

激活之前训练成功的GPU Python环境，然后安装本项目代码，不重新解析/升级依赖：

```bash
python -m pip install -e "$STAGE2_PROJECT/umamba" --no-deps
python -c "import torch, nnunetv2, mamba_ssm; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(nnunetv2.__file__)"
nvidia-smi
```

检查`nnunetv2.__file__`指向新目录。沿用服务器已验证的Torch/CUDA/Mamba组合；本机`output/stage2_venv`仅供CPU诊断，不上传作训练环境。

## 2. 恢复数据和冻结协议

以下数据路径沿用旧项目布局；恢复到其他位置时修改环境变量，不能指向空目录：

```bash
export nnUNet_raw=/hy-tmp/Hetero-UNet/data/nnUNet_raw
export nnUNet_preprocessed=/hy-tmp/Hetero-UNet/data/nnUNet_preprocessed
export nnUNet_results=/hy-tmp/stage2_results
export nnUNet_compile=false
export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=0
cd "$STAGE2_PROJECT/umamba"
python -m unittest discover -s nnunetv2/tests/unit_tests -p 'test_*.py' -q
cd "$STAGE2_PROJECT"
python scripts/run_stage2.py --model C --results-root /hy-tmp/stage2_results --seed 1234 --smoke --dry-run
```

使用原CT Dataset701的`nnUNetPlans.json`、`dataset.json`及预处理数据；正式/全尺寸smoke入口检查`3d_fullres`、`fold=all`、物理batch=2、patch=`[40,224,192]`，不符时直接停止。默认正式预算仍为1000 epochs、250训练/50验证迭代、lr=0.01；12层TransMamba与原MHA保留。`--dry-run`只展示路径，不能验证实际plans或GPU。

## 3. 首先验收真实CUDA短程训练

```bash
python scripts/run_stage2.py --model F --seed 1234 --results-root /hy-tmp/stage2_results --smoke
python scripts/run_stage2.py --model C --seed 1234 --results-root /hy-tmp/stage2_results --smoke
python scripts/run_stage2.py --model F-aux --seed 1234 --results-root /hy-tmp/stage2_results --smoke
```

每组2 epochs，每epoch 2训练迭代+1验证迭代；仍使用完整patch和batch=2，检查真实Mamba、AMP、有限loss/梯度、显存、保存best/final及验证输出。smoke位于`stage2_F_seed1234_smoke`等独立目录，绝不能拿它与完整A比较。再次运行相同目录时会保护已有checkpoint；确实续跑加`--continue`，重新查错可换一个`--results-root`。2 epochs运行完成后续跑不会自动增加epochs。

完整C需确认训练日志中有`segmentation`与`boundary_unweighted`；验证仅使用分割loss，best仍按原在线Dice选择。保留原物理batch；参数小不代表激活显存一定小，实际GPU占用另行记录。

## 4. 开发划分与冻结参数

外部50例不用于反复选择组大小/辅助权重等超参。开发使用内部数字fold并明确标记，例如：

```bash
python scripts/run_stage2.py --model C --fold 0 --development --seed 1234 --results-root /hy-tmp/stage2_results --smoke
```

去掉`--smoke`才按完整训练预算运行开发实验；本例只是内部划分短程查错。目录带`_dev_fold0`，不混入主实验。首轮代码组大小3、边界隐藏通道8、辅助权重0.1；如修改Trainer超参，先在开发集决定并记录Git版本，再给所有相关组冻结一致协议。开发运行允许单独plans，但其结果不能冒充原CT协议结果。

## 5. 正式训练参考：F、C、F-aux

先通过上述验收，再在tmux会话中运行以免SSH断线打断：

```bash
tmux new -s stage2
cd "$STAGE2_PROJECT"
python scripts/run_stage2.py --model F --seed 1234 --results-root /hy-tmp/stage2_results --skip-final-validation
python scripts/run_stage2.py --model C --seed 1234 --results-root /hy-tmp/stage2_results --skip-final-validation
```

单GPU按顺序运行；这些命令各为1000 epochs。`--skip-final-validation`只跳过训练结束后的整病例导出，**保留每epoch验证和best选择**。因为`fold=all`的最后整病例验证使用训练病例，不是外部50例，外评在下一节单独执行。断线后用`tmux attach -t stage2`；中断恢复用相同命令追加`--continue`，没有检查点时会报错而非从头重训。

若需要区分辅助监督和边界注入，正式训练F-aux：

```bash
python scripts/run_stage2.py --model F-aux --seed 1234 --results-root /hy-tmp/stage2_results --skip-final-validation
```

首次F/C结果可用于探索；关键结论随后按工作计划预先记录的同一组≥3种子确认A/F/C。已有A能否计入取决于种子和协议记录。新增seed时修改`--seed`，目录自动隔离，不改原入口硬编码参数。

## 6. 外部50例统一推理与评价

**已完成记录**：F/C已在独立`/hy-tmp/stage2_eval_20261024/`完成评价和归档；协议、日志及结果见[F/C评价记录](F_C_EVALUATION_2026-10-24.md)。本轮F-aux将复用相同runtime与评价实现。以下为手动参考，不与现有后台队列重复启动。

训练完成后，先设置实际外部图像和GT路径（不要把训练imagesTr作为外部测试集）：

```bash
export CT50_IMAGES=/实际恢复位置/外部50例/images
export CT50_GT=/实际恢复位置/外部50例/labels
export STAGE2_EVAL=/hy-tmp/stage2_eval
mkdir -p "$STAGE2_EVAL"
# 手动参考要求此runtime已按归档恢复并核验，内含原等价低内存导出补丁。
export STAGE2_RUNTIME=/hy-tmp/stage2_eval_20261024/runtime
export PYTHONPATH="$STAGE2_RUNTIME${PYTHONPATH:+:$PYTHONPATH}"
export nnUNet_results=/hy-tmp/stage2_results/stage2_F_seed1234
nnUNetv2_predict -i "$CT50_IMAGES" -o "$STAGE2_EVAL/F_seed1234" -d 701 -c 3d_fullres -f all -tr nnUNetTrainerMambaUNetBlockAttnRes -chk checkpoint_best.pth --disable_tta -npp 1 -nps 1
export nnUNet_results=/hy-tmp/stage2_results/stage2_C_seed1234
nnUNetv2_predict -i "$CT50_IMAGES" -o "$STAGE2_EVAL/C_seed1234" -d 701 -c 3d_fullres -f all -tr nnUNetTrainerMambaUNetBGBlockAttnRes -chk checkpoint_best.pth --disable_tta -npp 1 -nps 1
```

必须同时匹配Trainer和结果根目录。F-aux使用`stage2_F-aux_seed1234`及`nnUNetTrainerMambaUNetBlockAttnResBoundarySupervision`。推理器现在遵守fold/checkpoint参数；不再偷偷改成固定best，因此本手册显式传`-f all -chk checkpoint_best.pth`。保持相同滑窗设置（默认step_size=0.5）与无TTA；不要不同组单独调整推理协议。

原低内存导出补丁已于10月21日从OSS恢复，位置为`/hy-tmp/stage2_audit_2026-10-21/pred50_artifacts/backup50/export_prediction_lowmem_patch.py.bak`。10月24日已核对AST确认只改变重采样转换函数，并在FP16/FP32及两种spacing的4组检查中得到逐元素相同的分割和概率。F/C评价已在独立runtime中统一使用该补丁，本轮F-aux将复用，不覆盖训练仓库；A使用补丁处理0050的既有记录保留。若导出未完成，不把残缺病例列表直接求均值当作50例结果。

使用同一评价脚本、13器官顺序和NSD容差，例：

```bash
cd "$STAGE2_PROJECT"
python evaluation/abdomen_DSC_Eval.py --gt_path "$CT50_GT" --seg_path "$STAGE2_EVAL/F_seed1234" --save_path "$STAGE2_EVAL/F_seed1234_DSC.csv"
PYTHONPATH="$STAGE2_PROJECT/evaluation:${PYTHONPATH:-}" python "$STAGE2_REPO/evaluation/nsd_parallel.py" --gt_path "$CT50_GT" --seg_path "$STAGE2_EVAL/F_seed1234" --save_path "$STAGE2_EVAL/F_seed1234_NSD.csv" --workers 2
```

把以上F路径统一替换为C/F-aux路径评价对应配置。并行NSD沿用现有脚本及裁剪/空器官规则；`PYTHONPATH`指向内层evaluation以找到`SurfaceDice.py`。确认50个完整预测均有对应GT、形状/空间信息一致、两个CSV各50例×13器官、没有遗漏/无穷/NaN；保留逐病例配对结果。后处理如有变化须统一并记录。

## 7. 必须保存的运行产物

每组实际路径形如：

```text
/hy-tmp/stage2_results/stage2_C_seed1234/
  Dataset701_实际名称/
    nnUNetTrainerMambaUNetBGBlockAttnRes__nnUNetPlans__3d_fullres/
      fold_all/
        checkpoint_best.pth / checkpoint_final.pth
        stage2_config.json
        stage2_run.json
        stage2_run_manifests/各次调用.json
        stage2_console_各次调用.log
        原训练日志和曲线
```

入口记录Git commit/dirty、种子、plans和关键源代码SHA256、训练预算、checkpoint及调用状态；不会哈希所有图像内容，数据完整性需另外归档。保存环境`pip freeze`、GPU信息、峰值显存、预测文件、CSV及OSS校验清单。关机前将这些文件与导出补丁备份到持久存储。`status=completed`表示请求的训练调用完成，不代表外部评价或阶段二整体已经验收。
