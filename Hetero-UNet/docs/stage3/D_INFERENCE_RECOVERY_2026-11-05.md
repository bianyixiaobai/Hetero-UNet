# 2026-11-05 D组推理恢复

用户反馈D训练已完成、推理异常。按原授权恢复D外评→E训练/外评→归档回读→关机，不重训D。

## 已核验的训练产物

- 训练在北京时间11月5日12:05完成1000轮，最后train_loss −0.7848、val_loss −0.8106。内部fold_all指标不解释为独立外部表现。
- best/final均为epoch1000；原审计检查权重、优化器及1000条有限train/val记录通过。
- best：725112050字节，SHA256 `889e941b665b51dbac10321b3d6689ad9d4ccae37e0a94b7e0e9ea3674dc684d`。
- final：725113134字节，SHA256 `2629d2be2d8cb0266d2b6a6a1ad37a9374b582918101d058cc6d403893cd17cb`。
- `D_training.tar`已上传`oss://backup/stage3_20261103/D_E/`并回读通过；1451724800字节，SHA256 `b0450773b299aa82d49889fdbf9bbd488e197617441461d6b807ad702bc17af2`。

## 两处故障与修复

1. 推理console入口使用`importlib.metadata`查找nnunetv2发行包，但清空临时盘后原editable安装信息失效。训练显式设置PYTHONPATH所以能够运行，console入口未在开训前得到验证。这是部署漏检。
2. 在固定源码`/hy-tmp/Hetero-UNet-stage3/Hetero-UNet/umamba`执行`pip install --no-deps --no-build-isolation -e`，只修复包登记；`nnUNetv2_predict_from_modelfolder --help`退出0。没有升级依赖、改变模型或重采样逻辑。
3. 修复入口后的首次完整调用发现ZIP附带的两个macOS AppleDouble文件被误当成NIfTI输入；并非主机内存不足，不能因末尾“workers died”泛化报错就判为OOM。
4. 两个`._*.nii.gz`文件均核验AppleDouble魔数`00051607`，记录大小和SHA后移动至`recovery_20261105/quarantine/`，没有删除。imagesVal现有50个真实NIfTI名称与原protocol一致；恢复进程再次逐文件核验真实影像/GT哈希。

本机通用恢复脚本已新增安装登记和真实console入口预检，并跳过ZIP中的`__MACOSX`与`._`元数据。**这些改动未拉取到已经完成D训练的冻结checkout**；其代码、权重、数据描述符和推理runtime沿用原版本。

## 从哪个步骤继续

- 新恢复控制器：`scripts/resume_stage3_after_D_20261105.py`，部署在工作目录`recovery_20261105/`，不替换冻结训练源码。
- 恢复前检查D完整训练manifest、代码提交、全部命名源文件SHA、数据描述符、外评50例图像/GT、两份权重哈希以及原D训练包回读收据。
- 先恢复D的best/无TTA/滑窗0.5/单预处理导出进程推理，再按原DSC/NSD规则评价、检查50例预测几何和标签、上传回读。之后E从头1000轮，继承原D/E协议；最终所有归档通过后才关机。
- 后台tmux：`stage3_DE_recovery_20261105_v2`。恢复日志：`/hy-tmp/stage3_20261103/recovery_20261105/recovery.log`；预测日志仍追加`D_predict.log`，旧失败日志另存。
- 8项恢复/关机闸门测试通过，覆盖D未完训、源码不匹配、未验证备份等拒绝条件。

## 证据与后续状态

完整记录在本机`output/stage3_D_recovery_20261105/`及服务器`/hy-tmp/stage3_20261103/recovery_20261105/`；小型原始凭据随本目录session归档。恢复控制器在最终`records.tar`中包含恢复日志、安装日志、失败快照和隔离清单。

恢复后已实际导出4例预测。抽查前两例（FLARETs_0001、FLARETs_0002），预测尺寸分别为512×512×406、512×512×171，均与GT尺寸及affine一致；凭据见`session_2026-11-05/first_export_check.json`。这确认完整读取、模型推理、重采样和NIfTI导出链路已恢复。

当前50例推理仍在进行，不能将上述抽查解释为完整评价完成。D正式DSC/NSD及E结果，以后续summary和最终回读收据为准。代码与修复记录已在本机提交；GitHub推送暂因连接超时/重置失败，服务器已独立部署修复，不受该同步故障影响。
