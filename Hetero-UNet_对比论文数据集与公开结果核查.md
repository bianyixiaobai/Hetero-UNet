# 对比论文数据集与公开结果核查

核查日期：2026-11-03。目的：找出可以复用公开指标、预测或训练权重的对比，减少重复训练。未连接服务器、未启动训练，也未下载全部公开权重或184.7 MB预测压缩包。

## 1. 结论：确实有能省下训练的资源

**最直接的是U-Mamba及其在同一基准上运行的经典对照，另有Hetero-UNet原文结果。** 不需要为了引用这些公开结果而把所有方法从头训练一遍。需要区分三种来源：原论文报告值、公开权重推理后的实测、公开预测重评后的实测。

本次找到的证据不仅是数据集同名：

- U-Mamba原文§3.1/Table 1明确采用FLARE2022的50例训练、50例测试、13器官；§3.2–3.3为patch `[40,224,192]`、batch2、1000 epochs、关闭TTA。[原文](https://arxiv.org/html/2401.04722v1)
- U-Mamba官方数据入口中的`U-Mamba-data`，正是Hetero-UNet官方README链接的同一个Drive文件夹，ID均为`18QSSiABS8H3qtx8SZA6RQb3aH1nbc3iF`。[U-Mamba官方仓库](https://github.com/bowang-lab/U-Mamba)、[Hetero-UNet官方仓库](https://github.com/ZhilingYan/Hetero-UNet)、[共享数据目录](https://drive.google.com/drive/folders/18QSSiABS8H3qtx8SZA6RQb3aH1nbc3iF)
- 该目录的`Dataset701_AbdomenCT/labelsVal`公开文件名共50个，恰为本项目评价ID `FLARETs_0001.nii.gz`至`FLARETs_0050.nii.gz`。[公开标签目录](https://drive.google.com/drive/folders/1cNhKCIPFNI0wx4CZGaEoCVNSzBqoXhnM)
- 下载的U-Mamba官方DSC/NSD脚本从`label_tolerance`起至文件结尾与本机脚本一致，`SurfaceDice.py`归一化换行后全文一致，包括空类计分、GT spacing、label 5/6/10的z范围规则。[官方DSC脚本](https://github.com/bowang-lab/U-Mamba/blob/main/evaluation/abdomen_DSC_Eval.py)、[官方NSD脚本](https://github.com/bowang-lab/U-Mamba/blob/main/evaluation/abdomen_NSD_Eval.py)

**证据边界**：本次确认公开分发来源、病例名和评价代码一致，没有重新下载全部训练/测试影像与GT逐体素对照，也没有独立验证作者每次训练的实际文件哈希。因此可作为同基准文献对照，不能写成本项目已重新运行了所有外部方法。

## 2. 可以先直接引用的数值

### 2.1 U-Mamba原作者统一基准结果

下表从[作者项目页Table 3图片](https://u-mamba.github.io/static/images/Table3.png)人工目视核对，0–1值转换为百分数。来源为原文Table 3；本表只摘录均值，不将原表的±项解释为多种子标准差。

| 模型 | Abdomen CT DSC % | NSD % | 来源性质 |
|---|---:|---:|---|
| nnU-Net | 86.15 | 89.72 | U-Mamba原文报告 |
| SegResNet | 79.27 | 82.57 | U-Mamba原文报告 |
| UNETR | 68.24 | 70.04 | U-Mamba原文报告 |
| SwinUNETR | 75.94 | 76.63 | U-Mamba原文报告 |
| U-Mamba_Bot | 86.83 | 90.49 | U-Mamba原文报告，瓶颈版 |
| U-Mamba_Enc | 86.38 | 89.80 | U-Mamba原文报告，编码器版 |

上述SegResNet/UNETR/SwinUNETR是U-Mamba作者在该基准上适配运行的结果，不能写成这些方法各自原始论文的FLARE成绩，也不能认定它们代表后来所有改进版本的最高表现。不同优化器等架构适配应遵循来源说明。[U-Mamba §3.3](https://arxiv.org/html/2401.04722v1)

### 2.2 Hetero-UNet本地原论文Table 1

来源：`E:/文献阅读/医学/Hetero-UNet.pdf`，PDF第5页§4.1–4.2、第6页Table 1，已提取文本并目视核对。原文报告采用与U-Mamba/nnU-Net相同实验设置；[公开论文入口](https://openreview.net/pdf?id=lLc8qTQxuO)。

| 模型 | CT DSC % | NSD % | 来源性质 |
|---|---:|---:|---|
| nnUNet | 86.0 | 89.3 | Hetero-UNet原文报告 |
| SegResNet | 79.4 | 82.9 | Hetero-UNet原文报告 |
| UNETR | 68.3 | 70.4 | Hetero-UNet原文报告 |
| U-Mamba（原表未区分Bot/Enc） | 86.2 | 89.8 | Hetero-UNet原文报告；不擅自补变体名 |
| Hetero-UNet | 86.7 | 90.5 | Hetero-UNet原文报告 |

两篇来源的对照数字有差别，不是都能视为四舍五入差异；不能为了突出本文优势，逐行选择较低的来源。建议以U-Mamba Table 3作为一组来源完整的经典参照，再增加Hetero-UNet原文行，并在来源列明确两篇文献。也可以分表展示。

本项目A为自行训练的86.7638%/90.5529%，是内部改进的直接基线；F/C/F-aux沿用原实测。**不能用原论文A替换本项目A来计算模块增益。**

## 3. 比引用更有用：作者确实公开了模型和分割结果

### 3.1 精确入口

| 资源 | 本次确认内容 | 可节省的工作 |
|---|---|---|
| [U-Mamba总入口](https://drive.google.com/drive/folders/1DmyIye4Gc9wwaA7MVKFVi-bWD2qQb-qN) | 包含数据、模型和视频教程；数据子目录与Hetero-UNet共享 | 不必重新寻找不同版本数据 |
| [模型总目录](https://drive.google.com/drive/folders/1rwfK_rzEXRB-cysYapmBnAmNUscADwj4) | 有CT/MR等任务目录、`old-v0`、`readme.txt`和预测ZIP | 有机会不训练，直接推理或重评 |
| [old-v0腹部CT目录](https://drive.google.com/drive/folders/1ITlpPkg7V43fky9yGzfHFbfFTRrUcAux) | nnUNet、SegResNet、UNETR、SwinUNETR、U-MambaBot、U-MambaEnc六个Trainer目录 | 优先核验原论文对应权重，然后只推理；具体权重文件尚未逐个下载/加载 |
| [新版腹部CT目录](https://drive.google.com/drive/folders/1IEp0gjRTzDlRlo-KqB3YyXo0XDs5-zWK) | 同样六个模型目录；SegResNet含`2xFeat`变体，部分目录已看到`fold_all`、plans和dataset描述符 | 可作为另行标明版本的公开模型比较 |
| [U-MambaSegResults.zip](https://drive.google.com/file/d/1qlzTym3YdyCt3eR8J90h636it4cCDCt8/view) | 公开目录显示约184.7 MB；作者说明其为新版实现对应分割结果 | 若包内有完整所需CT预测，可跳过训练和GPU推理，只用CPU重评 |
| [作者readme.txt](https://drive.google.com/file/d/1o2EoCxe6TMOgg6EnIe0ezLu3G1RZSmHV/view) | 明确区分原论文对应old-v0和更新实现；新版涉及模型规模与残差块调整 | 防止错误声称新版预测复现了旧论文指标 |

**重要：发现了压缩包不等于已核验其内部。** 当前没有下载并解包ZIP，因此还不能保证六个模型、CT任务和每组全部50例预测都在里面；也未核验NIfTI几何、标签范围或预测内容。目录名不等于已经成功加载了权重。

### 3.2 最省算力的使用顺序

1. **马上可做文献对照**：直接引用第2节的原文报告结果，清楚标注出处，不要求先训练经典方法。
2. **优先检查公开预测ZIP**：核对模型版本、完整50例、形状/affine/spacing、标签映射和公开GT内容。若通过，用本项目同一脚本重算DSC/NSD并保存50×13 CSV。此路线不需要GPU，也不需要从头训练。
3. **预测缺失再用公开权重推理**：优先匹配old-v0代码/Trainer/plans，确认训练数据不含测试病例，再统一无TTA推理和评价。这需要GPU推理，但仍免训练。不能把新版权重塞进旧实现假装相同。
4. **既无预测也无合适权重才从头训练**：尤其是本项目自定义G/H及内部控制，外部别的数据集成绩不能替代。

公开预测重评的论文表述可写为：“采用作者公开分割结果，在本研究评价流程下重新计算指标，未重新训练该模型。”只引用表格时则写“数值引自原文”。前者可得到配对病例指标，后者不能凭均值推导配对显著性或逐病例图。

## 4. 全部16项候选逐项判定

“可引用”在这里指可以用作**当前FLARE50例/13器官基准的文献性能对照**；不同任务论文当然仍可放Related Work，但不能直接混入这张性能排名表。

| 原候选 | 原论文/已核对适配使用的数据与设置 | 与当前CT基准是否匹配 | 本轮处理结论及来源 |
|---|---|---|---|
| nnU-Net | 原始方法覆盖多任务；U-Mamba明确在本基准重训 | 有匹配基准结果 | 可引用U-Mamba作者的适配结果；公开模型目录有对应条目。[§3.3](https://arxiv.org/html/2401.04722v1) |
| SegResNet | 原始研究为BraTS；U-Mamba在FLARE50/50适配 | 有匹配基准结果 | 可引用适配报告值，不拿原始BraTS成绩比较；注意新版`2xFeat`。[§3.3](https://arxiv.org/html/2401.04722v1) |
| UNETR | 原始论文含BTCV/MSD；U-Mamba另有FLARE适配结果 | 有匹配基准结果 | 可用同基准适配值，原论文BTCV值不直接替代。[§3.3](https://arxiv.org/html/2401.04722v1) |
| Swin-UNETR | 原文主任务为脑肿瘤；U-Mamba另有FLARE适配 | 有匹配基准结果 | 可用U-Mamba Table 3值，无需为了引用而重训。[§3.3](https://arxiv.org/html/2401.04722v1) |
| U-Mamba | FLARE2022，50训练/50测试，13器官，DSC/NSD | **公开分发和协议匹配最充分** | Bot/Enc分列；有原文、模型和预测资源，版本须区分。[原文](https://arxiv.org/html/2401.04722v1) |
| Hetero-UNet | 沿用U-Mamba腹部任务及数据入口，1000轮、DSC/NSD | 匹配当前基座 | 原文可引用；本项目A已实测，不需要再做一次首轮A。本文公开仓库未在本次找到同样明确的预测包。[仓库](https://github.com/ZhilingYan/Hetero-UNet) |
| SAMA-UNet | 所核v1为Synapse/BTCV 30卷、18训练/12测试，2D，宏平均取8器官 | **不匹配** | 85.38/87.82不能与FLARE13器官平均排名；不是改成8器官求平均就能解决病例不同问题。[§4.1与Table 2](https://arxiv.org/html/2505.15234v1) |
| Mamba-HoME | PANORAMA、AMOS CT/MRI、FeTA、MVSeg等；AMOS腹部平均15器官，另有大规模预训练 | **不匹配** | AMOS-CT与FLARE不是同一批病例；即使都分腹部器官也不能替代。[§3.1、附录F Table 23](https://arxiv.org/html/2507.06363v4) |
| DAMamba | ImageNet、COCO、ADE20K等通用视觉任务 | **不匹配** | 没有该原文可直接引用的FLARE分割分数；不把同名医学变体当原方法。[作者仓库](https://github.com/ltzovo/DAMamba) |
| XAttnRes | Synapse 30卷、18/12，8器官；另有ColonDB/ClinicDB/ISIC，U-Net/EMCAD骨干 | **不匹配** | 原文成绩不能代替G/H。若要论证Hetero基座上的相对机制贡献，仍需适配和本项目实验。[§4.1–4.2](https://arxiv.org/html/2604.03297v1) |
| DM-SegNet | Synapse与BraTS2023；文中70%/10%/20%随机划分，DSC/HD95 | **不匹配** | 85.44不是本项目FLARE结果，也不能把HD95当NSD。[§4.1–4.3](https://arxiv.org/html/2506.05297v1) |
| EViM相关混合模型 | Synapse 30卷、8器官、224×224切片；另有ACDC，DSC/HD95 | **不匹配** | 已核原文§4.1–4.2全文XML；82.67等数字不可直接排名。[原文](https://www.mdpi.com/1424-8220/25/21/6785) |
| MambaCAFU | Synapse18/12的8器官及BTCV13类，另有ACDC/ISIC/GlaS/MoNuSeg | **不匹配** | BTCV的13类不等于FLARE的13类：其包含门静脉/脾静脉类别，FLARE包含十二指肠；病例也不同。[§4.1–4.2](https://arxiv.org/html/2510.03786v2) |
| TranSamba | BraTS、KiTS、LASC；弱监督体积分割 | **不匹配** | 数据、任务、监督方式不同，只作相关文献，不用公开成绩替代全监督FLARE对照。[会议页面](https://papers.miccai.org/miccai-2026/0479-Paper4858.html) |
| DAMamba-UNet3D | BraTS2020，369例、四模态、ET/TC/WT、五折 | **不匹配** | 不是腹部CT；现有权重/结果不能直接替代本基准。[§2.1](https://arxiv.org/html/2607.22718v1) |
| BiSegMamba | 颈动脉CTA、BraTS2023、ACDC、AMOS-CT；AMOS15器官，DSC/HD95 | **不匹配** | AMOS数据和标签范围不同，没有从所核版本确认FLARE50例结果。[§IV-A/B/C](https://arxiv.org/html/2605.30972v1) |

**数量解释**：原16项中6项有本项目相同公开基准的报告值（其中4项通过U-Mamba/Hetero论文中的适配实验获得），不代表6篇各自原始论文都使用本项目数据。Bot/Enc拆开后可形成更多模型行，但不是新增独立论文。

## 5. 额外找到的同基准文献线索

[CPNet: 3D Semantic Relation and Geometry Context Prior Network for Multi-organ Segmentation，ECAI 2024](https://andrewchiyz.github.io/vision.medseg.cpnet/)在[原文§4.1及Table 1](https://ebooks.iospress.nl/pdf/doi/10.3233/FAIA240483)说明沿用FLARE50训练/50测试、13器官，报告CT 86.95% DSC/90.78% NSD。可作为**额外同基准候选**；本次原文在线访问部分受限，数据/指标细节来自出版方索引文本与作者项目页，未完成逐病例/预测核验。它不自动加入强制训练矩阵，也不替换原16项。

## 6. 对当前排程的实际影响

- D/E仍按原安排等待服务器做真实CUDA短程和完整实验，本次不改模型代码。
- **经典参照不必全部从头训练**：先用标注来源的原文表格；希望得到更强的配对比较时优先检查作者预测，再考虑公开权重推理。
- 原重点候选保留，不因数据不同就删除文献；实测与否依具体论文论点、适配可行性及预算登记。不能把原文不匹配的数据硬搬到主表来凑模型数。
- G/H属于自定义同基座比较，不能由XAttnRes的Synapse表格代替。A/F/C/F-aux/D/E内部消融也不能由别人的结果代替。
- 对文献比较，只能报告已公开的汇总；对官方预测重评，需先检查完整病例与标签再生成本项目CSV。尚未做的事情在论文中保持待完成，不把“有下载入口”写成“已复评”。

## 7. 证据保存与复核范围

完整本机材料保存在`output/comparison_dataset_audit_20261103/`：本地原论文逐页提取文本、第5/6页截图、U-Mamba Table 3原图、官方评价脚本及空差异文件、Drive目录快照/结构化清单、作者版本说明、EViM原文XML与数据段落。原文材料仅作为阅读证据，不批量纳入Git或重新发布。

关键结构化证据：`public_artifact_discovery.json`含50例标签名和作者版本说明；`old_CT_listing.json`含old-v0六模型目录；`evaluation_equivalence.json`记录代码比对、源文件哈希及50例ID匹配。

本次未取得：作者预测包内清单/文件哈希、全部公开权重可加载性、原始GT与本机GT逐体素一致性、外部方法重新评价的CSV。部分出版网页/API有访问限制，已用官方公开代码、作者项目图和原文公开XML补充；没有根据第三方解读杜撰缺失结果。
