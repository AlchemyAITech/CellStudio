# 基于深度学习的医学显微图像分析：分类、检测与分割多任务对比研究及实验分析报告

## 摘要
本报告详细阐述了基于 CellStudio 框架构建的医学显微图像三大核心视觉任务（图像分类、目标检测、图像分割）的全流程实验设计与统计分析方案。本实验旨在对目前主流的深度学习模型进行严谨的横向对比，评估其在细胞及有丝分裂目标分析中的性能差异。所有实验将严格遵循医学图像处理规范，包含完整的超参数控制变量法、科学的数据集划分、以及临床医学顶刊标准的统计学显著性分析。

---

## 第一章 实验数据集与预处理方案

### 1.1 数据集来源与背景
1. **细胞有丝分裂分类与检测数据集 (MIDOG)**：选用 MIDOG 数据集，该数据集包含了不同扫描仪下产生的乳腺癌组织病理学图像（WSI），标注了有丝分裂象（Mitotic figures）。用于评估模型在复杂组织背景和不同染色、扫描仪域偏移情况下的泛化能力。
   - **分类子集**：从MIDOG中裁切出以标注点为中心的 `64×64` 或 `128×128` 像素图像块 (Patches)，标注为"有丝分裂阳性 (Mitosis)" 和 "阴性 (Non-Mitosis)"两类。
   - **检测子集**：使用原始 `512×512` Tile 切片，标注框标注有丝分裂象的精确位置。
2. **细胞核与细胞质分割数据集 (Cellpose)**：包含多样化的荧光细胞图像以及组织切片图像，标注了高精度的细胞核/细胞质轮廓（Instance mask）。
3. **(可选扩展) NuCLS / PanNuke 数据集**：多类别细胞核分类与分割公共基准集，用于进一步验证框架的泛化能力和模型在不同组织类型中的鲁棒性。

### 1.2 数据集划分策略 (6:2:2 Stratified Split)
考虑到医学图像中经常存在类别极其不平衡的问题（例如背景远多于目标细胞、不同分级的样本差异巨大）：
- 所有数据集在进入训练管线前，均通过 `cellstudio/datasets/splitter.py` 严格按照 **训练集(Train) 60%、验证集(Validation) 20%、测试集(Test) 20%** 的比例进行**分层抽样 (Stratified Split)**。
- 确保测试集中各个域（如不同扫描仪）和类别分布与训练集保持一致，以防止评估结果出现严重的过拟合偏移。
- 数据集统计摘要表（含各类别样本数、正负样本比率、图像尺寸分布）将在实验前置生成。

### 1.3 数据增强 (Data Augmentation)
为提升模型的鲁棒性，使用针对显微医学图像的专用增强管线：
- **Geometry**：随机旋转 (RandomRotate90)、翻转 (Horizontal/Vertical Flip)、弹性形变 (ElasticTransform, 针对细胞形态极度有效)、随机缩放裁剪 (RandomResizedCrop)。
- **Photometry**：色彩抖动 (ColorJitter)、高斯噪声 (GaussianNoise)、随机模糊 (GaussianBlur)、染色标准化 (Stain Normalization, Macenko 方法)、随机灰度化 (RandomGrayscale)。
- **Mixup / CutMix**：针对分类任务的正则化增强策略，降低模型对单一样本的依赖。

---

## 第二章 实验模型选择与网络架构设计

为了保证论文的丰富程度与对比的全面性，在每一个视觉任务中，我们均挑选了 **5~7 种**最具代表性或最新的 SOTA (State-of-the-Art) 模型进行横向盲评，涵盖 CNN 范式、Transformer 范式以及面向医学影像的专用算法。

### 2.1 任务一：细胞图像分类 (Classification) — 6 个模型

| # | 模型 | 范式 | 选择理由 |
|---|---|---|---|
| 1 | **ResNet50** | 经典 CNN | 残差网络基准线 (Baseline)，具备较强的特征穿透能力 |
| 2 | **EfficientNet-B4** | 复合缩放 CNN | 基于 Compound Scaling 的轻量高精度网络，在小样本医学图像中往往表现优异 |
| 3 | **ConvNeXt-Tiny** | 现代化 CNN | 吸收 Transformer 设计理念的纯卷积架构，代表 CNN 的最新进化形态 |
| 4 | **ViT-Base** | Vision Transformer | 基于全局自注意力机制，验证其在缺乏归纳偏置时对细胞纹理的表征效果 |
| 5 | **YOLOv8m-cls** | 工业级 CNN | Ultralytics 最新工业级分类头，验证检测框架的分类迁移能力 |
| 6 | **YOLOv26m-cls** | 最新 YOLO | 2026 年最新分类架构，对标最新 SOTA |

### 2.2 任务二：病灶与细胞检测 (Detection) — 7 个模型

| # | 模型 | 范式 | 选择理由 |
|---|---|---|---|
| 1 | **YOLOv8m** | One-Stage Anchor-free | 极致速度与高精度的工业级代表 |
| 2 | **YOLOv26m** | One-Stage 最新架构 | 2026 年最新检测架构 |
| 3 | **Faster R-CNN** | Two-Stage | 经典两阶段检测器黄金标杆，RPN 机制在密集小目标医学检测中查全度高 |
| 4 | **FCOS** | One-Stage Anchor-free | 完全无锚框、逐像素预测的检测器，适合不规则形状目标 |
| 5 | **RTMDet-M** | One-Stage | 实时多尺度检测器，在小目标上有突出表现 |
| 6 | **DETR** | Transformer 端到端 | 基于匈牙利匹配的端到端检测器，彻底消除 NMS 后处理 |
| 7 | **RetinaNet** | One-Stage + Focal Loss | Focal Loss 的提出者，专门解决组织病理图像中极度常见的前景-背景类别不平衡 |

### 2.3 任务三：细胞实例与语义分割 (Segmentation) — 6 个模型

| # | 模型 | 范式 | 选择理由 |
|---|---|---|---|
| 1 | **UNet (ResNet34 Backbone)** | 编码-解码 | 医学图像分割绝对统治地位架构，Skip-Connection 完美还原细胞边缘低级语义 |
| 2 | **DeepLabV3+ (ResNet50 Backbone)** | 扩张卷积 | ASPP 多尺度上下文信息捕获，大小不一的细胞群适应性好 |
| 3 | **YOLOv8m-seg** | 实例分割 | Ultralytics 工业级实例分割，兼顾速度与精度 |
| 4 | **Cellpose** | 向量流 | 专为细胞设计的基于 Vector Flows 的实例分割算法，彻底解决细胞粘连 |
| 5 | **Cellpose-SAM** | 向量流 + Transformer | 在 Cellpose 基础上融合 SAM 的 Transformer 编码器 |
| 6 | **Mask R-CNN** | Two-Stage 实例分割 | 经典两阶段实例分割方法，在 RoI 级别同时预测框与掩码 |

---

## 第三章 实验超参数设定与硬件环境

为确保**控制变量法**的严谨性，所有模型的训练环境和基础超参数保持高度一致。

### 3.1 实验环境设定
| 项目 | 配置 |
|---|---|
| 操作系统 | Windows 11 / Linux Ubuntu 22.04 兼容构建 |
| GPU | NVIDIA RTX 4090 24GB / A100 (CUDA 自动加速) |
| 深度学习框架 | PyTorch >= 2.0.0 |
| Python 版本 | 3.11 |
| 实验管理框架 | CellStudio v0.2.0 |

### 3.2 训练超参数统一配置 (Hyperparameters)

| 超参数 | 值 | 说明 |
|---|---|---|
| Epochs | 100 | 统一训练轮数 |
| Optimizer | AdamW | 具备权重衰减正则化 |
| Initial LR | $1 \times 10^{-4}$ | 初始学习率 |
| Weight Decay | $5 \times 10^{-4}$ | 防止过拟合 |
| Scheduler | ReduceLROnPlateau | 验证 Loss 连续 10 Epoch 未降 → LR × 0.5 |
| Early Stopping | patience=20 | 核心指标 20 轮未改善则停止 |
| Batch Size | 16 (cls), 8 (det), 4 (seg) | 按任务类型调整 |
| Input Size | 224×224 (cls), 512×512 (det), 256×256 (seg) | 按任务标准尺寸 |
| Random Seed | 42 | 全局种子固定，保证可复现 |

### 3.3 预训练策略
- 所有支持 ImageNet 预训练的模型均使用 **ImageNet-1K 预训练权重** 作为初始化。
- Cellpose 使用其官方发布的细胞学预训练权重 (`cyto2` model)。
- 冻结策略：前 5 个 Epoch 冻结 Backbone，仅训练分类/检测/分割 Head (Warm-up)。

---

## 第四章 医学级评价指标与统计分析方法

对于毕业论文级的数据分析，仅计算精确度是远远不够的。本研究基于国际医学影像顶级期刊（如 *TMI, Medical Image Analysis, Nature Methods*）的标准，构建了以下详尽的统计学评价与分析方法。

### 4.1 各任务核心评价指标 (Evaluation Metrics)

#### 4.1.1 图像分类指标 — 完整体系

| 指标 | 英文 | 意义 |
|---|---|---|
| 准确率 | Accuracy | 总体正确率 |
| AUC-ROC | Area Under ROC Curve | 综合区分能力（阈值无关） |
| 敏感度 | Sensitivity / Recall | 阳性检出率（漏诊率的互补值） |
| 特异度 | Specificity | 阴性正确排除率（误诊率的互补值） |
| 精确率 | Precision | 阳性预测的可信度 |
| F1 分数 | F1-Score | 精确率与召回率的调和均值 |
| Cohen's Kappa | Kappa Coefficient | 消除偶然一致性后的预测-金标准一致度 |
| PR-AUC | Area Under PR Curve | 在极端不平衡数据中比 ROC-AUC 更可靠 |
| 混淆矩阵 | Confusion Matrix | 各类别的 TP/TN/FP/FN 全量可视化 |

#### 4.1.2 目标检测指标 — 完整体系

| 指标 | 英文 | 意义 |
|---|---|---|
| mAP@0.5 | Mean AP at IoU=0.5 | 宽松评价标准 |
| mAP@0.5:0.95 | Mean AP at IoU=0.5~0.95 | COCO 标准严格评价 |
| Precision | Detection Precision | 检出正确率 |
| Recall | Detection Recall | 查全率 |
| F1 | Detection F1 | 查准查全平衡率 |
| FROC 曲线 | Free-response ROC | **医学检测核心指标**：FPI=0.5/1.0/2.0/4.0/8.0 下的敏感度 |
| Count Error | 绝对计数误差 | 预测目标数与 GT 数的差值 |
| 推理速度 | FPS / Inference Time | 模型实时性评估 |

#### 4.1.3 图像分割指标 — 完整体系

| 指标 | 英文 | 意义 |
|---|---|---|
| Dice | Dice Coefficient / F1-Score for Sets | 预测掩码与真实掩码的空间重叠度 |
| 95% HD | 95th-percentile Hausdorff Distance | **苛刻边界指标**：轮廓距离的 95 分位数 |
| mIoU | Mean Intersection over Union | 分割区域交并比均值 |
| AJI | Aggregated Jaccard Index | 实例级聚合 Jaccard 指数（评估单个细胞分离质量） |
| PQ | Panoptic Quality | 全景质量 = SQ × RQ，同时评估分割准确度与匹配度 |
| Pixel Accuracy | 逐像素准确率 | 基础像素级评价 |
| Boundary IoU | 边界区域 IoU | 专注评价轮廓附近区域的分割精度 |

### 4.2 统计显著性测试与数据分析方案

在测试集上获得所有模型的预测结果后，运行以下独立的统计分析测试，以得出具有说服力的科学结论：

1. **置信区间估计 (95% Confidence Intervals, CI)**
   利用 Bootstrap 重抽样技术（在测试集上随机有放回抽样 1000 次），估计所有指标均值的 95% 置信区间，绘制带有 Error Bar 或阴影带的图表。

2. **假设检验与 P-value 计算**
   - 配对检验：使用 Wilcoxon signed-rank test（非参数配对检验），比较两两模型在同一测试样本上的性能差异。
   - 独立检验：使用 Mann-Whitney U test，比较不同架构范式（CNN vs Transformer）间的群体性能差异。
   - 显著性判定：$p < 0.05$ 为显著，$p < 0.01$ 为高度显著，$p < 0.001$ 为极度显著。

3. **Friedman 检验 + Nemenyi 后验（多模型联合对比）**
   当同时比较 ≥ 3 个模型时，先进行 Friedman 检验确认是否存在全局显著差异，再通过 Nemenyi post-hoc test 确定哪些模型对之间存在显著差异，绘制 **CD (Critical Difference) 图**。

4. **临床可解释性分析 (Model Interpretability)**
   - **Grad-CAM 热力图**：对最佳分类模型提取网络最后一层的类激活映射，验证模型是否真正关注了细胞核的有丝分裂特征而非背景伪特征。
   - **t-SNE 特征可视化**：提取各模型倒数第二层的高维特征向量，通过 t-SNE 降维至 2D，观察不同类别样本的聚类效果。
   - **分割结果三联图**：`Source Image` + `Ground Truth Mask` + `Predicted Mask` 的并排比较图。

5. **消融实验与鲁棒性验证 (Ablation Study)**
   - 数据增强消融：对比关闭 "弹性形变"、"颜色抖动"、"Stain Normalization" 前后的模型性能差异。
   - 输入分辨率消融：128×128 vs 256×256 vs 512×512。
   - 主干网络替换消融：在 UNet 中切换 ResNet18/34/50 不同深度的 Backbone。
   - 预训练权重消融：ImageNet 预训练 vs 从零训练 (From Scratch)。

### 4.3 K 折交叉验证 (K-Fold Cross Validation)
为彻底消灭数据划分本身带来的偶然性并为统计学检验提供多组独立数据：
- 核心定准模型将进行 **5-Fold Cross Validation**。
- 每个 Fold 独立训练、验证与测试，汇报 $Mean \pm Std$ 以及各 Fold 的 Box Plot。
- 交叉验证的结果同时用于 Friedman 检验的输入数据。

---

## 第五章 实验全量模型矩阵总览

### 5.1 完整实验矩阵 (19 个模型 × 3 个任务 = 全量实验组)

| 任务 | 模型总数 | 模型列表 |
|---|---|---|
| Classification | 6 | ResNet50, EfficientNet-B4, ConvNeXt, ViT-Base, YOLOv8-cls, YOLOv26-cls |
| Detection | 7 | YOLOv8, YOLOv26, Faster R-CNN, FCOS, RTMDet, DETR, RetinaNet |
| Segmentation | 6 | UNet, DeepLabV3+, YOLOv8-seg, Cellpose, Cellpose-SAM, Mask R-CNN |

### 5.2 论文预期产出图表清单

| 图表类型 | 数量 | 说明 |
|---|---|---|
| Loss / LR / Metric 训练曲线 | 19 组 | 每个模型一组 |
| 混淆矩阵 | 6 张 | 分类任务每个模型一张 |
| ROC 曲线 (多模型叠加) | 1 张 | 分类任务全模型 AUC 对比 |
| PR 曲线 (多模型叠加) | 1 张 | 分类任务全模型 PR-AUC 对比 |
| FROC 曲线 (多模型叠加) | 1 张 | 检测任务医学核心图 |
| 检测结果渲染图 | 7 组 | 带预测框和置信度的检测可视化 |
| 分割三联图 | 6 组 | Source + GT + Pred 并排对比 |
| Grad-CAM 热力图 | 6 张 | 分类最佳模型可解释性分析 |
| t-SNE 特征散点图 | 6 张 | 各模型特征聚类效果 |
| 横评柱状图 (带 Error Bar) | 3 张 | 各任务所有模型核心指标 Bar Plot |
| Box Plot (K-Fold 分布) | 3 张 | 核心模型 K-Fold 性能分布箱线图 |
| CD 图 (Nemenyi 后验) | 3 张 | 多模型统计学显著性排名 |
| 消融对比表 | 4 张 | 数据增强/分辨率/Backbone/预训练消融结果 |
| 性能汇总表 (LaTeX) | 3 张 | 各任务全指标横评大表 |

---

## 第六章 实验全流程执行代码设计总结 (Execution Pipeline)

为满足科研代码工程级别的"一键复现 (Reproducibility)"，目前的管线 (`CellStudio`) 设置如下，这部分系统代码将直接支持论文实验的产出并可作为论文附录源码：

1. `task/cls-MIDOG/run_task.py` → 包含前置模块（DatasetsLoader）调用、拉起不同参数下的 6 个分类模型。引入了 5-Fold 生成、训练早停，结束时输出模型基于 Grad-CAM 的诊断热图、混淆矩阵、ROC/PR 曲线图、t-SNE 特征散点图及 CSV 汇总报告。
2. `task/det-MIDO/run_task.py` → 执行 7 个检测模型的对比实验，自动保存最优权重、输出带有预测置信度的检出框渲染图，以及横评分析所必须的 mAP 汇总与 FROC 图。
3. `task/seg-cellpose/run_task.py` → 运行 6 个分割模型，运行过程中即时记录训练过程的 Dice/mIoU/PQ 评估指标，结束时进行苛刻的 95%HD 与 AJI 验证，并在结果目录里保存三联对拼图。
4. `task/stats/run_statistics.py` → 全量跑通后自动运行 Bootstrap 置信区间、假设检验、Friedman + Nemenyi 后验测试，产出结构化 CSV/LaTeX 表格。
5. **统一结果回溯**：全量跑通后，上述代码自动产生的全部图表 (`.png`/`.csv`/`.tex`) 及权重 (`.pth`) 均会被锁定在对应 `results/` 子目录，供毕业答辩及论文撰写随时调用溯源。

*(本报告在模型训练完毕后，其中的理论数字与框架将被真实产生的预测数据、FROC图表、置信区间及可解释性图像填满，构成一篇完整翔实的医学病理学图像评测论文。)*
