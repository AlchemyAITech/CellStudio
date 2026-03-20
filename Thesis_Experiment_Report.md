# 基于深度学习的医学显微图像分析：分类、检测与分割多任务对比研究及实验分析报告

## 摘要
本报告详细阐述了基于 CellStudio 框架构建的医学显微图像三大核心视觉任务（图像分类、目标检测、图像分割）的全流程实验设计与统计分析方案。本实验旨在对目前主流的深度学习模型进行严谨的横向对比，评估其在细胞及有丝分裂目标分析中的性能差异。所有实验将严格遵循医学图像处理规范，包含完整的超参数控制变量法、科学的数据集划分、以及临床医学顶刊标准的统计学显著性分析。

---

## 第一章 实验数据集与预处理方案

### 1.1 数据集来源与背景
1. **细胞有丝分裂分类与检测数据集 (MIDOG)**：选用 MIDOG 数据集，该数据集包含了不同扫描仪下产生的乳腺癌组织病理学图像（WSI），标注了有丝分裂象（Mitotic figures）。用于评估模型在复杂组织背景和不同染色、扫描仪域偏移情况下的泛化能力。
2. **细胞核与细胞质分割数据集 (Cellpose)**：包含多样化的荧光细胞图像以及组织切片图像，标注了高精度的细胞核/细胞质轮廓（Instance mask）。

### 1.2 数据集划分策略 (6:2:2 Stratified Split)
考虑到医学图像中经常存在类别极其不平衡的问题（例如背景远多于目标细胞、不同分级的样本差异巨大）：
- 所有数据集在进入训练管线前，均通过 `cellstudio/datasets/splitter.py` 严格按照 **训练集(Train) 60%、验证集(Validation) 20%、测试集(Test) 20%** 的比例进行**分层抽样 (Stratified Split)**。
- 确保测试集中各个域（如不同扫描仪）和类别分布与训练集保持一致，以防止评估结果出现严重的过拟合偏移。

### 1.3 数据增强 (Data Augmentation)
为提升模型的鲁棒性，使用针对显微医学图像的专用增强管线：
- **Geometry**：随机旋转 (RandomRotate90)、翻转 (Horizontal/Vertical Flip)、弹性形变 (ElasticTransform, 针对细胞形态极度有效)。
- **Photometry**：色彩抖动 (ColorJitter)、高斯噪声 (Gaussian Noise)、随机模糊 (Blur)，以模拟显微成像时的失焦和染色差异。

---

## 第二章 实验模型选择与网络架构设计

为了保证论文的丰富程度与对比的全面性，在每一个视觉任务中，我们均挑选了3种最具代表性的 SOTA(State-of-the-Art) 模型进行横向盲评。

### 2.1 任务一：细胞图像分类 (Classification)
探究提取局部纹理特征的 CNN 与关注全局上下文的 Transformer 架构在显微图像分类上的特征表达能力。
1. **ResNet50**：残差网络的经典基准 (Baseline)，具备较强的特征穿透能力。
2. **EfficientNet-B4**：基于复合缩放 (Compound Scaling) 的轻量高精度网络，在小样本医学图像中往往表现优异。
3. **ViT-Base (Vision Transformer)**：基于全局自注意力机制的模型，验证其在缺乏归纳偏置 (Inductive Bias) 时对细胞纹理的表征效果。

### 2.2 任务二：病灶与细胞检测 (Detection)
对比 One-Stage 头与 Two-Stage 头在微小目标（如病理切片中的有丝分裂细胞）上的优劣。
1. **YOLOv8**：极致速度与高精度的代表，One-Stage 锚框无关 (Anchor-free) 架构。
2. **Faster R-CNN**：两阶段 (Two-Stage) 检测器的黄金标杆，其 RPN 机制在密集的小目标医学检测中依然具有极高的查全度。
3. **RetinaNet**：引入 Focal Loss 的单机段检测器，用于解决组织病理图像中极度常见的前景-背景类别不平衡问题。

### 2.3 任务三：细胞实例与语义分割 (Segmentation)
细胞实例彼此贴合、边界模糊，是本任务的巨大挑战。
1. **UNet (ResNet34 Backbone)**：医学图像分割的绝对统治地位架构，其 Skip-Connection 能完美还原细胞边缘的低级语义信息。
2. **DeepLabV3+**：采用空洞空间金字塔池化 (ASPP)，能够捕获多尺度上下文信息，对于大小不一的细胞群有良好的适应性。
3. **Cellpose**：一种专为细胞设计的基于向量流 (Vector Flows) 的实例分割算法，彻底解决细胞彼此“粘连”的问题。

---

## 第三章 实验超参数设定与硬件环境

为确保**控制变量法**的严谨性，所有模型的训练环境和基础超参数保持高度一致。

### 3.1 实验环境设定
- **操作系统**: Windows / Linux 兼容构建
- **核心计算硬件**: NVIDIA GPU (CUDA 自动加速)
- **深度学习框架**: PyTorch >= 2.0.0

### 3.2 训练超参数统一配置 (Hyperparameters)
- **Epochs**: 统一设置为 100 轮 (Epoch=100)。
- **Optimizer**: 采用 AdamW 优化器，具备较好的正则化效果。
- **Learning Rate (LR)**: 初始学习率 $1 \times 10^{-4}$。
- **Scheduler**: 引入 `ReduceLROnPlateau`，当验证集 Loss 连续 10 个 Epoch 未下降时，学习率衰减 `factor=0.5`。
- **Early Stopping**: 设定 `patience=20`，如果验证集核心指标（如分类的 AUC、分割的 Dice）连续 20 轮未见改善，则触发早停，防止模型过拟合，并保存最优模型权重 (`best_model.pth`)。

---

## 第四章 医学级评价指标与统计分析方法（重要）

对于毕业论文级的数据分析，仅计算精确度是远远不够的。本研究基于国际医学影像顶级期刊（如 *TMI, Medical Image Analysis*）的标准，构建了以下详尽的统计学评价与分析方法：

### 4.1 各任务核心评价指标 (Evaluation Metrics)

#### 4.1.1 图像分类指标
除基础的 **Accuracy (准确率)** 外，引入：
- **AUC-ROC (Receiver Operating Characteristic - Area Under Curve)**：衡量模型在不同阈值下的综合判断阈值能力。
- **Sensitivity (敏感度/Recall)** & **Specificity (特异度)**：对于疾病/异常细胞筛查，Sensitivity 至关重要（要求极低的漏诊率）。
- **Cohen's Kappa 系数**：评估模型预测结果与金标准之间的一致性，专门对抗数据不平衡的虚假繁荣。

#### 4.1.2 目标检测指标
- **mAP (Mean Average Precision) @ IoU=0.5~0.95**：通用检测评估基准。
- **FROC 曲线 (Free-response ROC)**：**（医学检测独有核心指标）**
  *在组织病理切片中，由于图像幅面巨大（WSI分辨率极高），每张图像上允许出现的“假阳性个数 (False Positives per Image, FPI)”是医学专家极度关注的值。FROC 曲线横坐标为 FPI，纵坐标为检测敏感度(Sensitivity)。我们将重点报告 FPI=0.5, 1.0, 2.0 等截断值下的敏感度。*

#### 4.1.3 图像分割指标
- **Dice Coefficient (Dice 系数，F1-Score for Sets)**：衡量预测掩码与真实掩码的空间重叠度。
- **95% HD (Hausdorff Distance 95%)**：**非常苛刻的边界指标**，计算预测轮廓与真实轮廓之间的最大距离的 95 分位数，专门用于评估模型对细胞边界细节（如毛刺、突起）的分割能力。

### 4.2 统计显著性测试与数据分析方案
在测试集上获得所有模型的预测结果后，运行以下独立的统计分析测试，以得出具有说服力的科学结论：

1. **置信区间估计 (95% Confidence Intervals, CI)**
   利用 Bootstrap 重抽样技术（比如在测试集上随机有放回抽样 1000 次），估计所有指标（如 AUC、Dice）均值的 95% 置信区间，绘制带有 Error Bar（误差棒）或阴影带的图表。
2. **假设检验与 P-value 计算**
   使用独立双样本 T 检验 (Student's t-test) 或非参数检验 (Wilcoxon signed-rank test)，比较两两模型（如 UNet 与 Cellpose，或者 ResNet 与 ViT）之间预测结果分布是否存在显著的统计学差异 ($p < 0.05$ 判定为统计学显著)。
3. **临床可解释性分析 (Model Interpretability & Grad-CAM)**
   鉴于深度学习在医学组织学上的“黑盒”属性面临严重的信任危机，本实验特别设计**模型可解释性章节**：对于最佳表现的分类模型和分割模型，提取其最后一层特征图进行可视化，或使用 Grad-CAM (Gradient-weighted Class Activation Mapping) 提取网络的热力图，探寻模型究竟是“关注到了细胞核的有丝分裂染色质形状”还是“基于背景染色造假提取了伪特征”。

4. **消融实验与鲁棒性验证 (Ablation Study)**
   引入模型及环境消融研究：
   - 数据增强消融：对比关闭“弹性形变”与“颜色抖动”前后的模型性能差异（论证数据增强对于防过拟合的真正效用）。
   - 主干网络替换消融（如在 UNet 中切分不同体量的 Backbone）。

### 4.3 K折交叉验证 (K-Fold Cross Validation)
为了彻底消灭数据划分本身带来的偶然性并为统计学检验提供多组随机分布数据，除了初始的 6:2:2 划分，核心定准模型将进行 **5-Fold Cross Validation**，并给出每个折叠的表现，最终汇报 $Mean \pm Standard Deviation$。这种严苛的控制方法远超一般普通实验。

---

## 第五章 实验全流程执行代码设计总结 (Execution Pipeline)

为满足科研代码工程级别的“一键复现 (Reproducibility)”，目前的管线 (`CellStudio`) 设置如下，这部分系统代码将直接支持论文实验的产出并可作为论文附录源码：
1. `task/cls-MIDOG/run_task.py` -> 包含前置模块（DatasetsLoader）调用、拉起不同参数下的3个分类模型。引入了 5-Fold 生成、训练早停，结束时输出模型基于 Grad-CAM 的诊断热图、混淆矩阵、ROC/PR 曲线图及 CSV 汇总报告。
2. `task/det-MIDO/run_task.py` -> 执行检测试验对比及消融实验分析，自动保存最优权重、输出带有预测置信度的检出框渲染图，以及横评分析所必须的 mAP 汇总与 FROC 图。
3. `task/seg-cellpose/run_task.py` -> 处理不规则胞体的分割场景，运行过程中即时记录训练过程的 Dice 评估指标，结束时进行苛刻的 95%HD 验证，并在结果目录里保存 `Source Image` + `Ground Truth Mask` + `Predicted Mask` 的三联对拼图用于直接插入论文排版。
4. **统一结果回溯**：全量跑通后，上述代码自动产生的全部图表 (`.png`/`.mat`/`.csv`) 及权重 (`.pth`) 均会被锁定在对应 `results/` 子目录，供毕业答辩及论文撰写团队随时调用溯源。

*(本报告在模型训练完毕后，其中的理论数字与框架将被真实产生的预测数据、FROC图表、置信区间及可解释性图像填满，构成一篇完整翔实的医学病理学图像评测论文。)*
