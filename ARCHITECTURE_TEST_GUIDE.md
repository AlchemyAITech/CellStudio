# CellStudio 架构级端到端测试与验证指南 (Global Architecture Test Specification)

## 摘要
本指南基于《基于深度学习的医学显微图像分析：分类、检测与分割多任务对比研究及实验分析报告》制定。其核心目标是：在真实大盘实验开启前，利用本机 **Tiny 极限微缩数据集**，以“极速过拟合 (Overfit-Driven)” 和 “短路验证 (Short-circuit Simulation)” 的方式，全面、严苛地验证 CellStudio 框架内 **19 种 SOTA 模型架构、5 大数据流引擎、涵盖 20+ 项医学评价指标的计算逻辑，以及所有可视化 Hook 的贯通性**。

这是一份工业级、全覆盖的白盒测试指南，旨在把运行中可能因模型适配器不兼容、张量形状不对齐、评估脚本爆内存而导致的 Bug 卡载提早暴露并扫清。

---

## 第一章 核心测试集与数据栈环境

为兼顾“跑得快”与“测得全”，全部子模块强制绑定在 Tiny 环境。
验证原则为：在 Tiny 特供验证集上，不考虑泛化能力，若验证通过，**所有模型的首要泛化核心指标（如 Accuracy、mAP、Dice）必须能随着 Epoch 的推进而平滑逼近 1.0 (100%)**。

### 1.1 微缩数据集矩阵配置 (Dataset Matrix)
| 分析任务 | Dataloader 引擎 | 测试根目录 (Root) | 数据规模 | 验证方法 |
|---|---|---|---|---|
| **分类 (Cls)** | `StandardClassificationDataset` | `datasets/classfication/MIDOG_tiny` | Train:100, Test:20 | Train/Val 跑同一份 `splits/train.json` |
| **检测 (Det)** | `TileMIDODataset` | `datasets/detection/MIDO_tiny` | Train:10, Test:2 | Train/Val 跑同一份 `splits/train.json` |
| **分割 (Seg)** | `CellposeSegmentationDataset` | `datasets/segmentation/cellpose_tiny` | Train:43, Test:18 | Train/Val 跑同一份 `splits/train.json` |

---

## 第二章 基础数据栈与图元结构测试流 (Data Foundation Testing)

在模型真正前向传播前，必须对底层的 UDF (Universal Data Format) 和图形基元进行严酷的单元测试与张量边界测试。这是整个架构不崩溃的基石。

### 2.1 基础图元实体边界测试 (Primitive Graph Objects)
确保脱离了模型本身的基类具有极强的自校验与反序列化解析能力。
- [ ] **空间基元拉伸**：对 `CSUPoint`, `CSUBBox`, `CSUPolygon`, `CSUMask` 强制输入非法值（如越界负数、塌缩的零面积框），验证系统抛出的防护级别而非静默报错。
- [ ] **UDF JSON 脱水测试**：测试解析体积达到 GB 级的伪造 JSON（模拟多张带有 10 万+细胞的多边形 `Total-Part` 拆分文件），必须满足**内存平缓且无 OOM 卡死**。
- [ ] **多边形 RDP 瘦身压缩率**：针对复杂边缘细胞生成的 `CSUPolygon`，运行 `shapely.simplify`，要求边缘还原度保真（视觉偏移 < 1 个像素级）的同时，点位数组压缩率达到 60% 以上。

### 2.2 相似度与缓存匹配引擎 (MatchCache & IOU)
验证从“算法预测数据 (`source_type='algorithm'`)” 向 “人工金标准 (`source_type='human'`)” 发起匹配合并的核心机制。
- [ ] **极限坐标算子测试**：传入极小重叠、完全包含、十字交叉等极端几何拓扑，验证 `compute_udf_iou` 在 BBox 与 Polygon 体系下的精确数值。
- [ ] **ID 挂载逻辑**：当 `IoU > thresh` 时，确保算法生成的图元正确外键绑定 `match_info` 字典。

### 2.3 WSI 切图与增强流引擎 (WSI & Data Pipeline)
- [ ] **图像域桥接**：在同一代码管线下，随机分发 `.svs` (基于 `OpenSlide`) 和巨幅原生 `.tiff` (基于 `TiffFile`)，验证 `Dataset` 是否能做到上层无感化滑窗提取 (Slide Window Extraction)。
- [ ] **安全的数据增强（Data Augmentation）边界**：开启弹性形变 (`ElasticTransform`)、旋转错位等，确认多边形映射或边缘切割坐标不会像之前的 `visual_aug.py` 抛出 `IndexError` 溢出。

---

## 第三章 全维度模型适配器测试流 (Adapter Validation)

本测试流旨在彻底验证 19 套模型是否能在 CellStudio 系统内顺畅无阻地被实例化、输入特征切片、正确反向传递并在最后完成 Checkpoints 权重落盘。

### 2.1 图像分类模型簇 (Classification Adapters)
**总数量：6 套** | **要求**：在 `configs/classify/` 中通过 `--debug` 拉起测试配置文件。
- [ ] 测试 `TimmClassifier` 的前向流转 (ResNet50, EfficientNet-B4, ConvNeXt-Tiny, ViT-Base)。
  - *观测指标*：`CrossEntropyLoss` 下降；验证 ViT 的输入通道归纳偏置是否被异常阻拦。
- [ ] 测试 `UltralyticsClsAdapter` 工业级流转 (YOLOv8m-cls, YOLOv26-cls)。
  - *观测指标*：测试 `tools/train.py` 是否正常承接了 Ultralytics 引擎库内的参数。

### 2.2 目标检测模型簇 (Detection Adapters)
**总数量：7 套** | **要求**：检测目标多且密集，重点查验 BBox 和类别对齐。
- [ ] 测试 `UltralyticsDetAdapter` 架构 (YOLOv8m, YOLOv26m)。
  - *观测指标*：原生 Tile 切图能否被裁剪缩压后正确投影 bbox，Loss 是否下降。
- [ ] 测试 `MMDetAdapter` 架构 (Faster R-CNN, FCOS, RTMDet, DETR, RetinaNet)。
  - *观测指标*：Two-Stage 与 One-Stage RPN 网络层的加载；Transformer (DETR) `objectness_loss` 匹配无异常。

### 2.3 医学图像分割模型簇 (Segmentation Adapters)
**总数量：6 套** | **要求**：医学分割的核心难点是内存占用，需验证并行与张量解析。
- [ ] 测试 `SMPAdapter` 架构体系 (UNet, DeepLabV3+)。
  - *观测指标*：Skip connections、ASPP 激活情况；Dice/BCE Loss 平滑下降。
- [ ] 测试 `UltralyticsSegAdapter` 架构 (YOLOv8m-seg)。
- [ ] 测试细胞原生库级联 `CellposeAdapter` (Cellpose, Cellpose-SAM)。
  - *观测指标*：SAM Transformer 头的位置编码 (Positional Embeddings) 和 Vector Flow 解算正常工作，不抛溢出报错。

---

## 第三章 医学验证流控及钩子系统测试 (Metrics & Hooks Pipeline)

这是毕业论文分析的“护城河”，需要保证即使是在 Tiny 数据集上，这 20 多项极为严苛的医学指标与图表绘图仪依然完备触发。

### 3.1 医学图表与可解释性分析仪 (Visualizers & Plotters)
每个模块需通过修改配置设定 `val_interval: 1` 并在 10 个 Epoch 后检查 `results/` 工程化目录：
- [ ] **分类图表测试**：确认生成 `Confusion_Matrix.png` 且类别归位准确；产生多阈值 `ROC_Curve.png` 与 `PR_Curve.png`。
- [ ] **可解释性探针测试**：触发 `Grad-CAM` 绘图器 Hook，检查生成的热力图（Heatmap）是否成功叠层于原生医学切片之上。
- [ ] **检测与分割可视化**：检查测试后生成的对比图。必须产出 `Source` + `Ground Truth` + `Predicted` 级别的同框三联拼接图。

### 3.2 严苛统计学指标数值边界测试 (Extremum Validation)
对下列医学期刊要求的特殊指标，在纯过拟合状态下，检查代码实现是否正确：
- [ ] **分类极限**：当 Train==Val 时，检查 Cohen's Kappa, Sensitivity, Specificity, AUC 是否都严格达到了极值 (`1.0` 或 `0.99+`)。
- [ ] **检测极限**：检测代码内的 FROC (Free-response ROC) 绘图器函数。对于完全复习过的数据，在 `FPI=0.5, 1.0` 的节点上，检测阈值 Sensitivity 必须飙红达到最高点。
- [ ] **分割极限**：测试 `95% HD` (豪斯多夫距离) 必须下降趋近于 `0`。如果依旧维持在几千像素，则表明边界函数对齐在源码层面存在破坏；验证 `AJI` 和 `PQ` 是否正确产出。

---

## 第四章 OOM 防止与鲁棒性打压测试 (Robustness Stress Test)

确保我们在大盘挂载医学级别巨婴影像数据 (WSI / 50X倍镜 TIFF) 时代码不崩溃。
- [ ] **Data Pipeline 安全探针**：修改 `tiny` 模型配置，全量开启所有的 Data Augmentations (包含最高难度的 `ElasticTransform` 和 `CutMix`)，确保之前修复的 `visual_aug.py` `IndexError` bug 彻底绝迹，所有预处理变换流能承受医学图像的边界极值。
- [ ] **VRAM/RAM 显存保活测试**：在拉起任意分割网络迭代期间，打开 `nvidia-smi` 观测，显存用量曲线必须维持直线阻尼震荡。严禁出现因 Pytorch Loader 每次保留梯度树或 Tensor 不下放而造成的**台阶式的渐进内存泄漏**。
- [ ] **Early Stopping 与 Scheduler**：设定极为短暂的 `patience=3`，故意干预 Loss 让它不下降，测试系统是否能在符合打断规范时切断训练保护最佳权重，验证点落在 `best_model.pth`。

---

## 开始您的测试旅程 (Execution Scripts)

在此处，我们将基于这套架构体系逐步拉满实测脚本。
第一步（基座健全度探针验证）：**我们已为您预设了 3 大完全屏蔽增强的 `_tiny_test.yaml`**。
请使用这三条指令拉起**短路级过拟合检测**。只需 5 分钟，所有绿灯必须亮起：

```bash
# 1. 验证分类器体系、Grad-CAM与精度拉升
python tools/train.py --config configs/classify/resnet18_mido_tiny_test.yaml --debug

# 2. 验证检测流、Tile拼图截切与FROC钩子
python tools/train.py --config configs/detect/yolov8m_mido_tiny_test.yaml --debug

# 3. 验证分割实例重建与边界重投影
python tools/train.py --config configs/segmentation/unet_mido_tiny_test.yaml --debug
```
