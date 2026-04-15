# CellStudio 架构端到端联调测试指南 (Overfit-Driven Test Guide)

## 摘要
本指南旨在为 CellStudio 重构后的 Monorepo 架构进行**最快速的、端到端的打通测试**。测试核心目标并非追求模型的泛化精度，而是通过在极小数据集（Tiny datasets）上故意**过拟合（Overfit）**，从工程侧验证整个数据加载栈（Dataloader）、各模型适配器流转（Model Adapters）、损失函数（Loss）、评价指标流控（Evaluator）及权重落盘功能的完备性与正确性。

---

## 第一章 测试环境与数据集设定

### 1.1 数据集分布说明
测试统一采用本机开发环境中的 `tiny` 级极限微缩数据集。为了实现快速验证模型架构是否连通，我们在数据切分逻辑上做了特殊设定：
- **Train (训练集) == Val (验证集)**：将训练集的标注和图像集合作为验证集输入。如果在该设定下模型的泛化指标（Accuracy、mAP、Dice）不能随着 Epoch 训练快速攀升至 1.0 (100%)，则标志着网络的前向传播、标签映射或 Loss 梯度回传存在**架构断链级别的 Bug**。
- **Test (测试集)**：作为完全独立的数据，在此阶段仅充当 Inference 流转通过的占位数据块，不考量其真实得分。

| 任务类型 | 数据加载器 (Dataloader) | 数据集根目录 (Root) | 数据规模 | 验证方法 |
|---|---|---|---|---|
| **Classification (分类)** | `StandardClassificationDataset` | `datasets/classfication/MIDOG_tiny` | Train:100, Test:20 | Train/Val 跑同一份 `splits/train.json` |
| **Detection (检测)** | `TileMIDODataset` | `datasets/detection/MIDO_tiny` | Train:10, Test:2 | Train/Val 跑同一份 `splits/train.json` |
| **Segmentation (分割)** | `CellposeSegmentationDataset` | `datasets/segmentation/cellpose_tiny` | Train:43, Test:18 | Train/Val 跑同一份 `splits/train.json` |

*(注：在本次验证中，测试所用的 `_tiny_test.yaml` 配置文件均已显式将 `train_dataloader` 和 `val_dataloader` 的 `ann_file` 统一定向至 `splits/train.json`，确保验证集在数学上 100% 重叠，严格遵循极速过拟合测试标准。)*

---

## 第二章 测试超参数与配置脚本

为满足一键复现，已为您在此次验证中专门生成了三个最小化、移除了全部 Data Augmentation 的极速测试配置文件：
- 分类测试配置：`configs/classify/resnet18_mido_tiny_test.yaml`
- 检测测试配置：`configs/detect/yolov8m_mido_tiny_test.yaml`
- 分割测试配置：`configs/segmentation/unet_mido_tiny_test.yaml`

这些配置文件遵循以下「过拟合级验证参数」：
- **Epochs**：极少轮次 (10 Epochs)，快速验证。
- **Batch Size**：2 或 4 (对显存极度友好)。
- **Learning Rate**：`1e-3` (放大初始学习率暴力过拟合)。
- **Workers**: `0` (单发线程防死锁调试)。

---

## 第三章 阶段性任务测试流执行标准

### 3.1 分类任务连通性测试 (Classification)
1. **执行入口**：`python tools/train.py --config configs/classify/resnet18_mido_tiny_test.yaml --debug`
2. **测试观测点**：
   - `StandardClassificationDataset` 是否成功解析了 `MIDOG_tiny` 下 `splits/train.json` 中的 `cls_labels`。
   - `CrossEntropyLoss` 是否随着 Epoch 急剧下降至 0.0X 级别。
   - 训练结束后，验证集指标 **Accuracy 必须逼近 1.0 (100%)**。

### 3.2 检测任务连通性测试 (Detection)
1. **执行入口**：`python tools/train.py --config configs/detect/yolov8m_mido_tiny_test.yaml --debug`
2. **测试观测点**：
   - `TileMIDODataset` 是否能在不越界的前提下成功处理大型 `*.tiff` 原生切片的坐标切割重投影。
   - BBox 映射逻辑是否健康（预处理阶段 `visual_aug.py` 不再抛出对齐异常）。
   - 由于 Train==Val，验证集指标 **mAP@0.5 急速攀升**。

### 3.3 分割任务连通性测试 (Segmentation)
1. **执行入口**：`python tools/train.py --config configs/segmentation/unet_mido_tiny_test.yaml --debug`
2. **测试观测点**：
   - `CellposeSegmentationDataset` 对 `Mask` 的像素级提取和高精度还原能否被 PyTorch Loader 正确重组为 Tensor。
   - 验证集的 **Dice 系数与 mIoU 是否能逼近 1.0**。

---

## 第四章 架构测试验收标准 (Acceptance Criteria)

一次完备的 Monorepo 工程基建连通性测试应包含如下交付结项标志：

- [ ] **1. 无崩溃（Crash-free）**：从数据读取、预处理 `visual_aug.py` 转换、前向传播、Loss组合、方向传播、Metrics记录到最后 `save_checkpoint` 全生命周期无任何 Python 语法或计算资源引起的中断流阻。
- [ ] **2. 100% 内存/显存防泄漏**：由于 Tiny 集体量极小，显存 (VRAM) 与 CPU 主存必须保持绝对的一条直线平稳，未出现随着多批次推流显式递增导致的内存泄漏（OOM/RAM 剧增）现象。
- [ ] **3. 完美过拟合（Perfect Overfit）**：证明梯度的向后传递工作有效——网络能毫无阻力地"死记硬背"完全无干预的训练集数据。核心泛化评估指标极短时间内封顶。
- [ ] **4. 规范化工件落盘**：`results/` 工程目录中如约呈现对应的权重存档文件 (`best_model.pth` 或 `last.pth`)；无头绪地留存出各类具有回溯对比价值的图表。
