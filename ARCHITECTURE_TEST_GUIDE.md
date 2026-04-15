# CellStudio 架构端到端联调测试指南 (Overfit-Driven Test Guide)

## 摘要
本指南旨在为 CellStudio 重构后的 Monorepo 架构进行**最快速的、端到端的打通测试**。测试核心目标并非追求模型的泛化精度，而是通过在极小数据集（Tiny datasets）上故意**过拟合（Overfit）**，从工程侧验证整个数据加载栈（Dataloader）、各模型适配器流转（Model Adapters）、损失函数（Loss）、评价指标流控（Evaluator）及权重落盘功能的完备性与正确性。

---

## 第一章 测试环境与数据集设定

### 1.1 数据集分布说明
测试统一采用本机开发环境中的 `tiny` 级极限微缩数据集。为了实现快速验证模型架构是否连通，我们在数据切分逻辑上做了特殊设定：
- **Train (训练集) == Val (验证集)**：将训练集的标注和图像集合作为验证集输入。如果在该设定下模型的泛化指标（Accuracy、mAP、Dice）不能随着 Epoch 训练快速攀升至 1.0 (100%)，则标志着网络的前向传播、标签映射或 Loss 梯度回传存在**架构断链级别的 Bug**。
- **Test (测试集)**：作为完全独立的数据，在此阶段仅充当 Inference 流转通过的占位数据块，不考量其真实得分。

| 任务类型 | 使用的数据集目录 | 数据量规模 | 拆分策略核验状态 |
|---|---|---|---|
| **Classification (分类)** | `datasets/classfication/MIDOG_tiny` | 极小量 | Train=2, Val=2 (100% 重叠), Test=2 (独立) |
| **Detection (检测)** | `datasets/detection/MIDO_tiny` | 极小量 | Train=2, Val=2 (100% 重叠), Test=2 (独立) |
| **Segmentation (分割)** | `datasets/segmentation/cellpose_tiny` | 极小量 | Train=2, Val=2 (100% 重叠), Test=2 (独立) |

*(注：系统自检脚本已验证当前目录下的 `splits/` 配置，其中 `train.json` 与 `val.json` 的索引内容完全一致，完全符合急速过拟合测试标准。)*

---

## 第二章 测试超参数与流程配置

在验证架构流转健康度时，模型不需要经历漫长而复杂的真实大盘收敛期，只需遵循以下「过拟合极速验证参数标准」：

### 2.1 联调级超参数配置表 (Hyperparameters for Debug)

| 参数项 | 取值 | 配置目的 |
|---|---|---|
| **Epochs** | 20 ~ 50 | 次数减少，以最快速度看指标拉升状态 |
| **Batch Size** | 2 或 4 | Tiny 集的总量极小，缩小 Batch 让它跑出完整的 Iteration |
| **Learning Rate** | `1e-3` | 放大初始学习率，暴力加速过拟合 |
| **Data Augmentation** | 全部关闭 (Off) | 移除所有数据增强（旋转、噪声、颜色抖动等），让模型只学习极其固定的图像像素 |
| **Pretrained Weights** | 从零训练 (From Scratch) | 选配。若想彻底测网络连通性，可以不加载预训练，看 Loss 是否正常下降 |
| **Early Stopping** | 禁用 | 确保架构能完整走完指定 Loop，测试最终的 Model Saver 环节 |

---

## 第三章 阶段性任务测试流执行标准

### 3.1 分类任务连通性测试 (Classification)
1. **执行入口**：`python tools/train.py --config configs/classify/yolo_v8m_cls_mido.yaml --debug` (需配合具体实现的 config 名)
2. **测试观测点**：
   - DataLoader 是否成功解析了 `MIDOG_tiny` 下切图的 Label 字典。
   - `CrossEntropyLoss` 是否随着 Epoch 急剧下降至 0.0X 级别。
   - 训练结束后，验证集指标 **Accuracy, AUC 必须逼近 1.0 (100%)**。
   - 检查 `results/` 目录下是否生成了没有报错挂起的分类混淆矩阵图。

### 3.2 检测任务连通性测试 (Detection)
1. **执行入口**：`python tools/train.py --config configs/detect/yolo_v8m_det_mido_tile.yaml --debug`
2. **测试观测点**：
   - BBox 映射逻辑是否健康（是否出现坐标越界越权报错，尤其是数据预处理阶段 `visual_aug.py` 的索引）。
   - Box Loss 与 Objectness Loss 是否双双下降。
   - 在由于 Train==Val 的前提下，验证集指标 **mAP@0.5 急速攀升至 0.95 以上**。
   - Dataloader 是否能平稳承接原生切片的切割拼接推流。
   - 测试结束时产生的可视检出图片中，绿色的框应极为精准地裹紧目标。

### 3.3 分割任务连通性测试 (Segmentation)
1. **执行入口**：`python tools/train.py --config configs/segmentation/unet_mido_seg.yaml --debug`
2. **测试观测点**：
   - 面向 `Mask` 的像素级提取、针对多边形的高精度还原或者下采样特征降维图构建不脱离正确对位。
   - 验证集的 **Dice 系数与 mIoU 是否能逼近 1.0**，**95% HD 是否逼近极小值 (近乎 0)**。
   - 检查生成的推理重绘样本中，预测吐出的掩码是否与基准 GT **像素级吻合**。

---

## 第四章 架构测试验收标准 (Acceptance Criteria)

一次完备的 Monorepo 工程基建连通性测试应包含如下交付结项标志：

- [ ] **1. 无崩溃（Crash-free）**：从数据读取、预处理 `visual_aug.py` 转换、前向传播、Loss组合、方向传播、Metrics记录到最后 `save_checkpoint` 全生命周期无任何 Python 语法或计算资源引起的中断流阻。
- [ ] **2. 100% 内存/显存防泄漏**：由于 Tiny 集体量极小，显存 (VRAM) 与 CPU 主存必须保持绝对的一条直线平稳，未出现随着多批次推流显式递增导致的内存泄漏（OOM/RAM 剧增）现象。
- [ ] **3. 完美过拟合（Perfect Overfit）**：证明梯度的向后传递工作有效——网络能毫无阻力地"死记硬背"完全无干预的训练集数据。核心泛化评估指标极短时间内封顶。
- [ ] **4. 规范化工件落盘**：`results/` 工程目录中如约呈现对应的权重存档文件 (`best_model.pth` 或 `last.pth`)；无头绪地留存出各类具有回溯对比价值的图表。
