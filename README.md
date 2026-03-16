# CellStudio 🔬

这是一个**病理细胞深度学习综合项目**。本项目致力于收集并整合各类开源病理细胞数据集，构建高质量的细胞图像标准库，并在此基础上建立从基础到前沿的各类深度学习任务基线（Baseline）。

## 🌟 核心特性 (Features)

- **🗂️ 标准数据集构建**：持续收集各类开源病理细胞数据集，进行统一数据清洗、结构化与标注格式化，构建标准化的细胞病理数据集库。
- **📊 基础视觉任务基线**：
  - **分类 (Classification)**：细胞类型识别、良恶性判断。
  - **目标检测 (Object Detection)**：病理切片中的特定细胞或病灶精确定位。
  - **分割 (Segmentation)**：细胞核、细胞质及相关区域的精细实例或语义分割。
- **🧠 前沿模型与范式**：
  - **大模型 (Large Models)**：引入视觉大模型（如 SAM、DINOv2 等）进行特征提取、微调或零样本泛化。
  - **多模态 (Multi-modality)**：结合病理图像与临床文本报告进行多模态预训练与推理。
  - **少样本与零样本学习 (Few-shot & Zero-shot Learning)**：探索在极少甚至无标注目标领域的模型泛化能力。

## 📁 建议目录结构 (Project Structure)

```text
CellStudio/
├── data/                  # 数据集存放目录（不在版本控制内）
│   ├── raw/               # 原始数据集
│   └── processed/         # 处理后标准格式数据集
├── configs/               # 配置文件目录 (YAML/JSON)
├── cellstudio/            # 核心源码目录
│   ├── data/              # 数据加载与增强 (Dataset, DataLoader)
│   ├── models/            # 网络模型定义 (网络结构、损失函数等)
│   ├── engine/            # 训练、验证、测试逻辑
│   └── utils/             # 工具函数 (指标计算、可视化、日志等)
├── scripts/               # 各种任务的训练、推理入口脚本
├── notebooks/             # Jupyter Notebooks (用于探索性数据分析、可视化)
├── requirements.txt       # 环境依赖
└── README.md              # 项目说明文档
```

## 🛠️ 安装说明 (Installation)

1. 克隆仓库
```bash
git clone https://github.com/AlchemyAITech/CellStudio.git
cd CellStudio
```

2. 创建并激活虚拟环境（推荐使用 Conda）
```bash
conda create -n cellstudio python=3.10 -y
conda activate cellstudio
```

3. 安装依赖项
```bash
pip install -r requirements.txt
```
*(注意：请根据您的硬件环境（如 GPU 型号、CUDA 版本），前往 [PyTorch 官网](https://pytorch.org/) 再次确认并安装对应版本的 PyTorch 生态组件)*

## 🚀 快速开始 (Quick Start)

*(建设中)* 各任务的训练与推理脚本及说明文档正在逐步完善。未来您将能够通过类似以下的命令快速启动不同任务：

```bash
# 基线分类训练示例
python scripts/train_classification.py --config configs/cls_baseline.yaml

# 目标检测推理示例
python scripts/infer_detection.py --weights weights/det_best.pt --source data/test_images/
```

## 🤝 贡献说明 (Contributing)

欢迎提交 Issue 和 Pull Request 来完善数据集和算法基线。在提交代码前，请确保遵循本项目的代码规范。
