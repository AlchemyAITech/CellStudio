# CellStudio 🔬

**病理细胞深度学习综合框架** — 面向细胞病理学的全生命周期深度学习开发平台。

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)

## 🌟 核心特性

- **🗂️ 标准数据集** — 统一的 MIDO 格式数据集接口，支持分类、检测、分割三大任务
- **🔧 插件化架构** — 基于统一 Registry 的组件注册机制，模型/数据集/指标/可视化全部可热插拔
- **📊 三大视觉任务**
  - **分类** — ResNet / EfficientNet / ConvNeXt / ViT / YOLOv8-cls 等
  - **目标检测** — YOLOv8 / YOLO11 / YOLO26 / Faster R-CNN / DETR / FCOS / RTMDet
  - **实例分割** — Cellpose / Cellpose-SAM / UNet
- **⚡ Config-Driven** — 一个 YAML 文件定义完整实验，`_base_` 继承机制复用公共配置
- **🌐 生产部署** — 内置 FastAPI 服务端，支持模型热加载和异步训练调度

## 📁 项目结构

```text
CellStudio/
├── cellstudio/            # 核心框架源码
│   ├── core/              #   通用 Registry 基类
│   ├── engine/            #   Runner, Hooks, Config 引擎
│   ├── tasks/             #   任务编排器 (Classification/Detection/Segmentation)
│   ├── datasets/          #   数据集抽象与 MIDO 适配
│   ├── models/            #   模型适配器 (forward_train / forward_test)
│   ├── backends/          #   第三方后端隔离层 (Ultralytics / timm / Cellpose)
│   ├── metrics/           #   评估指标 (Accuracy, mAP, Dice, PQ, ...)
│   ├── plotting/          #   可视化 (ROC, PR, Confusion Matrix, ...)
│   ├── evaluation/        #   评估编排器
│   ├── inference/         #   推理引擎
│   ├── pipeline/          #   数据变换 DAG (Compose + Transform Nodes)
│   └── structures/        #   标准数据结构 (DataSample, InferResult)
├── configs/               # YAML 配置文件
│   ├── _base_/            #   公共运行时配置
│   ├── classify/          #   分类任务配置
│   ├── detect/            #   检测任务配置
│   └── segmentation/      #   分割任务配置
├── tools/                 # CLI 工具入口
│   ├── train.py           #   训练入口
│   ├── cli.py             #   统一命令行
│   └── ...                #   分析 / 基准测试 / 数据处理工具
├── api/                   # FastAPI 推理服务
├── docs/                  # MkDocs 文档站
├── tests/                 # 单元测试
├── weights/               # 预训练权重 (.pt, .pth)
└── sandbox/               # 本地调试脚本 (gitignored)
```

## 🛠️ 安装

```bash
# 1. 克隆仓库
git clone https://github.com/AlchemyAITech/CellStudio.git
cd CellStudio

# 2. 创建环境 (Python 3.10+)
conda create -n cellstudio python=3.10 -y
conda activate cellstudio

# 3. 安装 PyTorch (根据 CUDA 版本选择)
# 参考: https://pytorch.org/get-started/locally/

# 4. 安装 CellStudio
pip install -r requirements.txt
# 或使用 pyproject.toml (可选依赖按需安装):
# pip install -e ".[all]"
```

## 🚀 快速开始

### 训练

```bash
# 分类
python tools/train.py --config configs/classify/timm_resnet50_mido.yaml

# 检测 (Tile 模式)
python tools/train.py --config configs/detect/yolo_v8m_det_mido_tile.yaml

# 分割
python tools/train.py --config configs/segmentation/cellpose_mido_seg.yaml
```

### 推理

```python
from cellstudio.inference.inferencer import CellStudioInferencer

infer = CellStudioInferencer(
    config_path='configs/classify/timm_resnet50_mido.yaml',
    weight_path='runs/best.pth',
)
result = infer('path/to/image.png')
print(result)
```

### API 服务

```bash
cd api && python main.py
# POST /predict/{model_id} — 图片推理
# POST /train              — 异步训练
# GET  /status/{job_id}    — 查询任务状态
```

### 调试与测试集构建

框架内置了快速提取 tiny 数据集的工具，非常适合用于流水线连通性测试与快速 Dubug：

```bash
# 从完整数据集中提取 120/12/60 规模的极小分类/检测/分割数据集
python tools/create_tiny_datasets.py

# 验证测试集的完整性
python sandbox/verify_tiny.py
```

## 📖 文档

完整 API 文档位于 `docs/` 目录，使用 MkDocs 构建：

```bash
pip install mkdocs-material
mkdocs serve
```

## 🤝 贡献

欢迎提交 Issue 和 Pull Request。提交前请确保：

```bash
ruff check cellstudio/  # 代码检查
pytest tests/            # 单元测试
```
