# CellStudio Development Tasks & Timeline

此任务看板基于 `PROJECT_PLAN.md` 生成。作为本地工程的追踪表，不仅列出了颗粒度的 Task 原语，也给出了预期的时间估算（以纯净无打扰的研发时间计，共约 5 周 / 25 个工作日）。
请自由调整优先级与排期。

**状态图例**：
- [ ] 待开发 (To Do)
- [/] 进行中 (In Progress)
- [x] 已完成 (Done)

---

## Phase 0: 架构切分与核心工程基建设 (预计: 2 days) ✅
*目标：将现有的面条代码剥离为独立强耦合的基座。*
- [x] 重写并落实基础依赖架构（拆分 `cellstudio-core`, `server`, `web` 概念边界）。
- [x] 设计基础统一入口基类（如 `CSUos` 等）。
- [x] 构建自动化的 CI/CD 框架（Lint、Typing 检查）。
- [x] 搭建 Git Submodule 架构（将三个子模块独立为 GitHub 远端仓库）。
- [x] 制定并写入分支命名规范、Conventional Commits、Tagging 策略。
- [x] 清理 `.gitignore`（防止模型权重、`__pycache__`、`.egg-info` 入库污染）。
- [x] 从 Git 历史中清除 1GB+ 的误提交权重文件。

## Phase 1: 基础数据栈重构 - Data Foundation (预计: 3 days) 🔧
*目标：打通高自由度的"图形基类"与"UDF去中心化极简存储"。*
- [x] 实现并重构基础图元类（`CSUPoint`, `CSUBBox`, `CSUPolygon`, `CSUMask`, `CSULabel`）。
- [x] 完成 "Total-Part" 的 UDF JSON 去中心化加载拆分（解决大 JSON 爆内存问题）。
- [x] 实现 RDP (`shapely.simplify`) 算法多边形瘦身存储。
- [x] 为 `CSUProperties` 增加 `source_type` (human/algorithm) 与 `match_info` 字段。
- [x] 构建相似度匹配引擎 `MatchCache` (`compute_udf_iou`) 并可视化验证。
- [x] **(✅ 已修复)** 解决 `visual_aug.py` 中 `IndexError` (因 Augmentation 裁剪造成的 bbox 与 label 索引偏移错位)。
- [ ] 完善 `UDFDataset` 及统一的 Dataloader 读取流验证（对三种任务类型跑通端到端 DataLoader 迭代）。
  - **[新增]** 构建基础张量还原绘图探针：提取 DataLoader Batch，逆归一化并叠绘 BBox/Mask。
  - **[新增]** UDF字典一致性可视化：实现 JSON 原始解析（红线）与 Dataloader 提取流（绿线）的 100% 同屏像素对齐校验。
- [ ] 构建数据增广 (Data Augmentations) 可视化打靶引擎 (`tools/debug_visual_aug.py`)。
  - 强制启动大尺度空间扭曲 (`ElasticTransform` / 旋转偏移等)。
  - 并排比较 Source Image 与 Augmented Image，彻底用肉眼排查因像 `IndexError` 类矩阵越界或漂移导致标注“脱靶”的多边形漏洞。
- [ ] 各类型 WSI 读取引擎桥接（`OpenSlide`/`TiffFile`）。
  - **[新增]** 滑窗连通性（Seams）拼图可视化查验，确保切割与重组无特征断裂。

## Phase 2: 模型训练闭环引擎 - Trainer & Evaluator (预计: 5 days) 📋
*目标：整合各 SOTA 视觉大模型引擎基类，配合论文实验全量跑通。*

### 2.1 模型适配器验证
- [ ] Classification: 验证 `timm_adapter` 对 ResNet18/50, EfficientNet-B3/B4, ConvNeXt, MobileNetV3, ViT 的挂载。
- [ ] Classification: 验证 `ultralytics_adapter` 对 YOLOv8-cls, YOLOv11-cls, YOLOv26-cls 的挂载。
- [ ] Detection: 验证 `ultralytics_adapter` 对 YOLOv8-det, YOLOv26-det 的挂载。
- [ ] Detection: 验证 `mmdet_adapter` 对 Faster R-CNN, DETR, FCOS, RTMDet, RetinaNet 的挂载。
- [ ] Segmentation: 验证 `smp_adapter` 对 UNet, DeepLabV3+ 的挂载。
- [ ] Segmentation: 验证 `ultralytics_seg_adapter` 对 YOLOv8-seg 的挂载。
- [ ] Segmentation: 验证 `cellpose_adapter` 对 Cellpose, Cellpose-SAM 的挂载。

### 2.2 Loss 体系补全
- [ ] 分类 Loss：CrossEntropy, Label Smoothing CE, Focal Loss (for class imbalance)。
- [ ] 检测 Loss：Classification Loss + Box Regression Loss (GIoU/CIoU) + Objectness Loss (完整多分支)。
- [ ] 分割 Loss：Dice Loss + BCE Loss + Boundary Loss。

### 2.3 Metrics 模块补全 (对齐论文需求)
- [ ] 分类：Accuracy, AUC-ROC, Sensitivity, Specificity, F1, Cohen's Kappa, PR-AUC, Confusion Matrix。
- [ ] 检测：mAP@0.5, mAP@0.5:0.95, FROC 曲线, Precision, Recall, F1, Count Error。
- [ ] 分割：Dice, 95% HD, mIoU, AJI (Aggregated Jaccard Index), PQ (Panoptic Quality)。

### 2.4 训练流控
- [ ] `tools/train.py` 跑通从入参、Loss、Metric 到权重保存的全链路。
- [ ] 训练过程可视化：Loss 曲线、LR 曲线、Metric 曲线实时输出。
- [ ] 验证 Early Stopping 与 ReduceLROnPlateau 逻辑生效。

## Phase 2.5: 论文实验全量矩阵执行 (预计: 3 days) 📊
*目标：满足毕业论文所要求的完整模型对比实验矩阵。*
- [ ] Classification: 跑完 ResNet50, EfficientNet-B4, ViT-Base, ConvNeXt, YOLOv8-cls, YOLOv26-cls (6 模型 × 100 epochs)。
- [ ] Detection: 跑完 YOLOv8-det, YOLOv26-det, Faster R-CNN, FCOS, RTMDet, DETR, RetinaNet (7 模型 × 100 epochs)。
- [ ] Segmentation: 跑完 UNet, DeepLabV3+, YOLOv8-seg, Cellpose, Cellpose-SAM, Mask R-CNN (6 模型 × 100 epochs)。
- [ ] 5-Fold 交叉验证 (对核心锚定模型执行)。
- [ ] Bootstrap 置信区间估计 (1000 次重抽样)。
- [ ] 假设检验 (T-Test / Wilcoxon) 模型间两两对比 P-value。
- [ ] 消融实验：数据增强开关、Backbone 替换。
- [ ] Grad-CAM 可解释性热力图生成。

## Phase 3: 推理引擎与结果重构 - Inferencer (预计: 4 days)
*目标：实现端到端的长链预测与大图超分辨率拆解拼合。*
- [ ] 编写 `BaseInferencer` 基类与单图前向方法。
- [ ] 实现核心超大 WSI 切图预测与后处理合并重构（`sliding_window_inference` + NMS / Block Blend）。
- [ ] 将 Tensor 结果强制反向序列化挂载回 `CSUFeatureCollection`。
- [ ] 推理热力图 (Heatmap) / 特征重投影绘制输出引擎构建。
- [ ] 封装独立测试 `tools/test.py`。

## Phase 4: Nuitka 发版预编译与 IP 保障 - Exporter/Deployer (预计: 3 days)
*目标：构建跨场景交付与纯二进制"死代码"导出流水线。*
- [ ] 编写并测试 Nuitka C++ 一键式编译导出脚本 `builder/build_core.py`。
- [ ] 编写模型权重的加密/内存映射解密流程。
- [ ] 场景容器化支持（打包标准的 `Dockerfile`）。
- [ ] 内网 SaaS 场景：增加宿主机唯一识别符检查探针（UUID/MAC 验证拦截器）。

## Phase 5: 分析和医学特征模块 - Analytics/Stats (预计: 2 days)
*目标：医学影像报告特供。*
- [ ] 统计层：集成 T-Test, 方差分析代码块。
- [ ] 可视化层：加入混淆矩阵、ROC/AUC 计算生成件。
- [ ] 空间特征挖掘：增加细胞族群分型，聚类等算法引擎扩展（TME环境统计）。

## Phase 6 & 7: 业务外环与后端 API / Web Server (预计: 2~3 迭代)
*此部分剥离于当前纯算法工作区以外，建议并行或后期推进。*
- [ ] 构建 `cellstudio-server` 引擎：FastAPI 后端挂载 `core` 接口。
- [ ] 开发权限 RBAC 与多租户数据落盘分离挂载引擎。
- [ ] WSI 前后端大屏交互流控（整合 WebSocket 与重媒体传输）。
- [ ] WSI "Human-in-the-Loop" 人工修正与自动发流。
