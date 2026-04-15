# CellStudio Development Tasks & Timeline

此任务看板基于 `PROJECT_PLAN.md` 生成。作为本地工程的追踪表，不仅列出了颗粒度的 Task 原语，也给出了预期的时间估算（以纯净无打扰的研发时间计，共约 4 周 / 20 个工作日）。
请自由调整优先级与排期。

**状态图例**：
- [ ] 待开发 (To Do)
- [/] 进行中 (In Progress)
- [x] 已完成 (Done)

---

## Phase 0: 架构切分与核心工程基建设 (预计: 2 days)
*目标：将现有的面条代码剥离为独立强耦合的基座。*
- [x] 重写并落实基础依赖架构（拆分 `cellstudio-core`, `server`, `web` 概念边界）。
- [x] 设计基础统一入口基类（如 `CSUos` 等）。
- [x] 构建自动化的 CI/CD 框架（Lint、Typing 检查）。

## Phase 1: 基础数据栈重构 - Data Foundation (预计: 3 days)
*目标：打通高自由度的“图形基类”与“UDF去中心化极简存储”。*
- [x] 实现并重构基础图元类（`CSUPoint`, `CSUBBox`, `CSUPolygon`, `CSUMask`, `CSULabel`）。
- [x] 完成 "Total-Part" 的 UDF JSON 去中心化加载拆分（解决大 JSON 爆内存问题）。
- [x] 实现 RDP (`shapely.simplify`) 算法多边形瘦身存储。
- [/] **(🚨当前阻塞任务)** 解决 `visual_aug.py` 中 `IndexError` (因 Augmentation 裁剪造成的 bbox 与 label 索引偏移错位)。
- [ ] 完善 `UDFDataset` 及统一的 Dataloader 读取流验证。
- [ ] 构建相似度匹配引擎 `MatchCache` (`compute_udf_iou` - 算法预测与人工 GT 合并基石)。
- [ ] 各类型 WSI 读取引擎桥接（`OpenSlide`/`TiffFile`）。

## Phase 2: 模型训练闭环引擎 - Trainer & Evaluator (预计: 4 days)
*目标：整合各 SOTA 视觉大模型引擎基类。*
- [ ] 测试数据集转化与接入 (Classification, Detection, Segmentation 各一套 `tiny` 极限数据集)。
- [ ] 实现 `cellstudio.models` 路由表与各类算法拓扑映射架构。
- [ ] 补全并联调各类 Loss（包括检测的多维 Loss 分支补充）。
- [ ] 搭建 `Metrics` 模块系统，补充所有性能验证算子（Dice, HD95, mIoU, AP）。
- [ ] 构建 `tools/train.py` 跑通从入参、Loss、Metric到权重的验证流控，并产出 Training Log。

## Phase 3: 推理引擎与结果重构 - Inferencer (预计: 4 days)
*目标：实现端到端的长链预测与大图超分辨率拆解拼合。*
- [ ] 编写 `BaseInferencer` 基类与单图前向方法。
- [ ] 实现核心超大 WSI 切图预测与后处理合并重构（`sliding_window_inference` + NMS / Block Blend）。
- [ ] 将 Tensor 结果强制反向序列化挂载回 `CSUFeatureCollection` 。
- [ ] 推理热力图 (Heatmap) / 特征重投影绘制输出引擎构建。
- [ ] 封装独立测试 `tools/test.py`。

## Phase 4: Nuitka 发版预编译与 IP 保障 - Exporter/Deployer (预计: 3 days)
*目标：构建跨场景交付与纯二进制“死代码”导出流水线。*
- [ ] 编写并测试 Nuitka C++ 一键式编译导出脚本 `builder/build_core.py` (研发态隔离，测试发版态)。
- [ ] 编写模型权重的加密加密/内存映射解密流程。
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
