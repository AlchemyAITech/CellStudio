# CellStudio 总体研发流程图与项目进度树

本计划大纲聚焦于具体的工程化实现、API抽象以及可编程模块的业务逻辑构建，去除冗余概念，落实到软件框架层面。

## 0. 系统分层总纲
- **(1) 底层计算域**：基础数据模块 (Dataloader/UDF)、训练模块 (Trainer)、推理模块 (Inferencer)、部署模块 (Exporter/Deployer)、统计模块 (Metrics/Analytics)。
- **(2) 逻辑转译层**：各类 CLI 入口 (如 `tools/train.py`)、API 业务接口 (FastAPI HTTP Server) 及工作流总控。
- **(3) 顶层表示层**：标注页面 Web 组件、训练监控操作盘、推理测试报告面板、部署一键发版端、统计分析图表前端引擎。

---

## 0. 代码仓库解耦与架构切分矩阵 (Repository Splitting Strategy)
鉴于 CellStudio 包含了极重的底层深度学习算力调度与复杂的交互级 GUI Web 界面，为了避免依赖冲突、流水线混乱以及前后端开发互相掣肘，强烈建议将当前架构解耦为 **3 个正交的独立代码仓库 (Repositories)**：

- **(A) `cellstudio-core` (底层纯算法库)**：只包含 Python 深度学习基座本身 (`models`, `datasets`, `engine`, `metrics`)。不包含任何 Web 接口和 GUI。以 `pip install cellstudio_core` 的形式作为底层 SDK 供其他服务调用。
  - **核心技术栈选型**：`PyTorch 2.x` (深度学习张量引擎), `OpenCV` / `Pillow` (像素级处理), `OpenSlide` / `TiffFile` (WSI 巨幅金字塔图像解析层), `Shapely` (底层矢量多边形与几何算法), `Nuitka` (编译发版态 C++ 二进制转换)。
- **(B) `cellstudio-server` (中间件与业务后台)**：Python 实现。基于 FastAPI 构建，封装对 `core` 库的资源调度任务。负责向外暴露标准 HTTP/gRPC API，并统筹数据库读写、租户鉴权体系、配额记账与长连任务排队。
  - **服务端技术栈选型**：`FastAPI` + `Uvicorn` (极速异步 REST API 引擎), `SQLAlchemy` + `PostgreSQL` (关系型业务元数据与 ORM), `Redis` (热数据缓存与多租户分布式锁), `Celery` (超长生命周期如训练/大图推理离线异步任务队列引擎), `JWT` (跨终端状态保持鉴权)。
- **(C) `cellstudio-web` (顶层可视化前端与前端工作站)**：前后端分离纯前端项目。包含科研数据看板、可拖拽流水线大屏以及带实时渲染反馈的高刷新率巨幅病理切片标注操作台。
  - **前端技术栈选型**：`Vue 3` + `TypeScript` + `Vite` (核心前端骨架与编译工具链), `TailwindCSS` (现代原子化 CSS 布局), `OpenSeadragon` (专门攻克 10 万像素级 WSI 无极缩放金字塔图层引擎), `ECharts` (医学统计全景图表), `WebSockets` (双工长连接以支撑 AI 辅助预打标的实时边渲染推送)。

### 版本控制架构：Git Submodule 治理规范

本项目采用 **Superproject + Submodule** 架构模式。父仓库 `CellStudio` 作为总指挥仓 (Superproject)，通过 Git Submodule 机制引用三个独立子仓库。每个子仓库拥有完全独立的 Git 历史、分支与版本号，父仓库则锁定各子模块到特定的 commit SHA 上，确保任何时刻 checkout 出来的整体系统都是配套可用的。

**仓库拓扑**：
```
CellStudio/                    ← Superproject (父仓库)
├── .gitmodules                ← Submodule 注册清单
├── cellstudio-core/           ← Submodule → repos/cellstudio-core.git
├── cellstudio-server/         ← Submodule → repos/cellstudio-server.git
├── cellstudio-web/            ← Submodule → repos/cellstudio-web.git
├── datasets/                  ← 数据资产 (不入版本管理)
├── work_dirs/                 ← 训练产物 (不入版本管理)
├── PROJECT_PLAN.md
└── DEVELOPMENT_TASKS.md
```

#### (1) 分支命名规范 (Scoped Branching)
所有功能分支必须携带模块作用域前缀，以便 Code Review 与 CI 时快速识别影响范围：
- `core/feat-rdp-compress` — 算法层功能研发
- `core/fix-resize-indexerror` — 算法层 Bug 修复
- `server/feat-jwt-auth` — 服务端功能研发
- `web/ui-wsi-canvas` — 前端界面研发
- `global/refactor-udf-schema` — 横跨多个子仓库的全链路改动（在父仓库中操作）

#### (2) 提交规范 (Conventional Commits)
每次 `git commit` 强制遵循 `type(scope): description` 格式：
- `feat(core): add MatchCache for computing feature IoU`
- `fix(core): resolve bbox-label index mismatch in Resize transform`
- `feat(server): implement async Celery task for WSI inference`
- `refactor(global): migrate to submodule architecture`

#### (3) 版本号与发布标签策略 (Tagging)
采用**独立发版**策略，各子仓库维护自身语义化版本号：
- `cellstudio-core` → Tag `core-v0.3.0`
- `cellstudio-server` → Tag `server-v0.1.0`
- `cellstudio-web` → Tag `web-v0.1.0`

父仓库在里程碑节点合并时打统一系统版本 Tag（如 `v1.0.0`），锁定当时三个子模块的 SHA 快照。

#### (4) 日常开发流程 (Workflow)
```bash
# 首次克隆（递归拉取所有子模块）
git clone --recurse-submodules <repo-url>

# 进入子模块独立开发
cd cellstudio-core
git checkout -b core/feat-new-metric
# ... 编辑、提交 ...
git push origin core/feat-new-metric

# 回到父仓库，更新子模块指针
cd ..
git add cellstudio-core
git commit -m "chore(global): bump cellstudio-core to latest"

# 同步所有子模块到最新
git submodule update --remote --merge
```

#### (5) CI/CD 路径触发规则 (Path Filtering)
- 当 `cellstudio-core/**` 变动时 → 触发 PyTorch/CUDA 算法测试流水线
- 当 `cellstudio-server/**` 变动时 → 触发 FastAPI 接口测试与 Docker 构建
- 当 `cellstudio-web/**` 变动时 → 触发 Node.js/Vite 前端构建与部署

#### (6) 迁移至远端 (GitHub/GitLab)
当前子模块 remote 指向本地裸仓库 (`e:\workspace\AlchemyTech\repos\*.git`)。迁移到远端只需在各子仓库中执行：
```bash
git remote set-url origin https://github.com/AlchemyAITech/cellstudio-core.git
git push -u origin master
```
父仓库的 `.gitmodules` 文件中的 URL 也需同步更新。

---

## 1. 基础数据模块：通用数据结构构建 (Data Foundation)
- **(0) 统一底层空间图元数据基类 (Primitive Data Structures)**：摒弃以“任务（分类/检测/分割）”为导向的固化架构。底层仅抽象实现各类纯粹的基础空间与属性类型（如 `CSUPoint`, `CSUBBox`, `CSUPolygon`, `CSUMask`, `CSULabel`）。将各类任务的输出结果回归到这些基本图元的组合上进行装配适配，从而做到真正的底层模块化与高自由度。
- **(1) 模型训练用标注数据结构（人工数据）**：依托上述图元组件构建的真值集合（标识 `source_type='human'`），负责流转业务专家产生的数据图元。
- **(2) 模型推理用数据结构（算法数据）**：依托上述图元组件对模型推理结果进行强制映射（标识 `source_type='algorithm'`），无论多么复杂的下游任务输出，最终都降维转换为以上标准图元类型的组合输出。
- **(3)** 支持算法数据和人工数据的相似度匹配，对于匹配上的算法数据，加上人工标注的 ID 标识，并记录相似度。
  - *工程落实*：在底层增加 `MatchCache` 工具，提供 `compute_udf_iou(pred_feature, gt_feature)` 接口，当 `IoU > thresh` 时映射 ID 作为外键保存，并留出 `confidence` 和 `match_metric` 属性（该部分剥离为通用库 `cellstudio/utils/match.py`，供未来前后端预打标复用修正）。
- **(4)** 各类型 WSI 读取显示：封装底层滑窗与金字塔层级引擎（接入 `OpenSlide` 或 `TiffFile` API）。
- **(5)** 数据增广：在 Pipeline 层实现可配拔插的 Transforms 函数链 (`Resize`, `RandomFlip`, 参数色彩偏移等)。
- **(6)** 构建全类型测试数据，并按照当前设计的 JSON 数据结构实现数据集自动化实例化存储脚本。
- **(7)** 测试验证数据结构正确：编写并运行各类 `tests/integration` 端到端回归测试用例。
- **(8)** 逐步支持各类型数据（分类框、实例分割多边形、关键点等）流式特征标注反写模块。

## 2. 通用模型训练/验证构建 (Trainer & Evaluator)
- **(1)** 按架构树顺序逐步开发基类支持：图像分类 (`cls`)、目标检测 (`det`)、图像分割 (`seg`)、多属性学习、MIL (多实例)、反卷积重建、回归预测、非监督聚类、空间分析、Zero-shot/Few-shot 推演等。
- **(2)** 构建对应算法的各类模型工厂 (Model Registry)：
  - 支持将算法拓扑与执行参数独立保存（剥离为高内聚 `yaml` 或 `json`），以便后续前后端 API 能以反序列化的途径直接将模型转置为可视化配置表单。
  - 对于类型相同/跨任务的 Common Args（如 `lr`, `batch_size`, `optimizer`）做抽象合并。
- **(3)** 针对每个模型任务池，自动化产出或构建独立的、验证模型连通性的极限微缩测试集 (Tiny-sets)。
- **(4)** 构建全量训练模型的评价指标组件 (`metrics`)：代码层面完整实现如 Dice, HD95, mIoU, AP 等函数，支持 `Config` 层动态控制需要统计输出哪些指标计算器。
- **(5)** 构建训练用 CLI (`tools/train.py`)、并针对 Python 后端导出对应的 `train_api(...)` 编程黑盒接口。
- **(6)** 训练调试系统搭建与可视化流控：中间数据存储输出 (通过 `dump_hooks`)，包括全流程对齐如原始 Image、前置 Image+Label 覆盖掩膜图、Loss 数学曲线日志输出系统。
- **(7)** 全类型模型回归遍历：将 (1) 中涵盖的所有单一算法引擎，进行统一的自动化单元通过测试。

## 3. 通用模型测试/推理 (Inferencer)
- **(1) 基础可编程推理接口 (`BaseInferencer`)**：支持底层直接挂载单张图像 `predict_image()` 和 批量图像流 `predict_batch()` 的逻辑引擎。
- **(2) WSI 超大图推理拼接层**：实现程序滑窗切块 (`sliding_window_inference`) 预处理，执行推理后再以 NMS（非极大值抑制）对拼缝输出的特征重构组合。
- **(3) 输出规范化处理**：推理输出结果对象化，把零散 Tensor 强制序列化清洗，输出回统一的 `CSUFeatureCollection` 结构。
- **(4) 推理引擎工具与可配层**：实现支持输出分类热力图表现 (`heatmap`)，及掩码覆盖混合层渲染器。
- **(5) 构建测试用 CLI/API**：如 `tools/test.py` 及后端开放的 `FastAPI` 预留接口（如 `/api/v1/models/{id}/predict`），能够同时应对轻量图回调与大图异步任务结果下放的逻辑流转。

## 4. 通用模型部署与全场景兼容架构 (Model Exporter, Deployer & IP Protection)

在架构设计上，底层 `cellstudio-core` 与外部 `cellstudio-server`/`cellstudio-web` 的强隔离解耦是实现跨场景兼容的核心。基于此解耦，系统需原生涵盖以下三种业态下的代码与模型 IP 保护及分发能力。

> **[IMPORTANT] 关于 研发态 (Development) 与 发版态 (Release) 的底层形态界定**
> - **研发态**：`cellstudio-core` 全程保持纯正 Python 原生代码。保障算法迭代、断点调试逻辑的极致灵活性，便于科学家研究与调优。
> - **交付发版态**：触发 Exporter 流水线打最终产品包时，全量 Python 业务代码将被调用 **Nuitka 编译器**，由 C++ 编译器降级压铸成操作系统原生的二进制动态库 (Windows `.pyd` / Linux `.so`)。最终客户环境内**彻底不存在任何 Python 源码**，实现彻底的“死代码”重工业级防逆向交付。

- **场景 A: 纯净 To-C 本机部署 (研发者/科研单机桌面端)**
  - *部署形态*：直接在本地启动 `core` 和轻量 `server` 进程。可利用 Electron 将 Web UI 原生包覆为桌面软件，实现断网离线态使用。
  - *源码及 IP 保护*：强依赖前端编译级防御。使用 **Nuitka 转译编译方案** 将所有 Python 业务文件锁死成 C++ 二进制动态库（阻止窥视与逆向）；模型文件 (Weights) 采用物理文件 AES-256 强加密，辅以后台启动时的**内存直写级解密 (In-Memory Decryption)**，确保大模型资产不落盘。

- **场景 B: 公网 SaaS 大体量云端部署 (商用公众开放平台)**
  - *部署形态*：标准的云原生集群架构 (AWS/阿里云)。`server` 群接入全流量高可用网关与 Nginx 集群，`core` 推理部分退居二线成为纯并用算力 Worker（借助 RabbitMQ / Celery 等消息列队），应对庞大 WSI 的阻塞。`web` 则投递于全球加速 CDN。
  - *源码及 IP 保护*：物理级绝对隔离防御。采用最优的**不交付代码隔离策略**。客户无权访问执行环境，仅拥有端点与页面交互权。鉴于环境高控，云集群内部的模型直接转存为极速的裸 TensorRT Engine/ONNX 格式执行以榨取极限并发。

- **场景 C: 医院本地 SaaS 内网私有化部署 (医疗机构局域网硬解压一体机)**
  - *部署形态*：单机或边缘端 K8s 的“缩微版独立云架构”。一套基于 Docker Compose 的打包，把场景 B 所有的微服务下放到单台实体机中，直接与医院的 PACS / LIS 系统通过网线协议对话。 
  - *源码及 IP 保护*：交织型复合防御。在 Nuitka 掩码与 Docker 私有化封禁之上，强切入 **机器及算力指纹绑定 (Hardware Fingerprinting)**。引擎启动探针须强制扫描并核对主机所在网卡的硬 MAC 地址或所在搭载英伟达 GPU 的原生 UUID 芯片验证。若发生容器跨设备克隆偷盗，核心将自锁或阻断初始化。

- **(1) 资产转化与加固编译器 (Exporter/Obfuscator)**：构建一键式的预发布清洗脚本栈。完成模型解绑 (To TensorRT/ONNX)、权重打码加密、与 Python 业务代码的库预编译指令调用。
- **(2) 高可用环境容器化建设 (Containerization)**：输出全场景贯通兼容的标准 `Dockerfile` 与配置表。
- **(3) 高并发队列与协议桥接层**：封装统一的 Redis 流控制任务管道以对应 WSI 的极长吞吐时延；同时利用桥接件支持现有第三方服务网关挂接 (如 Triton Inference Server / KServe)。

## 5. 科研统计模块 (Analytics & Stats)
- **(1) 纯医学统计算子库封装**：实现（或在服务端桥接 Python/R 包接口例如 `rpy2` 或 `scipy.stats`）以支持各种 T检验 (T-Test), 方差分析 (ANOVA) 与卡方检验业务类方法。
- **(2) 深度性能报表生成器**：实现基于模型输出数据集级的混淆矩阵生成、ROC/AUC 曲线计算绘图代码，以及特异度与敏感度报表等逻辑层代码实现。
- **(3) 数据透视异常值监控功能**：利用孤立森林或极值抛光算法实现结构数据噪音识别，剥离异样输出/非法检测区域，并执行数据表格的空值规整清洗。
- **(4) 生存与聚类挖掘集成**：封装 Kaplan-Meier 生存评估时间计算模型；集成聚类表征测算用于肿瘤微环境（TME）细胞分群占比等空间业务。

## 6. 可视化操作界面端 (GUI & Middleware Services)
- **(1) Pipeline 面板驱动器层**：实现打通从“构建标注配置”、“超参拖拉更改”、“发起测训任务”到“监控返回统计表盘”数据闭环交互的前端对接中间件。
- **(2) WSI 重媒体切片标注中心引擎**：实现集成类似 `OpenSeadragon` 的前端画板核心，挂接 WebSockets 使人工前端轮廓绘制能和后端 Python 推理端点的 `BBox` 发送识别产生“一键高亮交互”(Human-In-The-Loop)。
- **(3) 应用商店管理中心枢纽 (Model Zoo Services)**：针对现存或不训练仅调用的 AI 基础大模型，实装基于模型表的统一资源纳管。对模型定义“资产属性”(Name, Version, Engine Type) 的关系型库结构，提供拉起使用的前端钩子。
- **(4) 部署状态面板系统**：支持前线直接监测推流 Docker 的 API 状态响应及健康检测探针 (`health_checks`)。

## 7. 完善业务支撑逻辑 (Business Infrastructure)
- **(1) 组织与数据切分容器 (Tenants API)**：实现基于多租户库隔离的数据管理层 API，确保数据上传与使用的跨机构沙盒安全。
- **(2) 鉴权与业务角色引擎 (RBAC Policy)**：代码化实现 JWT Token 分发方案、以及“打标人员”、“模型管理员”、“病理审查人”三种以上层级鉴权路由装饰隔离。
- **(3) 记账和调度锁**：在算力/时效层下构建 GPU 限时 Hook 和配网锁 (Limit quotas)，保障对消耗硬件的服务实现计价或阻断监控逻辑。
