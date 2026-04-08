# CellStudio MLOps 工业平台

![License](https://img.shields.io/badge/License-MIT-blue.svg) ![Framework](https://img.shields.io/badge/Framework-FastAPI%20%7C%20PyTorch-red.svg)

欢迎来到 **CellStudio MLOps 平台** 官方文档站！这一套从零打磨的企业级深度学习视觉开发栈，摒弃了传统随意的科研脚本习惯，专为高标准的工业场景打造。

## 我们的基座哲学 (Zenith Architecture)
很多开源架构将 “训练”、“测试” 和 “部署” 割裂成不重叠的代码仓，使得模型脱机使用异常艰难。
而在我们的架构中：
- **`EpochRunner` 层面深度解耦**：一切均由 Hooks 定义，完全剥离前行迭代逻辑。不管是 YOLO 还是 Timm。
- **配置即真理 (Config-Driven)**：一切超参数全量收敛到 `.yaml`。业务方不再需要读 Python 源文件。
- **全生命周期接管**：通过内置的 `tools/cli.py` 统御训练、评估，通过内置的 `api` 端点自动接管热缓存。

## 探索功能
左侧有更详尽的内容体系：
- [统一命令行矩阵 (CLI)](cli.md): 查阅如何用一行 Bash 启动训练或完成跑批测试。
- [Web 云微服务 (API)](api.md): 查阅如在没有代码基础的业务端架设后台守护服务器并启动异步炼丹。
- [配置说明辞典 (Config Guide)](config.md): 查阅模型 `.yaml` 里的奇特字段，如 `class_weights` 等干预手段。
