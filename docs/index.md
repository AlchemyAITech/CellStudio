# CellStudio 文档

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/Framework-FastAPI%20%7C%20PyTorch-red.svg)](https://pytorch.org)

欢迎来到 **CellStudio** 官方文档站 — 面向细胞病理学的全生命周期深度学习开发框架。

## 架构设计

CellStudio 采用 **Registry + Hook + Config** 三位一体的模块化架构：

- **Registry 插件机制** — 模型、数据集、指标、可视化组件通过统一的 `Registry` 类注册，实现完全解耦的热插拔扩展。
- **Hook 生命周期** — 训练循环通过 `EpochBasedRunner` 驱动，日志、优化器、检查点、评估等横切关注点全部由 Hook 注入。
- **Config-Driven** — 一切超参数收敛到 `.yaml` 配置文件，支持 `_base_` 继承和 OmegaConf 变量插值。

## 快速导航

- [统一命令行 (CLI)](cli.md) — 一键启动训练、评估和批量实验
- [REST API 服务](api.md) — FastAPI 部署端点：推理、异步训练、任务监控
- [配置参数指南 (Config)](config.md) — YAML 配置字段详解
- **API Reference** — 核心框架接口文档
  - [Engine (Runner & Hooks)](api/engine.md)
  - [Tasks](api/tasks.md)
  - [Datasets](api/datasets.md)
  - [Metrics](api/metrics.md)
  - [Backend Adapters](api/backends.md)
  - [Inference](api/inference.md)
  - [Data Structures](api/structures.md)
