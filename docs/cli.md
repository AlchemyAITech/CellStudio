# 统一网关与命令行 (Unified CLI)

`CellStudio` 不支持您直接运行杂乱的 `.py` 文件！一切终端活动都强制由**唯一的终端入口网关 `tools/cli.py`** 接管查收。这种中心化的调度方式可以有效隔离配置环境变量带来的混乱。

## 通用格式
```bash
python tools/cli.py <子命令> [必选参数] [--可选参数]
```
我们支持的子命令有三大项：`train` (训练)、`eval` (测试与矩阵计算)、`infer` (极速脱机端点验证)。

---

## 1. 启动大模型训练 (`train`)

**适用环境：** 您希望开启模型训练循环。

**终端范例：**
```bash
python tools/cli.py train configs/classify/timm_resnet18_mido.yaml
```

**参数详解：**
- `config` (必要): 指向欲拉起的蓝图 YAML 配置。
- `--work-dir` (可选): 强制指定本次训练落盘的日志、参数、权重位置。若不覆盖，默认存放至 `work_dirs/[config_name]` 目录下。

---

## 2. 脱机全量评估 (`eval`)

**适用环境：** 训练已彻底完成。你想单独在一个测试集里批量跑完上千张图，并算出 ROC、FPS 和 FLOPs 数学参数时使用！

**终端范例：**
```bash
python tools/cli.py eval configs/classify/timm_resnet18_mido.yaml work_dirs/timm_resnet18_mido/best.pth
```
- 命令会将当前传入的模型文件实例化，并从测试集的 `DataLoader` 里获取数据。

---

## 3. 工具级极速打分 (`infer`)

**适用环境：** 当你只需要快速判定某**一张**特定图片的结果，不依赖验证循环的开销，此项完全轻量化处理您的入参。

**终端范例：**
```bash
python tools/cli.py infer configs/classify/timm_resnet18_mido.yaml work_dirs/timm_resnet18_mido/best.pth --image 123.jpg --device cuda --out logs.json
```
- 直接向本地磁盘导出 JSON 格式推断概率簇结构。
