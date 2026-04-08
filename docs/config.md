# CellStudio Deployment Configuration Guide (配置说明篇)

无论是进行算法端全量评测，还是提供给业务方进行 FastAPI 服务化预热，配置表单（YAML Configuration）都是连接两端的唯一通行证。
本文档解析这套系统中对外暴露的所有配置文件结构极其配置含义。

---

## 一、服务器环境预热配置 (`api/config.yaml`)

业务侧只需关注由微服务掌管的这唯一一个总入口枢纽文件。

### 1.1 `server` 节点块 (Server Block)
定义了 FastAPI uvicorn 底层运行环境。
```yaml
server:
  host: "0.0.0.0"   # 服务端暴漏的 IP 域，0.0.0.0 允许广域/局域网外网透传
  port: 8000        # 对外映射 HTTP 端口号
  workers: 1        # 多进程承载数目（在纯深度学习推理中推荐设小，防止显存 OOM。因为 Uvicorn worker 会复制整个缓存池）
```

### 1.2 `models` 节点块 (Model Registry Block)
真正决定了服务端启动后，对外挂载了多少把“兵器”。
其中每一个顶层字典键名 (`resnet18_test` 等) 都隐式生成了一个独立的 API 端点 `/predict/resnet18_test`。

```yaml
models:
  # 你想叫什么端点名都可以（注意不要带特殊符号）
  resnet18_test:
    config: "../configs/classify/timm_resnet18_mido.yaml"   # 模型配套的构架与流解析文件，要求写相对 api/ 所在根的路径
    checkpoint: "../work_dirs/timm_resnet18_mido/best.pth"  # 实打实的物理权重
    device: "cuda"                                          # 加载到显卡(cuda)还是内存(cpu)
    
  # 你可以同时平行挂载 YOLO，甚至另外一个业务方向的模型
  mido_yolo:
    config: "../configs/classify/yolo_v8m_cls_mido.yaml"
    checkpoint: "../work_dirs/yolo_v8m_cls_mido/best.pth"
    device: "cuda"
```

---

## 二、模型本体训练/推理架构配置 (`configs/xxx.yaml`)

模型本体配置文件随权重复刻，作为 API 提取**数据预处理手段（Transforms）**的核心来源。
作为使用者，通常只需要检查并关注两个区域：

### 2.1 数据均衡参数 (`class_weights`)
位于 `model` 定义块下。如果您想干预前向损失：
```yaml
model:
  type: TimmClassifier
  architecture: resnet18
  # NMF（负样本）较多，AMF较少，可以利用权重补偿：[NMF_WH, AMF_WH]
  # 这一段在训练时起作用，API 推断时该数组仍被静默解析但不干预前向推理结果
  class_weights: [1.0, 2.0]
  num_classes: 2
```

### 2.2 前置管道配置 (`val_dataloader.dataset.pipeline`)
部署推断时（Inferencer），代码逻辑**只读取**这段管道配置！
如果你在服务端上线前更改了均值方差或前置尺寸，你的 API 对同一张图片的判断结果可能完全不同。
```yaml
val_dataloader:
  dataset:
    pipeline:
      - type: LoadImageFromFile
      - type: Resize
        size: [224, 224]            # Web 端上传图片将无条件被强制拉伸到 224
      - type: Normalize             # 底层预热时的标准化偏移
        mean: [123.675, 116.28, 103.53]
        std: [58.395, 57.12, 57.375]
      - type: PackInputs
        keys: ['img']
```
