# 生产端全周期微服务 (FastAPI Server)

我们的服务端 API 系统并不仅满足于让前端上传一张照片反馈结果。
这是一个名副其实的 **AI MLOps 全生命周期管理服务器**！它甚至支持以异步非阻塞形式在服务器后台自动新开训练，并将状态不断回传！

**极速避撞启动方案**：
```bash
# 默认占用工业服务大端 18080：
python api/main.py 
```

一旦启动，强烈建议访问本地自带的纯网页交互站：  
> 👉 **http://localhost:18080/docs** (自动构建的 OpenAPI Swagger 服务台)

---

## 1. 实时高速预测 (`POST /predict/{model_id}`)
最常用的零延迟前向判定端点。
在启动的瞬间，此节点会读取 `api/config.yaml` 自动将数十个大型网络常驻加载入 VRAM 显存。
上传一张 `UploadFile` 二进制图片即可瞬间下发带有最高置信度的判定。

**CURL 例子：**
```bash
curl -X 'POST' \
  'http://localhost:18080/predict/resnet18_test' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@my_slide.jpg;type=image/jpeg'
```

---

## 2. 异步训练托管 (`POST /train`)
无需登录远程服务器跑 SSH！向此 Endpoint 发送一份合法的 YAML 配置纯文本，服务层将于后台 OS 层启动 `subprocess` 触发原生框架流水线。调用端绝不会阻塞，系统会立即丢回带有追查权的 `job_id` 序列号。

**应用场景：** 让业务中台系统跨库对模型发起热训练请求。

---

## 3. 异步测试托管 (`POST /evaluate`)
同理训练流，将测试逻辑推入后台操作系统线程池并发执行。

---

## 4. 进度轮询总控 (`GET /status/{job_id}`)
配合前两个端点使用！前端可通过 WebSocket 或每隔两秒通过此函数去索取最新的服务器进度。
该节点除了会返回 `RUNNING/SUCCESS/FAILED` 物理层枚举，更将一并**抽离尾部的 50 行核心输出日志（Tail Logs）**给前端充当虚拟控制台界面！
