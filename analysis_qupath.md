# QuPath 源码级架构探查报告 (数据与图像处理专报)

根据您的要求，我对已克隆的 QuPath 核心源码（基于 `qupath-core/src/main/java/qupath/lib`）进行了针对底层**图像载入引擎**与**标记物存储拓扑**的拆解解析。

---

## 一、图像数据处理存储架构 (Image Processing & Tile Caching)

QuPath 能在极低内存消耗下流畅漫游 10万乘10万像素的医学全尺寸大图 (WSI)，完全得益于其精妙的**“接口级金字塔+多线程按需切片提取 (Tile-based on-demand pyramid loading)”**模式。

### 1. 核心接口与调度链路
源码入口核心位于 `qupath.lib.images.servers.AbstractTileableImageServer`：
- 整图不进入内存：任何提取操作必须经过 `readRegion(RegionRequest request)`。`RegionRequest` 打包了 `x, y, width, height, downsample`。
- **瓦块分发缓存池 (Tile Cache)**：遇到针对一个特大范围（如缩小了16倍的大全景）的请求时，底层会通过 `TileRequestManager` 将其打散为无数个小 `TileRequest`。
- **并发锁与线程池 (`pendingTiles` Cache)**：在 `getTile(TileRequest)` 中，它天才地写了一个基于 `FutureTask<BufferedImage>` 的 `ConcurrentHashMap`。如果有多个界面同时滚到了同一个小方块，系统仅仅放行一个底层 I/O 线程去硬盘读取，其余线程全部 `futureTask.get()` 挂起等待共享这一小块内存块。

### 2. 核心源码截取片段验证
```java
// from AbstractTileableImageServer.java: getTile()
var futureTask = pendingTiles.computeIfAbsent(tileRequest, t -> new TileTask(Thread.currentThread(), () -> readTile(t)));
var myTask = futureTask.thread == Thread.currentThread();
try {
    if (myTask) futureTask.run(); // 唯独当前拿到钥匙的线程去解压图像
    imgCached = futureTask.get(); // 其他线程安全地拿取内存拷贝
}
```

> **对 CellStudio 的工程意义**:
> 如果您想在平台上搭建 Web 标注界面或巨型 WSI 前向推理管道，我们必须移植这套逻辑套路——用 Python 的 `concurrent.futures` 加上 LRU 显存/内存缓存，搭建相同的 `TileRequestManager`。坚决杜绝让 OpenCV 或 PIL 去强吃全玻片文件。

---

## 二、标注数据的处理存储架构 (Data Annotation Hierarchy)

在这边，QuPath 采用了非常“重量级”的 Java 面向对象设计，但是其输入输出则完全靠向了纯粹的 Web 阵营。

### 1. 面向对象的标记物内存树 (`qupath.lib.objects`)
QuPath 将所有在图上画出的东西强制纳入同一基类 `PathObject`，并在内存里通过 `parent` 与 `childList` 维护绝对的父子关系挂载树：
- **`PathAnnotationObject`**: 外层级宏观标注（如画个大肿瘤边框）。
- **`PathDetectionObject`**: 被塞进 Annotation 内侧的 AI 检出目标。
- **`PathCellObject` (核心特化)**: 继承自 `PathDetectionObject`。对于细胞，它会在里面开辟出两块截然不同的几何空间：一个是 `roi`（代表细胞质边界），另偷偷包裹了一个专属方法用来调用 `getNucleusROI()`。

### 2. 化繁为简的 I/O 落地架构 (`qupath.lib.io.FeatureCollection`)
在落地到硬盘（无论是 `.qpdata` 还是外发的 GeoJSON）时，QuPath `GsonTools.java` 绝不输出任何带指针的 Java 对象层级树！
由于第三方开源地图规范 (GeoJSON) 只承认单个地理多边形（Feature），QuPath 聪敏地做了一层**平铺降维映射**：
- 它会把整个 `childList` 打平为一个一维的 `features: []` 列表。
- 然后把特殊的属性（比如是不是核心，面积是多少测量值，属于啥分类类型），全部塞进 GeoJSON Feature 结构内预留的 `properties: {}` 杂项字典里。
- 把细胞核的第二重边界编码成 `properties -> nucleusGeometry -> coordinates`.

> **对 CellStudio 的工程意义**:
> 正是因为看破了这段 `PathObject` 打平到 `GeoJSON` 的映射动作。所以在此前 Rev.2 的 CSUOS Python 字典设计里，我果断弃用复杂的 OOP 指针写法，直接让咱们的系统从诞生起，**就把“打平的 GeoJSON 节点”当成本命传输协议**。只需要给每个点加上 `parent_id` 字典项，就瞬间完成了比 Java 更轻量化、且 100% 同构的数据表达。
