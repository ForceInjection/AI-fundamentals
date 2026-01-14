# LMCache 源码分析指南

欢迎阅读 LMCache 源码分析文档集合。为了帮助您更高效地理解 LMCache 的架构设计与代码实现，我们建议按照以下顺序进行阅读。

## 推荐阅读顺序

### 第一阶段：全貌认知

从宏观视角理解 LMCache 的核心价值、分层架构及组件交互。

**1. [LMCache 架构概览](./lmcache_overview.md)**

- **核心内容**: 系统定位、四层存储架构 (L1-L4)、核心能力范式、整体组件架构图。
- **目标**: 建立对 LMCache 的全局认知。

### 第二阶段：核心链路

深入理解 LMCache 如何集成到推理引擎（如 vLLM）中，以及核心的 I/O 路径。

**2. [LMCacheConnector 源码分析](./lmcache_connector.md)**

- **核心内容**: vLLM 集成入口、请求拦截、KV Cache 视图转换。
- **目标**: 理解 LMCache 如何“劫持”并接管推理引擎的 KV Cache 操作。

**3. [LMCacheEngine 源码分析](./lmcache_engine.md)**

- **核心内容**: 核心控制流、I/O 编排、元数据管理 (TokenDatabase)。
- **目标**: 掌握数据流如何在不同组件间流转。

### 第三阶段：分布式控制平面

探索 LMCache 如何管理大规模集群的元数据与节点协调。

**4. [LMCache Controller 源码分析](./lmcache_controller.md)**

- **核心内容**: 控制平面架构、ZMQ 三通道通信模型、元数据一致性管理 (KVController)、集群编排指令。
- **目标**: 理解分布式环境下的节点发现、元数据同步与状态机管理。

### 第四阶段：存储子系统

聚焦于 LMCache 最核心的多级存储与调度机制。

**5. [LMCache 分层存储架构与调度机制详解](./lmcache_storage_overview.md)**

- **核心内容**: StorageManager 调度器、Write-All 策略、Waterfall 检索机制。
- **目标**: 理解数据如何在 L1/L2/L3/L4 之间调度与迁移。

### 第五阶段：后端实现细节

深入最底层的存储后端实现，探究极致性能优化的细节。

**6. [LocalCPUBackend 源码分析](./local_cpu_backend.md)** (L1 内存层)

- **核心内容**: 内存分配器 (Allocator)、锁粒度优化、NUMA 亲和性。
- **目标**: 理解高性能内存管理与并发控制。

**7. [LocalDiskBackend 源码分析](./local_disk_backend.md)** (L3 磁盘层)

- **核心内容**: 扁平化文件布局、O_DIRECT 直通 I/O、异步流水线。
- **目标**: 理解基于磁盘的高吞吐扩展实现。

**8. [LMCache Remote Connector 源码分析](./lmcache_remote_connector.md)** (L4 远程层)

- **核心内容**: RemoteConnector 抽象接口、Redis/S3/Mooncake 等多后端实现、零拷贝与异步 I/O 机制。
- **目标**: 理解如何适配异构远程存储系统以实现数据共享与持久化。

---

## 文档索引

| 文档名称                                                     | 描述                                                    |
| :----------------------------------------------------------- | :------------------------------------------------------ |
| [lmcache_overview.md](./lmcache_overview.md)                 | 系统的整体架构与核心概念介绍。                          |
| [lmcache_connector.md](./lmcache_connector.md)               | 与推理引擎对接的连接器实现分析。                        |
| [lmcache_engine.md](./lmcache_engine.md)                     | 核心引擎逻辑与数据流编排分析。                          |
| [lmcache_controller.md](./lmcache_controller.md)             | 控制平面架构与分布式元数据管理分析。                    |
| [lmcache_storage_overview.md](./lmcache_storage_overview.md) | 多级存储架构与 StorageManager 调度逻辑分析。            |
| [local_cpu_backend.md](./local_cpu_backend.md)               | L1 本地 CPU 内存后端及内存分配器分析。                  |
| [p2p_backend.md](./p2p_backend.md)                           | L2 弹性互联层及跨节点传输机制分析。                     |
| [local_disk_backend.md](./local_disk_backend.md)             | L3 本地磁盘后端及 I/O 优化分析。                        |
| [lmcache_remote_connector.md](./lmcache_remote_connector.md) | L4 远程存储连接器及 Redis/Server/Mooncake/S3 实现分析。 |
