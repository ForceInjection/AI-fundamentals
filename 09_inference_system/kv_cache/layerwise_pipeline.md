# KV Cache 层级流水线并行 (Layerwise Pipelining)

层级流水线并行 (Layerwise Pipelining) 是 LMCache 和 DualPath 等先进 LLM 推理系统为了解决 **KV Cache 存取延迟** 和 **显存容量限制** 而采用的核心优化机制。其核心思想是将 KV Cache 的加载/存储与 GPU 的模型层级计算进行 **并行化处理**，从而掩盖 I/O 延迟并最大化系统吞吐量。

本文基于 LMCache 源码解读文档 [1] 和 DeepSeek DualPath 论文 [2] 对该技术进行详细阐述。

## 1. 核心概念

层级流水线并行是一种 **计算与 I/O 重叠 (Computation-I/O Overlap)** 的策略。

在传统的推理过程中，系统通常采用 **"全部加载-计算-全部存储"** 的串行模式：

1. **加载阶段**: 等待所有层的 KV Cache 加载完成；
2. **计算阶段**: 执行所有层的推理计算；
3. **存储阶段**: 将生成的 KV Cache 全部回写存储。

这种模式会导致严重的 GPU 空闲，且受限于单张 GPU 的显存容量 (HBM Capacity)，限制了 Batch Size 的提升。

**层级流水线** 则利用了 LLM **逐层计算 (Layer-by-Layer Execution)** 的特性：

- 在 GPU 计算第 $i$ 层时，系统在后台 **同步加载** 第 $i+1$ 层的数据；
- 或者在后台 **异步持久化** 第 $i-1$ 层的数据。

---

## 2. 基本原理

层级流水线并行的实现依赖于 LLM 推理的计算局部性、硬件资源的解耦以及显存管理的优化。该技术主要依赖以下三个基本原理：

1. **计算局部性 (Locality)**:
   推理中的每一层计算仅需要该层特有的 KV Cache 数据。这意味着系统不需要在显存中同时保存所有层的 KV Cache，可以实现 **按需加载 (On-demand Loading)** 和 **即时释放 (Immediate Freeing)**。

2. **资源解耦 (Resource Decoupling)**:
   - 利用 GPU 的 **DMA 引擎** 进行显存与主机内存 (H2D/D2H) 的传输；
   - 利用 **存储网卡 (SNIC)** 进行主机与远程存储的 I/O；
   - 利用 **计算网卡 (CNIC)** (如 RDMA) 进行节点间传输。
     这些 I/O 操作可以与 GPU 的 Tensor Core 计算 **异步执行**，互不干扰。

3. **显存节省 (Memory Savings)**:
   通过逐层分配和释放 KV Cache 空间，显存只需持有当前正在计算的那一层 (或几层) 数据。这显著降低了显存占用，使得系统能够支持更大的 Batch Size，从而提高 GPU 计算单元的利用率。

---

## 3. LMCache 中的工程实现

LMCache [1] 在集成到 vLLM 等引擎时，通过以下技术手段实现了这一机制，主要侧重于 **降低首字延迟 (TTFT)**。

### 3.1 生成器模式 (Generator Pattern)

LMCache 的存储和检索接口 (如 `store_layer` 和 `retrieve_layer`) 并不执行单一的阻塞操作，而是采用 Python 的 `yield` 机制构建流水线。

- **存储流程**: 将存储任务拆解为细粒度步骤嵌入在模型每一层的计算循环中：
  1. 初始化 (计算 Hash)；
  2. 等待上一层 D2H 完成；
  3. 启动当前层 D2H；
  4. 提交持久化任务。

- **检索流程**: 构建 **"生产者-消费者"** 模型。
  - **生产者**: 负责从后端 (L1-L4 存储) 读取数据；
  - **消费者**: GPU 连接器 (GPU Connector) 负责将数据拷贝至显存。

```python
# 伪代码示例：LMCache 的层级检索生成器
def retrieve_layer(self, tokens):
    # 预取第 0 层和第 1 层
    prefetch(0); prefetch(1)

    for i in range(num_layers):
        # 等待第 i 层数据就绪 (可能已经在后台加载完毕)
        layer_data = wait_for_layer(i)

        # 触发第 i+2 层的预取 (保持流水线充盈)
        if i + 2 < num_layers:
            prefetch(i + 2)

        # 将第 i 层数据 yield 给计算引擎
        yield layer_data
```

### 3.2 预取策略 (Prefetching)

为了最大化重叠效果，LMCache 采用了 **"Prefetch-2"** 策略：

1. **启动阶段**: 在开始计算前，立即触发 **第 0 层** 和 **第 1 层** 的加载任务。
2. **掩盖延迟**: 当引擎开始计算第 0 层时，数据已在传输中；且计算第 0 层的时间可以用来掩盖第 1 层的 I/O 延迟。
3. **流式协调**: 后续通过 `wait_for_layer_load` 在每一层计算前进行流式协调，确保数据就绪。

### 3.3 显存与内存管理

为了确保流水线在数据传输过程中不会因为频繁的内存分配而抖动，LMCache 实施了严格的内存管理策略：

- **全局预分配 (Global Pre-allocation)**: 在流水线启动前，LMCache 会一次性预分配所有层所需的 CPU 内存 (`MemoryObj`)，避免在推理关键路径上因频繁申请内存导致抖动。
- **同位置约束 (Same-location Constraint)**: 为了保证流水线的稳定性，系统强制要求同一 Key 的所有层级数据必须存储在同一个后端位置 (如都在磁盘或都在远程)，不支持跨介质碎片化存储。

---

## 4. DualPath 中的演进

DualPath [2] 将层级流水线并行从 **单机 I/O 优化** 扩展到了 **集群级带宽调度**，旨在解决 PD 分离 (Prefill-Decode Disaggregation) 架构下的 **存储带宽瓶颈**。

### 4.1 核心问题：存储 I/O 瓶颈

在 Agentic (智能体) 等长上下文、多轮对话场景中，Prefill 阶段的 KV Cache 加载需求巨大。传统的加载路径导致 **Prefill 节点的存储网卡 (SNIC) 饱和**，而 **Decode 节点的 SNIC 却处于空闲状态**。

### 4.2 双路径加载 (Dual-Path Loading)

为了平衡存储 I/O 压力，DualPath 引入了第二条加载路径，利用空闲的 Decode 节点带宽来辅助加载数据。两条路径的对比如下：

| 路径名称                    | 数据流向                                                                     | 适用场景                                                   |
| :-------------------------- | :--------------------------------------------------------------------------- | :--------------------------------------------------------- |
| **PE Read Path** (传统路径) | Storage $\rightarrow$ PE Buffer $\rightarrow$ PE HBM $\rightarrow$ DE Buffer | 标准加载，利用 Prefill 节点的 SNIC                         |
| **DE Read Path** (新增路径) | Storage $\rightarrow$ DE Buffer $\rightarrow$ PE HBM $\rightarrow$ DE Buffer | **借用 Decode 节点的 SNIC**，通过 RDMA 回传给 Prefill 节点 |

通过动态调度这两条路径，DualPath 将存储 I/O 压力在整个集群中进行了再平衡 (Rebalancing)。

### 4.3 结合层级流水线

DualPath 在这两条路径中都深度应用了层级流水线技术，以确保数据传输与计算的高度重叠：

- **细粒度传输 (Layer Blocks)**: KV Cache 被切分为细粒度的层级块 (Layer Blocks)。
- **跨网络流式传输**: 在 "DE Read Path" 中，数据从 Decode 节点加载到内存后，会立即以层为单位通过 **计算网络 (Compute Network/RDMA)** 流式传输给 Prefill 节点。
- **流量隔离**: 采用以 NIC 为中心的流量管理 (NIC-centric Traffic Management)，确保 KV Cache 的传输流量不会干扰模型推理的关键通信 (如 Collective Ops)。

---

## 5. 性能收益

通过层级流水线并行及相关的架构优化，系统获得了显著的性能提升：

1. **降低首字延迟 (TTFT)**: LMCache 通过 I/O 与计算的重叠，消除了显式的数据加载等待时间。
2. **提升吞吐量**:
   - **显存效率**: 逐层加载释放了显存压力，允许更大的 Batch Size。
   - **带宽利用率**: DualPath 通过聚合集群内所有节点的存储带宽，解决了单节点 I/O 瓶颈。
3. **实测数据**:
   - 在智能体 (Agentic) 等高缓存命中率场景下，DualPath 配合该技术将端到端吞吐量提升了高达 **1.87 倍** (离线推理) 和 **1.96 倍** (在线服务) [2]。

## 6. 参考文献

1. LMCache Documentation: [LMCache 架构概览](lmcache/lmcache_overview.md), [LMCacheEngine 核心引擎代码分析](lmcache/lmcache_engine.md).
2. DualPath Paper: _DualPath: Rethinking KV Cache Loading for Agentic LLM Inference_ (arXiv:2602.21548v2).
