# vLLM KV Offloading Connector 与 LMCacheConnector：架构设计与性能深度对比

## 1. 引言

随着大语言模型 (LLM) 支持的上下文窗口 (Context Window) 日益增长，Key-Value (KV) Cache 对显存资源的占用已成为制约推理系统吞吐量和并发能力的关键瓶颈。特别是在长文本推理场景下，庞大的 KV Cache 往往占据了绝大部分显存，导致 Batch Size 受限甚至引发 OOM (Out-Of-Memory) 错误。为了突破物理显存的限制，"KV Cache Offloading"（KV 缓存卸载）技术应运而生。

当前在 vLLM 生态中，主要存在两种主流的 KV 缓存管理与卸载方案：

1. **vLLM Native KV Offloading Connector**: vLLM 官方在 0.11.0 版本引入的原生功能，专注于通过利用 Host CPU 内存来透明扩展 GPU 显存容量。
2. **LMCacheConnector**: 基于 LMCache 开源项目的集成方案，旨在构建支持跨实例共享、多层级存储（CPU/Disk/Remote）的 KV 数据管理系统。

本文将结合 vLLM 官方设计理念、LMCache 技术文档以及核心代码实现，对这两种方案进行深入剖析与对比。

---

## 2. KV Offloading Connector 深入分析

本章将详细解析 vLLM 原生的 KV Offloading Connector，探讨其设计目标、核心架构以及在实际应用中的性能表现。

### 2.1 设计初衷与背景

KV Offloading Connector 的设计旨在解决两大核心痛点：

- **消除 Preemption 带来的计算浪费**: 在高负载场景下，vLLM 调度器可能会抢占（Preempt）部分低优先级请求以释放显存。若无卸载机制，被抢占请求的 KV Cache 将被直接丢弃，导致其在恢复运行时必须重新执行昂贵的 Prefill 计算。KV Offloading 允许将这些 Cache 暂时移至 CPU，从而避免了重复计算，显著降低了恢复延迟。
- **高效利用层级存储架构**: 现代服务器通常配备大容量的系统内存 (DRAM)，其成本远低于 HBM 显存。随着 PCIe 4.0/5.0 技术普及，CPU 与 GPU 之间的数据传输带宽已大幅提升。将 CPU 内存作为 GPU 显存的高速 Swap 区域，是一种在成本与性能之间取得平衡的有效扩容手段。

### 2.2 核心架构与组件

该 Connector 采用了 **Scheduler-Worker** 分离的架构，深度集成于 vLLM 的调度循环中。

#### 2.2.1 关键组件

KV Offloading Connector 的实现依赖于几个核心组件的协同工作，分别负责策略定义、状态管理和数据传输。

1. **OffloadingSpec**: 定义卸载策略的基类，位于 [`vllm/v1/kv_offload/spec.py`](./vllm/v1/kv_offload/spec.py)。它支持配置 `offloaded_block_size`（通常大于 GPU 的 `block_size` 以优化传输带宽），并负责实例化 Manager 和 Handler。

   ```python
   class OffloadingSpec(ABC):
       """Spec for an offloading connector"""

       def __init__(
           self, vllm_config: "VllmConfig", kv_cache_config: "KVCacheConfig | None"
       ):
           # ... (initialization logic)
           self.gpu_block_size = vllm_config.cache_config.block_size
           self.offloaded_block_size = int(
               self.extra_config.get("block_size", self.gpu_block_size)
           )
           assert self.offloaded_block_size % self.gpu_block_size == 0

       @abstractmethod
       def get_manager(self) -> OffloadingManager:
           pass
   ```

2. **OffloadingManager (Scheduler 侧)**:
   - 驻留在 Scheduler 进程中，维护一个全局的 Block 状态表。
   - 提供了多种淘汰策略的实现：
     - **LRU (Least Recently Used)**: 默认策略，由 [`LRUOffloadingManager`](./vllm/v1/kv_offload/lru_manager.py#L16) 实现。

     ```python
     class LRUOffloadingManager(OffloadingManager):
         """
         An OffloadingManager with a pluggable backend, which evicts blocks by LRU.
         """
         def __init__(self, backend: Backend, enable_events: bool = False):
             self.backend: Backend = backend
             # block_hash -> BlockStatus
             self.blocks: OrderedDict[BlockHash, BlockStatus] = OrderedDict()
             # ...
     ```

     - **ARC (Adaptive Replacement Cache)**: 自适应策略，由 [`ARCOffloadingManager`](./vllm/v1/kv_offload/arc_manager.py#L16) 实现，能够在 Recency (最近访问) 和 Frequency (频繁访问) 之间动态平衡。

     ```python
     class ARCOffloadingManager(OffloadingManager):
         """
         An OffloadingManager implementing the ARC (Adaptive Replacement Cache)
         eviction policy with a pluggable backend.

         Data Structures:
             T1: Recent cache containing blocks accessed once.
             T2: Frequent cache containing blocks accessed multiple times.
             B1/B2: Ghost lists tracking recently evicted blocks from T1/T2.
             target_t1_size: Adaptive target size for the T1 partition.
         """
         # ...
     ```

   - 负责计算每次调度步骤中的 `reqs_to_load` (需要加载的块) 和 `reqs_to_store` (需要卸载的块)。

3. **OffloadingWorker (Worker 侧)**:
   - 驻留在 GPU Worker 进程中，由 [`OffloadingWorker`](./vllm/v1/kv_offload/worker/worker.py#L80) 类管理。

   ```python
   class OffloadingWorker:
       """
       OffloadingWorker class for managing asynchronous KV data transfers
       using multiple OffloadingHandlers
       """
       def __init__(self):
           self.handlers: set[OffloadingHandler] = set()
           self.transfer_type_to_handler: dict[TransferType, OffloadingHandler] = {}

       def register_handler(self, src_cls, dst_cls, handler) -> None:
           # ...
   ```

   - 通过具体的 [`OffloadingHandler`](./vllm/v1/kv_offload/worker/worker.py#L18) 执行实际的数据传输。

   ```python
   class OffloadingHandler(ABC):
       """
       OffloadingHandler class for managing asynchronous KV data transfers
       """
       @abstractmethod
       def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
           pass
   ```

   - 利用 vLLM 的 **异步 Connector API**，在 GPU 执行 Model Forward 的同时，通过 DMA (Direct Memory Access) 并行进行数据拷贝，最大程度掩盖 I/O 延迟。

#### 2.2.2 工作流程

Offloading Connector 的工作流程主要包含数据的卸载（Store）和加载（Load）两个反向过程，它们与 vLLM 的调度器紧密配合。

1. **Store (卸载)**: 当一个请求生成新的 KV Block，或者被抢占时，Scheduler 标记这些 Block 需要卸载。这部分逻辑在 [`OffloadingConnectorScheduler._get_reqs_to_store`](./vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py#L276) 中实现：

   ```python
   def _get_reqs_to_store(self, scheduler_output: SchedulerOutput):
       reqs_to_store: dict[ReqId, TransferSpec] = {}
       # iterate over both new and cached requests
       for req_id, new_block_id_groups, preempted in yield_req_data(scheduler_output):
           if preempted:
               self._request_block_ids[req_id] = []

           if new_block_id_groups:
               new_block_ids = new_block_id_groups[0]
               self._request_block_ids[req_id] += new_block_ids
           # ...
   ```

2. **Load (加载)**: 当请求被重新调度时，Scheduler 识别出该请求在 CPU 中有缓存的 Block，并生成加载指令。这部分逻辑在 [`OffloadingConnectorScheduler.update_state_after_alloc`](./vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py#L234) 中处理：

   ```python
   def update_state_after_alloc(
       self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
   ):
       # ...
       src_spec = self.manager.prepare_load(block_hashes)
       dst_spec = GPULoadStoreSpec(block_ids[num_computed_gpu_blocks:])

       self._reqs_to_load[request.request_id] = (src_spec, dst_spec)
       # ...
   ```

   通过从 CPU 加载 KV Cache，速度远快于重新计算 (Recompute)，从而显著降低 **TTFT (Time-To-First-Token)**。

**性能收益**: 除了降低延迟，在高负载下，Offloading 机制允许系统接纳更多的并发请求，从而显著提升了整体 **Throughput (吞吐量)**。

### 2.3 局限性与边界

尽管 KV Offloading Connector 在缓解显存压力方面效果显著，但作为 vLLM 的原生组件，其设计权衡也带来了一些局限性：

- **单机视角的局限**: 该方案本质上是 **Scale-up**（单机垂直扩展）策略，利用本地 CPU 内存扩容。它无法跨越物理节点，不支持多实例间的 KV Cache 共享或迁移。
- **存储层级受限**: 核心设计聚焦于利用 Host DRAM 作为高速 Swap。虽然理论上可扩展，但缺乏对远程分布式存储（如 S3/Redis）的原生支持，限制了其存储容量的上限和持久化能力。
- **调度与状态紧耦合**: 卸载逻辑与 Block Table 状态管理深度嵌入在 Scheduler 循环中。这意味着 KV Cache 的生命周期与计算实例绑定，无法实现真正的“存算分离”。

这些局限性也正是 **LMCache/LMCacheConnector** 试图解决的核心问题。

---

## 3. LMCache 生态与 LMCacheConnector 深入分析

本章将视角从单一的 Connector 扩展到整个 **LMCache** 生态系统。KV Offloading Connector 主要解决的是单机显存扩展问题，而 LMCache 则致力于构建一个跨实例、多层级的 KV 数据管理平台。LMCacheConnector 是该系统在 vLLM 中的接入桥梁。

### 3.1 LMCache 系统概览

LMCache 是一个专为大语言模型服务设计的 KV Cache 管理系统，旨在通过多级存储架构降低 Time-To-First-Token (TTFT) 并提升系统吞吐量。其核心架构包含两个关键维度：

#### 3.1.1 多层级存储架构

LMCache 将存储介质划分为四个层级 (L1-L4)，实现了从本地极速访问到全局共享的全覆盖：

1. **L1 极速内存层 (Memory Tier)**:
   - **GPU Memory**: 保存当前活跃的 KV Cache。
   - **Local CPU**: 类似于 vLLM Offloading，利用 Host 内存扩容。

2. **L2 弹性互联层 (P2P Tier)**:
   - **RDMA/TCP**: 利用 RDMA/TCP 实现计算节点间的点对点 (P2P) 数据传输，延迟低于本地磁盘，适合集群内热点数据共享。

3. **L3 本地持久层 (Disk Tier)**:
   - **NVMe SSD**: 利用本地 NVMe SSD 进行持久化，支持利用 **GDS (GPUDirect Storage)** 技术实现零拷贝加载。

4. **L4 远程共享层 (Remote Tier)**:
   - **Redis/S3等**: 对接 Redis, S3 或专有 LMCache Server，支持跨地域、跨集群的数据共享与 Serverless 实例冷启动。

#### 3.1.2 核心能力范式

基于上述存储层级，LMCache 实现了三种核心应用范式：

1. **本地复用 (Local Reuse)**: 扩展单机显存，利用 CPU/Disk 换取更大的 Batch Size。
2. **集群共享 (Cluster Sharing)**: "一次计算，全局可用"。支持跨实例负载均衡和 Context 迁移。
3. **流水线传输 (Pipeline Transmission)**: 专为 **Prefill-Decode 分离** 架构设计，实现 KV Cache 从 Prefill 实例到 Decode 实例的低延迟流转。

### 3.2 LMCacheConnector：vLLM 的接入桥梁

在 vLLM 中，**LMCacheConnector** 充当了物理内存视图与 LMCache 逻辑视图之间的智能适配器。其核心实现类 [`LMCacheConnectorV1`](./vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py#L72) 采用适配器模式，根据配置动态加载底层的 LMCache 实现（通常通过 `vllm_v1_adapter` 进行桥接）。

```python
class LMCacheConnectorV1(KVConnectorBase_V1):
    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(
            vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config
        )
        assert vllm_config.kv_transfer_config is not None
        # ... (adapter loading logic)
        self._lmcache_engine = cls(vllm_config, role, self)
```

#### 3.2.1 核心组件实现

LMCacheConnector 通过以下核心组件实现了 vLLM 与底层存储后端的高效解耦：

- **LMCacheEngine**: 核心引擎，负责管理底层的存储后端（Local CPU, Disk, Redis, P2P 等）。在 vLLM 适配层中，通过 [`LMCacheConnectorV1Impl`](./vllm/distributed/kv_transfer/kv_connector/v1/lmcache_integration/vllm_v1_adapter.py#L210) 进行封装。

  ```python
  class LMCacheConnectorV1Impl:
      def __init__(
          self,
          vllm_config: "VllmConfig",
          role: KVConnectorRole,
          parent: KVConnectorBase_V1,
      ):
          # ...
          if role == KVConnectorRole.SCHEDULER:
              # Create lookup client using factory
              self.lookup_client = LookupClientFactory.create_lookup_client(
                  vllm_config, config
              )
          else:
              self.lmcache_engine = _init_lmcache_engine(
                  config,
                  vllm_config,
              )
          # ...
  ```

- **RequestTracker**: 跟踪每个请求的状态，记录请求对应的 Token 序列以及在 vLLM 中分配的 Block ID。源码可见 [`RequestTracker`](./vllm/distributed/kv_transfer/kv_connector/v1/lmcache_integration/vllm_v1_adapter.py#L121)。

  ```python
  @dataclass
  class RequestTracker:
      # Request id
      req_id: str

      # Total prompt token length
      prompt_len: int

      # The token ids that has been scheduled so far
      token_ids: list[int]

      # The block ids that has been allocated so far
      # NOTE: allocated blocks could be more than the number of tokens
      allocated_block_ids: list[int]
      # ...
  ```

### 3.3 关键技术机制

为了实现物理与逻辑的解耦，LMCache 引入了多项关键技术。

#### 3.3.1 视图转换 (View Transformation)

这是连接 vLLM 与 LMCache 的核心机制：

- **物理视图 (vLLM)**: vLLM 使用 `PagedAttention`，内存是离散的 Block。
- **逻辑视图 (LMCache)**: LMCache 使用 Token 序列作为索引。
- **转换机制**: 通过 `ReqMeta` 和 `RequestTracker` 维护的映射关系，LMCache 能够将逻辑的 Token 位置精确转换为 GPU 显存的物理地址 (Block ID + Offset)。这使得 LMCache 可以直接通过 CUDA Kernel 读写 vLLM 的显存，而无需 vLLM 感知底层存储细节。

#### 3.3.2 智能索引与共享

LMCache 不依赖物理地址索引，而是使用 **Token Content Hash**。

- **Prefix Hashing**: 将 Token 序列切分并计算哈希值作为全局唯一的 Key。
- **Context Sharing**: 只要两个请求的前缀 Token 相同，无论它们位于哪个 vLLM 实例，生成的 Key 都一致。这天然支持了跨实例的 **KV Cache 复用**，极大减少了重复计算。

#### 3.3.3 性能优化特性

- **Layerwise Pipelining (层级流水线)**: 利用 Transformer 的层级结构，在计算第 $i$ 层时，异步预取第 $i+1$ 层的 KV 数据。这主要通过 [`wait_for_layer_load`](./vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py#L143) 接口实现，允许 Attention Layer 阻塞等待特定层的加载，实现 I/O 与计算的完美重叠。

```python
def wait_for_layer_load(self, layer_name: str) -> None:
    """
    Block until the KV for a specific layer is loaded into vLLM's
    paged buffer. This is called from within attention layer to ensure
    async copying from start_load_kv is complete.
    """
    self._lmcache_engine.wait_for_layer_load(layer_name)
```

- **LMCBlender (混合器)**: 支持将部分 KV Cache 放在 GPU，部分放在远程，通过 Blender 机制在推理时动态合并。该功能通过适配层中的 [`LMCBlenderBuilder`](./vllm/distributed/kv_transfer/kv_connector/v1/lmcache_integration/vllm_v1_adapter.py#L17) 进行集成，在启用时负责初始化混合引擎。

  ```python
  from lmcache.v1.compute.blend import LMCBlenderBuilder
  # ...
              if self.enable_blending:
                  self.blender = LMCBlenderBuilder.get_or_create(
                      ENGINE_NAME,
                      self.lmcache_engine,
                      self.lmcache_engine.gpu_connector,
                      config,
                  )
  ```

### 3.4 局限性与挑战

尽管 LMCache 提供了强大的跨实例共享能力，但在实际落地中也面临以下挑战：

1. **部署运维复杂性**:
   - 相比于仅需一个参数即可开启的 Native Offloading，LMCache 需要额外部署存储后端（如 Redis、MinIO 或 LMCache Server）并配置复杂的网络拓扑，运维成本较高。

2. **网络带宽依赖**:
   - 在 L2 (P2P) 和 L4 (Remote) 层级，KV Cache 的传输性能高度依赖集群网络带宽。如果网络基础设施较弱（如非 RDMA 环境），跨实例传输的延迟可能抵消 KV Cache 复用带来的收益。

3. **计算资源开销**:
   - **CPU 开销**: 实时计算 Token Hash、序列化/反序列化以及维护索引树需要消耗显著的 Host CPU 资源，可能与模型推理（如 CPU Offload 时的计算）产生竞争。
   - **内存开销**: LMCache 引擎本身需要维护大量的元数据（ReqMeta, RequestTracker），在大规模并发下会占用可观的 Host 内存。

4. **一致性与驱逐策略**:
   - 在分布式场景下，如何确保不同实例间的 KV Cache 一致性，以及在存储空间受限时制定全局最优的驱逐策略 (Global Eviction Policy)，仍是系统设计的难点。

---

## 4. 架构与性能的深度对比

基于前文的技术拆解，本章将超越简单的功能列表，从**架构设计哲学**与**性能权衡逻辑**两个核心维度，对这两种方案展开深度的横向对比。我们将揭示它们如何在“单机垂直扩展”与“分布式水平扩展”之间做出的根本性取舍。

### 4.1 核心差异总览

下表总结了 KV Offloading Connector 与 LMCacheConnector 的关键差异：

| 特性维度         | KV Offloading Connector (vLLM Native)                                                                           | LMCacheConnector (LMCache Ecosystem)                                                                                                                             |
| :--------------- | :-------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **核心定位**     | **垂直扩展 (Scale-Up)**：利用单机多级存储 (CPU/NVMe) 扩展显存容量。                                             | **水平扩展 (Scale-Out)**：构建分布式 KV 数据湖，支持跨实例、跨地域共享。                                                                                         |
| **数据管理方式** | **显式物理管理**：Scheduler 明确记录 Block 在 CPU 还是 GPU，逻辑紧耦合。                                        | **隐式逻辑管理**：通过 Token Hash 索引，底层存储位置对 vLLM 透明。                                                                                               |
| **通信机制**     | **进程内/共享内存**：主要依赖 PCIe 进行 H2D/D2H 拷贝，延迟极低。                                                | **多协议支持**：支持 TCP/IP, RDMA, gRPC 等，适应分布式网络环境。                                                                                                 |
| **存储后端**     | **单一 (Host Memory)**：架构上支持磁盘，但目前核心实现聚焦于 CPU DRAM 扩展。                                    | **丰富 (Multi-Tier)**：Local CPU, Local Disk, Redis, MinIO (S3), P2P 网络等。                                                                                    |
| **跨实例能力**   | **弱**：主要用于单实例内的 Preemption 恢复，无法跨进程共享。                                                    | **强**：天然支持 Disaggregated Serving (Prefill-Decode 分离) 和 Context Sharing。                                                                                |
| **实现复杂度**   | **轻量级**：代码量少，依赖 vLLM 内部 Block Table，无额外元数据开销。                                            | **重量级**：独立复杂的索引、序列化、网络模块，需维护 ReqMeta 和 RequestTracker。                                                                                 |
| **适用场景**     | 1. **单机高并发**: 解决 OOM 和频繁 Preemption 问题。2. **本地长文本**: 本地 CPU 内存充足，需扩展 Context 长度。 | 1. **分离式推理**: 专门的 Prefill 集群和 Decode 集群。2. **多轮对话复用**: 多个请求共享相同的 System Prompt 或 History。3. **Serverless 推理**: 实例快速冷启动。 |

### 4.2 架构设计哲学：紧耦合 vs. 松耦合

两种 Connector 的根本差异源于其对“KV Cache”这一实体的定义不同：前者将其视为**计算过程的附属品**（Runtime State），后者将其视为独立的**数据资产**（Data Asset）。这种认知差异导致了截然不同的架构设计：

1. **KV Offloading Connector (紧耦合)**:
   - 深度集成于 `Scheduler`。调度器不仅决定“何时”卸载，还精确控制“卸载到哪里”（Block 粒度）。
   - **优势**: 极致的本地性能，零开销的元数据管理，对 vLLM 核心流程侵入性低（复用现有的 Block Table）。
   - **代价**: 难以扩展到多机环境，受限于单机物理资源，且无法感知数据的内容（Content-Unaware）。

2. **LMCacheConnector (松耦合)**:
   - 引入了独立的 `LMCacheEngine` 和逻辑地址空间（Token Hash）。vLLM 只需通过 `Connector` 接口发起请求，无需关心数据最终落在 L1 还是 L4。
   - **优势**: 存储后端可插拔，感知数据内容（Content-Aware），天然支持全局去重和共享。
   - **代价**: 引入了视图转换（View Transformation）和哈希计算的开销，系统复杂度显著增加。

### 4.3 性能权衡分析

在实际应用中，架构的差异直接体现为性能曲线的不同。我们需要在“极致的单机延迟”与“全局的系统吞吐”之间寻找平衡点：

1. **延迟 (Latency)**:
   - **Swap 操作**: KV Offloading Connector 走 PCIe DMA，延迟在微秒级，几乎不影响推理。LMCacheConnector 若命中本地盘或远程存储，检索延迟较高（毫秒级）。
   - **TTFT (Time-To-First-Token)**:
     - **Local Hit (Preemption)**: KV Offloading 表现极佳，通过 CPU->GPU 快速回加载，完全消除重计算开销。
     - **Global Hit (Cold Start)**: 这是 LMCacheConnector 的杀手锏。它能将新实例的冷启动 TTFT 从“全量计算时间”降低为“网络传输时间”，在长文本场景下收益巨大（例如：从 10s 降低到 0.5s）。

2. **吞吐量 (Throughput)**:
   - KV Offloading Connector 通过扩大有效 Batch Size（将暂停的请求移出显存）来提升单机吞吐。
   - LMCacheConnector 通过 **全局复用**（一次计算，多处使用）减少了集群的总计算负载，从而提升了整个集群的有效吞吐量。

---

## 5. 总结与展望

KV Cache 管理技术正在经历从“单机显存优化”向“分布式数据资产管理”的范式转变。

**KV Offloading Connector** 是这一演进过程中的**战术性基石**。它以极低的工程侵入性，成功解决了单机显存瓶颈问题，是提升单实例吞吐量的“特效药”。对于大多数追求极致性价比的单机推理服务，它是目前的最优解。

**LMCacheConnector** 则代表了**战略性愿景**。它创造性地将 KV Cache 定义为独立于计算实例的一级资产 (First-class Asset)，通过解耦计算与存储，为构建大规模、弹性伸缩的推理集群奠定了基石。在分离式推理 (Disaggregated Serving) 和 Serverless 推理日益普及的今天，LMCache 提供的跨实例共享能力将成为下一代推理系统的核心竞争力。

**决策建议**:

- **路径 A (单机效能优先)**: 若业务主要运行在独立节点，且首要目标是缓解显存压力、减少 Preemption，请直接启用原生 `--kv_offloading_backend native`。
- **路径 B (集群弹性优先)**: 若业务正处于构建支持自动扩缩容、存算分离或多级缓存的复杂推理平台阶段，集成 `LMCacheConnector` 将为您带来长期的架构红利。

---

**参考资料**:

1. [vLLM Blog: Inside vLLM’s New KV Offloading Connector](https://blog.vllm.ai/2026/01/08/kv-offloading-connector.html)
2. [LMCache 架构概览](https://github.com/ForceInjection/AI-fundermentals/blob/main/09_inference_system/lmcache/lmcache_overview.md)
3. [LMCacheConnector 代码分析](https://github.com/ForceInjection/AI-fundermentals/blob/main/09_inference_system/lmcache/lmcache_connector.md)
