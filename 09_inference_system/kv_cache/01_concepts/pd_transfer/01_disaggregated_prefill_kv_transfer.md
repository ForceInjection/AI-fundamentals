# PD 分离架构下的 KV Cache 传输：Push 还是 Pull？算完再传还是边算边传？

PD 分离（Prefill-Decode Disaggregation）把 Prefill 放到高算力 GPU 上，Decode 放到大显存 GPU 上——两端分开后，核心问题从"显存放不放得下"变成了**"KV Cache 怎么从 Prefill 节点搬到 Decode 节点"**。一个 32K token 的 prompt 产生约 10 GB KV Cache，这个搬运过程直接决定了 PD 分离架构的 TTFT 和吞吐上限。

本文从传输时序、发起方、传输内容三个维度，对比 vLLM KV Connector V1、LMCache PD Backend 和 Mooncake 三种方案在 Push/Pull、Eager/Pipelined、完整/增量上的不同选择及其代价。

> **前置阅读**：[KV Cache 原理简介](../basic/kv_cache_basics.md) — Prefill 与 Decode 的计算特性差异；[Chunked Prefill 如何改变 KV Cache 管理](../scheduling/01_vllm_chunked_prefill.md) — PD 分离的前置概念。
>
> **配合阅读**：本文是概念层面的方案对比。各系统的完整源码分析见 [LMCache PD Backend](../../02_systems/lmcache/pd_backend.md)、[Mooncake 架构](../../02_systems/mooncake/mooncake_architecture.md)、[层级流水线并行](../offloading/02_layerwise_pipeline.md)。

---

## 一、PD 分离的动机与 KV 传输的必然性

### 1.1 两阶段的硬件需求错配

Prefill 一次性处理全部输入 token，Attention 计算量为 $O(S^2)$，属于 **compute-bound**——需要高算力（如 H100/H200）。Decode 每步只处理一个新 token，但需要反复读取完整的 KV Cache，属于 **memory-bound**——需要大显存带宽。

当 Prefill 和 Decode 绑在同一张 GPU 上时，两种需求互相妥协：算力浪费在 Decode 的空转上，显存被 Prefill 的 KV Cache 峰值撑满。PD 分离将两者解耦：Prefill 节点用高算力 GPU，快速处理 prompt 后释放；Decode 节点用大显存 GPU，专门承载长上下文的 KV Cache 读取。

### 1.2 分离的代价：KV Cache 必须跨节点传输

解耦的必然结果是 **KV Cache 不在同一张 GPU 上了**。Prefill 节点算出的 K 和 V 必须被搬到 Decode 节点。以 LLaMA-2 70B（GQA, 8 KV heads, FP16）为例，单 token 单层 KV 约为 4 KB，80 层 × 4 KB = 320 KB/token，32K prompt 共约 **10 GB**。这个传输量对 PCIe 5.0（~64 GB/s）单向理论极限约 156ms，对跨机 RDMA（以 400Gbps 网卡为例，有效带宽 ~50 GB/s）约 200ms。实际延迟还需加上接收端写入、网络协议栈开销、多节点竞争等因素，通常比理论值高 20-50%。注意这是全部 80 层 KV 的总和。在 Pipelined 模式下，每次只传输一层（约 128 MB），10 GB 的总传输量被分散到各层计算之间，避免一次性阻塞。

**KV 传输的延迟直接加在 TTFT 之上**。如果等 Prefill 全部完成再传，TTFT = Prefill 计算时间 + KV 传输时间 + 首个 Decode 时间。这是 PD 分离架构设计空间的核心约束。

---

## 二、KV 传输的三个设计维度

KV Cache 从 Prefill 节点搬到 Decode 节点，不是一个单纯的"拷贝"操作。它涉及三个需要独立决策的维度：

### 2.1 谁发起传输？Push vs Pull

| 模式     | 发起方       | 流程                                                                         |
| -------- | ------------ | ---------------------------------------------------------------------------- |
| **Push** | Prefill 节点 | Prefill 完成后主动推送到 Decode 节点；Decode 节点被动接收并写入本地 KV Cache |
| **Pull** | Decode 节点  | Decode 节点在需要时从 Prefill 节点（或共享存储）拉取；类似缺页中断，按需加载 |

Push 的优势是 Decode 可以立即开始——KV 已经在本地了。代价是 Prefill 节点需要知道推给谁（需要外部编排），且如果 Decode 节点还没准备好接收，数据无处存放。

Pull 的优势是实现简单——Decode 节点自己管理拉取时机，不需要外部协调。代价是首个 Decode 步骤被 KV 加载延迟阻塞，TTFT 中包含拉取时间。Pull 的拉取目标（从哪个 Prefill 节点或存储层拉）可以由外部调度器决定——Mooncake 的 Conductor 正是扮演了这个角色：维护全局缓存索引，指导 Decode 节点从最优位置拉取，而非盲目向所有 Prefill 节点发起请求。

**LMCache PD Backend 选择了 Push**：Sender（Prefill 节点）主动申请 Receiver 端内存、推送数据、通知完成。**Mooncake 选择了 Pull-with-cache**：Conductor 调度器根据全局缓存位置决定从哪个节点的哪个存储层拉取，Decode 节点按需通过 RDMA 批量拉取。**vLLM KV Connector V1 没有预设方向**——它提供了 Push（`save_kv_layer`）和 Pull（`start_load_kv`）两套 API，由外部编排器决定用哪个。

### 2.2 什么时候传？Eager vs Pipelined vs Lazy

| 策略          | 传输时机                                                         |        TTFT 影响         | 实现复杂度 |
| ------------- | ---------------------------------------------------------------- | :----------------------: | :--------: |
| **Eager**     | 全部 Prefill 完成后，一次性传输所有 KV                           | 传输时间完全加在 TTFT 上 |     低     |
| **Pipelined** | 每算完一层，立即开始传输该层的 KV（与后续层的 Prefill 计算重叠） |  传输时间部分被计算隐藏  |     中     |
| **Lazy**      | 只在 Decode 实际需要某层 KV 时才加载该层                         |     逐层摊薄传输延迟     |     高     |

Pipelined（层级流水线）是最常用的折中：当 GPU 计算 Layer 3 的 Attention 时，Layer 2 的 KV 正在通过 RDMA 传输——计算和 I/O 重叠，TTFT 中额外增加的只有最后几层的传输延迟。

vLLM KV Connector V1 的 `start_load_kv` + `wait_for_layer_load(layer_name)` API 正是为 Pipelined 设计的——worker 在 forward 开始前触发异步加载，每层计算前等待该层数据就绪。LMCache PD Backend 和 Mooncake CPP 同样采用了层级流水线。

### 2.3 传什么？完整 KV vs 增量传输

| 策略         | 传输内容                                                   |   带宽需求   |
| ------------ | ---------------------------------------------------------- | :----------: |
| **完整 KV**  | 所有的 K 和 V，FP16/BF16 原始精度                          |     最高     |
| **增量传输** | 只传 prefix cache 未命中的部分（命中部分 Decode 节点已有） | 取决于命中率 |

> 量化传输（FP8/INT8）是另一个正交维度——它减小每个 token 的体量，但不改变传输策略本身。本文聚焦在完整/增量的策略选择，压缩技术的讨论见 [KV Cache 压缩技术详解](../compression/kv_cache_compression.md)。

三种方案都支持某种形式的增量传输：

- vLLM KV Connector V1 通过 `get_num_new_matched_tokens()` 让 Decode 节点的 scheduler 知道哪些 token 的 KV 已经存在于远程缓存中，只传输增量。
- LMCache PD Backend 的 `AllocRequest` 包含 `already_sent_indexes`——Receiver 回复 Sender 哪些 key 已经有了，避免重复传输。
- Mooncake 的 Conductor 维护全局 KV Cache 位置索引，调度 Decode 请求到已缓存前缀的节点上，减少跨节点传输。

---

## 三、三种实现的选择矩阵

### 3.1 vLLM KV Connector V1

vLLM V1 的 KV Connector API（PR #15960，2025 年 4 月合入 v0.8.5）定义了一套**抽象接口**，而非一种具体传输方案。它的核心设计选择是**不替用户做决定**：

```text
KVConnectorBase_V1:
  Scheduler 侧：
    get_num_new_matched_tokens()  → 远程缓存命中多少？
    update_state_after_alloc()    → 分配本地 buffer
    build_connector_meta()        → 构建传输指令

  Worker 侧：
    start_load_kv()               → 开始异步加载（Pull）
    wait_for_layer_load(layer)    → 等待某层就绪
    save_kv_layer(layer, kv)      → 开始异步保存（Push）
    wait_for_save()               → 等待全部保存完成
```

- **Push/Pull**：可插拔。`save_kv_layer` 在 Prefill 节点每层 KV 计算完成后触发推送（Push），`start_load_kv` / `wait_for_layer_load` 在 Decode 节点触发异步预取和逐层同步。这套 API 的设计以 **Push + Pipelined** 为主流用法——Prefill 侧推送、Decode 侧配合接收；Connector 实现者也可以基于它们构建纯 Pull 模式（此时无需 Prefill 侧 `save_kv_layer`），但当前生产部署以 Push 路径为主。
- **时序**：支持 Pipelined（`wait_for_layer_load` 在每层计算前同步）和 Eager（全部 load 完再开始 forward）。vLLM 本身不做传输——它给 connector 一个在每层计算前后插入 I/O 操作的"挂钩点"。
- **传输内容**：增量传输。`get_num_new_matched_tokens` 计算出远程已缓存的 token 数，调度器只为增量部分分配 block。

> **与 Chunked Prefill 的交互**：vLLM V1 的 PD 分离基于 Chunked Prefill 和 Prefix Caching 语义实现——PD 分离被视为"远程 prefix cache"。这意味着 Chunked Prefill 中的 block 增量分配、只在第一个 chunk 查找 prefix cache 等约束，同样适用于 PD 分离场景下的 KV 传输：Decode 节点首次调度时通过 `get_num_new_matched_tokens` 查询远程缓存命中数，后续 chunk 不再重复查询。详见 [vLLM Chunked Prefill 与 KV Cache](../scheduling/01_vllm_chunked_prefill.md)。

### 3.2 LMCache PD Backend

LMCache 的 PD Backend 是一个**具体的 Push 模式实现**，不做抽象——它直接规定了传输方向和协议[^1]：

- **Push**：明确的 Sender/Receiver 角色。Prefill 节点作为 Sender，主动向 Receiver 申请内存（`AllocRequest`），推送数据（RDMA/NIXL），通知完成。
- **时序**：支持 Pipelined。可以在 Prefill 逐层计算时同步推送——每层 Attention 完成后立即触发该层 KV 的传输。
- **传输内容**：支持增量。`AllocResponse` 返回 `already_sent_indexes`，Receiver 告诉 Sender 哪些 key 已经有了。

LMCache PD Backend 的独特之处在于它取代了 `LocalCPUBackend` 充当内存分配器——Receiver 端的 KV Cache 接收缓冲由 PDBackend 直接管理 Host 内存，而非走传统的 vLLM block table 分配路径。这意味着 KV 传输的内存管理被从 vLLM 的核心分配逻辑中抽离出来，交由 PD Backend 独立控制。

### 3.3 Mooncake

Mooncake 的 KV 传输是**全局调度 + 多层存储池**的副产品[^2]：

- **Pull-with-cache**：Conductor 维护全局 KV Cache 位置索引。Decode 请求被调度到已缓存所需前缀的节点上（减少传输）。如果需要跨节点 KV，Decode 节点通过 RDMA 从目标节点的 GPU/CPU/SSD 存储层拉取。
- **时序**：Pipelined。分块管道并行（CPP）将 Prefill 按层和块切分，KV 在计算过程中通过 RDMA 异步流水线传输。
- **传输内容**：完整 KV + 增量。有全局缓存索引的节点不走传输；无缓存的节点从最近的副本拉取完整数据。

Mooncake 与其他方案的关键区别是**KV 传输不是独立决策，而是全局资源调度的一部分**。Conductor 同时考虑缓存位置、GPU 负载、网络拓扑来决定"请求发给哪个 Decode 节点"和"KV 从哪个存储层拉"。

---

## 四、取舍

### 4.1 三维对比

| 维度           |                 vLLM KV Connector V1                 |          LMCache PD Backend          |                    Mooncake                     |
| -------------- | :--------------------------------------------------: | :----------------------------------: | :---------------------------------------------: |
| **Push/Pull**  |       可插拔（API 同时提供 Push 和 Pull 挂钩）       |       Push（Sender 主动推送）        |     Pull-with-cache（Conductor 决定从哪拉）     |
| **传输时序**   |     Pipelined（`wait_for_layer_load` 逐层同步）      |  Pipelined（逐层 Attention 后触发）  |          Pipelined（CPP 分块管道并行）          |
| **传输内容**   |  增量（`get_num_new_matched_tokens` 计算远程命中）   | 增量（`already_sent_indexes` 去重）  |           完整或增量（全局索引决定）            |
| **编排方式**   |    外部编排（connector 由外部 orchestrator 控制）    | 内置角色（Sender/Receiver 自动协商） |       全局调度（Conductor 统筹所有节点）        |
| **实现复杂度** | 低（vLLM 侧只定义 API，传输逻辑在 connector 实现中） |    中（固定 Push 模式，协议简单）    | 高（需要全局调度器 + 多层存储池 + RDMA 传输层） |
| **适用规模**   |            单对单 PD 分离（vLLM 实例对）             |        单对单或小规模 PD 分离        |             大规模集群（数十节点）              |

### 4.2 一句话总结

**PD 分离的 KV 传输本质上是在 Push 和 Pull、Eager 和 Pipelined、完整和增量三条轴上做排列组合。vLLM 选择了"不做选择"——提供 API 让社区自由选择组合；LMCache 选择了最简路径——固定 Push + Pipelined；Mooncake 选择了最全局的路径——让调度器统筹一切。** 三个方案适用于不同的规模：vLLM KV Connector 适合需要灵活对接不同传输后端的场景，LMCache PD Backend 适合快速搭建单对单 PD 分离，Mooncake 适合大规模集群下的全局优化。

---

## 相关阅读

- [KV Cache 层级流水线并行](../offloading/02_layerwise_pipeline.md) — 计算与 KV I/O 重叠的详细机制
- [vLLM Chunked Prefill 与 KV Cache](../scheduling/01_vllm_chunked_prefill.md) — PD 分离的调度基础
- [LMCache PD Backend 源码分析](../../02_systems/lmcache/pd_backend.md) — Sender/Receiver 角色与 AllocRequest 协议
- [Mooncake 架构概览](../../02_systems/mooncake/mooncake_architecture.md) — CPP 分块管道并行与 Conductor 全局调度
- [NIXL 网络传输库](../../02_systems/nixl/nixl_introduction.md) — RDMA、GDS 与跨节点传输抽象层

[^1]: LMCache PD Backend 源码分析见 [`pd_backend.md`](../../02_systems/lmcache/pd_backend.md)。核心设计：Push 模式，Sender 通过 AllocRequest 向 Receiver 申请远程内存，数据推送后通知 Proxy 完成。

[^2]: Mooncake 架构见 [`mooncake_architecture.md`](../../02_systems/mooncake/mooncake_architecture.md)。核心设计：Conductor 全局调度器维护 KV Cache 位置索引，RDMA-based Messenger 实现跨节点 KV 批量传输。
