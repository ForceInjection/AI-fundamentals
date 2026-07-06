# KV Cache Prefetching：三层预取如何隐藏 KV 访问延迟

Decode 阶段的每一轮 forward，GPU 都要从 HBM 中读取完整的 KV Cache。以 LLaMA-2 70B @ 32K context 为例，单次 decode 的 KV 读取量约为 10 GB——而 H100 的 HBM 带宽约 3.35 TB/s，这意味着单次 KV 读取至少需要 ~3ms，已经赶上了整个 decode forward 的耗时。实际上，attention kernel 的 compute 利用率通常不到 25%[^1]——GPU 的大量时间不是在计算，而是在**等 KV 数据从显存到位**。

Prefetching 的思路是：**不等数据到了再算，而是在需要之前就把数据提前搬到更近的地方。** 这个"提前搬"可以在三个层次上发生：kernel 内部从 HBM 搬到 L2 cache，跨节点从 Prefill 节点的 GPU 搬到 Decode 节点的 GPU，以及跨存储层级从 NVMe/远程存储搬到本地 GPU。

本文从这三个层次拆解 KV Cache Prefetching 的机制、实现与取舍。三个层次目前在业界没有单一框架统一实现——Kernel 层来自学术论文（框架无关的 CUDA 技术），系统层以 vLLM KV Connector V1 为代表，存储层以 SGLang HiCache 为代表。本文将它们放在同一个 prefetch 框架下对比，各节会标注对应的实现来源。

> **前置阅读**：[PD 分离架构下的 KV Cache 传输](pd_kv_transfer.md) — 跨节点的 KV 传输构成了第二层 prefetch 的基础；[KV Cache 层级流水线并行](layerwise_pipeline.md) — 计算与 I/O 重叠是 prefetch 的核心思想。

---

## 一、Decode 为什么需要 Prefetch？

### 1.1 等数据的时间远比算的时间长

Decode 每步只处理一个新 token 的 query，但它需要 attend 到**所有历史 token** 的 K 和 V。对于 32K context、80 层、GQA 8 heads 的模型：

$$\text{单步 KV 读取量} = 2 \times 80 \times 8 \times 128 \times 32768 \times 2 \text{ bytes} \approx 10 \text{ GB}$$

这只是解码一个 token 的访存量。Decode 阶段每个请求需要独立读取自己完整的 KV Cache，总读取量大致与 batch size 成正比——batch=32 时约 32 × 10 GB = 320 GB 的 KV 数据需要从 HBM 读出。Attention kernel 的 compute utilization（MFU）通常只有 20-30%[^1]，其余时间都在等 HBM。

### 1.2 三级延迟，三级 Prefetch

KV Cache 的访问延迟在三个层次上有数量级的差距：

| 层次          | 数据源 → 目标                      |                                     典型延迟                                      | 隐藏手段                                    |
| ------------- | ---------------------------------- | :-------------------------------------------------------------------------------: | ------------------------------------------- |
| **Kernel 层** | HBM → L2 cache                     |                                       ~1 μs                                       | L2 prefetch（`cp.async.bulk.prefetch.L2`）  |
| **系统层**    | Prefill 节点 GPU → Decode 节点 GPU | 取决于 KV 大小和网络带宽：~100 μs（NVLink 小数据）–200 ms（10 GB @ 50 GB/s RDMA） | PD KV 异步预取（`load_kv_async`）           |
| **存储层**    | NVMe/远程存储 → 本地 GPU           |      取决于存储类型和数据量：~1 ms（本地 NVMe 小块）–100 ms（远程存储大 KV）      | HiCache 预取策略（`best_effort`/`timeout`） |

三个层次的 prefetch 在机制上彼此独立，可以叠加使用。下表先给出各层的快速概览，后续三章逐一展开。

| 层次          | 核心机制                                                                         | 代表实现                                                                                                     | 硬件依赖         |
| ------------- | -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ | ---------------- |
| **Kernel 层** | 在计算当前 block 的 $Q \cdot K^T$ 时，提前将下一个 block 的 KV 从 HBM 搬到 L2    | `cp.async.bulk.prefetch.L2`（Hopper SM 9.0+）[^1]                                                            | H100/H200/B200   |
| **系统层**    | 在 PD 分离下，Decode 节点提前从 Prefill 节点拉取 KV，与 attention 计算流水线重叠 | vLLM `load_kv_async` + KV Connector[^2]（支持逐层流水线），LMCache PD Backend（Push 模式，层级流水线开发中） | RDMA/NIXL/NVLink |
| **存储层**    | 在 Decode 开始前，从 NVMe/远程存储预取 KV 到本地 GPU                             | SGLang HiCache `best_effort`/`timeout`/`wait_complete`[^3], LMCache 分层存储跨层预取                         | NVMe/RDMA        |

---

## 二、Kernel 层：从 HBM 到 L2 的异步预取

### 2.1 问题：Attention kernel 的 cache miss 噩梦

Decoder 的 attention kernel 处理逻辑是逐 block 遍历 KV Cache：`for block_i in range(num_blocks): Q @ K[block_i].T`。FlashAttention 通过分块（tiling）将计算拆分到 SRAM 中执行，解决了 HBM↔SRAM 的重复读写问题——但它不能消除一个根本事实：**每个 block 的 KV 数据仍需要从 HBM 加载**。在 decode 阶段，Q 只有 1 个 token，每次迭代加载一个 KV block（典型 4 KB，d_h=128, block_size=16）。由于 L2 cache 容量有限（H100 上约 50 MB），32K context 的 KV Cache（~10 GB）完全无法放入 L2。论文实测 XFormers 后端的 L2 hit rate 接近 0%，77% 的 warp cycle 消耗在 Stall Long Scoreboard（等待 HBM 数据）上[^1]。

### 2.2 方案：在算当前 block 时预取下一个 block

Hopper 架构引入的 `cp.async.bulk.prefetch.L2` PTX 指令允许程序**显式地将数据从 HBM 预取到 L2 cache**，而不阻塞当前 warp 的执行。核心思路是：

```text
传统 Attention（逐 block 无预取）：
  Block 0: load KV → compute QK^T → load KV → compute QK^T → ...
            ↑ HBM miss           ↑ HBM miss
每次 load 都等 ~1μs HBM 延迟

L2 Prefetch Attention：
  Block 0: load KV → compute QK^T ─┐
                 └→ prefetch block 1 ┘  compute 与 HBM load 重叠
  Block 1: load KV (L2 hit!) → compute QK^T ─┐
                                      └→ prefetch block 2 ...
  Block 2: load KV (L2 hit!) → compute ...
```

当一个 warp 正在计算当前 block 的 $Q \cdot K^T$ 时，硬件已经在后台将下一个 block 的 KV 从 HBM 搬到了 L2——等到实际需要加载 block 1 时，数据已经在 L2 中，访问延迟从 ~1 μs（HBM）降到 ~0.1 μs（L2）。

论文实现中，每个 thread block（128 线程 = 4 warps）专用于处理一个 attention head。Warp 0 负责 block 0, 4, 8...，Warp 1 负责 block 1, 5, 9...，以此类推。每个 warp 在计算当前 block 时预取的**不是自己的下一个 block，而是 stride = w（warp 总数）之后的 block**——即 Warp 0 算 block 0 时预取 block 4（其他 warp 将在下一轮使用的 block），Warp 0 算 block 4 时预取 block 8。这种跨 warp 的预取编排确保了 L2 中的数据在 warp 之间轮转复用。

### 2.3 效果与局限

论文在 LLaMA-2 7B 上实测 attention kernel 加速高达 2.15×，端到端吞吐提升 1.97×，L2 hit rate 从 0% 提升到 43-83%[^1]。

但这一技术有三个关键局限：

**GQA 衰减**：prefetch 的收益来自加速每个 KV head 的 block 加载——每个 KV head 的每个 block 都可以被独立预取。在 MHA 下（32 Q heads = 32 KV heads），每次迭代有 32 个独立的预取机会。在 GQA 下（如 28 Q heads : 4 KV heads，7:1），同一个 KV block 被多达 7 个 Q head 复用——预取一次即可服务多个 Q head，但额外的 Q head 无法再通过新预取隐藏延迟。同时，预取的 KV 数据占据 L2 空间，可能在大量 Q head 轮转中被提前 evict。**prefetch 收益与 KV head 数量成正比**：Qwen2.5-7B（7:1 GQA，仅 4 个 KV heads）上论文观测到 2-5% 的退化——预取带来的微小收益被 L2 eviction 开销和额外的 warp 编排成本抵消。

**L2 容量墙**：预取的总数据量不能超过 L2 大小（~50 MB）。batch size 较大或 context 较长时，预取的数据互相 evict，收益递减。

**与 FlashAttention-3 的定位差异**：FA3 利用 Hopper 的 Tensor Core（WGMMA 指令）、TMA（Tensor Memory Accelerator）异步数据搬运和 FP8 低精度计算来加速 attention——这些优化与 KV head 数量无关，对 MHA 和 GQA 同样有效。L2 Prefetch 的收益则与 KV head 数量成正比——KV head 越多，可预取的独立数据流越多。两者不是替代关系：在 MHA 上 L2 Prefetch 显著优于 FA3（实测 +15-110%，视模型而定[^1]），在激进 GQA 上 FA3 更稳定。理想的实现可以将两者叠加——TMA 负责 SRAM 级数据搬运，`cp.async.bulk.prefetch.L2` 负责 HBM→L2 预取。

**Hopper 独占**：`cp.async.bulk.prefetch.L2` 仅在 SM 9.0+（H100/H200/B200）上可用，A100 不可用。

---

## 三、系统层：PD 分离下的 KV 异步预取

### 3.1 问题：跨节点 KV 加载阻塞 Decode

在 PD 分离架构中，Decode 节点在开始推理之前，必须从 Prefill 节点获取 KV Cache。如果等全部 KV 传输完成才开始 forward——就等于把传输延迟直接加在 TTFT 上。10 GB KV 在 50 GB/s RDMA 下需要 ~200ms，这对 TTFT 是不可接受的。

### 3.2 方案：层级流水线预取

vLLM V1 的 KV Connector API 提供了 `start_load_kv` + `wait_for_layer_load(layer_name)` 两个挂钩点[^2]，专门用于实现异步层级预取：

```text
层级流水线预取（Pipelined Prefetch）：
  Step 1: start_load_kv()        → 触发所有层的 KV 异步加载
  Step 2: 开始 Layer 0 forward
          wait_for_layer_load(0) → Layer 0 KV 已就绪 → 计算
          save_kv_layer(0)       → 保存 Layer 0 输出的 KV（push 模式）
  Step 3: 开始 Layer 1 forward
          wait_for_layer_load(1) → Layer 1 KV 在 step 1-2 期间已加载完成
          ...
```

关键是在 GPU 计算 Layer 0 的 attention 时，Layer 1 的 KV 正在从 Prefill 节点通过网络传输——计算和 I/O **完全重叠**。最终 TTFT 中额外增加的只有第一层的加载延迟，而非全部 80 层的传输时间。

vLLM scheduler 中的 `load_kv_async` 标志正是控制这一行为：当 `load_kv_async = True` 时，调度器在 `allocate_slots()` 中为异步加载的 token 提前分配 block，但延迟执行 `cache_blocks()` 直到 connector 确认 KV 已就绪。LMCache PD Backend 在 Push 模式下由 Sender（Prefill 节点）主动推送 KV 到 Receiver（Decode 节点），数据传输与计算可以在逐层粒度上重叠。不过截至当前版本，LMCache 的层级流水线预取仍在开发中（`TODO: Add layerwise support`）——当前 PD Backend 推送的是完整的 chunk KV，而非逐层独立传输。

### 3.3 与 Speculative Decoding 的交互

当 `load_kv_async` 与 Eagle speculative decoding 同时启用时，vLLM 会减少或临时禁用 speculative lookahead——调度器限制 `lookahead_tokens`，直到关键层的 KV 异步加载完成。因为投机 token 的生成依赖完整的 KV Cache，异步加载未完成时不能安全地进行投机解码[^2]。这是 prefetch 激进程度与 speculative decoding 加速效果之间的典型权衡。

---

## 四、存储层：从 NVMe/远程到 GPU 的跨层预取

### 4.1 问题：冷启动的 KV Cache 从哪来？

当 PD 分离下的 Decode 节点冷启动时（无本地缓存、无 Prefill 节点预热），KV Cache 只能从 L3 存储层（NVMe 或远程分布式存储）加载。在某些部署中，Prefill 节点本身也需要从存储层加载历史 KV——此时存储层预取发生在 Prefill 端，Prefill 完成后再通过系统层传输到 Decode 端。NVMe 的延迟（~100 μs 随机读）和带宽（~7 GB/s 顺序读）远不如 GPU HBM，直接从 L3 加载全部 KV 可能导致秒级延迟。

### 4.2 方案：SGLang HiCache 的三级预取策略

SGLang HiCache 针对存储层的 cross-layer prefetch 设计了三种策略[^3]：

| 策略                      | 行为                                                                   |    TTFT 影响     | 适用场景                             |
| ------------------------- | ---------------------------------------------------------------------- | :--------------: | ------------------------------------ |
| **`best_effort`**（默认） | 发起预取后立即返回，不等数据到位。如果数据未及时到达，GPU 重算缺失部分 |     最低延迟     | TTFT 敏感，宁可重算不等磁盘          |
| **`timeout`**             | 等待预取完成或超时。部分数据就绪即可放行，剩余部分 GPU 重算            | 兼顾延迟与命中率 | 适中场景                             |
| **`wait_complete`**       | 阻塞等待全部预取数据就绪后才开始计算                                   |     最高延迟     | 重算成本极高（长 context），不介意等 |

预取触发条件是 L3 命中 KV Cache 的连续长度超过 `prefetch_threshold`（默认 256 token）——小于此阈值不值得发起 I/O，直接在 GPU 上重算更经济。

### 4.3 三阶段流水线

HiCache 的预取分三个阶段并行执行：

```text
prefetch_from_storage(req_id)
  → prefetch_queue → prefetch_thread (L3 hit 查询, TP all_reduce)
  → prefetch_buffer → I/O 线程 (NVMe→CPU 同步读)
  → tree insert → 将加载数据插入 host radix tree (scheduler 线程)
```

三个阶段在独立的线程中流水线执行——查询、I/O、插入三者在不同请求之间重叠，最大化存储带宽利用率。LMCache 的分层存储架构（L1 GPU → L2 CPU → L3 Disk → L4 Remote）同样依赖跨层预取：当请求需要的 KV 位于 L3/L4 时，后台 I/O 线程提前将数据加载到 L1/L2，与 GPU 计算流水线重叠。

---

## 五、三层 Prefetch 的协作与冲突

### 5.1 三层可以叠加

三层 prefetch 作用于不同的延迟源，可以同时启用：

```text
存储层 prefetch (HiCache best_effort):
  NVMe → CPU → GPU，Decode 开始前触发
    ↓
系统层 prefetch (vLLM load_kv_async):
  Prefill GPU → Decode GPU，层级流水线
    ↓
Kernel 层 prefetch (L2 prefetch):
  HBM → L2 cache，attention kernel 内部
```

一个部署在 PD 分离 + Hopper GPU + HiCache 上的系统可以三层全开：HiCache 从 NVMe 预取到 GPU，KV Connector 在 Decode 和 Prefill 节点间做层级流水线预取，attention kernel 内部再做 L2 prefetch。

### 5.2 潜在的冲突

**缓存污染**：Kernel 层的 L2 prefetch 是 blind prefetch——每个 block 都预取，不考虑它是否真的会被用到。如果 batch size 很大，过多的 prefetch 数据互相 evict，反而降低有效 L2 hit rate。

**带宽竞争**：当存储层使用远程存储（如 LMCache 的 L4 Remote 或 HiCache 的远端 L3）且走与 PD 传输相同的 RDMA/NVLink 链路时，系统层的 KV 传输和存储层的 prefetch 会产生带宽竞争。若存储层从本地 NVMe 加载，则不涉及网络竞争。

**激进预取 vs 重算成本**：`best_effort` 策略下，预取的数据可能不完整——GPU 需要重算缺失部分。在长 context 下重算成本高，可能需要切换到 `timeout` 或 `wait_complete`。

---

## 六、取舍

### 6.1 三层 Prefetch 的适用条件

Kernel 层 L2 prefetch 最适合 **MHA 或低 GQA ratio** 的模型在 **Hopper GPU** 上运行。GQA ratio 高于 4:1 时收益递减，A100 不可用。

系统层 PD KV prefetch 是 **PD 分离部署的必备优化**——没有层级流水线预取，TTFT 将在传输延迟上百毫秒级别。与 Spec Decoding 同时启用时需注意 `lookahead_tokens` 限制。

存储层 HiCache prefetch 适合 **冷启动或缓存 miss 场景**。`best_effort` 是 TTFT 敏感场景的默认选择；如果 KV 重算成本极高（超长 context、复杂模型），考虑 `wait_complete`。

### 6.2 一句话总结

**KV Cache Prefetching 的本质是在三个层次上——kernel、系统、存储——将"等数据"与"做计算"重叠。从 Hopper 的 L2 预取指令到 vLLM 的层级异步加载到 HiCache 的跨存储预取，核心逻辑都是同一个：提前把数据搬到更近的地方，让 GPU 不再空转。** 三层可以叠加，但各自有硬件依赖和适用边界——L2 prefetch 要 Hopper、PD prefetch 要 RDMA/NIXL、存储 prefetch 要 NVMe。选择哪个层次，取决于你的硬件栈和延迟预算。

---

## 相关阅读

- [PD 分离架构下的 KV Cache 传输](pd_kv_transfer.md) — 跨节点 KV 传输的 Push/Pull 与 Pipelined 策略
- [KV Cache 层级流水线并行](layerwise_pipeline.md) — 计算与 KV I/O 重叠的详细机制
- [SGLang HiCache 深入详解](../../../sglang/hicache_deep_dive.md) — 三级存储架构与三种预取策略的完整分析
- [投机解码如何与 KV Cache 交互](spec_decode_kv_cache.md) — Prefetch 与 Spec Decoding 的 `lookahead_tokens` 交互

[^1]: Zhao et al., "Asynchronous KV Cache Prefetching for LLM Inference," arXiv:2504.06319v2, 2025. — 提出基于 `cp.async.bulk.prefetch.L2` 的 L2 cache 异步预取方案。在 LLaMA-2 7B 上 attention kernel 加速 2.15×，端到端吞吐提升 1.97×，L2 hit rate 从 ~0% 提升至 43–83%。GQA ratio ≥ 4:1 时收益递减。FA3 对比见 Shah et al., "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision," 2024。FlashAttention 基础见 Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness," 2022。

[^2]: vLLM V1 调度器 [`vllm/v1/core/sched/scheduler.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py) — `load_kv_async` 控制异步 KV 加载，配合 `start_load_kv`/`wait_for_layer_load` 实现层级流水线预取。与 Eagle spec decoding 同时启用时限制 `lookahead_tokens`。

[^3]: SGLang HiCache 预取机制见 [`hicache_deep_dive.md`](../../../sglang/hicache_deep_dive.md) — `--hicache-storage-prefetch-policy` 控制 `best_effort`/`timeout`/`wait_complete` 三种策略，`prefetch_threshold` 默认 256 token。
