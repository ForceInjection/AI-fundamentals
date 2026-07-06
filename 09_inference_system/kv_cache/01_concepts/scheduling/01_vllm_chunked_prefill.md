# vLLM Chunked Prefill 如何改变 KV Cache 管理

> 本文以 vLLM V1 调度器源码为基准。SGLang 的 Chunked Prefill 在调度策略（Prefill 优先、chunk 连续执行）和 Prefix Caching 交互（stash 机制支持跨 chunk 命中）上做出了不同选择，详见 [SGLang Chunked Prefill 原理与实现](../../../sglang/chunked-prefill.md)。

Preill 是一次性把整个 prompt 的 KV 全算完，还是切成小块逐步算？这看起来只是计算调度的问题，但 vLLM 的 Chunked Prefill 实际上深刻改变了 KV Cache 的分配时序、与 Prefix Caching 的交互方式，以及 Preemption 的触发窗口。

这篇文章从 KV Cache 管理的视角拆解 vLLM 的 Chunked Prefill 实现——它如何把"一次性申请全部 block"变成"逐步申请逐批增长"，以及这个看似简单的变化在工程上引出的三个约束：prefix cache 只在第一个 chunk 查找、chunk 大小必须对齐 block_size、以及 chunk 之间的 Preemption 风险。

> **前置阅读**：[PagedAttention 原理介绍](../basic/paged_attention.md) — block table 与按需分配；[Prefix Caching 原理分析](../prefix_caching/prefix_caching.md) — hash chaining 与 block 级复用。

---

## 一、一次算完的代价

### 1.1 Prefill 的算力特性

Prefill 需要处理全部输入 token——序列长度 $S$ 的 prompt，每层 Attention 的计算量是 $O(S^2)$，全部层的总 FLOPs 与 $S^2$ 成正比。一个 32K token 的 prompt，Prefill 的算力开销约为 4K token 的 64 倍。

更重要的是，Prefill 是 **compute-bound**——GPU 的计算单元被占满，但显存带宽并未饱和。而 Decode 是 **memory-bound**——每步只算一个新 token，瓶颈在读 KV Cache 的带宽。

### 1.2 传统 Prefill 对 KV Cache 的双重冲击

一次性 Prefill 意味着一次性为整个 prompt 的 **所有 token** 分配 KV Cache block：

```text
传统 Prefill（无 Chunked）：
  请求到达（32K token prompt）
    ↓
  一次性分配所有 block：32K ÷ 16 = 2000 blocks
    ↓
  GPU 持续 O(S²) 计算，Decode 被阻塞
    ↓
  全部 Prefill 完成后，才开始第一个 token 的生成
```

两个问题同时出现：

- **Block 分配峰值高**：2000 个 block 瞬间被占用。如果此时 GPU 上还有其他请求正在 Decode，可能导致 block 池耗尽，触发 Preemption。
- **Decode 被阻塞**：用户感知的 TTFT（Time To First Token）等于整个 Prefill 的耗时。32K token 的 Prefill 可能需要数秒——用户盯着空白屏幕等第一个字。

### 1.3 Chunked Prefill 的核心思路

既然 Prefill 算得慢又占得多，那就不等它全算完——切成小块，每步只算一部分 token，与 Decode 交替执行。vLLM V1 用统一的 token budget 调度所有请求，chunk 之间的间隙自然被其他请求的 Decode 步骤填充。[^1]

```text
Chunked Prefill（max_num_scheduled_tokens=2048）：
  请求到达（32K token prompt）
    ↓
  Step 1: Prefill chunk 0-2047    → 分配 128 blocks → 计算 → 留在 running 队列
  Step 2: Decode 其他请求          → 生成 token
  Step 3: Prefill chunk 2048-4095  → 分配 128 blocks → 计算 → 留在 running 队列
  Step 4: Decode 其他请求          → 生成 token
  ...
  Step N: Prefill 最后一个 chunk   → 分配 128 blocks → 计算
  全部 Prefill 完成后，num_computed_tokens >= num_tokens，切换到 Decode（`num_computed_tokens` 详见 §2.1）
```

Chunked Prefill 把算力开销和 block 分配分散到多步中，让**其他请求的 Decode** 有机会在本请求的 chunk 之间执行——请求自身仍需完成所有 chunk 后才能进入 Decode，但它不再需要等排在它前面的长请求全部 Prefill 完。TTFT 的改善来自"不被其他请求长时间阻塞"，而非"自己的第一个 chunk 就能解码"。

---

## 二、Chunked Prefill 对 KV Cache 的三个改变

### 2.1 Block 分配：从一次性到增量式

这是最核心的变化。无 Chunked Prefill 时，`allocate_slots()` 一次性为整个 prompt 分配全部 block。Chunked Prefill 下，每次只分配当前 chunk 所需的 block：

```text
32K token prompt, block_size=16, chunk_size=2048:

传统 Prefill:
  allocate_slots(request, num_new_tokens=32768)
  → 一次性分配 2048 个 block

Chunked Prefill:
  Step 1: allocate_slots(request, num_new_tokens=2048)  → 128 blocks
  Step 2: allocate_slots(request, num_new_tokens=2048)  → 128 blocks  (追加)
  ...
  Step 16: allocate_slots(request, num_new_tokens=2048) → 128 blocks  (追加)
```

从 KV Cache 管理的角度看，这有三个直接影响：

**内存峰值降低**。如果请求之间有交错——Decode 在此请求的 chunk 之间处理其他请求——单个请求不会在任意时刻独占所有 block。当然，如果只有这一个请求，总 block 数不变，但峰值的时间分布更平滑。

**Preemption 窗口期拉长**。传统 Prefill 下，Preemption 只能在 Prefill 开始前或完成后发生。Chunked Prefill 在两个 chunk 之间也会检查 block 可用性：如果 Step 3 的 `allocate_slots()` 返回 `None`，调度器可以选择 Preempt 优先级更低的请求，或者将当前请求暂时搁置。

**`num_computed_tokens` 成为状态指针**。在 vLLM V1 调度器中，一个请求是"正在 Prefill"还是"正在 Decode"，由 `num_computed_tokens` 与总 token 数的关系决定：`is_prefill_chunk = num_computed_tokens < num_tokens`。每完成一个 chunk，`num_computed_tokens` 向前推进——它既是"已计算了多少 token"的计数器，也是"下一次从哪开始 Prefill"的指针。请求在 `num_computed_tokens >= num_tokens` 之前，`is_prefill_chunk` 条件为真，每个调度步均走 Prefill 处理路径——直到全部 token 计算完毕后才切换到 Decode。

### 2.2 Prefix Caching：只在第一个 chunk 查找

这里 vLLM 有一个当前版本的重要约束：**Prefix Caching 的命中检查只在请求首次被调度时进行**[^1]。

```python
# vLLM V1 scheduler: 前缀缓存查找的入口条件
if request.num_computed_tokens == 0:
    # 首次调度：查找 prefix cache
    new_computed_blocks, num_new_local_computed_tokens = \
        self.kv_cache_manager.get_computed_blocks(request)
    # ...基于命中结果决定跳过多少 token
else:
    # 后续 chunk：跳过 prefix cache 查找，直接从上一次中断处继续
    # num_computed_tokens 已经记录了进度
```

这意味着：如果请求 A 和请求 B 共享同一个长 System Prompt，但 A 先被调度，已经占用了前 2048 个 token 的 block，B 在第一个 chunk（`num_computed_tokens==0`）可以命中 A 的缓存。但 A 的第二个 chunk（`num_computed_tokens > 0`）不会去查 B 可能已经算好的后续 block——无论缓存是否存在，A 的后续 chunk 都会自己重新 Prefill。

这不是设计缺陷，而是一个工程权衡。跨 chunk 做 prefix cache 查找的障碍不仅在于代码复杂度——还包括更实质的工程问题：缓存命中后需要判断命中的 block 是否与当前请求已分配的物理页冲突、部分命中时需要重组 block table 映射、以及处理"缓存的 block 已被其他请求淘汰"的竞态。当前版本选择了一条状态机更可控的路径：只在 `num_computed_tokens == 0` 时做一次缓存匹配，后续 chunk 沿着已建立的 block table 继续追加。

> **对比 SGLang**：SGLang 采取了不同的设计——通过 `stash_chunked_request()` 将每个 chunk 的部分 KV 写回 Radix Tree，后续 chunk 调用 `init_next_round_input()` 重建 `fill_ids` 后重新匹配前缀，使得跨 chunk 的 HiCache 命中成为可能。两种设计体现了同一个权衡的不同选择：vLLM 选择了调度逻辑的简单性，SGLang 选择了缓存利用率的完整性。详见 [SGLang Chunked Prefill 原理与实现](../../../sglang/chunked-prefill.md)。

### 2.3 Block 对齐：chunk 大小必须是 block_size 的整数倍

Chunked Prefill 与 Prefix Caching 同时启用时，vLLM 要求 `max_num_batched_tokens`（即 chunk size）必须能被 `block_size` 整除[^2]。

```python
# vllm/v1/core/sched/scheduler.py: chunked prefill + prefix caching 的约束
# 如果 token budget 无法完整容纳剩余 token：
# 只调度 (budget // block_size) * block_size 个 token
num_new_tokens = (budget // block_size) * block_size
```

原因在于 Prefix Caching 需要**完整的 block** 做 hash 匹配。如果一个 chunk 在 block 中间结束（比如 block_size=16，chunk 只处理了 20 个 token，恰好卡在第二个 block 的第 4 个 token），这第二个 block 处于"半满"状态——它的 hash 无法用于缓存查找，因为它还缺 12 个 token 才能形成一个完整 block。

这个约束有一个实际影响：token budget 可能无法被充分利用。例如 `block_size=16, budget=133`，只能调度 `(133//16)*16 = 112` 个 token——浪费了 21 个 token 的预算额度。PR #7753 的作者注明了这是一个"临时约束"，理论上可以通过支持 partial block hashing 来解除，但截至当前版本仍然有效。

---

## 三、三个派生问题

### 3.1 第一个 chunk 的 hit 与后续 chunk 的 miss 并存

结合 §2.2 和 §2.3，一个在实际部署中常见的场景：

```text
请求 A: System Prompt (4096 token) + User Query (256 token)
请求 B: System Prompt (4096 token) + User Query (512 token)

Chunk size = 2048, block_size = 16 (128 blocks/chunk)

请求 A 先到达：
  Chunk 1 (token 0-2047): 首次调度 → 查 prefix cache → 无命中 → 分配 128 blocks，计算
  Chunk 2 (token 2048-4095): num_computed_tokens=2048 > 0 → 跳过查找 → 分配 128 blocks，计算
  Chunk 3 (token 4096-4352): User Query → 分配 16 blocks，计算（同样不会触发 prefix cache 查找）
  Preill 完成，开始 Decode

请求 B 随后到达：
  Chunk 1 (token 0-2047): 首次调度 → 查 prefix cache → 命中 A 的 chunk 1 的 block！→ 复用
  Chunk 2 (token 2048-4095): num_computed_tokens=2048 > 0 → 跳过查找 → 但 A 的 chunk 2 的缓存明明存在……

  B 的 Chunk 2 会自己重新 Prefill 后 2048 个 token——即使 A 已经算过且缓存了相同的 block。
```

这个场景揭示了当前实现的一个局限：**只有从位置 0 开始的请求才能从 prefix cache 中受益**。如果共享前缀正好跨越了一个 chunk 边界，第一个 chunk 之后的缓存全部浪费。

### 3.2 Chunk 之间的 KV Cache 碎片

Chunked Prefill 分配 block 是增量的——每个 chunk 追加到已有的 block table 中。但如果两个 chunk 之间，其他请求分配并释放了中间的一些 block，这个请求的 block table 在物理显存上是不连续的。PagedAttention 通过 block table 的虚拟→物理映射天然处理了这种不连续性，不会影响正确性。但极端的物理碎片可能降低 block table 遍历时的缓存局部性——虽然在实际中，PagedAttention 的虚拟→物理映射对 compute kernel 完全透明，且 block_size=16 或更大时这个影响极小。

### 3.3 long_prefill_token_threshold：长 prompt 的特殊对待

vLLM 提供了一个 `long_prefill_token_threshold` 参数用于标注"长" prompt：超过此阈值的 prompt，每个调度步最多只能处理 `long_prefill_token_threshold` 个 token，禁止一个 prompt 吃掉全部 token budget[^3]。这个参数配合 `max_long_partial_prefills`（默认 1）使用，允许短 prompt 跳过排在前面的长 prompt 优先被调度——"短 prompt 插队"。

从 KV Cache 角度，这引入了更复杂的 block 分配交错模式。但本文不展开讨论——它更多是调度策略而非 KV Cache 机制本身的改变。

---

## 四、与 Preemption 的交互：窗口期变宽了

传统 Prefill 下，Preemption 只有一个触发点：`allocate_slots()` 为整个 prompt 分配 block 时，发现空闲 block 不够。Chunked Prefill 把"一次分配"变成"多次分配"，每个 chunk 的 `allocate_slots()` 调用都可能触发 Preemption。

```text
Chunked Prefill 的 Preemption 窗：
  Chunk 1: allocate_slots → 成功 → 计算 →
  Chunk 2: allocate_slots → 失败（block 池耗尽）
           → 调度器检查 running 队列
           → 选择 victim（FCFS: self.running.pop()，PRIORITY: max priority）
           → _preempt_request() → 释放 victim 的全部 block
           → allocate_slots 重试 → 成功 → 继续计算
```

这比传统 Prefill 多了一个好处：**Preemption 不再需要一次性释放足够的 block 来容纳整个 prompt**。传统 Prefill 下，如果 prompt 需要 2000 个 block，必须 Preempt 足够多的请求来腾出 2000 个 block。Chunked Prefill 只需要腾出当前 chunk 所需的 block（如 128 个）——门槛大幅降低。

但同时多了一个风险：**请求可能在半路被 Preempt**。同一个 `schedule()` 调用内，`allocate_slots()` 失败时，victim 通常选自 `self.running` 队列中的其他请求，而非当前正在被调度的请求本身。但如果后续调度轮次中显存压力持续，该请求优先级较低，则可能在后续轮次中被选为 victim——此时已计算的 3 个 chunk（384 blocks, num_computed_tokens=6144）全部被释放，恢复时 `num_computed_tokens` 归零，需要从头 Prefill。Chunked Prefill 让请求在"半完成"状态下暴露了更长的 Preemption 窗口——单次调度中通常不会 preempt 当前请求，但跨调度轮次的竞争中，半完成的请求仍面临被牺牲的风险。

---

## 五、取舍

### 5.1 三个维度的变化

| 维度                      |             传统 Prefill              |                 Chunked Prefill                  |
| ------------------------- | :-----------------------------------: | :----------------------------------------------: |
| **Block 分配峰值**        |         高（一次性分配全部）          |             低（按 chunk 增量分配）              |
| **TTFT（首 token 延迟）** | 差（长 prompt 需等全部 Prefill 完成） |            改善（Decode 可交错执行）             |
| **总吞吐**                |       高（无 chunk 间调度开销）       |               略低（调度开销增加）               |
| **Prefix Cache 利用率**   |       高（一次查找，完整命中）        |            中（仅第一个 chunk 查找）             |
| **Preemption 门槛**       |   高（需腾出整个 prompt 的 block）    |        低（只需腾出单个 chunk 的 block）         |
| **Preemption 代价**       |            重算整个 prompt            | 重算已完成的所有 chunk（如果请求自己被 Preempt） |

### 5.2 Chunk Size 的选择困境

- **大 chunk**（如 4096）：接近传统 Prefill 的行为——分配峰值高、Decode 等待久，但 prefix cache 利用率高（共享前缀更可能完整落在一个 chunk 内）
- **小 chunk**（如 512）：分配更平滑、其他请求的 Decode 能更频繁地被穿插执行（请求自身仍需完成全部 chunk 才能开始 Decode），但第一个 chunk 更小意味着"从位置 0 开始的缓存查找"能复用的 token 更少——共享前缀可能在第一个 chunk 之后还在延续，但调度器不会再查缓存
- **vLLM 默认**：`max_num_batched_tokens` 不设硬上限，由 token budget 自然约束。PR #7753 的 benchmark 显示 chunk_size=2048 配合 block_size=16 在 Llama-2 上取得了最佳吞吐（3929.7 tok/s）

### 5.3 一句话总结

**Chunked Prefill 把 KV Cache 的分配从"一次性大宗申请"变成了"多次小额分期"，降低了峰值压力和 Preemption 门槛，但付出的代价是 Prefix Caching 只在第一个 chunk 生效——共享前缀如果跨越 chunk 边界，后续的缓存将白白浪费。** 这个约束源自一个工程选择：为简单性牺牲跨 chunk 的缓存查找。如果未来 vLLM 支持 partial block hashing 或跨 chunk 缓存索引，Chunked Prefill 的内存效率还会有成倍的提升空间。

---

## 相关阅读

- [PagedAttention 原理介绍](../basic/paged_attention.md) — Block 级内存管理：block table、按需分配、碎片率
- [Prefix Caching 原理分析](../prefix_caching/prefix_caching.md) — Hash chaining 与 block 级复用机制
- [Attention Sinks 与 KV Cache 淘汰策略](../eviction/attention_sinks_and_eviction.md) — KV Cache 满了怎么淘汰
- [为什么 GPU 生成每个 token 时利用率不到 5%？——Prefill 与 Decode 深度拆解](../../../prefill_decode/prefill_decode_qkv_calculation.md) — compute-bound vs memory-bound 的数学推导
- [SGLang Chunked Prefill — 原理与代码实现](../../../sglang/chunked-prefill.md) — SGLang 的不同设计选择：Prefill 优先、chunk 连续执行、stash 机制支持跨 chunk 前缀缓存命中

[^1]: vLLM V1 调度器源码 [`vllm/v1/core/sched/scheduler.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py) — `if request.num_computed_tokens == 0:` 守卫条件。后续 chunk 跳过前缀缓存查找，直接从上一次中断处继续。

[^2]: vLLM PR #7753 ["Enable chunked prefill and prefix caching together"](https://github.com/vllm-project/vllm/pull/7753) — 引入 block 对齐约束（`budget // block_size * block_size`），并要求 `max_num_batched_tokens % block_size == 0`。作者注明此为临时约束。

[^3]: vLLM 调度器配置 [`vllm/config/scheduler.py`](https://github.com/vllm-project/vllm/blob/main/vllm/config/scheduler.py) — `long_prefill_token_threshold` 默认 0（禁用），`enable_chunked_prefill` 默认 `True`，`max_long_partial_prefills` 默认 1。
