# PagedAttention：当 KV Cache 遇到操作系统分页

你还记得操作系统课上一个经典的场景吗：一块 4 GB 的物理内存，同时跑着浏览器、IDE、终端十几个进程，每个进程都以为自己独占全部地址空间。这个"幻觉"靠的是**虚拟内存分页**——物理内存切成 4 KB 的页框，进程看到的是连续的虚拟地址，背后由页表偷偷翻译成散落在物理内存各处的真实地址。

现在把镜头从 CPU 内存切到 GPU 显存。一块 H100 有 80 GB HBM，上面跑着推理服务，同时处理几十个并发请求，每个请求都在往 KV Cache 里追加数据。**同样的问题出现了**：连续预分配引发大量碎片，有效利用率只有 20-40%。

PagedAttention 的思路直截了当：**既然操作系统用分页解决了内存碎片，GPU 显存为什么不能？** 把 KV Cache 切成固定大小的 block，每个请求维护一张 block table，按需分配、动态映射——有效利用率从 20-40% 提升到接近 100%。

> **交互可视化**：[PagedAttention — 传统预分配 vs 分页管理](paged_attention_visual.html) — 6 步并排对比，键盘 ← → 翻页。

---

## 一、KV Cache 让推理成为可能，但也制造了一个新瓶颈

### 1.1 为什么需要 KV Cache

自回归生成中，每产生一个新 token，模型都需要与历史上所有 token 做注意力计算。如果每步都重新算一遍所有历史 token 的 K 和 V，单步注意力计算量是 $O(N^2)$——当前序列长度 $N$ 越大，每步的矩阵乘法越重，不可接受。

KV Cache 的做法很自然：**把已经算好的 K 和 V 缓存起来，新 token 只算自己的 Q、K、V，然后从缓存中读取历史的 K 和 V**。Decode 单步的计算量从 $O(N^2)$ 降到了 $O(N)$。代价是**每生成一个 token，缓存就大一圈**。

### 1.2 这个"大一圈"究竟有多大

以 Llama-2 70B（GQA, num_kv_heads=8, head_dim=128, 80 layers, FP16）为例，每个 token 每层的 KV Cache 是 4 KB。乘上 80 层，每个 token 的完整 KV Cache 约 320 KB。再乘上序列长度：

| 序列长度 | 单请求 KV Cache | batch=8 | batch=32 |
| :------: | :-------------: | :-----: | :------: |
|  2,048   |     0.6 GB      |  5 GB   |  20 GB   |
|  8,192   |     2.5 GB      |  20 GB  |  80 GB   |
|  32,768  |      10 GB      |  80 GB  |  320 GB  |

模型权重本身约 140 GB（FP16）。当 batch=8, seq_len=8192 时，KV Cache（20 GB）还是可控的附加开销。但 push 到 batch=32, seq_len=32768——**KV Cache 膨胀到 320 GB，是模型权重的 2.3 倍**。

这意味着推理服务的显存瓶颈不再是模型权重，而是那个动态增长的 KV Cache。**你不仅需要存它，还需要高效地管理它。**

---

## 二、预分配：最直觉的方案，最严重的浪费

### 2.1 一个看似合理的做法

KV Cache 需要存在 GPU 显存里。面对"一块 tensor 要不断往尾部追加数据"这个需求，最直接的方案是：在推理开始时，给每个请求预分配一块连续显存，大小为 `max_model_len × per_token_bytes`。

这很自然。GPU 上的 tensor 不就是连续内存吗？`cudaMalloc` 返回的不就是连续地址吗？一次分配比反复分配+拷贝简单得多。

### 2.2 但这个方案有两个制度性缺陷

**预留浪费（reservation waste）**[^1]：预分配的是**最大可能长度**，但实际生成的 token 数通常远小于上限。`max_model_len=32768`，而用户 prompt 只有 2,000 token——那 30,768 token 的空间被白白锁定。vLLM 论文将这种"为未来预留但从未使用"的空间定义为 reservation waste。即使系统知道精确输出长度，这种浪费也不可避免。

**外部碎片（external fragmentation）**：不同请求的完成时间不同。短请求先结束，释放的空间留下空洞——这些空洞可能不够容纳下一个请求，即使**总空闲空间是够的**。

这正是操作系统在 1960 年代就遇到过的问题：连续内存分配下，程序来来去去，内存里布满碎片。

### 2.3 操作系统在半个世纪前给出了答案

操作系统解决这个问题的方式今天已成为常识：把物理内存切成固定大小的**页**（通常是 4 KB），程序看到的是连续虚拟地址，背后由**页表**将每个虚拟页映射到任意物理页框。一个 100 MB 的程序，不需要找一块 100 MB 的连续物理内存——只要有 25,000 个空闲页框，散落在任何位置都行。

页表就是翻译器：CPU 发出虚拟地址，MMU 查页表，找到对应的物理页框。程序完全不知道它的数据在物理上是如何分布的。

### 2.4 GPU 遇到了完全一样的问题，但没人给它做 MMU

GPU 显存面临同样的碎片困境。不同的是，GPU 没有硬件 MMU——CUDA kernel 看到的是线性地址空间，`cudaMalloc` 返回连续地址。

但"没有硬件 MMU"并不意味着"不能做分页"。硬件做不了的事，**软件可以做**。这，就是 PagedAttention 的起点。

[^1]: vLLM 论文严格区分了三种浪费：**reservation waste**（为未来 token 预留的空间）、**internal fragmentation**（单个 block 内未填满的 slot，PagedAttention 下最多 1 block/请求）、**external fragmentation**（不同大小块导致的不可用空洞）。预分配方案的主要浪费来自 reservation waste，PagedAttention 消除的也是它。本文 §五.3 讨论的才是论文定义的 internal fragmentation。

---

## 三、PagedAttention：在 GPU 上手工实现分页

### 3.1 三步走——切块、映射、按需分配

既然 CPU 内存的分页机制已经解决了碎片问题，PagedAttention 的设计直接翻译这个方案：把硬件 MMU 换成软件实现的 block table。

**第一步：切块。** KV Cache 不再以"整个序列"为单位分配，而是切成固定大小的 **KV block**。每个 block 存储 `block_size` 个 token 在某层的 K 和 V。以 GQA 模型（8 KV heads, head_dim=128, block_size=16, FP16）为例：

```text
一个 block 的物理内容（单层）:
  K: (16, 8, 128) × 2 bytes = 32 KB
  V: (16, 8, 128) × 2 bytes = 32 KB
  合计: 64 KB / block / layer
```

block 是 KV Cache 管理的**最小分配单位**——就像操作系统的 4 KB 页框。

**第二步：映射。** 每个请求维护一张 **block table**，记录逻辑 block 到物理 block 的对应关系：

```text
请求 A 的 Block Table:
  logical_block[0] → physical_block[#42]
  logical_block[1] → physical_block[#17]
  logical_block[2] → physical_block[#89]
```

请求 A "以为"它的 KV Cache 是 `[block_0, block_1, block_2, ...]` 这样连续的逻辑序列，但实际上这三个 block 可以散落在 GPU 显存的任意位置。**逻辑连续，物理分散**——分页的本质被完整保留。

**第三步：按需分配。** 请求开始时只分配第一个 block。Token 一个个生成，block 一个个追满。追满一个后，从空闲池中取一个新 block 挂到 block table 末尾。不再需要预测最终长度。

```text
OS 虚拟内存                      PagedAttention
─────────────────────────────────────────────────
物理页框 (Page Frame)     →    KV Block
页表 (Page Table)         →    Block Table
页表项 (PTE)              →    block_table[i] = physical_block_id
MMU 地址翻译              →    Attention kernel 查表 gather
按需调页 (Demand Paging)  →    按需 block 分配
```

### 3.2 注意力计算怎么办——软件 MMU 的开销

有了 block table，每个请求的 K 和 V 就可以散落存储。但 attention 计算怎么办？

OS 有 MMU 在硬件层面透明翻译地址——CPU 执行 `mov eax, [ebx]`，MMU 自动将虚拟地址 `ebx` 翻译为物理地址。**GPU 没有这个。** Attention kernel 必须自己查 block table，逐个 block 做 gather：

```text
for each logical_block_id in range(num_blocks):
    physical_block = block_table[logical_block_id]    // 软件查表
    k_block = load_k_from(physical_block)              // gather
    v_block = load_v_from(physical_block)

    scores = Q @ k_block.T                             // 分块计算
    output = update_online_softmax(output, scores, v_block)
```

vLLM 为此实现了专门的 CUDA kernel——本质上是**用软件实现了一个 GPU 版的 MMU**，翻译的是 "logical_block_id → physical_block_addr"。

### 3.3 Block Size 的选择——操作系统也要选页大小

Block size 决定了碎片粒度和管理开销的平衡点：

|  block_size   |   碎片粒度   | Block Table 大小 |    Kernel 效率     | 适用场景         |
| :-----------: | :----------: | :--------------: | :----------------: | ---------------- |
|   小 (8~16)   | 细，利用率高 | 大，VRAM 开销多  |   launch 次数多    | 通用场景         |
| 大 (128~256)  | 粗，可能浪费 |    小，开销低    | 单次处理更多 token | 长序列、压缩模型 |
| **16 (默认)** |      —       |        —         |         —          | GQA/MHA 模型     |
| **256 (V4)**  |      —       |        —         |         —          | CSA/HCA 压缩模型 |

vLLM 默认 block_size=16，单层每个 block 仅 64 KB。DeepSeek-V4 推荐 256——CSA/HCA 压缩后单个 token 的 KV 极小（平均 ~169 B），小 block 会导致 block table 条目激增。

---

## 四、效果：不只是利用率数字的变化

### 4.1 纸面数据

分页方案的效果有多显著？vLLM 论文的对比给出了答案：

| 方案                        | 有效 KV Cache 利用率 | 浪费来源               | 同等显存下的并发上限 |
| :-------------------------- | :------------------: | :--------------------- | :------------------: |
| Orca (Max)                  |        20.4%         | reservation + external |          1×          |
| Orca (Pow2)                 |        32.0%         | reservation + external |        ~1.6×         |
| Orca (Oracle, 已知输出长度) |        38.2%         | 纯 reservation waste   |        ~1.9×         |
| **vLLM (PagedAttention)**   |      **~100%**       | ≤ 1 block/请求         |       **2-4×**       |

> 数据来源：vLLM 论文 Table 1。即使系统提前知道每个请求的精确输出长度（Orca Oracle），预留浪费仍占 61.8%——预分配的浪费是**结构性的**，不是调参问题。PagedAttention 通过按需分配消除了 reservation waste，仅剩每个请求最后一个 block 未填满的 internal fragmentation（≤ `block_size - 1` 个 token）。

### 4.2 换个角度看：同一块 GPU 能多跑多少请求

假设 GPU 可分配 40 GB 用于 KV Cache，服务 Llama-2 70B：

```text
传统预分配 (max_len=8192):
  每请求预分配 = 8192 × 4 KB × 80 = ~2.5 GB
  最多并发 = 40 / 2.5 ≈ 16 个请求
  但实际平均长度只有 3000 → 实际占用仅 37%
  40 GB 中有 25 GB 白白浪费

PagedAttention:
  16 个请求实际占用 = 16 × 3000 × 4 KB × 80 ≈ 15 GB
  剩余 25 GB 还能服务更多请求
  按动态增长持续分配，最多可支持 ~40 个并发

  吞吐提升: 2.5×
```

**同样的硬件，同样的模型，吞吐翻了 2.5 倍。** 没有 GPU 升级，没有量化压缩，没有模型修改——仅仅改变了"怎么存"。

### 4.3 Block 抽象之上自然生长的能力

分页带来的不只是碎片消除。**写时复制（Copy-on-Write）**、**共享内存**、**按需调页**——这些 OS 特性都是在分页抽象上自然生长的。PagedAttention 也一样。

**Prefix Caching**：相同 prompt 前缀的 block 可以在多请求间共享——多个 block table 指向相同物理 block。vLLM 的 Automatic Prefix Caching (APC) 不需要额外架构，只是 block manager 分配时多做一次 hash 比对。

**灵活的内存分层**：Block 粒度让 offloading 不需要搬动整个序列。按 block 级别做 CPU/NVMe 换入换出，比搬动 GB 级连续 tensor 灵活得多。

---

## 五、权衡：软件 MMU 不是免费的

### 5.1 翻译开销——没有 TLB 的分页

OS 有硬件 MMU 和 TLB，查页表的开销被压到几乎为零。PagedAttention 每次 attention 计算都要遍历 block table 逐个 block gather。这个"软件 MMU"的开销直接体现在 kernel 执行时间上。

vLLM 的缓解手段：kernel 内预加载 block table 到寄存器或 shared memory，批量 gather 减少内存事务，分块 softmax 融合计算。好在 memory-bound 的 Decode 场景下，这个开销可以被内存延迟部分掩盖。

### 5.2 Block Table 自身也占显存

每个请求的 block table 也是一块显存。block_size=16, seq_len=32768 时，单请求 block table 有 `32768/16=2048` 条目 × 8 bytes = 16 KB。100 并发约 1.6 MB。block_size 减到 8 则翻倍至 3.2 MB。虽整体可控，但它是"碎片粒度 vs 管理开销"天平上的一个砝码。

### 5.3 Internal Fragmentation 的最后一公里

即使 block 粒度，序列长度不是 `block_size` 的整数倍时，最后一个 block 会有部分 slot 未使用——每个请求最多浪费 `block_size - 1` 个 token。block_size=16 时平均浪费 8 个 token（~32 KB），占单请求 KV Cache 的比例随序列增长趋近于零。量级上可接受，但不是零。

---

## 六、抽象层之上的天空

PagedAttention 的 block 抽象已成为事实标准。其他 KV Cache 优化都在这个基础上叠加：

```text
        ┌─────────────────────┐
        │    Prefix Caching   │ ← block 级共享，APC
        ├─────────────────────┤
        │    KV Offloading    │ ← block 粒度的 CPU/NVMe 换入换出
        ├─────────────────────┤
        │    KV 量化 (FP8)     │ ← block 内部压缩，kernel 反量化
        ├─────────────────────┤
        │    PagedAttention   │ ← 基础抽象层：碎片消除 + 按需分配
        └─────────────────────┘
```

Block 是这一切的**通用组织单位**。没有它，Prefix Caching 的共享需要额外数据结构，Offloading 需要搬动连续大块内存，量化需要重新设计整个内存布局。

PagedAttention 的设计者洞见到了这一点：**好的抽象不是叠加功能，而是让功能自然生长。**

---

## 七、小结

PagedAttention 回答了一个简单但影响深远的问题：**KV Cache 到底该怎么存？**

从操作系统半个世纪前的分页方案中借来洞见，用软件在 GPU 上实现了一个轻量级 "MMU"，把有效利用率从 20-40% 提升到接近 100%。Block 抽象成为 Prefix Caching、Offloading、量化等上层优化的共同基础。

> 预分配连续内存 → 利用率仅 20-40%（1960 年代 OS 碎片问题的翻版）
> 引入分页 → 利用率接近 100%（OS 的答案，在 GPU 上用软件再做一遍）
> Block 抽象 → Prefix Caching、Offloading、量化自然叠加

工程上最优雅的解决方案，往往不是发明一个新东西，而是**把另一个领域已经验证了半个世纪的方案，带到一个新场景里**。

---

## 相关阅读

- [交互可视化：传统预分配 vs PagedAttention](paged_attention_visual.html) — 6 步并排对比，键盘 ← → 翻页
- [KV Cache 原理简介](kv_cache_原理简介.md) — 本文依赖的 KV Cache 基础概念与显存公式
- [不同注意力类型的 KV Cache 到底长什么样](attention_kv_cache_formats.md) — GQA/MQA/MLA/CSA-HCA 下 block 物理形状的变化
- [Prefix Caching 原理分析](../prefix_caching/prefix_caching.md) — PagedAttention 之上的 block 级复用
- [KV Cache Offloading 分析](../offloading/01_kv_offloading.md) — block 粒度的存储层次迁移
- [vLLM 论文: Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — 原始论文
- [vLLM v1 架构: KVCacheGroupSpec](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/utils.py) — 多注意力类型 block 池的源码实现
