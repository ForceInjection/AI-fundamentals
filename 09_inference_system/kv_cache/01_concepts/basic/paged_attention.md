# PagedAttention：当 KV Cache 遇到操作系统分页

你还记得操作系统课上一个经典的场景吗：一块 4 GB 的物理内存，同时跑着浏览器、IDE、终端十几个进程，每个进程都以为自己独占全部地址空间。这个"幻觉"靠的是**虚拟内存分页**——物理内存切成 4 KB 的页框，进程看到的是连续的虚拟地址，背后由页表偷偷翻译成散落在物理内存各处的真实地址。

现在把镜头从 CPU 内存切到 GPU 显存。一块 H100 有 80 GB HBM，上面跑着推理服务，同时处理几十个并发请求，每个请求都在往 KV Cache 里追加数据。**同样的问题出现了**：连续预分配引发大量碎片，显存利用率只有 20-40%。

PagedAttention 的思路直截了当：**既然操作系统用分页解决了内存碎片，GPU 显存为什么不能？** 把 KV Cache 切成固定大小的 block，每个请求维护一张 block table，按需分配、动态映射——碎片率从 60-80% 降到 4% 以下。

本文从碎片问题的根因出发，用操作系统的视角重新理解 PagedAttention 的设计逻辑。

---

## 一、背景：KV Cache 增长得比你想象的快

### 1.1 KV Cache 从哪来

自回归生成中，每产生一个新 token，模型都需要与历史上所有 token 做注意力计算。如果每步都重新算一遍所有历史 token 的 K 和 V，单步注意力计算量是 $O(N^2)$——当前序列长度 $N$ 越大，每步的矩阵乘法越重。这很快就变得不可接受。

KV Cache 的做法很自然：**把已经算好的 K 和 V 缓存起来，新 token 只算自己的 Q、K、V，然后从缓存中读取历史的 K 和 V**。这个设计让 Decode 单步的计算量从 $O(N^2)$ 降到了 $O(N)$，但也意味着**每生成一个 token，缓存就大一圈**。

### 1.2 缓存究竟有多大

以 Llama-2 70B（GQA, num_kv_heads=8, head_dim=128, 80 layers, FP16）为例，每个 token 每层的 KV Cache 是 4 KB。乘上 80 层，每个 token 的完整 KV Cache 约 320 KB。再乘上序列长度：

| 序列长度 | 单请求 KV Cache | batch=8 | batch=32 |
| :------: | :-------------: | :-----: | :------: |
|  2,048   |     0.6 GB      |  5 GB   |  20 GB   |
|  8,192   |     2.5 GB      |  20 GB  |  80 GB   |
|  32,768  |      10 GB      |  80 GB  |  320 GB  |

一个 70B 模型本身权重约 140 GB（FP16）。当 batch=8, seq_len=8192 时，KV Cache（20 GB）还是可控的附加开销。但 push 到 batch=32, seq_len=32768——**KV Cache 膨胀到 320 GB，是模型权重的 2.3 倍**。

这意味着显存管理的重点不再是模型权重，而是那个动态增长的 KV Cache。

---

## 二、问题：预分配——最简单的方案，最严重的浪费

### 2.1 自然而然的做法

KV Cache 需要存在 GPU 显存里。最直接的方案：给每个请求在推理开始时预分配一块连续显存，大小为 `max_model_len × per_token_bytes`。

这很自然。GPU 上的 tensor 不就是连续内存吗？CUDA 的 `cudaMalloc` 返回的不就是连续地址吗？而且在已知最大长度的情况下，一次分配比反复分配+拷贝简单得多。

### 2.2 但这个方案有两个致命缺陷

**内部碎片**：每个请求预分配的是**最大可能长度**，但实际生成的 token 数通常远小于这个上限。`max_model_len=32768`，而用户的 prompt 只有 2,000 token——那 30,768 token 的空间被白白锁定，直到请求完成才能释放。

**外部碎片**：不同请求的完成时间不同。当一个短请求先结束，释放了它占用的空间，留下的空洞可能不够容纳下一个请求——即使**总空闲空间是够的**。

这正是操作系统在 1960 年代就遇到过的问题。那个年代，计算机还是用连续内存分配的：一个程序装入内存，占一块连续区域。程序结束，释放空间。几个程序来来去去之后，内存里布满碎片——总空闲空间够，但就是找不到一块足够大的连续区域来装入下一个程序。

### 2.3 OS 的答案：分页

操作系统的解决方案今天已成为常识：把物理内存切成固定大小的**页**（通常是 4 KB），把程序的地址空间切成同样大小的**虚页**。程序看到的是连续虚拟地址，背后由**页表**将每个虚页映射到任意的物理页框。一个程序 100 MB，不需要找一块 100 MB 的连续物理内存——只要 25,000 个空闲页框，散落在任何位置都行。

页表就是翻译器：CPU 发出虚拟地址，MMU（内存管理单元）查页表，找到对应的物理页框，完成访问。程序完全不知道、也不需要关心它的数据在物理上是怎么分布的。

### 2.4 GPU 显存面临完全一样的问题，但没人给它做 MMU

GPU 显存同样需要管理动态分配的内存，同样面临碎片问题。不同的是，GPU 没有硬件 MMU。CUDA kernel 看到的是线性地址空间，`cudaMalloc` 返回连续地址。

但"没有硬件 MMU"并不意味着"不能做分页"。硬件做不了的事，**软件可以做**——这正是 PagedAttention 的核心洞察。

---

## 三、方案：PagedAttention 的手工分页

### 3.1 三步走：切块、映射、按需分配

PagedAttention 的设计直接对应操作系统的分页机制，只是把硬件 MMU 换成了软件实现的 block table。

**第一步：切块。** KV Cache 不再以"整个序列"为单位分配，而是切成固定大小的 **block**。一个 block 存储 `block_size` 个 token 在某层的 K 和 V。以 GQA 模型（8 KV heads, head_dim=128, block_size=16, FP16）为例：

```text
一个 block 的物理内容（单层）:
  K: (16, 8, 128) × 2 bytes = 32 KB
  V: (16, 8, 128) × 2 bytes = 32 KB
  合计: 64 KB / block / layer
```

这就像操作系统的 4 KB 页框——block 是 KV Cache 管理的**最小分配单位**。

**第二步：映射。** 每个请求维护一张 **block table**，记录"我逻辑上第 i 个 block 实际在哪个物理 block 里"。这和页表的逻辑一模一样：

```text
请求 A 的 Block Table（类比页表）:
  logical_block[0] → physical_block[#42]
  logical_block[1] → physical_block[#17]
  logical_block[2] → physical_block[#89]
```

请求 A "以为"它的 KV Cache 是 `[block_0, block_1, block_2, ...]` 这样连续的逻辑序列，但实际上这三个 block 可能散落在 GPU 显存的任意位置。**逻辑连续，物理分散**——这正是分页的本质。

**第三步：按需分配。** 请求开始时，只分配第一个 block。Token 一个个生成，block 一个个追满。追满一个 block 后，从空闲池中取一个新的物理 block，挂到 block table 末尾。不再需要预测这个请求最终会生成多少 token。

```text
OS 虚拟内存                      PagedAttention
─────────────────────────────────────────────────
物理页框 (Page Frame)     →    KV Block
页表 (Page Table)         →    Block Table
页表项 (PTE)              →    block_table[i] = physical_block_id
MMU 地址翻译              →    Attention kernel 查表 gather
按需调页 (Demand Paging)  →    按需 block 分配
```

### 3.2 注意力计算怎么办

这是 PagedAttention 最关键的工程问题。操作系统有 MMU 在硬件层面做地址翻译，CPU 指令 `mov eax, [ebx]` 中的 `ebx` 是虚拟地址，MMU 自动查页表得到物理地址——对程序完全透明。

**GPU 没有 MMU。** Attention kernel 必须自己处理这个翻译。

传统的 attention 计算假设 K 和 V 在连续内存中，一个 `torch.matmul(Q, K.T)` 就完成。在 PagedAttention 下，K 和 V 散落在不同的物理 block 中，kernel 需要：

```text
for each logical_block_id in range(num_blocks):
    physical_block = block_table[logical_block_id]    // 软件查表
    k_block = load_k_from(physical_block)              // gather
    v_block = load_v_from(physical_block)

    scores = Q @ k_block.T                             // 分块计算
    output = update_online_softmax(output, scores, v_block)
```

逐个 block gather K/V，分块做 attention，用 online softmax 累积结果。vLLM 为此实现了专门的 CUDA kernel——这本质上就是**用软件实现了一个 GPU 版的 MMU**，只不过它翻译的不是 CPU 虚拟地址，而是 "logical_block_id → physical_block_addr"。

### 3.3 Block Size 的权衡

Block size 是 PagedAttention 暴露给上层的关键参数。它决定了碎片粒度和管理开销的平衡点——就像操作系统选页大小一样。

|  block_size   |   碎片粒度   | Block Table 大小 |    Kernel 效率     | 适用场景         |
| :-----------: | :----------: | :--------------: | :----------------: | ---------------- |
|   小 (8~16)   | 细，利用率高 | 大，VRAM 开销多  |   launch 次数多    | 通用场景         |
| 大 (128~256)  | 粗，可能浪费 |    小，开销低    | 单次处理更多 token | 长序列、压缩模型 |
| **16 (默认)** |      —       |        —         |         —          | GQA/MHA 模型     |
| **256 (V4)**  |      —       |        —         |         —          | CSA/HCA 压缩模型 |

vLLM 默认 block_size=16。对于标准的 Llama-2 70B，这意味单层每个 block 只有 64 KB——碎片粒度极细，即使序列长度不是 16 的整数倍，最多浪费 15 个 token 位置。

DeepSeek-V4 推荐 block_size=256，原因是 CSA/HCA 压缩后单个 token 的 KV 极小（平均 ~169 B），16 的 block 太小会导致 block table 条目激增和 kernel launch 开销过大。

---

## 四、效果：从 20% 到 90%，不只是数字

### 4.1 碎片率对比

vLLM 论文的实测数据给出了一个戏剧性的对比：

| 方案                                  | 内存利用率 | 碎片率  | 同等显存下的并发上限 |
| :------------------------------------ | :--------- | :------ | :------------------: |
| 传统预分配（FasterTransformer, Orca） | 20-40%     | 60-80%  |          1×          |
| **PagedAttention (block_size=16)**    | **>90%**   | **<4%** |       **2-4×**       |

60-80% 的碎片意味着：一块 80 GB 的 H100，在传统方案下实际只有 16-32 GB 真正用于存储有用数据，其余 48-64 GB 被空洞和预留空间占据。PagedAttention 把浪费压到 4% 以下——80 GB 中有超过 72 GB 是实实在在的 KV 数据。

### 4.2 一个具体的算例

假设 GPU 可分配 40 GB 用于 KV Cache，服务 Llama-2 70B（每 token 每层 4 KB, 80 layers, block_size=16）。

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

**同样的硬件，同样的模型，吞吐翻了 2.5 倍。** 这背后没有任何 GPU 升级，没有量化压缩，没有模型修改——仅仅是改变了"怎么存"这个看似简单的问题。

### 4.3 Block 抽象催生的附加能力

OS 分页带来的不只是碎片消除。**写时复制（Copy-on-Write）**、**共享内存**、**按需调页**——这些都是在分页抽象之上自然生长的功能。PagedAttention 也一样：

**Prefix Caching**：因为 KV Cache 按 block 组织，相同 prompt 前缀的 block 可以在多个请求间共享——多个请求的 block table 指向相同的物理 block。vLLM 的 Automatic Prefix Caching (APC) 不需要额外架构，只是 block manager 在分配时多做一次 hash 比对。

**灵活的内存分层**：Block 粒度让 offloading 不需要搬动整个序列的 KV Cache。哪些 block 冷、哪些 block 热——按 block 级别做 CPU/NVMe 换入换出，比搬动 GB 级的连续 tensor 灵活得多。

---

## 五、权衡：没有免费的午餐

### 5.1 翻译开销

操作系统有硬件 MMU 和 TLB（Translation Lookaside Buffer）来加速地址翻译，把查页表的开销压到几乎为零。PagedAttention 没有这个——每次 attention 计算，kernel 都要遍历 block table，逐个 block 做 gather。这个"软件 MMU"的开销直接体现在 GPU kernel 的执行时间上。

vLLM 的优化方向：kernel 内部预加载 block table 到寄存器或 shared memory，批量 gather K/V 以减少内存事务的次数，分块 softmax 时融合计算避免往返。虽然比不上硬件 MMU，但在 memory-bound 的 Decode 场景下，这个开销可以被内存延迟掩盖。

### 5.2 Block Table 自身也占显存

每个请求的 block table 本身也是一块显存。block_size 越小，block table 条目越多，这部分开销越大。以 block_size=16, seq_len=32768 为例，一个请求的 block table 有 `32768/16=2048` 个条目，每个条目 8 bytes（存储 physical block ID），共计 16 KB。100 个并发请求时约 1.6 MB——量级不大。但如果把 block_size 减到 8，条目数翻倍到 4096，单请求 32 KB；100 并发时约 3.2 MB。虽然整体可控，但它是 block_size 选择中"碎片粒度 vs 管理开销"天平上的一个砝码。

### 5.3 碎片不会完全消失

Block 粒度下仍然存在碎片——序列长度如果不是 block_size 的整数倍，最后一个 block 会有部分空间未使用。block_size=16 时，平均浪费 8 个 token 位置（~32 KB），这个量级在工程上可接受，但不是零。

---

## 六、PagedAttention 之上的优化栈

PagedAttention 的 block 抽象已经成为事实标准。其他 KV Cache 优化策略都在这个基础上叠加：

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

Block 是这一切的**通用组织单位**。没有 block 抽象，Prefix Caching 的共享需要额外数据结构，Offloading 需要搬动连续大块内存，量化需要重新设计整个内存布局。PagedAttention 的设计者洞见到了这一点——**好的抽象不是叠加功能，而是让功能自然生长**。

---

## 七、小结

PagedAttention 回答了一个看似简单但影响深远的问题：**KV Cache 到底该怎么存？**

从操作系统的分页历史中借来的洞见，用软件手段在 GPU 上实现了一个轻量级 "MMU"，把碎片率从 60-80% 压到 4% 以下。更重要的是，block 抽象成为上层优化（Prefix Caching、Offloading、量化）的共同基础。

回顾这条技术决策链：

> 预分配连续内存 → 碎片率 60-80%（这和 1960 年代 OS 遇到的问题是同一个问题）
> 引入分页 → 碎片率 <4%（60 年前 OS 给出的答案，现在 GPU 上用软件再做一遍）
> Block 抽象 → Prefix Caching、Offloading、量化自然叠加（好的抽象自己会生长）

工程上最优雅的解决方案，往往不是发明一个新东西，而是**把另一个领域已经验证了半个世纪的方案，带到一个新场景里**。

---

## 相关阅读

- [KV Cache 原理简介](kv_cache_原理简介.md) — 本文依赖的 KV Cache 基础概念与显存公式
- [不同注意力类型的 KV Cache 到底长什么样](attention_kv_cache_formats.md) — GQA/MQA/MLA/CSA-HCA 下 block 物理形状的变化
- [Prefix Caching 原理分析](../prefix_caching/prefix_caching.md) — PagedAttention 之上的 block 级复用
- [KV Cache Offloading 分析](../advanced/kv_offloading_analysis.md) — block 粒度的存储层次迁移
- [vLLM 论文: Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — 原始论文
- [vLLM v1 架构: KVCacheGroupSpec](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/utils.py) — 多注意力类型 block 池的源码实现
