# sglang KV Pool 管理

> 2026-07-24 | 基于 sglang v0.5.14 源码分析

sglang 的 KV cache 管理是一套 GPU 内存管理系统，负责在有限的 GPU HBM 中高效存储和复用所有请求的 K/V tensor。核心能力：

- **前缀共享**：通过 Radix Tree 实现跨请求的 KV cache 复用，共享前缀的请求引用同一组 GPU slot，避免重复计算。
- **引用计数保护**：通过 `lock_ref` 机制保证活跃请求依赖的 KV 不被逐出，prefill/decode 期间所有依赖数据必定在 GPU 中。
- **多级逐出**：HiCache 将 KV cache 从 GPU (L1) 逐出到 CPU DRAM (L2)，再逐出到 NVMe 或远程存储 (L3)，需要时通过 `load_back` 恢复。
- **Page 粒度管理**：`page_size` 个连续 token 组成一个 page，作为分配、hash、I/O 的原子单元，减少元数据开销。

实现分布在 `python/sglang/srt/mem_cache/` 目录下，核心文件包括 `radix_cache.py`（前缀树）、`memory_pool.py`（物理存储）、`hiradix_cache.py`（HiCache 集成）、`cache_controller.py`（多级 I/O 协调），以及 `allocator/` 子目录（slot 分配器）。

---

## 一、整体架构

三层数据结构协作完成上述功能：**KV Pool**（物理存储）、**Radix Tree**（逻辑索引，即 Prefix Cache）、**ReqToTokenPool**（请求视图）。

```text
                        Radix Tree                           ReqToTokenPool
                  (token序列 → slot索引)                      (req → slot序列)
                  ──────────────────────                   ─────────────────

                  [1,2,3]                              req[0]: [a, b, c, 0, 0]
                   ↓ ↓ ↓                                        ↑   ↑   ↑
                  value = [a, b, c] ──────────────→    req[1]: [a, b, c, d, e]
                     │                                           ↑   ↑   ↑   ↑
              ┌──────┴──────┐                                    │   │   │   │
              ▼             ▼                                    │   │   │   │
            [4,5]         [6,7]        ┌─────────────────────────┼───┼───┼───┼──┐
             ↓ ↓           ↓ ↓         │   KV Pool (GPU HBM)     │   │   │   │  │
            v=[d,e]       v=[f,g]      │                         ▼   ▼   ▼   ▼  │
              │             │          │ slot a: K/V of token [1] ← 共享前缀     │
              │             │          │ slot b: K/V of token [2]               │
              │             │          │ slot c: K/V of token [3]               │
              │             │          │ slot d: K/V of token [4] ← req[1] 独占  │
              │             │          │ slot e: K/V of token [5]               │
              ▼             ▼          │ slot f: K/V of token [6] ← 无活跃请求    │
                                       │ slot g: K/V of token [7]               │
                                       └────────────────────────────────────────┘
```

> 浏览器打开 [sglang-kv-pool-three-relation.html](assets/sglang-kv-pool-three-relation.html) 查看带颜色标注的版本。

**数据流是单向循环**：Radix Tree → ReqToTokenPool → KV Pool → Radix Tree。

1. `match_prefix` 时将 TreeNode.value 的 slot 索引复制到 ReqToTokenPool 的前缀部分
2. `forward` 时 attention kernel 按 ReqToTokenPool 索引读写 KV Pool
3. `cache_finished_req` 时将 ReqToTokenPool 的完整 slot 索引插入回 Radix Tree

请求流程概览：

```text
                        Request 到达
                          │
                          ▼
① match_prefix      radix tree 查找匹配前缀
                          │  TreeNode.value → device_indices (slot 索引)
                          ▼
② inc_lock_ref      沿 parent 链 lock_ref++，保护前缀 slot
                          │
                          ▼
③ alloc              尝试分配新 slot
                          │  失败? → evict (逐出 lock_ref==0 的节点) → 重试 alloc
                          ▼
④ 填充 ReqToTokenPool [前缀 slot | 新分配 slot] → req_to_token[req_idx, :]
                          │
                          ▼
⑤ forward            attention kernel 按 ReqToTokenPool 索引读写 KV Pool (prefill + decode)
                          │
                          ▼
⑥ cache_finished_req  插入 radix tree → dec_lock_ref (旧前缀 lock_ref--)
                          │  节点 lock_ref 归零 → 变为 evictable
                          ▼
                     ┌─── 异步 ───┐
                     │ backup 线程│
                     │ L1→L2→L3  │
                     └───────────┘
```

---

## 二、核心数据结构与关系

### 2.1 KV Pool — 物理存储（只有一份）

KV Pool 是 GPU HBM 中一块连续分配的 tensor。**没有 per-request 的物理隔离**——不同请求的 KV 可以交错存放在同一块内存中，通过 slot 索引区分。

```text
KV Pool 的逻辑视图（MHA，单层）:
┌──────┬──────┬──────┬──────┬──────┬──────┬──────┐
│slot 0│slot 1│slot 2│slot 3│slot 4│slot 5│ ...  │
│dummy │reqA  │reqA  │reqB  │reqA  │reqB  │      │
│write │tok0  │tok1  │tok0  │tok2  │tok1  │      │
└──────┴──────┴──────┴──────┴──────┴──────┴──────┘
```

shape: `[num_slots, head_num, head_dim] × layer_num`。每个 slot 存**一个** token 在所有 layer 和 head 上的 K/V。

slot 索引是 KV Pool 的"指针"——所有上层结构（Radix Tree、ReqToTokenPool）操作的都是这些索引，而非数据本身。

| Attention 类型 | 形状（per layer）                                    | Buffer 结构                           |
| -------------- | ---------------------------------------------------- | ------------------------------------- |
| MHA            | `[size+page_size, head_num, head_dim]`               | `k_buffer[layer]` + `v_buffer[layer]` |
| MLA            | `[size+page_size, 1, kv_lora_rank+qk_rope_head_dim]` | 单一 `kv_buffer[layer]`               |
| PageMajor      | `[page_num, page_size, head_num, head_dim]`          | page-major envelope                   |
| FP4            | uint8 存储，K/V 维度减半                             | `k_buffer` + `kv_scale_buffer`        |

### 2.2 Radix Tree (Prefix Cache) — 逻辑索引

定义在 `radix_cache.py:217`。按 **token 序列** 组织 slot 索引：

```text
Radix Tree 结构:
  root
   └── [1,2,3]          ← 前缀 [1,2,3] 的 KV 在 slots [a, b, c]
        ├── [4,5]        ← 扩展 [4,5] 后 KV 在 slots [d, e]
        │     └── [6]    ← 再扩展 [6] 后 KV 在 slot [f]
        └── [6,7]        ← 另一分支: 扩展 [6,7] 后 KV 在 slots [f, g]
             └── [8]     ← ... slot [h]
```

**关键**：每个节点存**自己** token span 的 slot 索引。`len(key) == len(value)`，`key[i]` → `value[i]` 一一对应。

| 节点       | key       | value     | 含义                                 |
| ---------- | --------- | --------- | ------------------------------------ |
| root child | `[1,2,3]` | `[a,b,c]` | token 1→a, 2→b, 3→c                  |
| child 1    | `[4,5]`   | `[d,e]`   | token 4→d, 5→e（路径 `[1,2,3,4,5]`） |
| child 2    | `[6,7]`   | `[f,g]`   | token 6→f, 7→g（路径 `[1,2,3,6,7]`） |

TreeNode 核心字段（`radix_cache.py:217`）：

```python
class TreeNode:
    key: RadixKey            # 此节点覆盖的 token 序列
    value: torch.Tensor      # KV pool slot 索引 (GPU, int64)
    host_value: torch.Tensor # KV pool slot 索引 (CPU host pool)
    lock_ref: int            # 引用计数 (>0 = 被活跃请求使用，不可逐出)
    host_ref_counter: int    # host 端引用计数
    parent: TreeNode         # 父节点
    children: defaultdict    # 子节点 (defaultdict(TreeNode))
    hash_value: List[str]    # 每个 page 的 hash 值 (内容寻址，L3 索引)
```

状态判断：

```python
@property
def evicted(self):
    return self.value is None       # GPU slot 已释放

@property
def backuped(self):
    return self.host_value is not None  # CPU 端有备份
```

关键特性：

- **共享**：请求 `[1,2,3,4,5]` 和 `[1,2,3,6,7]` 共享节点 `[1,2,3]` 的 slots `[a,b,c]`
- **分裂**：匹配只命中节点的一部分时，节点在边界分裂（只分裂索引，不复制数据）
- **引用计数**：`lock_ref` 沿 parent 链保护整条路径，防止活跃请求依赖的 slot 被逐出

Match/cache 流程：

```python
# radix_cache.py:355
match_prefix(params):
  1. key = params.key.page_aligned(page_size)   # 对齐到 page 边界
  2. _match_prefix_helper: 沿树收集匹配节点的 value
  3. value = torch.cat(values)                   # 拼接为连续 slot 索引
  4. return MatchResult(device_indices=value, last_device_node=matched_node)

# radix_cache.py:437
cache_finished_req(req):
  1. kv_indices = req_to_token_pool[req.req_pool_idx, :len(token_ids)]
  2. insert(key=radix_key, value=kv_indices.clone())
  3. 释放与树中已有节点重复的 slot (free duplicate)
```

MambaRadixCache（`mamba_radix_cache.py:425`）在 hybrid SSM 模型中额外管理：

```text
TreeNode 扩展字段:
  mamba_value / mamba_host_value / full_lock_ref / mamba_lock_ref / mamba_evicted
双 LRU 链表: full_lru_list (KV) + mamba_lru_list (SSM state)
不变量: full_lock_ref >= mamba_lock_ref
```

tombstone 机制：SSM state 被逐出但 KV 仍保留时，`mamba_value = None`，`value` 仍有效。后续请求需重算 SSM state，但 KV 可直接复用。

### 2.3 ReqToTokenPool — 请求视图

`memory_pool.py:240`。每个活跃请求占一行 `[max_running_reqs, max_context_len]` (int32)，存储 `token_pos → kv_slot_idx` 映射：

```text
req_to_token[req_idx, :] =
  前缀复用部分 (来自 TreeNode.value) + 新分配部分 (来自 Allocator.alloc)
```

attention kernel 按此行索引从 KV Pool 读写 KV，kernel 不关心 slot 是否共享——它只按索引寻址。

### 2.4 三者关系总结

|          | Radix Tree                        | ReqToTokenPool                   | KV Pool           |
| -------- | --------------------------------- | -------------------------------- | ----------------- |
| 维度     | 共享（按前缀）                    | 私有（按请求）                   | 物理存储          |
| 内容     | token 序列 → slot 索引            | token_pos → slot 索引            | slot → K/V tensor |
| 何时修改 | insert / evict / split            | forward 时读写                   | forward 时读写    |
| 引用关系 | 共享前缀时多个请求指向同一组 slot | 前缀部分 = TreeNode.value 的副本 | 只有一份          |

**三者存储的都是同一组 slot 索引**——只是组织维度不同。

---

## 三、lock_ref 与 KV 驻留保证

### 3.1 问题

> prefill 或 decode 时，请求依赖的 KV Cache 是否都要在 KV Pool (GPU) 中？

**是。这是正确性要求。** attention kernel 直接从 GPU HBM 的 KV Pool 读 K/V tensor，如果 slot 已被逐出（内容被覆盖或释放），kernel 会读到错误数据。

### 3.2 正确性保证

活跃请求依赖的 KV slot 分两部分：**共享前缀部分**受 `lock_ref > 0` 保护，evict 不会选中；**新分配部分**由 allocator 保证不在 free 列表中，请求独占。evict 只操作 `lock_ref == 0` 的叶子节点，因此永远不会碰到活跃请求正在使用的 slot。forward 时所有依赖的 KV 必定在 GPU 中。

### 3.3 逐出与恢复

`lock_ref` 归零后，节点进入 `evictable_leaves` 集合。逐出发生时：

```text
inc_lock_ref(node):  沿 parent 链向上，所有节点 lock_ref++
dec_lock_ref(node):  沿 parent 链向上，所有节点 lock_ref--

evict():
  找到 evictable_leaves 中 lock_ref == 0 的叶子节点
  → 按 LRU 选择候选
  → free(node.value) 归还 slot 到 allocator
  → 从树中移除 node
```

如果配置了 HiCache，逐出会触发异步备份，形成三层恢复路径：

```text
L1 (GPU):   活跃请求需要的 KV ← 必须在此（lock_ref 保证）
            lock_ref=0 的 KV 暂时保留，空间不够时逐出

L2 (CPU):   已被逐出但 host_value 还在的 KV ← 可快速恢复到 L1
            load_back: CPU→GPU (~500ns per token)

L3 (NVMe):  已被完全逐出的 KV ← 最慢恢复到 L1
            load_back: NVMe→CPU→GPU (~100μs+)
```

逐出链：GPU 不够 → `evict` → L2；CPU 不够 → `evict_host` → L3。

加载链：后续请求 `match_prefix` 命中已逐出节点 → `load_back` 将 KV 从 L2/L3 重新加载到 L1 → `inc_lock_ref` 保护 → forward。

**关键**：`load_back` 在 forward **之前**完成，全有或全无——不存在 "forward 时部分 KV 在 L2" 的情况。

---

## 四、内存池与分配器

KV Pool 有多种类型适配不同 attention 机制，分配器则管理 slot 的分配与回收。两者在初始化时根据模型特征选定。

### 4.1 KV Pool 类型

所有 Pool 类定义在 `python/sglang/srt/mem_cache/memory_pool.py`，形状见 [§2.1](#21-kv-pool--物理存储只有一份)：

```text
KVCache (抽象基类, line 1228)
  ├── MHATokenToKVPool (line 1331)         ← MHA: k_buffer + v_buffer per layer
  │     ├── NoOpMHATokenToKVPool (2027)    ← embedding-only 模型
  │     ├── MHATokenToKVPoolFP4 (2141)     ← FP4 量化
  │     └── PageMajorMHATokenToKVPool (2291) ← page-major envelope layout
  ├── MLATokenToKVPool (line 2712)         ← MLA: kv_buffer per layer (latent space)
  │     ├── MLATokenToKVPoolFP4 (2982)     ← FP4 量化
  │     └── DSATokenToKVPool (3122)        ← DSA: 额外 index_k_with_scale_buffer
  ├── HybridLinearKVPool (2445)            ← hybrid SSM: 内部持有 MHA/MLA pool + MambaPool
  ├── MHATokenToKOnlyPool (3451)           ← MiniMax: K-only
  └── MiniMaxSparseKVPool (3538)           ← MiniMax: MHA pool + index KV pool

ReqToTokenPool (line 240)                  ← [max_req, max_context_len] int32
  └── HybridReqToTokenPool (line 818)      ← 额外管理 MambaPool + mamba_allocator
```

| Pool 类型              | Attention       | 特殊字段                                                                  |
| ---------------------- | --------------- | ------------------------------------------------------------------------- |
| **MHATokenToKVPool**   | MHA             | NHD/HND/vectorized_5d 布局                                                |
| **MLATokenToKVPool**   | MLA             | 维度: `kv_lora_rank + qk_rope_head_dim`                                   |
| **DSATokenToKVPool**   | MLA + DSA       | page_size: NVIDIA 固定 64; AMD 需 16 的倍数 (需 preshuffle)，否则强制为 1 |
| **HybridLinearKVPool** | Attention + SSM | 代理到 MHA/MLA pool + MambaPool，双 LRU 驱逐                              |

MambaPool（`memory_pool.py:313`）：管理 conv state + temporal state，可独立于 attention KV 被逐出（tombstone 机制）。

### 4.2 分配器类型

分配器在 `python/sglang/srt/mem_cache/allocator/` 下（`MultiEndedAllocator` 在 `mem_cache/multi_ended_allocator.py`，`MambaSlotAllocator` 在 `allocator/mamba.py`）：

| 分配器                           | 文件                          | page_size | 场景                                 |
| -------------------------------- | ----------------------------- | --------- | ------------------------------------ |
| `TokenToKVPoolAllocator`         | `allocator/token.py:28`       | = 1       | token 级分配                         |
| `PagedTokenToKVPoolAllocator`    | `allocator/paged.py:105`      | > 1       | page 级分配，CUDA kernel 批量操作    |
| `SWATokenToKVPoolAllocator`      | `allocator/swa.py:20`         | 任意      | hybrid SWA，full + swa 两套子分配器  |
| `PureSWATokenToKVPoolAllocator`  | `allocator/swa.py:410`        | 任意      | 纯 SWA 模型                          |
| `MultiEndedAllocator`            | `multi_ended_allocator.py:99` | 任意      | UnifiedKVPool，virtual→physical 页表 |
| `HiSparseTokenToKVPoolAllocator` | `allocator/hisparse.py:15`    | 任意      | 稀疏索引                             |
| `MambaSlotAllocator`             | `allocator/mamba.py:30`       | N/A       | Mamba 固定大小 slot                  |

分配器选择（`model_runner_kv_cache_mixin.py:1162-1270`）：

```text
backward_compatible_mode → TokenToKVPoolAllocator
  非 backward_compatible:
    NPU (ascend / dsv4 / hybrid_gdn):
      hybrid_swa + dsv4 → DSV4NPUTokenToKVPoolAllocator
      hybrid_swa         → SWATokenToKVPoolAllocator
      非 hybrid           → NPUPagedTokenToKVPoolAllocator
    CUDA / ROCm:
      hybrid_swa + full==0 → PureSWATokenToKVPoolAllocator
      hybrid_swa + full>0  → SWATokenToKVPoolAllocator
      非 hybrid:
        hisparse             → HiSparseTokenToKVPoolAllocator
        page_size==1 + dcp==1 → TokenToKVPoolAllocator
        其他                  → PagedTokenToKVPoolAllocator
```

`PagedTokenToKVPoolAllocator` 核心操作：

```python
# alloc (paged.py:149): page 对齐分配
alloc(need_size)  # need_size % page_size == 0
  → out_pages = free_pages[:need_size // page_size]
  → out_indices = (out_pages[:, None] * page_size + arange(page_size)).reshape(-1)

# alloc_extend (paged.py:172): CUDA kernel 批量分配
alloc_extend(prefix_lens, seq_lens, ...)
  → alloc_extend_kernel 按 batch 计算各请求需要的 page 数量

# free (paged.py:261): page 级回收
free(free_index)
  → free_page_indices = torch.unique(free_index // self.page_size)
```

`MultiEndedAllocator`（`multi_ended_allocator.py:99`）：两个子池从同一块 byte buffer 两端向中间增长，virtual page table 做 virtual→physical 映射，不够时触发 compaction。

---

## 五、Page Size

### 5.1 什么是 page

`page_size` 将连续 token 分组为**原子管理单元**。一个 page 包含 `page_size` 个 token，分配、hash、I/O 都以 page 为最小粒度。page 内部的 token 在 KV Pool 中占据连续的 slot。

```text
page_size = 4 时:
  tokens: [1, 2, 3, 4 | 5, 6, 7, 8 | 9, 10, 11, 12 | ...]
  pages:  [  page 0   |  page 1   |   page 2     | ...]
  slots:  [a, b, c, d | e, f, g, h | i,  j,  k,  l | ...]
```

**page_size = 1 时，page 退化为单个 token**，所有 page 级机制退化为 token 级操作。

### 5.2 为什么需要 page

三个收益：

1. **内容寻址 (hash)**：对每 page 的 token 序列计算 hash，L3 存储用 hash 做 key 查找。没有 page 就没有稳定的内容寻址单元。
2. **I/O 粒度**：L3 读写以 page 为单位（每 page 一个文件或一条 ZMQ 消息），批量传输比逐 token 高效。
3. **Radix Tree 压缩**：`child_key(page_size)` 用每 page 的 hash 做子节点索引键，将 key 空间从 token 级压缩到 page 级。

代价：key 必须 page 对齐（末尾不足一个 page 的 token 被截断），引入少量内部碎片。

### 5.3 page_size 如何贯穿整个栈

| 层级           | 影响                                                                                                                                       |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| **Radix Tree** | `match_prefix` 前 key 截断为 `len(key) // page_size * page_size`；`child_key(page_size)` 用每 page hash 做子节点键                         |
| **Allocator**  | `page_size = 1` 用 `TokenToKVPoolAllocator`（逐 token），`page_size > 1` 用 `PagedTokenToKVPoolAllocator`（逐 page），分配数必须 page 对齐 |
| **KV Pool**    | 前 `page_size` 个 slot 预留为 padding（含 slot 0 的 dummy write）                                                                          |
| **Host Pool**  | 按 `page_num = size // page_size + 1` 分配                                                                                                 |
| **L3 Storage** | 每 page 存为一个文件（hash-indexed）或一条 ZMQ 消息（LMCache）                                                                             |
| **PageMajor**  | page 内采用 layer-major layout，page 数 = `(size + page_size) // page_size`                                                                |

分配路径分流（`common.py:479`）：

```python
if _alloc_page_size(batch) == 1:
    alloc_token_slots()                # 简单连续分配
else:
    alloc_paged_token_slots_extend()   # CUDA kernel 批量分配 + extend
```

### 5.4 典型配置

| 模型                      | page_size   | 原因                                                          |
| ------------------------- | ----------- | ------------------------------------------------------------- |
| Llama-3-8B (MHA)          | 1 或 256    | 256 适配 HiCache L3 page 粒度                                 |
| Qwen3.5-122B (hybrid SSM) | 256         | 须配 `--mamba-scheduler-strategy extra_buffer`，否则强制 1    |
| DeepSeek-V4 (MLA + DSA)   | 64 (NVIDIA) | DSA kernel 限制；AMD 需 16 的倍数 (需 preshuffle)，否则强制 1 |

---

## 六、HiCache 多层存储

### 6.1 三层架构

```text
L1 (GPU HBM):  原始 KV Pool，模型 forward 读写
  │             活跃请求依赖的 KV 必须在此（lock_ref 保证）
  │  evict + write_backup (lock_ref==0 的节点)
  ▼
L2 (CPU DRAM): Host Pool，GPU pool 的 CPU 镜像
  │             load_back: CPU→GPU 快速恢复
  │  evict_host + write_through / write_back
  ▼
L3 (Storage):  NVMe (file/hash_indexed) / Mooncake / LMCache MP Server
               load_back: NVMe→CPU→GPU 最慢恢复
```

两种写策略决定 L2→L3 的触发时机：

| 策略            | L2→L3 触发                         | 特点                                                      |
| --------------- | ---------------------------------- | --------------------------------------------------------- |
| `write_through` | L1→L2 时**同步**写入 L3            | 数据即时持久化，L3 始终有最新副本；写延迟叠加             |
| `write_back`    | L2 空间不足被逐出时**延迟**写入 L3 | L2 做 write buffer，减少 L3 写入次数；L2 满之前 L3 无副本 |

### 6.2 Host Pool (L2)

`pool_host/base.py:81` + 各 attention 类型子类。

内存布局（`pool_host/mha.py:130-166`）：

| 布局                | 形状                                                      | 适用场景                                                     |
| ------------------- | --------------------------------------------------------- | ------------------------------------------------------------ |
| `layer_first`       | `(2, layer_num, size, head_num, head_dim)`                | 逐层 transfer（默认），层内连续，适合大多数场景              |
| `page_first`        | `(2, size, layer_num, head_num, head_dim)`                | 单 page 内所有层连续，减少 page fault，适合小 batch 随机访问 |
| `page_first_direct` | `(2, page_num, layer_num, page_size, head_num, head_dim)` | 直接 page 级访问，无需计算偏移                               |
| `page_head`         | `(2, page_num, head_num, page_size, layer_num, head_dim)` | head 级粒度，优化极 small-batch 场景                         |

Transfer 路径（`load_to_device_per_layer`, line 209）：Ascend NPU → sgl_kernel C++ → JIT Triton → direct copy。

### 6.3 L3 Storage 后端

L3 是持久化存储层。L2 中的数据通过 `write_through` 或 `write_back` 写入 L3，以 page 为单位组织——每 page 的 token 序列计算出 hash 值（SHA256，`TreeNode.hash_value`），作为内容寻址的 key 存储在 L3 后端中。需要恢复时，通过 hash 查找对应的 page 数据，加载到 L2 再 transfer 到 L1。

| 后端                     | 存储位置                  | 共享 | 持久化 | 说明                                     |
| ------------------------ | ------------------------- | ---- | ------ | ---------------------------------------- |
| `HiCacheFile` (built-in) | 本地 NVMe 文件            | ❌   | ❌     | 每次重启清空，仅 S3 基准测试用           |
| `HashIndexedFileBackend` | 本地 NVMe 文件            | ❌   | ✅     | hash 命名文件，重启后仍可命中            |
| `MooncakeStore`          | 远程 RDMA 集群            | ✅   | ✅     | 大规模多节点共享，RDMA 传输              |
| 自定义后端               | 任意（如 LMCache Server） | ✅   | ✅     | 实现 `batch_exists/get/set` 接口即可接入 |

### 6.4 HiCacheController

`cache_controller.py:203`。两个 CUDA stream 独立运行，与 compute stream 并行，forward 不被 backup/prefetch 阻塞：

```text
write_stream (GPU→CPU)                load_stream (CPU→GPU)
────────────────────                  ────────────────────
backup 线程:                           prefetch 线程:
  device pool → host pool               host pool → device pool
     ↓ (write_through 时)                ↑ (L3 miss 触发)
  host pool → L3 storage                L3 storage → host pool
```

读写路径：

```text
读 (prefetch: L3 → L2 → L1):
  batch_exists(hashes) → batch_get(hashes, host_indices) → load_to_device_per_layer
  → TreeNode.value 恢复，其他请求 match_prefix 可命中

写 (backup: L1 → L2 → L3):
  backup_from_device_all_layer → batch_set(hashes, host_indices)
  → TreeNode.host_value = host_indices, TreeNode.value = None
```

---

## 七、总结

sglang 的 KV Pool 管理围绕一个核心问题展开：**如何在有限的 GPU HBM 中高效存储和复用所有请求的 KV cache。**

三层数据结构（KV Pool / Radix Tree / ReqToTokenPool）通过 slot 索引串联成一个单向数据流循环：Radix Tree → ReqToTokenPool → KV Pool → Radix Tree。slot 索引是贯穿全局的 "指针"，三者的不同组织维度（共享 / 私有 / 物理）使得前缀缓存共享成为可能。

正确性由 `lock_ref` 保证：活跃请求依赖的所有 KV slot 要么被引用计数保护，要么来自独占分配。evict 只看 `lock_ref == 0` 的节点，永远不会触碰正在使用的数据。

HiCache 在此基础上扩展了多级存储：L2 (CPU DRAM) 作为快速恢复层，L3 (NVMe/远程) 作为持久化层。`page_size` 将 token 分组为原子管理单元，使内容寻址、page 级 I/O 和 radix tree 压缩成为可能。`write_through` / `write_back` 两种策略提供了即时持久化与延迟批量写入之间的选择。
