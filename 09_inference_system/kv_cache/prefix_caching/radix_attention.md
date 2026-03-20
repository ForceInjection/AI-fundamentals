# RadixAttention 技术详解：从原理到 SGLang 实践及 vLLM APC 对比

本文深入剖析大语言模型推理优化中的 RadixAttention 技术，详述其基于 Radix Tree（基数树）自动复用 KV Cache 的核心原理，结合 SGLang 的落地实现解析其内存调度机制，并与 vLLM 的 APC 方案进行全方位对比，为您在生产环境中提供性能评估与选型参考。

## 1. 引言与背景

本章将介绍 LLM 推理中的重复计算问题以及 RadixAttention 提出的解决方案。

### 1.1 LLM 推理中的重复计算问题

大语言模型（LLM）的推理过程分为两个阶段：**Prefill（预填充）** 和 **Decode（解码）**。在 Prefill 阶段，模型需要处理完整的输入序列并生成对应的 KV Cache。根据 [KV Cache 原理简介](../basic/kv_cache_原理简介.md) 中的分析，由于 Transformer 的自注意力机制，Prefill 阶段的计算复杂度为 $O(N^2)$ ，其中 $N$ 为输入序列长度。

在实际的 LLM 应用中，大量请求存在**共享前缀**的现象：

- **System Prompt**：同一应用的所有请求使用相同的系统指令（通常 2K-8K tokens）。
- **多轮对话**：历史对话上下文在后续轮次中反复出现。
- **Few-shot Examples**：相同的示例在多个请求中复用。
- **RAG 场景**：检索到的文档片段被多个查询复用。

根据生产环境的统计数据 [1,3]，**40%-80% 的 token 序列存在可复用的共享前缀**。如果每次请求都从零开始计算这些重复前缀的 KV Cache，将造成巨大的计算资源浪费。

```text
场景示例：客服机器人

请求 A: [System Prompt: 8K tokens] + [用户问题 A: 256 tokens]
请求 B: [System Prompt: 8K tokens] + [用户问题 B: 128 tokens]
请求 C: [System Prompt: 8K tokens] + [用户问题 C: 512 tokens]

→ System Prompt 被重复计算 3 次，浪费约 95% 的 Prefill 计算量
```

### 1.2 为什么需要 KV Cache 复用

KV Cache 具有一个关键特性：**确定性**。对于相同的输入 token 序列，模型总是产生相同的 K（Key）和 V（Value）表征。这意味着：

> **相同的 token 序列 → 相同的 KV Cache → 只需计算一次，后续直接复用**

这一特性使得 KV Cache 复用成为可能。通过缓存已计算的 KV Cache 并在后续请求中复用，我们可以：

1. **降低 TTFT（Time-To-First-Token）**：跳过前缀的 Prefill 计算。
2. **提升吞吐量**：节省的 GPU 算力可服务更多请求。
3. **降低成本**：减少单位请求的计算资源消耗。

然而，这带来了一个核心的技术挑战：**如何高效地管理和检索 KV Cache？**

- 缓存的键（Key）是什么？
- 如何快速判断新请求是否命中缓存？
- 如何处理部分前缀匹配的情况？
- 如何在有限的显存中进行淘汰和替换？

### 1.3 现有方案的局限性

在 RadixAttention 提出之前，常见的 KV Cache 管理方案存在明显局限：

**方案 1：简单哈希表**:

将完整的 token 序列哈希后作为键，KV Cache 作为值存入哈希表。

```python
# 初始化缓存字典
cache = {}
# 将完整的 token 序列转换为元组后计算哈希值作为键
key = hash(tuple(token_ids))
# 将生成的 KV Cache 存入哈希表
cache[key] = kv_cache
```

**问题**：无法支持前缀匹配。即使两个请求共享 99% 的前缀，只要有一个 token 不同，就无法复用任何缓存。

**方案 2：静态前缀缓存**:

预先定义固定的 System Prompt，为其计算并缓存 KV Cache。

**问题**：缺乏灵活性，无法适应动态工作负载。多轮对话、用户特定上下文等场景无法受益。

**方案 3：逐 Token 查找**:

将每个 token 的 KV Cache 单独存储，按位置索引。

**问题**：管理开销巨大，无法利用前缀的结构化特性进行高效查找。

这些局限性促使 SGLang 团队提出了 **RadixAttention**——一种基于 Radix Tree（基数树）的 KV Cache 管理方案，能够自动、高效地识别和复用任意长度的共享前缀。

---

## 2. RadixAttention 核心原理

RadixAttention 的核心创新在于使用 **Radix Tree（基数树）** 这一数据结构来组织和管理 token 序列与 KV Cache 的映射关系。本章将深入介绍 Radix Tree 的原理及其在 KV Cache 管理中的应用。

### 2.1 Radix Tree 数据结构

本节将详细介绍 Radix Tree 数据结构的基本原理、核心特性以及时间复杂度分析。

#### 2.1.1 从 Trie 到 Radix Tree

**Trie（前缀树）** 是一种经典的树形数据结构，用于高效存储和检索字符串集合。在 Trie 中，每个节点代表一个字符，从根到叶子的路径表示一个完整的字符串。

```text
Trie 存储 ["hello", "help", "world"]：

        root
       /    \
      h      w
      |      |
      e      o
      |      |
      l      r
     / \     |
    l   p    l
    |        |
    o        d
```

Trie 的问题在于：当存在长的非分支路径时，会产生大量只有单个子节点的中间节点，浪费内存且增加遍历深度。

**Radix Tree（基数树）**，也称为 **压缩前缀树（Compressed Trie）** 或 **Patricia Tree**，通过**路径压缩**解决了这一问题。它将连续的、没有分支的节点合并为一个节点，存储整个字符串片段而非单个字符。

```text
Radix Tree 存储 ["hello", "help", "world"]：

        root
       /    \
    "hel"  "world"
     / \
   "lo" "p"

节点数：5（Trie 需要 11 个节点）
```

#### 2.1.2 Radix Tree 的核心特性

与传统 Trie 树相比，Radix Tree 具有以下几个核心特性，使其非常适合用于管理 KV Cache：

| 特性             | 说明                                   |
| ---------------- | -------------------------------------- |
| **路径压缩**     | 连续非分支节点合并，减少内存和遍历深度 |
| **前缀共享**     | 相同前缀自动合并到同一路径             |
| **动态更新**     | 支持高效的插入、删除和查找             |
| **最长前缀匹配** | 天然支持找到与查询序列匹配的最长前缀   |

#### 2.1.3 时间复杂度分析

设 $m$ 为查询/插入序列的长度， $k$ 为字符集大小（对于 token 序列， $k$ 为词表大小）：

| 操作         | 时间复杂度 | 说明                                   |
| ------------ | ---------- | -------------------------------------- |
| 查找         | $O(m)$     | 与序列长度线性相关，与树中总节点数无关 |
| 插入         | $O(m)$     | 最坏情况需要分裂现有节点               |
| 删除         | $O(m)$     | 可能触发节点合并                       |
| 最长前缀匹配 | $O(m)$     | 遍历直到无法继续匹配                   |

### 2.2 Token 序列到 KV Cache 的映射

在 RadixAttention 中，Radix Tree 的每个节点不仅存储 token 序列片段，还关联了对应的 **KV Cache Block**。

#### 2.2.1 节点结构设计

在 RadixAttention 的实现中，每个 Radix Tree 节点都需要保存 token 信息以及关联的缓存块索引。以下是节点的典型结构设计：

```python
class RadixNode:
    # 存储的 token 序列片段
    tokens: List[int]

    # 对应的 KV Cache Block 列表（每个 Block 固定大小，如 16 tokens）
    kv_cache_blocks: List[KVCacheBlock]

    # 子节点映射：第一个 token -> 子节点
    children: Dict[int, RadixNode]

    # 引用计数：当前有多少活跃请求使用此节点
    ref_count: int

    # 最后访问时间：用于 LRU 淘汰
    last_access_time: float
```

#### 2.2.2 树的构建过程

当新的 token 序列需要被缓存时，RadixAttention 执行以下步骤：

1. **从根节点开始匹配**：逐 token 比较，沿着匹配的路径向下遍历
2. **处理分歧点**：
   - 如果在某个节点内部出现不匹配，**分裂该节点**
   - 将公共部分保留，不同部分创建新的子节点
3. **创建新路径**：对于完全不匹配的后缀，创建新的节点链

```text
初始状态：树中已有序列 "hello world"

插入序列 "hello vllm"：

步骤 1: 匹配 "hello " (6 tokens) ✓
步骤 2: 在 "world" 节点处发现不匹配（w vs v）
步骤 3: 分裂节点，创建分支

结果：
        root
          |
      "hello "
        /    \
   "world"  "vllm"
```

#### 2.2.3 前缀共享机制

Radix Tree 的结构天然支持前缀共享：**所有具有相同前缀的序列都会经过相同的节点路径**。这意味着：

- 相同前缀的 KV Cache 只存储一份。
- 新序列插入时自动识别并复用已有前缀。
- 无需显式的“注册”或“声明”共享关系。

```text
请求 A: [System Prompt] + [Query A] → 缓存 System Prompt
请求 B: [System Prompt] + [Query B] → 自动复用 System Prompt 的 KV Cache
请求 C: [System Prompt] + [Query A] + [Response A] + [Query C]
        → 复用 "System Prompt + Query A" 的全部 KV Cache
```

### 2.3 前缀匹配与查找算法

前缀匹配是 RadixAttention 的核心操作，决定了能够复用多少已缓存的 KV Cache。本节介绍最长前缀匹配算法及其查找结果的处理方式。

#### 2.3.1 最长前缀匹配算法

当新请求到达时，RadixAttention 需要找到与请求 token 序列匹配的**最长前缀**，以最大化 KV Cache 复用。

```python
def match_prefix(root: RadixNode, query_tokens: List[int]) -> Tuple[int, List[KVCacheBlock]]:
    """
    查找与 query_tokens 匹配的最长前缀

    返回：(匹配长度, 对应的 KV Cache Block 列表)
    """
    current = root
    matched_length = 0
    matched_blocks = []
    query_pos = 0

    while query_pos < len(query_tokens):
        # 查找匹配的子节点
        first_token = query_tokens[query_pos]
        if first_token not in current.children:
            break  # 无匹配子节点，停止

        child = current.children[first_token]

        # 在子节点内逐 token 匹配
        node_pos = 0
        while node_pos < len(child.tokens) and query_pos < len(query_tokens):
            if child.tokens[node_pos] != query_tokens[query_pos]:
                break  # 节点内部不匹配，停止
            node_pos += 1
            query_pos += 1

        # 收集匹配的 KV Cache Blocks
        matched_length = query_pos
        matched_blocks.extend(child.kv_cache_blocks[:node_pos // BLOCK_SIZE])

        if node_pos < len(child.tokens):
            break  # 节点未完全匹配，停止

        current = child  # 继续向下遍历

    return matched_length, matched_blocks
```

#### 2.3.2 查找结果的处理

前缀匹配的结果决定了推理引擎的后续行为：

| 匹配情况 | 处理方式                                            |
| -------- | --------------------------------------------------- |
| 完全匹配 | 直接复用全部 KV Cache，跳过 Prefill                 |
| 部分匹配 | 复用匹配部分的 KV Cache，仅对未匹配后缀执行 Prefill |
| 无匹配   | 从头执行完整 Prefill，并将结果缓存                  |

```text
查询序列: [T0, T1, T2, T3, T4, T5, T6, T7]
树中已有: [T0, T1, T2, T3] → KV Blocks [B0, B1]

匹配结果: 长度=4, Blocks=[B0, B1]

Prefill 执行:
- 位置 0-3: 跳过计算，加载 [B0, B1]
- 位置 4-7: 执行 Prefill，生成新的 KV Cache
```

### 2.4 LRU 淘汰策略

GPU 显存有限，无法无限缓存 KV Cache。RadixAttention 采用 **LRU（Least Recently Used）** 策略进行缓存淘汰。

#### 2.4.1 引用计数机制

为了避免淘汰正在被使用的 KV Cache，每个节点维护一个**引用计数**：

- 当请求开始使用某节点的 KV Cache 时，`ref_count += 1`
- 当请求完成（生成结束或被取消）时，`ref_count -= 1`
- **只有 `ref_count == 0` 的节点才能被淘汰**

```python
def acquire_node(node: RadixNode):
    """请求开始使用节点"""
    node.ref_count += 1
    node.last_access_time = time.time()

def release_node(node: RadixNode):
    """请求释放节点"""
    node.ref_count -= 1
```

#### 2.4.2 淘汰算法

当显存不足时，RadixAttention 从**叶子节点**开始，按最后访问时间排序，淘汰最久未使用的节点：

```python
def evict_lru(root: RadixNode, target_free_blocks: int) -> int:
    """
    淘汰最久未使用的节点，释放指定数量的 KV Cache Blocks

    返回：实际释放的 Block 数量
    """
    freed_blocks = 0

    # 收集所有可淘汰的叶子节点（ref_count == 0）
    evictable_leaves = collect_evictable_leaves(root)

    # 按最后访问时间排序
    evictable_leaves.sort(key=lambda n: n.last_access_time)

    for leaf in evictable_leaves:
        if freed_blocks >= target_free_blocks:
            break

        # 释放 KV Cache Blocks
        freed_blocks += len(leaf.kv_cache_blocks)
        release_kv_blocks(leaf.kv_cache_blocks)

        # 从父节点移除
        remove_from_parent(leaf)

        # 如果父节点只剩一个子节点，合并路径
        merge_single_child_nodes(leaf.parent)

    return freed_blocks
```

#### 2.4.3 淘汰的级联效应

由于 Radix Tree 的前缀共享特性，淘汰需要谨慎处理：

- **只能从叶子节点开始淘汰**：中间节点被淘汰会导致所有后代节点失效。
- **淘汰后可能触发节点合并**：父节点只剩一个子节点时，可以合并以保持压缩特性。

```text
淘汰前：
    "hello "
     /    \
  "world" "vllm"

淘汰 "vllm" 后：
    "hello "
       |
    "world"

进一步压缩：
  "hello world"
```

---

## 3. SGLang 中的实现

SGLang（Structured Generation Language）是由 UC Berkeley 团队开发的高效 LLM 推理系统，RadixAttention 正是其核心优化技术之一。本章将分析 RadixAttention 在 SGLang 中的具体实现。

### 3.1 SGLang 系统架构简介

SGLang 的定位是**结构化语言模型程序的高效执行引擎**。它支持复杂的控制流（如循环、分支）、多次生成调用的复合任务，并通过 RadixAttention 实现跨调用的 KV Cache 自动复用。

```text
SGLang 架构概览：

┌───────────────────────────────────────────────────────────┐
│                     SGLang Frontend                       │
│       (Python DSL for structured LLM programs)            │
└───────────────────────────────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────┐
│                     SGLang Runtime                        │
│  ┌─────────────┐  ┌─────────────────┐  ┌─────────────┐    │
│  │  Scheduler  │◄─┤   RadixCache    ├─►│   Executor  │    │
│  │             │  │ (RadixAttention)│  │             │    │
│  └─────────────┘  └─────────────────┘  └─────────────┘    │
└───────────────────────────────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────┐
│                Model Execution Backend                    │
│           (CUDA Kernels, PagedAttention)                  │
└───────────────────────────────────────────────────────────┘
```

RadixAttention 位于 SGLang Runtime 的核心位置，负责：

1. 管理所有缓存的 token 序列与 KV Cache 的映射
2. 为每个请求提供前缀匹配查询
3. 协调 KV Cache Block 的分配与淘汰

### 3.2 核心数据结构

SGLang 中的 RadixAttention 实现涉及三个核心数据结构：RadixCache 主类、TreeNode 节点结构以及内存池管理器。

#### 3.2.1 RadixCache 类

`RadixCache` 是 RadixAttention 的主类，封装了 Radix Tree 及其操作：

```python
class RadixCache:
    def __init__(
        self,
        req_to_token_pool,      # Token Pool 管理器
        token_to_kv_pool,       # KV Cache Pool 管理器
        disable: bool = False,  # 是否禁用缓存
    ):
        self.root_node = TreeNode()  # Radix Tree 根节点
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool = token_to_kv_pool
        self.disable = disable

        # 统计信息
        self.total_cache_hit_tokens = 0
        self.total_cache_miss_tokens = 0

    def match_prefix(self, rid: int, key: List[int]) -> Tuple[TreeNode, int]:
        """查找最长匹配前缀"""
        ...

    def insert(self, node: TreeNode, key: List[int], value: List[int]):
        """插入新的 token 序列"""
        ...

    def evict(self, num_tokens: int) -> int:
        """LRU 淘汰指定数量的 tokens"""
        ...
```

#### 3.2.2 TreeNode 结构

每个 `TreeNode` 代表 Radix Tree 中的一个节点：

```python
class TreeNode:
    def __init__(self):
        # 子节点映射：token_id -> TreeNode
        self.children: Dict[int, TreeNode] = {}

        # 父节点引用
        self.parent: Optional[TreeNode] = None

        # 存储的 token 序列（压缩后的路径）
        self.key: List[int] = []

        # 对应的 KV Cache Block 索引
        self.value: List[int] = []

        # 引用计数
        self.lock_ref: int = 0

        # 最后访问时间
        self.last_access_time: float = 0.0
```

#### 3.2.3 内存池管理

SGLang 使用专门的内存池管理 KV Cache Blocks：

- **Token Pool**：管理 token 序列到物理位置的映射
- **KV Pool**：管理实际的 KV Cache 显存分配

```python
class TokenToKVPool:
    def __init__(self, size: int, dtype: torch.dtype):
        # 预分配的 KV Cache 显存
        self.k_buffer = torch.empty((size, ...), dtype=dtype, device='cuda')
        self.v_buffer = torch.empty((size, ...), dtype=dtype, device='cuda')

        # 空闲 Block 列表
        self.free_slots: List[int] = list(range(size))

    def alloc(self, num_blocks: int) -> List[int]:
        """分配 KV Cache Blocks"""
        ...

    def free(self, block_ids: List[int]):
        """释放 KV Cache Blocks"""
        ...
```

### 3.3 关键操作实现

本节详细分析 RadixCache 的四个关键操作：前缀匹配、插入、LRU 淘汰以及并发控制。

#### 3.3.1 前缀匹配：match_prefix()

前缀匹配是系统处理新请求时的首个操作。该方法通过遍历 Radix Tree，找到与输入序列匹配的最长前缀路径：

```python
def match_prefix(self, rid: int, key: List[int]) -> Tuple[TreeNode, int]:
    """
    查找与 key 匹配的最长前缀

    Args:
        rid: 请求 ID
        key: 输入的 token 序列

    Returns:
        (last_node, prefix_len): 最后匹配的节点和前缀长度
    """
    if self.disable:
        return self.root_node, 0

    current = self.root_node
    prefix_len = 0
    key_len = len(key)

    while prefix_len < key_len:
        # 查找匹配的子节点
        first_token = key[prefix_len]
        if first_token not in current.children:
            break

        child = current.children[first_token]
        child_key = child.key
        child_len = len(child_key)

        # 检查子节点的 key 是否完全匹配
        match_len = 0
        while match_len < child_len and prefix_len + match_len < key_len:
            if child_key[match_len] != key[prefix_len + match_len]:
                break
            match_len += 1

        prefix_len += match_len

        if match_len < child_len:
            # 部分匹配，需要在返回前处理
            break

        current = child

    # 更新统计
    self.total_cache_hit_tokens += prefix_len
    self.total_cache_miss_tokens += key_len - prefix_len

    return current, prefix_len
```

#### 3.3.2 插入操作：insert()

当处理完未命中的前缀并生成新的 KV Cache 后，系统需要将这些新数据插入到 Radix Tree 中。插入过程可能涉及节点的分裂：

```python
def insert(self, node: TreeNode, key: List[int], value: List[int]):
    """
    从指定节点开始，插入新的 token 序列

    Args:
        node: 起始节点（通常是 match_prefix 返回的节点）
        key: 要插入的 token 序列（未匹配部分）
        value: 对应的 KV Cache Block IDs
    """
    if self.disable or len(key) == 0:
        return

    current = node
    key_pos = 0

    while key_pos < len(key):
        first_token = key[key_pos]

        if first_token in current.children:
            child = current.children[first_token]
            child_key = child.key

            # 找到公共前缀长度
            common_len = 0
            while (common_len < len(child_key) and
                   key_pos + common_len < len(key) and
                   child_key[common_len] == key[key_pos + common_len]):
                common_len += 1

            if common_len < len(child_key):
                # 需要分裂节点
                self._split_node(child, common_len)

            key_pos += common_len
            current = child
        else:
            # 创建新节点
            new_node = TreeNode()
            new_node.key = key[key_pos:]
            new_node.value = value[key_pos:]
            new_node.parent = current
            current.children[first_token] = new_node
            break

    current.last_access_time = time.time()
```

#### 3.3.3 LRU 淘汰：evict()

当系统显存不足时，将触发 LRU 淘汰机制。淘汰操作总是从最久未使用的叶子节点开始，以释放 KV Cache 块：

```python
def evict(self, num_tokens: int) -> int:
    """
    淘汰至少 num_tokens 个 tokens 的缓存

    Returns:
        实际淘汰的 tokens 数量
    """
    if self.disable:
        return 0

    evicted = 0

    # 收集所有叶子节点
    leaves = self._collect_leaves()

    # 按最后访问时间排序（最旧的在前）
    leaves.sort(key=lambda n: n.last_access_time)

    for leaf in leaves:
        if evicted >= num_tokens:
            break

        # 跳过被锁定的节点
        if leaf.lock_ref > 0:
            continue

        # 释放 KV Cache
        evicted += len(leaf.value)
        self.token_to_kv_pool.free(leaf.value)

        # 从树中移除
        self._remove_node(leaf)

    return evicted
```

#### 3.3.4 并发控制

SGLang 使用**节点级锁**来支持并发访问：

```python
def inc_lock_ref(self, node: TreeNode):
    """增加节点的引用计数（请求开始使用）"""
    delta = 0
    while node is not None:
        if node.lock_ref == 0:
            delta += 1
        node.lock_ref += 1
        node = node.parent
    return delta

def dec_lock_ref(self, node: TreeNode):
    """减少节点的引用计数（请求结束）"""
    delta = 0
    while node is not None:
        node.lock_ref -= 1
        if node.lock_ref == 0:
            delta += 1
        node = node.parent
    return delta
```

### 3.4 与 PagedAttention 的集成

RadixAttention 与 PagedAttention 紧密集成，共同实现高效的 KV Cache 管理：

1. **Block 粒度对齐**：RadixCache 中的 value 存储的是 Block ID 而非原始显存地址。
2. **统一的内存池**：KV Cache Blocks 由 `TokenToKVPool` 统一管理。
3. **Block Table 更新**：Prefill 完成后，Block Table 自动更新以包含新生成的 KV Cache。

```text
请求处理流程：

1. match_prefix(tokens) → (node, prefix_len, block_ids)
2. 将 block_ids 加载到 Block Table 的前 prefix_len 位置
3. 仅对 tokens[prefix_len:] 执行 Prefill
4. 将新生成的 KV Cache 存入新分配的 Blocks
5. insert(node, tokens[prefix_len:], new_block_ids)
```

---

## 4. vLLM 的 Automatic Prefix Caching (APC) 对比

vLLM 从 v0.4.0 开始引入了 **[Automatic Prefix Caching (APC)](prefix_caching.md#3-vllm-的-automatic-prefix-caching-apc)** 功能，实现了类似 RadixAttention 的前缀缓存能力，但采用了不同的技术方案。本章将对比分析两者的异同。

### 4.1 vLLM 的技术选型：Hash Table vs Radix Tree

vLLM 的 APC 与 SGLang 的 RadixAttention 在功能上相似，但在底层数据结构上做出了不同的选择。本节分析 vLLM 选择哈希表的原因及其设计细节。

#### 4.1.1 为什么 vLLM 选择 Hash Table

vLLM 没有采用 Radix Tree，而是选择了**基于哈希表的增量哈希链**方案。主要考量包括：

1. **与 Block Manager 的集成**：vLLM 的 PagedAttention 已经以 Block 为粒度管理 KV Cache，哈希表可以直接以 Block Hash 为键。
2. **实现复杂度**：哈希表的实现比 Radix Tree 更简单，易于维护。
3. **内存开销**：哈希表的额外内存开销通常低于树结构。

#### 4.1.2 增量哈希链设计

vLLM 的 APC 使用**增量哈希**确保前缀依赖：

$$
H_0 = Hash(Chunk_0)
$$

$$
H_i = Hash(H_{i-1}, Chunk_i), \quad i > 0
$$

```python
def hash_block_tokens(
    parent_block_hash: Optional[int],
    curr_block_token_ids: Tuple[int, ...]
) -> int:
    """计算 Block 的哈希值"""
    return hash((parent_block_hash, curr_block_token_ids))
```

这种设计保证了：

- 相同的前缀 → 相同的哈希链
- 不同的前缀 → 不同的哈希值（即使当前 Block 内容相同）

#### 4.1.3 缓存查找流程

vLLM 的缓存查找过程基于增量哈希链，按 Block 粒度逐个匹配，一旦遇到未命中的 Block 即可提前终止查找：

```python
def find_cached_blocks(token_ids: List[int], block_size: int) -> List[int]:
    """查找已缓存的 Blocks"""
    cached_blocks = []
    parent_hash = None

    for i in range(0, len(token_ids), block_size):
        chunk = tuple(token_ids[i:i + block_size])
        block_hash = hash_block_tokens(parent_hash, chunk)

        if block_hash in block_cache:
            cached_blocks.append(block_cache[block_hash])
            parent_hash = block_hash
        else:
            break  # 一旦未命中，后续都无法复用

    return cached_blocks
```

### 4.2 两种方案的对比分析

SGLang 的 RadixAttention 与 vLLM 的 APC 在设计理念和数据结构上各有侧重，以下是这两种方案在多个维度上的详细对比：

| 维度           | RadixAttention (SGLang)      | APC (vLLM)                 |
| -------------- | ---------------------------- | -------------------------- |
| **数据结构**   | Radix Tree（压缩前缀树）     | Hash Table（哈希表）       |
| **索引粒度**   | Token 级别                   | Block 级别（16-32 tokens） |
| **查找方式**   | 树遍历，天然支持最长前缀匹配 | 逐 Block 哈希查找          |
| **内存开销**   | 较高（树节点、指针）         | 较低（仅哈希表条目）       |
| **动态前缀**   | 任意位置分支，灵活性高       | 需 Block 对齐，灵活性稍低  |
| **实现复杂度** | 较高（节点分裂、合并）       | 较低（标准哈希表操作）     |
| **并发支持**   | 需要细粒度锁                 | 通过引用计数管理           |

#### 4.2.1 适用场景对比

根据两种方案的特性差异，它们在不同的业务场景中各有优势。以下是它们的主要适用场景对比：

**RadixAttention 更适合**：

- 高度动态的工作负载（前缀频繁变化）
- 需要细粒度（token 级）缓存控制的场景
- 复杂的结构化生成程序（SGLang 的核心用例）

**vLLM APC 更适合**：

- 前缀相对固定的场景（如固定 System Prompt）
- 追求实现简洁性和与现有系统集成
- Block 对齐不造成显著浪费的场景

### 4.3 vLLM Router 中的 Radix Tree 应用

有趣的是，虽然 vLLM 的 KV Cache 管理使用哈希表，但其**路由组件（vLLM Router）** 却使用了 Radix Tree 来实现 **Cache Aware 路由策略**。本节介绍该策略的原理和动态平衡机制。

#### 4.3.1 Cache Aware 策略概述

在多实例部署场景中，如何将请求路由到最可能已缓存其前缀的 Worker 节点，是一个关键问题。vLLM Router 的 Cache Aware 策略：

1. 为每个 Worker 维护一棵**近似 Radix Tree**，追踪该 Worker 上已缓存的请求前缀。
2. 新请求到达时，计算其与各 Worker Radix Tree 的**前缀匹配率**。
3. 优先将请求路由到匹配率最高的 Worker。

```text
Router 中的 Radix Tree 结构：

Worker A 的 Radix Tree:
    root -> "You are a helpful" -> " assistant" -> " for coding"
                                -> " chatbot"

Worker B 的 Radix Tree:
    root -> "You are a helpful" -> " translator"
         -> "Summarize the"     -> " following text"

新请求: "You are a helpful assistant for Python"
→ 与 Worker A 匹配 "You are a helpful assistant"（更长）
→ 路由到 Worker A
```

#### 4.3.2 动态平衡机制

Cache Aware 策略并非总是选择匹配率最高的 Worker，它还会考虑负载均衡：

```python
def select_worker(request: Request, workers: List[Worker]) -> Worker:
    # 计算各 Worker 的匹配率
    match_rates = [worker.radix_tree.match_rate(request.tokens)
                   for worker in workers]

    # 获取当前负载
    loads = [worker.current_load for worker in workers]

    # 判断是否需要负载均衡
    if max(loads) - min(loads) > BALANCE_THRESHOLD:
        # 负载不均衡时，优先考虑负载均衡
        return workers[loads.index(min(loads))]
    else:
        # 负载均衡时，优先考虑缓存命中
        return workers[match_rates.index(max(match_rates))]
```

---

## 5. 进阶主题

本章探讨 RadixAttention 在生产环境中的进阶应用，包括多租户与分布式场景下的扩展设计、与其他推理优化技术的协同，以及非前缀复用场景的解决方案。

### 5.1 多租户与分布式场景

在实际生产环境中，RadixAttention 需要支持多租户隔离和跨实例的 KV Cache 共享。本节讨论相关的扩展设计。

#### 5.1.1 多租户支持

在多租户场景中，不同租户（或不同 Worker）可能对同一前缀有不同的访问模式。RadixAttention 通过在节点中维护 **per-tenant 访问时间** 来支持细粒度的淘汰策略：

```python
class TreeNode:
    # 每个租户的最后访问时间
    tenant_last_access_time: Dict[str, float] = {}

    def update_access(self, tenant_id: str):
        self.tenant_last_access_time[tenant_id] = time.time()
```

#### 5.1.2 跨实例 KV Cache 共享

在分布式推理系统中，跨实例共享 KV Cache 是一个重要优化方向。主要挑战包括：

- **一致性**：如何保证不同实例看到一致的缓存状态。
- **传输开销**：KV Cache 数据量大，网络传输成本高。
- **索引同步**：Radix Tree 结构需要在多实例间同步。

**[Mooncake](../mooncake/mooncake_architecture.md) 的方案**：

- 使用全局 Conductor 管理 KV Cache 元数据。
- KV Cache 存储在分布式存储池（GPU VRAM + CPU DRAM + SSD）。
- 调度时考虑缓存命中率，将请求路由到已缓存前缀的节点。

**[LMCache](../lmcache/lmcache_overview.md) 的方案**：

- 支持 P2P 模式和集中式存储模式。
- 使用 Cache Controller 维护全局索引。
- 支持跨实例的 KV Cache 传输和复用。

### 5.2 与其他优化技术的协同

RadixAttention 可以与多种推理优化技术协同工作，进一步提升系统性能。本节介绍三种典型的协同场景。

#### 5.2.1 Chunked Prefill

Chunked Prefill 将长 Prefill 拆分为多个小批次执行，与 Decode 请求交替调度。RadixAttention 可以与 Chunked Prefill 协同：

- 缓存命中的 Chunk 完全跳过。
- 仅对未命中的 Chunk 执行分块 Prefill。
- 进一步降低长前缀场景的调度延迟。

#### 5.2.2 Prefill-Decode 分离架构

Prefill-Decode 分离架构（PD 分离）通过将计算密集型的 Prefill 阶段和访存密集型的 Decode 阶段分配到不同的节点上执行，从而提升系统整体吞吐量。在 [PD 分离架构](../lmcache/pd_backend.md) 中：

- **Prefill 节点**：维护热门前缀的 RadixCache，专注于 Prefill 计算。
- **Decode 节点**：接收 Prefill 节点传来的 KV Cache，专注于 Decode 生成。

RadixAttention 在此场景下的价值：

- Prefill 节点可以通过 RadixCache 快速判断是否需要计算。
- 相同前缀的请求可以路由到同一 Prefill 节点，最大化复用。

#### 5.2.3 Speculative Decoding

推测执行需要为多个候选 token 序列准备 KV Cache。RadixAttention 可以：

- 缓存高概率分支的 KV Cache。
- 在验证阶段快速复用已计算的 KV Cache。
- 降低推测失败时的重算成本。

### 5.3 非前缀复用场景

标准的 [Prefix Caching](prefix_caching.md) 要求从序列开头进行严格的前缀匹配，但某些场景下存在非前缀的 KV Cache 复用需求。本节讨论相关挑战与解决方案。

#### 5.3.1 RAG 场景的挑战

在 RAG（检索增强生成）场景中，检索到的文档片段顺序可能不固定：

```text
请求 A: [System Prompt] + [Doc1] + [Doc2] + [Query]
请求 B: [System Prompt] + [Doc2] + [Doc1] + [Query]

由于 Transformer 的因果注意力：
- Doc1 在位置 2 和位置 3 的 KV Cache 是不同的
- 即使内容相同，位置不同导致无法直接复用
```

#### 5.3.2 CacheBlend：选择性重算

LMCache 提出的 [CacheBlend](../lmcache/cache_blend.md) 技术提供了一种解决方案：

1. **允许复用非前缀位置的 KV Cache**（存在位置不匹配）。
2. **选择性重算关键位置的 Attention**（修正位置偏差）。
3. **在复用率和计算开销之间动态平衡**。

```text
CacheBlend 流程：

1. 检索已缓存的 Doc2 的 KV Cache（位置 3）
2. 将其加载到位置 2
3. 重算部分 Attention Scores 以修正位置编码偏差
4. 得到近似但高效的结果
```

---

## 6. 性能分析与最佳实践

本章将从理论模型和实际测试两个方面分析 [Prefix Caching](prefix_caching.md) 的性能表现，并总结在生产环境中部署的最佳实践。

### 6.1 理论性能分析

本节从理论角度推导 [Prefix Caching](prefix_caching.md) 的加速比，并分析不同场景下的性能收益。

#### 6.1.1 加速比公式

在评估 [Prefix Caching](prefix_caching.md) 的性能时，我们可以通过建立简化的数学模型来推导其理论加速比。假设：

- $P$ ：前缀长度（tokens）
- $Q$ ：用户查询长度（tokens）
- Prefill 计算复杂度： $O(N^2)$ ，其中 $N = P + Q$

**无 [Prefix Caching](prefix_caching.md)**：

$$
\text{Cost}_{\text{total}} \approx \frac{1}{2}(P + Q)^2
$$

**有 [Prefix Caching](prefix_caching.md)（完全命中）**：

$$
\text{Cost}_{\text{cached}} \approx Q \cdot (P + Q) + \frac{1}{2}Q^2 \approx Q(P + Q)
$$

**加速比**：

$$
Speedup \approx \frac{(P + Q)^2 / 2}{Q(P + Q)} = \frac{P + Q}{2Q}
$$

#### 6.1.2 典型场景收益

根据上述公式，不同应用场景下 [Prefix Caching](prefix_caching.md) 能够带来的理论加速比有显著差异，具体如下表所示：

| 场景               | 前缀长度 P | 查询长度 Q | 理论加速比 |
| ------------------ | ---------- | ---------- | ---------- |
| 短 System Prompt   | 1K         | 256        | ~2.5x      |
| 标准 System Prompt | 4K         | 256        | ~8.5x      |
| 长 System Prompt   | 8K         | 256        | ~16.5x     |
| 多轮对话（5轮）    | 2K         | 128        | ~8.5x      |
| RAG（长文档）      | 16K        | 512        | ~16x       |

### 6.2 实测数据

本节汇总 SGLang 和 vLLM 在不同工作负载下的实测性能数据。

#### 6.2.1 SGLang 官方基准测试

根据 SGLang 论文 [1] 和官方文档 [4] 的基准测试：

| 工作负载         | 模型       | 吞吐量提升 | TTFT 降低 |
| ---------------- | ---------- | ---------- | --------- |
| 多轮对话         | Llama-2-7B | 2-3x       | 60-80%    |
| Few-shot 学习    | Llama-2-7B | 3-5x       | 70-85%    |
| 结构化生成       | Llama-2-7B | 4-6x       | 75-90%    |
| Tree of Thoughts | Llama-2-7B | 5-8x       | 80-95%    |

#### 6.2.2 与 vLLM APC 的对比

根据 SGLang 官方提供的测试数据 [4]，在相同硬件和模型配置下，RadixAttention 与 vLLM APC 的性能对比如下：

| 指标         | SGLang (RadixAttention) | vLLM (APC) |
| ------------ | ----------------------- | ---------- |
| 前缀命中率   | 85-95%                  | 80-90%     |
| 内存开销     | +5-10%                  | +2-5%      |
| 动态前缀支持 | 优秀                    | 良好       |
| 实现复杂度   | 高                      | 中         |

### 6.3 最佳实践

基于理论分析和实测经验，本节总结在生产环境中使用 [Prefix Caching](prefix_caching.md) 的关键优化建议。

#### 6.3.1 System Prompt 设计

为了最大化前缀缓存的命中率，建议在设计 System Prompt 时遵循“稳定内容在前，动态内容在后”的原则：

```text
推荐结构：

┌─────────────────────────────────────┐
│ [固定部分 - 放在最前面]                │
│ - 角色定义                           │
│ - 通用指令                           │
│ - 输出格式要求                        │
├─────────────────────────────────────┤
│ [半动态部分 - 中间位置]                │
│ - Few-shot 示例（按使用频率排序）       │
│ - 常用工具说明                        │
├─────────────────────────────────────┤
│ [动态部分 - 放在最后]                  │
│ - 用户名、时间戳                      │
│ - 会话特定上下文                      │
│ - RAG 检索结果                       │
└─────────────────────────────────────┘

原则：稳定内容在前，动态内容在后
```

#### 6.3.2 Chunk Size 选择

在使用哈希表方案（如 vLLM APC）时，Chunk Size 的大小直接影响缓存的匹配粒度和管理开销。不同大小的优缺点对比如下：

| Chunk Size | 优点       | 缺点             | 推荐场景     |
| ---------- | ---------- | ---------------- | ------------ |
| 64 tokens  | 细粒度匹配 | 哈希计算开销大   | 高度动态前缀 |
| 256 tokens | 平衡       | -                | **通用推荐** |
| 512 tokens | 低管理开销 | 粒度粗，浪费空间 | 前缀非常稳定 |

建议：**选择 vLLM Block Size 的整数倍**（如 256 = 16 × 16），便于与 PagedAttention 对齐。

#### 6.3.3 缓存预热策略

在服务启动或更新时，可以通过预热高频使用的 System Prompt 或上下文来避免冷启动带来的首次请求延迟：

```python
# 服务启动时预热常用前缀
def warmup_cache(common_prompts: List[str]):
    for prompt in common_prompts:
        tokens = tokenizer.encode(prompt)
        # 触发 Prefill 并缓存
        model.generate(tokens, max_new_tokens=1)
        print(f"Warmed up: {len(tokens)} tokens")

# 使用示例
warmup_cache([
    SYSTEM_PROMPT,
    SYSTEM_PROMPT + FEW_SHOT_EXAMPLES,
    COMMON_RAG_CONTEXT,
])
```

---

## 7. 总结

本章总结 RadixAttention 的核心价值、技术选型建议以及未来发展方向。

### 7.1 RadixAttention 的核心价值

RadixAttention 通过创新性地将 Radix Tree 数据结构应用于 KV Cache 管理，实现了：

1. **自动前缀识别**：无需手动声明，系统自动发现和复用共享前缀。
2. **高效查找**： $O(m)$ 时间复杂度的最长前缀匹配。
3. **细粒度管理**：支持任意位置的前缀分支和复用。
4. **动态适应**：通过 LRU 淘汰自动适应工作负载变化。

### 7.2 技术选型建议

针对不同的业务需求和系统架构，我们建议在实际部署中采用以下技术选型方案：

| 场景                     | 推荐方案                      |
| ------------------------ | ----------------------------- |
| SGLang 用户              | RadixAttention（内置）        |
| vLLM 单实例、显存充足    | APC（内置）                   |
| vLLM 单实例、需扩展缓存  | APC + LMCache                 |
| 多实例部署、需跨实例共享 | LMCache（P2P 或 Server 模式） |
| 生产级高可用             | LMCache + Redis/Mooncake 后端 |

### 7.3 未来发展方向

尽管 RadixAttention 已经极大地优化了 LLM 推理的效率，但 KV Cache 管理技术仍在不断演进，未来的发展方向主要包括：

1. **跨模态 KV Cache 复用**：视觉-语言模型中图像 token 的缓存。
2. **语义级缓存**：基于语义相似度而非精确匹配的缓存策略。
3. **分布式 Radix Tree**：支持多节点协同的全局 KV Cache 管理。
4. **硬件加速**：利用专用硬件加速 Radix Tree 操作。

---

## 8. 参考资料

### 8.1 延伸阅读

- [Prefix Caching 技术详解](prefix_caching.md) - Prefix Caching 基础概念与 vLLM/LMCache 实践
- [LMCache 架构概览](../lmcache/lmcache_overview.md) - 多级 KV Cache 管理系统
- [vLLM Router 架构解析](../../vllm/vllm_router.md) - Cache Aware 路由策略
- [Mooncake 架构详解](../mooncake/mooncake_architecture.md) - KV Cache 感知的分离式推理架构

### 8.2 学术论文

- [1] L. Zheng et al., "SGLang: Efficient Execution of Structured Language Model Programs," *arXiv preprint arXiv:2312.07104*, 2023.
  - **核心贡献**：提出 RadixAttention，通过 Radix Tree 自动管理和复用 KV Cache。

- [2] W. Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," *SOSP 2023*.
  - **核心贡献**：提出 PagedAttention，是 RadixAttention 的内存管理基础。

- [3] I. Gim et al., "Prompt Cache: Modular Attention Reuse for Low-Latency Inference," *arXiv preprint arXiv:2311.04934*, 2023.
  - **核心贡献**：系统阐述了 Attention 状态复用的模块化方法。

### 8.3 项目文档

- [4] SGLang Documentation: <https://sgl-project.github.io/>
- [5] vLLM Automatic Prefix Caching: <https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html>
- [6] LMCache Documentation: <https://docs.lmcache.ai>
