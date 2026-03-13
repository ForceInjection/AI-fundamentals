# Hybrid KV Cache Manager 深度解析

**Hybrid KV Cache Manager**（混合 KV 缓存管理器，在代码中简称 HMA，即 Hybrid Memory Allocator）是 vLLM V1 引擎专为**混合注意力架构模型**设计的一套显存优化机制。它解决了传统统一分配策略在混合模型上的严重内存浪费问题，使得 Gemma-3、Qwen 3.5 MoE、Llama 4 等采用 Sliding Window、Mamba、Local Chunked 等高效注意力机制的模型能够高效运行。本文将从 HMA 的设计思想、内存布局、Prefix Caching 算法、三层架构以及与 KV Transfer 的兼容性问题等方面进行全面解析。

## 1. 背景：混合注意力模型的兴起

近年来，越来越多的 LLM 在同一模型中混合使用多种注意力机制（hybrid attention），以此在长上下文能力与推理效率之间取得平衡。vLLM 目前支持以下三类主要的混合注意力模型：

| 注意力类型组合                  | 代表模型                         |
| ------------------------------- | -------------------------------- |
| Sliding Window + Full Attention | Gemma-2/3、Ministral、Cohere     |
| Mamba + Full Attention          | Bamba、Jamba、Minimax、Qwen3 MoE |
| Local Chunked + Full Attention  | Llama 4                          |

在这类模型中，**不同层需要缓存的 token 数量不同**：

- **Full Attention 层**：需要保存所有 token 的 KV Cache；
- **Sliding Window 层**：只需要保存最近 `sliding_window_size` 个 token；
- **Mamba 层**：使用固定大小的 state，不随序列长度增长；
- **Local Chunked Attention 层**：只保存当前 chunk 内的 token。

如果用“统一分配”的方式（即按 Full Attention 的规格为所有层分配相同数量的 KV Cache 块），Sliding Window 等高效注意力层就会浪费大量显存。**Hybrid KV Cache Manager** 正是为解决这一问题而设计的。

---

## 2. 核心思想：统一 Page Size + 差异化分配

Hybrid KV Cache Manager 的核心挑战是：**所有层共用一个物理内存池（block pool），要求每个 block 物理大小（page size）相同**，但不同类型的层需要不同数量的 block。

### 2.1 基本定义

以下是 Hybrid KV Cache Manager 中使用的核心术语定义：

| 术语               | 含义                                                      |
| ------------------ | --------------------------------------------------------- |
| **KV hidden size** | 单层单 token 的 KV Cache 字节数                           |
| **block size**     | 一个 block 能容纳的 token 数量                            |
| **page size**      | 一个 block 的物理内存大小 = `block_size × kv_hidden_size` |
| **KV Cache Group** | 具有相同注意力类型的层的集合，共享同一套 block table      |

注意：这里的 `page_size` 是单层单 block 的大小，与代码中的 `KVCacheSpec.page_size_bytes`（= `block_size × kv_hidden_size`）定义一致。

### 2.2 KV Cache Group 分组策略

vLLM 将模型中的层按注意力类型分组，称为 **KV Cache Groups**。每个 group 内的层共享相同的 block table，由同一个 [SingleTypeKVCacheManager](../vllm/v1/core/single_type_kv_cache_manager.py) 管理。

分组的两个核心约束：

1. 同一 group 内，所有层的注意力类型相同；
2. 所有 group 的 page size 必须相同（保证物理内存连续分配）。

#### Case 1：所有层注意力类型相同

当模型仅使用一种注意力类型（如全部为 Full Attention）时，所有层属于同一个 group，共享一张 block table。这是最简单的情况，由 `_get_kv_cache_groups_uniform_type` 直接处理，无需 Hybrid KV Cache Manager 介入。

#### Case 2：同 kv_hidden_size + 规律比例

以 10 个 Full Attention 层 + 20 个 Sliding Window 层为例，比例为 1:2，可分为 3 个 group：

- **Group 0**：10 个 Full Attention 层（full.0 – full.9）
- **Group 1**：10 个 Sliding Window 层（sw.0 – sw.9）
- **Group 2**：10 个 Sliding Window 层（sw.10 – sw.19）

调度时，一个请求的 block 会被差异化分配：Full Attention group 分到更多 block，Sliding Window group 只分配覆盖窗口所需的 block。

#### Case 3：同 kv_hidden_size + 无规律比例（如 Gemma-3）

Gemma-3-27B 有 52 个 Sliding Window 层和 10 个 Full Attention 层，比例不整除。此时取最小层数 `min(52, 10) = 10` 作为 group size，分为：

- Group 0：10 个 Full Attention 层
- Group 1~5：各 10 个 Sliding Window 层
- Group 6：2 个 Sliding Window 层 + **8 个 padding 层**（为对齐 Group Size 填充，存在内存浪费）

这个启发式算法在大多数场景下效果良好。此外，当各类型的层数差异较小时（`max_num_layers < min_num_layers * 1.25`），算法会使用最大层数而非最小层数作为 group size，以减少 padding 层的数量。对于极端比例的模型可能需要进一步调整。

#### Case 4：不同 kv_hidden_size（Mamba 混合模型）

Mamba 层的状态大小（state size）通常远大于 Attention 层的 kv_hidden_size。由于必须保证所有 group 的 page size 相同，当前算法是：

1. 增大 Attention 层的 `block_size`，直到：
   $$\texttt{block\_size} \times \texttt{kv\_hidden\_size}_{\text{attn}} \ge \texttt{state\_size}_{\text{mamba}}$$
2. 将 Mamba 的 state 填充（pad）到 `block_size × kv_hidden_size_attn`；
3. 再按 Case 3 的策略分组。

> 已知问题：这种方法可能导致 Attention 层的 `block_size` 超过 400，过大的 block_size 会降低调度灵活性。更优的策略（基于多层聚合的 padding）目前仍在开发中。

#### Case 5：KV Sharing（如 Gemma-3n）

部分模型的某些层会复用另一层的 KV Cache（KV sharing）。KVCacheManager 对这些层忽略独立分配，只为需要独立 KV Cache 的层分配，Model Runner 中再进行 block table 的映射。

---

## 3. 内存布局

对于包含 N 个 KV Cache Group、每 group M 层的模型，物理内存被划分为 M 个 `KVCacheTensor` 缓冲区。每个缓冲区由来自不同 group 的各 1 层共享，按 `block_size × kv_hidden_size` 的粒度切分。

以 10 Full + 20 SW（3 个 group，每 group 10 层）为例：

- `KVCacheTensor 0` 由 full.0（Group 0）、sw.0（Group 1）、sw.10（Group 2）共享；
- 一个请求分到 11 个 block（block_id 0-10），其中 0-6 分配给 Group 0，7-8 给 Group 1，9-10 给 Group 2；
- 物理上，逻辑 block_id 映射到对应 `KVCacheTensor` 中的不同 offset，互不干扰。

核心代码见 [vllm/v1/core/kv_cache_utils.py](../vllm/v1/core/kv_cache_utils.py) 中的 `_get_kv_cache_groups_uniform_page_size` 函数。

---

## 4. Prefix Caching 支持

Hybrid KV Cache Manager 对 Prefix Caching 提供分层支持。

### 4.1 Block Pool 的 group 感知

Block Pool 使用 `(block_hash, group_id)` 作为缓存键，封装在 `BlockHashToBlockMap` 类中：

```python
# BlockHashToBlockMap 内部结构（支持同一 hash 映射到多个 block）
# dict[BlockHashWithGroupId, KVCacheBlock | dict[int, KVCacheBlock]]
cached_block_hash_to_block: BlockHashToBlockMap = BlockHashToBlockMap()
```

即：同一 token 序列在不同 group 中的缓存可以独立命中和逐出，互不影响。

### 4.2 Full Attention 的前缀命中

从左到右扫描，遇到第一个缓存 miss 即停止——标准的前缀匹配语义。

### 4.3 Sliding Window 的前缀命中

由于 sliding window 只关心最近 `sliding_window_size` 个 token，有效的缓存命中前缀是任意满足“结尾 `sliding_window_size` 个 token 均已缓存”的前缀。因此：

- 从**右到左**扫描，遇到第一个 miss 停止；
- 可能存在多个有效前缀（越长的越节省计算），取最长者。

### 4.4 混合模型的前缀命中（HybridKVCacheCoordinator）

对于 Full + X 的组合，算法为：

1. 先对 Full Attention group 从左到右找最长命中长度 `L_full`；
2. 再对 X group 从右到左，在 `[0, L_full]` 范围内找最长命中 `L_x`；
3. 取 `min(L_full, L_x)` 作为最终前缀命中长度。

> 当前限制：`HybridKVCacheCoordinator` 只支持**恰好两种注意力类型**（1 Full + 1 X）。超过两种类型的模型（如 Full + SW + Mamba）需要关闭 Prefix Caching 使用 `KVCacheCoordinatorNoPrefixCache`。

代码实现见 [vllm/v1/core/kv_cache_coordinator.py](../vllm/v1/core/kv_cache_coordinator.py)。

---

## 5. 三层架构总览

Hybrid KV Cache Manager 的整体架构分为三层，从上到下分别是 KVCacheManager（调度器接口）、KVCacheCoordinator（跨 group 协调）和 SingleTypeKVCacheManager（单 group 管理）：

```text
KVCacheManager                 ← 调度器与 KV Cache 系统的接口
    │
    └── KVCacheCoordinator     ← 跨 group 协调，选择以下三种之一：
            ├── KVCacheCoordinatorNoPrefixCache   (无 prefix cache)
            ├── UnitaryKVCacheCoordinator         (单 group，简化逻辑)
            └── HybridKVCacheCoordinator          (Full + X，两 group)
                    │
                    └── SingleTypeKVCacheManager  ← 每个 group 一个实例
                            ├── FullAttentionManager
                            ├── SlidingWindowManager
                            ├── MambaManager
                            ├── ChunkedLocalAttentionManager
                            └── CrossAttentionManager
```

核心文件：

| 文件                                                                                            | 职责                                  |
| ----------------------------------------------------------------------------------------------- | ------------------------------------- |
| [vllm/v1/core/kv_cache_manager.py](../vllm/v1/core/kv_cache_manager.py)                         | KVCacheManager 入口                   |
| [vllm/v1/core/kv_cache_coordinator.py](../vllm/v1/core/kv_cache_coordinator.py)                 | 三类 Coordinator                      |
| [vllm/v1/core/single_type_kv_cache_manager.py](../vllm/v1/core/single_type_kv_cache_manager.py) | 各注意力类型的 Manager                |
| [vllm/v1/core/kv_cache_utils.py](../vllm/v1/core/kv_cache_utils.py)                             | 分组算法、unify_hybrid_kv_cache_specs |
| [vllm/v1/core/block_pool.py](../vllm/v1/core/block_pool.py)                                     | 物理 block 的分配与 LRU 逐出          |

---

## 6. KV Transfer（LMCache）的兼容问题

本章讨论 Hybrid KV Cache Manager 与 KV Transfer（特别是 LMCache）之间的兼容性问题，包括当前存在的死局场景、根本原因以及降级行为。

### 6.1 问题现象

以 Qwen 3.5 MoE（Mamba + Full Attention 混合架构）为例，使用 LMCache 开启 KV Transfer 时会遇到一个**死局**：

**场景 1：保持 HMA 开启（默认行为）**：

vLLM 的 `KVConnectorFactory`（[vllm/distributed/kv_transfer/kv_connector/factory.py](../vllm/distributed/kv_transfer/kv_connector/factory.py)）会在启动时强制检查 connector 是否支持 HMA：

```python
hma_enabled = not config.scheduler_config.disable_hybrid_kv_cache_manager
if hma_enabled and not supports_hma(connector_cls):
    raise ValueError(
        f"Connector {connector_cls.__name__} does not support HMA but "
        f"HMA is enabled. Please set `--disable-hybrid-kv-cache-manager`."
    )
```

LMCache 的 `LMCacheConnectorV1` 仅继承自 `KVConnectorBase_V1`，未实现 `SupportsHMA`，因此直接报错拒绝启动。

**场景 2：按提示关闭 HMA（`--disable-hybrid-kv-cache-manager`）**

对于 Sliding Window + Full 的模型（如 Gemma-3），关闭 HMA 后会退化为全量分配（详见 6.3），虽然浪费显存但能运行。然而对于 **Mamba + Full 的模型**（如 Qwen 3.5 MoE、Jamba），`MambaSpec` 无法被统一为 `FullAttentionSpec`，直接抛出异常：

```text
ValueError: Hybrid KV cache manager is disabled but failed to convert
the KV cache specs to one unified type.
```

这意味着：**Mamba 混合模型在开启 KV Transfer 时，既不能开 HMA 也不能关 HMA，完全无法启动**。

### 6.2 根本原因

LMCache 的 connector 实现**假设只有一个 KV Cache Group**。

在 LMCache 的 connector 类定义中（无论是 vLLM 内置的 `LMCacheConnectorV1` 还是 LMCache 独立发布的 `LMCacheConnectorV1Dynamic`），均仅继承自 `KVConnectorBase_V1`，未继承 `SupportsHMA`：

```python
# LMCache 的 connector 类继承关系
class LMCacheConnectorV1(KVConnectorBase_V1):    # 未继承 SupportsHMA
    ...
```

在 [vllm/distributed/kv_transfer/kv_connector/v1/lmcache_integration/vllm_v1_adapter.py](../vllm/distributed/kv_transfer/kv_connector/v1/lmcache_integration/vllm_v1_adapter.py) 中，建立 request tracker 时直接取 `block_ids[0]`：

```python
# According to the vLLM code, only one KVCacheGroup is supported in connector for now.
unfolded_block_ids = new_request.block_ids[0].copy()
```

而 Hybrid KV Cache Manager 下一个请求会持有**多个 group 的 block_ids**（`block_ids` 是一个 tuple，每个元素对应一个 group）。LMCache 当前只读取第一个 group 的 block，无法感知其他 group 的存在，导致 KV Transfer 的数据不完整或错误。

> LMCache 官方文档也明确要求使用 `--disable-hybrid-kv-cache-manager` 标志启动 vLLM，并标注为 **mandatory**（强制）。

### 6.3 HMA 关闭后的降级行为

当 Hybrid KV Cache Manager 被关闭时，调用 [unify_hybrid_kv_cache_specs](../vllm/v1/core/kv_cache_utils.py) 将所有 `SlidingWindowSpec` 和 `ChunkedLocalAttentionSpec` **强制转换为 `FullAttentionSpec`**：

```python
# unify_hybrid_kv_cache_specs 的核心逻辑
if has_full_attention and (has_sliding_window or has_chunked_local_attention):
    for layer_name, spec in kv_cache_spec.items():
        if isinstance(spec, SlidingWindowSpec):
            kv_cache_spec[layer_name] = FullAttentionSpec(...)
        elif isinstance(spec, ChunkedLocalAttentionSpec):
            kv_cache_spec[layer_name] = FullAttentionSpec(...)
```

这一降级策略对不同模型类型的影响差异巨大：

| 模型类型                         | 关闭 HMA 后的影响                                                      |
| -------------------------------- | ---------------------------------------------------------------------- |
| Sliding Window + Full（Gemma-3） | SW 层退化为 Full 分配，显存浪费可达 50%+，但**可以运行**               |
| Chunked Local + Full（Llama 4）  | Local 层退化为 Full 分配，显存浪费，但**可以运行**                     |
| Mamba + Full（Qwen 3.5 MoE）     | `MambaSpec` 无法统一为 `FullAttentionSpec`，**直接报错，完全无法运行** |

> 注意：即便关闭了 Hybrid KV Cache Manager，Sliding Window 层的**计算**仍然是正确的（attention mask 还是窗口范围内），只是**KV Cache 的内存**没有节省。

### 6.4 如何为 KV Connector 添加 HMA 支持？

vLLM 预留了扩展接口 [`SupportsHMA`](../vllm/distributed/kv_transfer/kv_connector/v1/base.py)，connector 开发者只需：

1. 继承 `SupportsHMA` 并实现 `request_finished_all_groups` 方法；
2. 在该方法中处理多个 KV Cache Group 的 block_ids；
3. 启动时使用 `--no-disable-hybrid-kv-cache-manager` 显式开启。

```python
class SupportsHMA(ABC):
    @abstractmethod
    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],   # 每个 group 的 block_ids
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        在请求的所有 KV Cache group 完成后调用（block 释放前）。
        返回 True 表示由 connector 异步释放 block。
        """
        raise NotImplementedError
```

调度器中的判断逻辑（[vllm/v1/core/sched/scheduler.py](../vllm/v1/core/sched/scheduler.py)）：

```python
if not isinstance(self.connector, SupportsHMA):
    # 非 HMA connector，断言只有一个 KV Cache Group
    assert len(self.kv_cache_config.kv_cache_groups) == 1
```

> vLLM 仓库的 PR #25712（_\[Core\]\[Hybrid allocator + kv connector 1/n\] Enable hybrid allocator + KV cache connector_）已于 2025 年 10 月合并，移除了 vLLM 侧对 HMA + Connector 的显式限制。然而，LMCache connector 本身尚未完全实现 `SupportsHMA` 接口（commit `919fe9b` 标注为 "remove hma support from LMCache for now"）。因此，在当前状态下开启 `--kv-transfer-config` 仍会自动关闭 HMA，导致 Mamba 混合模型（如 Qwen 3.5 MoE）依然无法使用。完整的 HMA + LMCache 支持仍需等待 LMCache connector 实现多 group 的 block_ids 处理逻辑。

---

## 7. 关键限制与已知问题

当前 Hybrid KV Cache Manager 存在以下已知限制：

| 限制                                           | 说明                                                                                                           |
| ---------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| HybridKVCacheCoordinator 只支持 2 种注意力类型 | Full + X，不支持 Full + SW + Mamba 等三类混合                                                                  |
| Mamba block_size 可能过大                      | Case 4 的 padding 策略可导致 block_size > 400，内存调度粗粒度                                                  |
| Chunked Local Attention + Eagle 不支持         | Llama 4 + Eagle 组合时 HMA 被强制关闭                                                                          |
| Chunked Local Attention 存在延迟回归           | 默认通过 `VLLM_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE=1` 手动开启                                       |
| ~~KV Transfer 与 HMA 互斥~~                    | **部分解决**（PR #25712，2025 年 10 月）vLLM 侧限制已移除，但 LMCache connector 尚未完全实现 `SupportsHMA`     |
| Mamba 混合模型 + KV Transfer 死局              | **仍然存在**LMCache connector 未实现多 group block_ids 处理逻辑                                                |
| Native KV Offloading 不支持 HMA                | vLLM 原生的 KV Cache Offloading（`--kv-offloading-size`）假设所有层共享同一 block table，无法处理多 group 场景 |
| LRU 全局共享                                   | 所有 KV Cache Group 共享同一个 LRU 逐出队列，无法按类型差异化逐出                                              |

---

## 8. 配置参数速查

以下是与 Hybrid KV Cache Manager 相关的配置参数及其含义：

| 参数                                                   | 说明                                    | 默认     |
| ------------------------------------------------------ | --------------------------------------- | -------- |
| `--disable-hybrid-kv-cache-manager`                    | 显式关闭 HMA                            | 自动判断 |
| `--no-disable-hybrid-kv-cache-manager`                 | 显式开启 HMA（配合 KV Transfer 使用）   | —        |
| `VLLM_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE=1` | 允许 Llama 4 使用 HMA（有延迟回归风险） | 关闭     |

当满足以下任一条件时，HMA 会被**自动关闭**：

- 使用了 `--kv-transfer-config`（且 connector 未实现 `SupportsHMA`）
- 开启了 KV Events（`--kv-events-config`）
- 在非 GPU 平台运行
- 使用 Chunked Local Attention + Eagle（Llama 4 + 投机解码）

---

## 9. 总结

Hybrid KV Cache Manager 是 vLLM V1 为混合注意力模型专门设计的显存优化机制。它通过 **KV Cache Group 分组 + 差异化 block 分配 + 跨 group Prefix Caching 交集算法**，避免了为 Sliding Window、Mamba、Local Chunked 等高效注意力层过度分配显存，可显著降低这类模型的 KV Cache 占用。

然而，由于 KV Transfer（尤其是 LMCache）的 connector 当前只支持单个 KV Cache Group，**混合注意力模型无法同时开启 KV Transfer 和 HMA**。对于 Sliding Window + Full 模型（Gemma-3 等），关闭 HMA 可以降级运行但浪费显存；而对于 **Mamba + Full 模型（Qwen 3.5 MoE、Jamba 等），则完全无法启动**——这是当前最严重的兼容性死局。要解决这一问题，需要 connector 开发者实现 `SupportsHMA` 接口，处理多 group 场景下的 block_ids 传递与生命周期管理。

尽管 vLLM 引擎侧已解除了对 HMA 与 Connector 联用的限制（PR #25712），但主流 Connector（如 LMCache）尚未完成适配。因此，目前混合注意力模型仍无法同时开启 KV Transfer 和 HMA，Mamba 混合模型的兼容性问题仍待 Connector 侧升级解决。

---

## 参考文献

- [1] vLLM 项目仓库：[vllm-project/vllm](https://github.com/vllm-project/vllm)
- [2] PR #25712 — \[Core\]\[Hybrid allocator + kv connector 1/n\] Enable hybrid allocator + KV cache connector：[vllm-project/vllm#25712](https://github.com/vllm-project/vllm/pull/25712)
- [3] LMCache 项目仓库：[LMCache](https://github.com/LMCache/LMCache)
- [4] LMCache 官方文档（vLLM 集成说明）：[LMCache Documentation](https://docs.lmcache.ai)
