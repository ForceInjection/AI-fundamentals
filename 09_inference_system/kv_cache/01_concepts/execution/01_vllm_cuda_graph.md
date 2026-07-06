# CUDA Graph 与 KV Cache：静态执行图如何容纳动态 Block Table？

CUDA Graph 将一系列 GPU 操作录制为静态执行图，然后反复重放——省去了 CPU launch kernel 的逐次开销，将 decode 延迟降低 20-40%。但 KV Cache 的 block table 是动态的：每次 decode step，新的 token 可能触发 block 分配、已完成的请求释放 block、prefix cache 命中改变 block 映射——这些变化改变了 attention kernel 需要访问的物理地址。CUDA Graph 录制时 block table 的内容，到重放时已经过时了。

vLLM V1 通过 **Piecewise CUDA Graph** 和 **block table 作为 Graph Input** 两条路径来调和这对矛盾。本文以 vLLM 源码为基准，拆解 Full / Piecewise / FULL_AND_PIECEWISE 三种模式如何与 KV Cache 的动态性共存。

> **前置阅读**：[PagedAttention 原理介绍](../basic/paged_attention.md) — block table 的虚拟→物理映射是本文讨论的"动态性"的核心来源。

---

## 一、CUDA Graph 为什么对 KV Cache "不友好"？

### 1.1 CUDA Graph 的基本约束

CUDA Graph 将一串 GPU kernel launch 录制成一个不可变的 DAG。录制期间，所有的输入指针、张量形状、内存地址都被固化为图的静态参数。重放（replay）时，CPU 只需将新的输入数据填入已录制的内存位置，GPU 即按图执行——不再逐次 launch kernel。

核心约束：**图录制时看到的内存布局必须与每次重放时一致。** 如果中间有新的内存分配、旧内存释放、或者输入张量的形状改变，图就是 stale 的，必须重新录制。

### 1.2 Block Table 是"图的外部变量"

PagedAttention 的核心是 block table——一个 `(num_requests, max_blocks)` 的映射表，将逻辑 token 位置映射到物理 KV block 地址。每次 decode step：

```text
Step N:   block_table = [block_0, block_1, block_2, ...]  ← decode token N 的 KV 追加
Step N+1: block_table = [block_0, block_1, block_2, block_3, ...]  ← 可能新增 block_3
          attention kernel 需要知道 block_table 才能找到 K 和 V 的位置
```

传统的 CPU launch 模式下，每次 step 都重新设置 kernel 参数，block_table 变化不是问题。但在 CUDA Graph 模式下，block_table 被录制为图的静态输入——录制时的 block_table 内容在重放时已经陈旧。如果 decode step N+1 恰好触发了新 block 分配（block_size=16 时每 16 步一次），block_table 就变了。

### 1.3 两条调和路径

vLLM V1 提供了两条路径来解决这个矛盾：

| 路径                                     | 思路                                                                                             |       代表模式       |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------ | :------------------: |
| **Block Table 作为 Graph Input**         | 把 block_table 声明为图的"可变输入"，每次重放前更新其内容                                        |   Full CUDA Graph    |
| **只 Capture 不依赖 block table 的计算** | 把 KV Cache 访问相关的操作（slot_mapping 计算、block_table 查找）留在图外，只 capture 纯计算部分 | Piecewise CUDA Graph |

两条路径在 vLLM V1 的默认配置 `FULL_AND_PIECEWISE` 下同时使用——Full 用于纯 decode batch，Piecewise 用于混合 batch（prefill + decode 共存）。

---

## 二、Full CUDA Graph：Block Table 作为可变输入

### 2.1 机制

Full CUDA Graph 录制整个 decode forward pass，但将 block_table 和 slot_mapping 定义为图的**可变输入缓冲区**——每次重放前由 CPU 更新其内容，而非录制时固化。每次重放前，CPU 将新的 block_table 数据填入已录制的 buffer，GPU 再按图执行[^1]。

```text
录制阶段：
  capture batch_size=8 的 decode forward
  block_table → 作为 "graph input" 录制（不固化具体值）
  slot_mapping → 作为 "graph input" 录制

重放阶段（每次 decode step）：
  CPU 更新 block_table 内容（新 block 分配 / 旧 block 释放）
  CPU 重新计算 slot_mapping[query_pos] → physical_addr
  GPU replay CUDA Graph（使用更新后的 block_table 和 slot_mapping）
```

关键是在重放前，slot_mapping 被重新计算——它建立了"query 的第 i 个 token 需要访问物理地址 addr 处的 KV block"的映射关系。attention kernel 在图中按 slot_mapping 查找 KV，不直接关心 block_table 是否变化。

### 2.2 约束：固定 Batch Size

Full CUDA Graph 要求 batch size 在录制和重放之间保持不变。因为图中所有的中间张量形状都依赖于 batch size——Q 的第一维是 batch_size，attention score 的形状是 `(batch_size, num_heads, 1, seq_len)`。

vLLM 通过 **多尺寸预录制** 来应对 batch size 变化：预先为常见的 batch size（如 `[1, 2, 4, 8, 16, 24, 32, ...]`，配置项 `cudagraph_capture_sizes`）各录制一份 CUDA Graph。运行时根据当前实际的 batch size 选择最接近的预录制尺寸，不足的部分用 padding 填充：

```text
当前 batch 有 5 个请求 → 选择 capture_size=8 的图
  req 0-4: 真实请求
  req 5-7: padding（slot_mapping 指向 dummy block，attention 结果被忽略）
```

padding 引入了一定的计算浪费（~40% 在上例中），但换来了 CPU launch 开销的消除。`cudagraph_capture_sizes` 的默认集合为 `[1, 2, 4] + list(range(8, 256, 8))`，覆盖了从单请求到大批量的常见场景[^1]。

### 2.3 触发 Re-capture 的条件

当以下任一条件满足时，已录制的 CUDA Graph 变为 stale，必须重新录制：

- **Batch size 变化超出预录制范围**：当前 batch 中有 257 个请求，但最大 capture_size 是 256
- **Attention backend 的 block table 格式变化**：如启用新的量化模式（FP8→NVFP4），block table 的物理布局改变
- **请求的 decode query length 变化**：如果开启了 speculative decoding，`num_spec_tokens` 改变导致 query_len 不同

Re-capture 的代价不低（~100-200ms），因此 vLLM 选择了 Piecewise 作为混合场景下的保底方案，避免频繁 re-capture。

---

## 三、Piecewise CUDA Graph：只 Capture 纯计算

### 3.1 机制

Piecewise CUDA Graph 不录制整个 forward pass——它只 capture attention 中不依赖 block table 的计算密集部分。具体来说：

```text
Full Graph 录制的范围：
  [slot_mapping 计算] → [block_table 查找 + QKV 加载] → [Q·K^T] → [SoftMax] → [A·V] → [output projection]

Piecewise Graph 录制的范围：
  （slot_mapping 计算：动态，不录制）
  （block_table 查找 + QKV 加载：动态，不录制）
  [Q·K^T] ────────────┐
  [SoftMax]            ├── 录制为 Piecewise CUDA Graph
  [A·V]                │
  [output projection] ─┘
```

Piecewise 的核心思想是：**只录制那些"输入张量形状固定、与 block table 内容无关"的数学运算**。Q、K、V 已经通过动态路径从 block table 中加载到寄存器/SRAM 中，后续的矩阵乘法、SoftMax、加权求和、输出投影等操作不再关心 block table——它们只关心"K 和 V 已经就位"这个事实。

### 3.2 与 KV Cache 的交互

Piecewise 对 KV Cache 管理的意义在于：**它彻底消除了 re-capture 的需求**。无论 block table 如何变化、batch size 如何波动、prefill chunk 和 decode 如何混合——Piecewise 录制的纯计算图总是有效的。代价是每次重放前仍需要 CPU 动态计算 slot_mapping 和 block_table 查找，这部分 CPU 开销无法消除。

在 vLLM V1 中，`FULL_AND_PIECEWISE` 是默认模式——它根据 batch 的类型自动切换：

- 纯 decode batch（`uniform_decode`）：使用 Full CUDA Graph，batch size 固定，最大化性能
- 混合 batch（prefill + decode 共存）：回退到 Piecewise，适应动态性

### 3.3 `requires_piecewise_for_cudagraph`

KV Connector 实现者可以通过 `requires_piecewise_for_cudagraph` 属性告知 vLLM：本 connector 的异步 KV 加载操作无法被 CUDA Graph 捕获，必须走 Piecewise 路径。例如，如果一个 connector 在 attention 计算中间需要动态等待网络数据（`wait_for_layer_load`），这个等待操作不能被录制为 CUDA Graph——因为 CUDA Graph 不能包含 host-device 同步或外部事件等待节点——此时必须使用 Piecewise 模式[^2]。

这是 KV Cache 的异步传输（Prefetch）与 CUDA Graph 之间的直接交互点：Prefetch 的动态性迫使 CUDA Graph 退化为更灵活的模式。

---

## 四、取舍

### 4.1 Full vs Piecewise vs FULL_AND_PIECEWISE

| 模式                           |         录制范围          |            性能             |                Re-capture 风险                | 适用场景                      |
| ------------------------------ | :-----------------------: | :-------------------------: | :-------------------------------------------: | ----------------------------- |
| **Full**                       |    整个 decode forward    | 最高（CPU launch 完全消除） | 高（batch size / block table 格式变化时触发） | 稳态大批量 decode             |
| **Piecewise**                  | 仅 attention 计算密集部分 |  中（保留部分 CPU launch）  |                      无                       | 混合 prefill/decode，抖动场景 |
| **FULL_AND_PIECEWISE**（默认） |        自适应切换         |       最优（自动化）        |                 仅 Full 部分                  | 所有生产场景                  |
| **NONE**                       |             —             |            最低             |                       —                       | Debug / 兼容性                |

### 4.2 Padding 的显存代价

Full CUDA Graph 的 padding 不仅浪费计算，还占用 KV Cache 空间——padding 请求的 slot_mapping 指向专用的 dummy block，这些 block 虽然不存储有效数据，但仍占据物理页。在 `cudagraph_capture_sizes` 包含 256 的大尺寸时，padding 的块浪费在极端情况下（当前 batch 仅 1 个请求，但使用 capture_size=256 的图）可达 ~255 个 dummy block。vLLM 通过 `cudagraph_capture_sizes` 的细粒度配置（`[1, 2, 4, 8, 16, ...]`）来限制 padding 的最大浪费不超过 ~50%（最差情况下上一个 capture_size 翻倍前的间隙）。

### 4.3 一句话总结

**CUDA Graph 让 GPU 不再每次等 CPU launch kernel，代价是要求计算图静态不变。KV Cache 的 block table 每步都可能变——Piecewise 把 block table 相关的操作留在图外解决静态性约束，Full 把 block table 作为图的输入参数绕过录制固化，FULL_AND_PIECEWISE 在两者之间自适应切换。** 理解这个"静态图 + 动态数据"的调和机制，是理解推理引擎底层执行模型如何与 KV Cache 管理交互的关键。

---

## 相关阅读

- [PagedAttention 原理介绍](../basic/paged_attention.md) — Block table 的虚拟→物理映射机制
- [投机解码如何与 KV Cache 交互](../scheduling/02_vllm_spec_decode.md) — `num_spec_tokens` 改变 query_len 导致 re-capture
- [KV Cache Prefetching](../offloading/03_kv_cache_prefetching.md) — `requires_piecewise_for_cudagraph` 与异步 KV 加载的交互
- [vLLM CUDA Graph 深度解析](../../../vllm/module_analysis/cuda_graph_deep_dive.md) — 从模型执行视角的完整 CUDA Graph 分析

[^1]: vLLM 编译配置 [`vllm/config/compilation.py`](https://github.com/vllm-project/vllm/blob/main/vllm/config/compilation.py) — `CUDAGraphMode` 枚举定义 `NONE`/`PIECEWISE`/`FULL`/`FULL_DECODE_ONLY`/`FULL_AND_PIECEWISE` 五种模式；`cudagraph_capture_sizes` 默认 `[1, 2, 4] + list(range(8, 256, 8))`；`FULL_AND_PIECEWISE` 为 V1 默认。

[^2]: vLLM V1 KV Connector API [`vllm/distributed/kv_transfer/kv_connector/v1/base.py`](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/kv_transfer/kv_connector/v1/base.py) — `requires_piecewise_for_cudagraph` 属性允许 connector 实现者告知 vLLM 必须使用 Piecewise 模式；当 connector 的异步操作无法被 CUDA Graph 捕获时返回 `True`。
