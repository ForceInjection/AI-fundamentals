# vLLM 内置 KV Cache Offloading 模块解析

## 1. 概述

vLLM V1 引擎提供了**原生的 KV Cache CPU Offloading 功能**（简称"native offloading"），允许将 GPU 上的 KV Cache 卸载到 CPU 内存，从而在有限的 GPU 显存上运行更大的模型或处理更长的序列。

---

## 2. 核心架构

Native Offloading 模块分为 Scheduler 侧的 OffloadingManager（决策层，负责缓存淘汰和 block 追踪）和 Worker 侧的 OffloadingHandler（执行层，负责 GPU-CPU 异步数据传输）两个核心层次。

### 2.1 组件层次

以下是 Native Offloading 模块的组件层次结构：

```text
Scheduler (OffloadingManager)
    ├── ARCOffloadingManager (ARC 缓存淘汰策略)
    └── LRUOffloadingManager (LRU 缓存淘汰策略)
            │
            └── Backend (CPUBackend)
                    │
Worker (OffloadingHandler)
    ├── CpuGpuOffloadingHandlers
    │   ├── gpu_to_cpu_handler (GPU → CPU offload)
    │   └── cpu_to_gpu_handler (CPU → GPU load)
    └── SingleDirectionOffloadingHandler
```

### 2.2 关键文件路径

Native Offloading 的核心代码分布在以下文件中：

| 文件                                                                                                                                           | 职责                                          |
| ---------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------- |
| [vllm/v1/kv_cache_interface/abstract.py](https://github.com/vllm-project/vllm/blob/v0.18.0/vllm/v1/kv_cache_interface/abstract.py)             | `OffloadingManager` 抽象基类定义              |
| [vllm/v1/kv_cache_interface/arc_manager.py](https://github.com/vllm-project/vllm/blob/v0.18.0/vllm/v1/kv_cache_interface/arc_manager.py)       | ARC (Adaptive Replacement Cache) 淘汰策略实现 |
| [vllm/v1/kv_cache_interface/lru_manager.py](https://github.com/vllm-project/vllm/blob/v0.18.0/vllm/v1/kv_cache_interface/lru_manager.py)       | LRU 淘汰策略实现                              |
| [vllm/v1/kv_cache_interface/backends/cpu.py](https://github.com/vllm-project/vllm/blob/v0.18.0/vllm/v1/kv_cache_interface/backends/cpu.py)     | CPU 后端存储管理                              |
| [vllm/v1/kv_cache_interface/worker/cpu_gpu.py](https://github.com/vllm-project/vllm/blob/v0.18.0/vllm/v1/kv_cache_interface/worker/cpu_gpu.py) | GPU-CPU 数据传输处理器                        |
| [vllm/v1/kv_cache_interface/spec.py](https://github.com/vllm-project/vllm/blob/v0.18.0/vllm/v1/kv_cache_interface/spec.py)                     | Offloading 配置规范                           |
| [vllm/v1/kv_cache_interface/mediums.py](https://github.com/vllm-project/vllm/blob/v0.18.0/vllm/v1/kv_cache_interface/mediums.py)               | `BlockIDsLoadStoreSpec` 及子类定义            |
| [vllm/config/cache.py](https://github.com/vllm-project/vllm/blob/v0.18.0/vllm/config/cache.py)                                                 | `CacheConfig` 中的 offloading 配置项          |

---

## 3. 配置参数

Native Offloading 通过 `CacheConfig` 中的参数进行配置，同时支持命令行方式启用。

### 3.1 基本配置

在 `CacheConfig` 中定义：

```python
@config
class CacheConfig:
    kv_offloading_size: float | None = None
    """KV cache offloading buffer 总大小（GiB）。当 TP > 1 时，这是所有 TP rank 的总和。
    默认为 None（不启用）。设置后，vLLM 将使用指定的 backend 启用 KV cache CPU offloading。"""

    kv_offloading_backend: KVOffloadingBackend = "native"
    """KV cache offloading 的后端选择：
    - "native": vLLM 原生 CPU offloading
    - "lmcache": LMCache connector
    """
```

### 3.2 Native Backend 配置

当 `kv_offloading_backend="native"` 时，vLLM 会自动配置 `OffloadingConnector`：

```python
# vllm/config/vllm.py 中的自动配置逻辑
if kv_offloading_backend == "native":
    self.kv_transfer_config.kv_connector = "OffloadingConnector"
    self.kv_transfer_config.kv_connector_extra_config.update(
        {"cpu_bytes_to_use": kv_offloading_size * (1 << 30)}
    )
```

### 3.3 命令行使用示例

```bash
# 启用 native offloading，分配 10GB CPU 内存用于 KV cache
vllm serve google/gemma-3b \
    --kv-offloading-size 10 \
    --kv-offloading-backend native

# 配合混合模型使用（需要关闭 HMA）
vllm serve google/gemma-3-27b-it \
    --kv-offloading-size 20 \
    --kv-offloading-backend native \
    --disable-hybrid-kv-cache-manager
```

---

## 4. 工作原理

Native Offloading 的工作流贯穿 Scheduler 侧的决策和 Worker 侧的异步数据传输两个层面。

### 4.1 Scheduler 侧（OffloadingManager）

`OffloadingManager` 提供以下核心原语：

1. **lookup()**：查找请求的 prefix 中有多少 block 已经 offloaded 到 CPU
2. **prepare_store()**：决定哪些 GPU block 需要 offload，返回 eviction 列表
3. **prepare_load()**：准备从 CPU 加载 block 到 GPU
4. **touch()**：标记 block 为最近使用，更新 LRU/ARC 状态

### 4.2 Worker 侧（CpuGpuOffloadingHandlers）

使用 CUDA 流异步执行 GPU-CPU 数据传输：

```python
class SingleDirectionOffloadingHandler:
    def transfer_async(self, job_id, transfer_spec):
        src_spec, dst_spec = transfer_spec

        # 从 pool 中复用或按需创建 CUDA stream
        stream = self._stream_pool.pop() if self._stream_pool else torch.cuda.Stream()

        # 等待前一个 job 完成（保证顺序）
        if self._transfers:
            last_event = self._transfers[-1].end_event
            stream.wait_event(last_event)

        # 执行 swap_blocks（底层调用 ops.swap_blocks）
        with torch.cuda.stream(stream):
            for src_tensor, dst_tensor, block_size_in_bytes in zip(
                self.src_tensors, self.dst_tensors, self.block_size_in_bytes
            ):
                ops.swap_blocks(src_tensor, dst_tensor,
                                block_size_in_bytes, src_to_dst_tensor)
```

### 4.3 底层操作

核心是 [`swap_blocks`](https://github.com/vllm-project/vllm/blob/v0.18.0/vllm/_custom_ops.py#L2580-L2606) CUDA kernel：

```python
def swap_blocks(
    src: torch.Tensor,         # 源 tensor（GPU 或 CPU）
    dst: torch.Tensor,         # 目标 tensor
    block_size_in_bytes: int,  # 每个 block 的字节数
    block_mapping: torch.Tensor,  # shape: (num_blocks, 2), dtype=int64
) -> None:
    """
    将 src 中的特定 blocks 复制到 dst。
    block_mapping[i] = [src_block_id, dst_block_id]
    """
    torch.ops._C_cache_ops.swap_blocks(src, dst, block_size_in_bytes, block_mapping)
```

---

## 5. 缓存淘汰策略

Native Offloading 提供 LRU 和 ARC 两种缓存淘汰策略，默认使用 LRU，可通过 `kv_connector_extra_config` 中的 `eviction_policy` 参数切换为 ARC。

### 5.1 LRU (Least Recently Used)

最简单的策略：淘汰最久未使用的 block。

### 5.2 ARC (Adaptive Replacement Cache)

更智能的自适应策略，动态平衡"最近使用"和"频繁使用"：

```python
class ARCOffloadingManager:
    # T1: 最近访问（只访问一次）
    # T2: 频繁访问（访问多次）
    # B1: T1 的 ghost list（记录被 T1 淘汰的 block hash）
    # B2: T2 的 ghost list（记录被 T2 淘汰的 block hash）

    def touch(self, block_hashes):
        for block_hash in reversed(list(block_hashes)):
            if block_hash in self.t1:
                block = self.t1.pop(block_hash)
                if block.is_ready:
                    # 从 T1 移动到 T2（promotion）
                    self.t2[block_hash] = block
                else:
                    # 未就绪的 block 保留在 T1（刚存入尚未完成传输）
                    self.t1[block_hash] = block
            elif block_hash in self.t2:
                # T2 内移动到末尾（MRU 位置）
                self.t2.move_to_end(block_hash)
            elif block_hash in self.b1:
                # B1 hit: 自适应增加 T1 目标大小
                delta = max(1, len(self.b2) / len(self.b1))
                self.target_t1_size = min(self.target_t1_size + delta, self.cache_capacity)
            elif block_hash in self.b2:
                # B2 hit: 自适应减少 T1 目标大小
                delta = max(1, len(self.b1) / len(self.b2))
                self.target_t1_size = max(self.target_t1_size - delta, 0)

    def prepare_store(self, block_hashes):
        # 自适应决策：从 T1 还是 T2 淘汰
        if len(self.t1) >= int(self.target_t1_size):
            # 从 T1 淘汰（跳过 ref_cnt > 0 的 block），加入 B1
            ...
        else:
            # 从 T2 淘汰，加入 B2
            ...
```

---

## 6. 与 Hybrid KV Cache Manager 的关系

本章讨论 Native Offloading 与 Hybrid KV Cache Manager 之间的兼容性限制及其原因。

### 6.1 当前限制

**Native Offloading 目前不支持 Hybrid KV Cache Manager**。原因：

1. OffloadingManager 假设所有 layer 共享同一个 block table
2. Hybrid 模型有多个 KV Cache Group，每个 group 有不同的 block allocation
3. 当前的 `BlockIDsLoadStoreSpec` 只支持单个 `block_ids` 数组

### 6.2 代码证据

在 [vllm/v1/kv_cache_interface/mediums.py](https://github.com/vllm-project/vllm/blob/v0.18.0/vllm/v1/kv_cache_interface/mediums.py) 中，`CPULoadStoreSpec` 继承自 `BlockIDsLoadStoreSpec`，仅支持单个 `block_ids` 列表：

```python
class BlockIDsLoadStoreSpec(LoadStoreSpec, ABC):
    """Spec for loading/storing KV blocks from given block numbers."""
    def __init__(self, block_ids: list[int]):
        # 内部转换为 numpy 数组，只支持单个 block_ids 列表
        self.block_ids = np.array(block_ids, dtype=np.int64)

class CPULoadStoreSpec(BlockIDsLoadStoreSpec):
    """Spec for loading/storing a KV block to CPU memory."""
    ...
```

对比 Hybrid KV Cache Manager 需要的格式：

```python
# Hybrid 模型需要 tuple[list[int], ...]
block_ids: tuple[list[int], ...]  # 每个 group 一个 list
```

### 6.3 官方文档说明

在 [docs/design/metrics.md](https://github.com/vllm-project/vllm/blob/v0.18.0/docs/design/metrics.md#kv-cache-offloading) 中提到：

> In this mode, when a request is preempted (e.g. to make room in KV cache to complete other requests), we swap kv cache blocks out to CPU memory. This is also known as "KV cache offloading" and is configured with `--swap-space` and `--preemption-mode`.

这是 V0 时代的 legacy 描述。V1 的 native offloading 是更通用的机制，不仅用于 preemption，也用于扩展 KV cache 容量。

---

## 7. 测试用例

Native Offloading 的测试覆盖了单元测试和端到端两个层面。

### 7.1 基础功能测试

文件：[tests/v1/kv_cache_interface/test_cpu_manager.py](https://github.com/vllm-project/vllm/blob/v0.18.0/tests/v1/kv_cache_interface/test_cpu_manager.py)

```python
def test_cpu_manager():
    """测试 LRUOffloadingManager + CPUBackend 的基础功能"""
    block_size = 256
    cpu_backend = CPUBackend(block_size=block_size, num_blocks=4)
    cpu_manager = LRUOffloadingManager(cpu_backend, enable_events=True)

    # prepare_store 将 block [1, 2] 标记为待存储
    prepare_store_output = cpu_manager.prepare_store(to_hashes([1, 2]))

    # complete_store 标记存储完成，block 变为 ready 状态
    cpu_manager.complete_store(to_hashes([1, 2]))

    # lookup [1, 2] -> 此时 block 已 ready，应返回命中数 2
    assert cpu_manager.lookup(to_hashes([1, 2])) == 2
```

### 7.2 端到端测试

文件：[tests/v1/kv_cache_interface/test_cpu_offloading.py](https://github.com/vllm-project/vllm/blob/v0.18.0/tests/v1/kv_cache_interface/test_cpu_offloading.py)

```python
def test_cpu_offloading(cpu_block_size, attn_backend):
    """测试 OffloadingConnector + CPUOffloadingSpec 的端到端功能"""
    kv_transfer_config = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "cpu_bytes_to_use": 500 << 20,   # 500 MiB
            "block_size": cpu_block_size,
        },
    )

    llm = LLM(
        model="meta-llama/Llama-3.2-1B-Instruct",
        kv_transfer_config=kv_transfer_config,
        gpu_memory_utilization=0.5,
    )

    # 生成输出，验证 offloading 功能正常
    outputs = llm.generate(prompts, sampling_params)
```

---

## 8. 性能优化建议

以下是使用 Native Offloading 时的性能调优建议。

### 8.1 块大小选择

CPU 侧的 `offloaded_block_size` 通过 `kv_connector_extra_config` 中的 `block_size` 参数配置，必须是 GPU `block_size` 的整数倍：

```python
# OffloadingSpec.__init__ 中的块大小配置
# 默认与 GPU block_size 相同，也可通过 extra_config 自定义
self.offloaded_block_size = int(
    self.extra_config.get("block_size", self.gpu_block_size)
)
assert self.offloaded_block_size % self.gpu_block_size == 0
```

较大的 CPU block size 可以减少元数据开销，但会降低粒度。推荐配置：

- GPU block_size: 16 或 32
- offloaded_block_size: 与 GPU 相同或 2-3 倍

### 8.2 传输并行化

`SingleDirectionOffloadingHandler` 使用 CUDA stream pool 实现并行传输：

```python
class SingleDirectionOffloadingHandler:
    def __init__(self, ...):
        # stream pool 初始为空，按需创建并在完成后回收复用
        self._stream_pool: list[torch.cuda.Stream] = []
        self._event_pool: list[torch.Event] = []

    def transfer_async(self, job_id, transfer_spec):
        # 优先复用已完成的 stream，否则创建新的
        stream = self._stream_pool.pop() if self._stream_pool else torch.cuda.Stream()
        ...
```

stream 在传输完成后会自动回收到 pool 中，实现按需创建 + 复用的策略。

### 8.3 避免与 PagedAttention 冲突

Native offloading 使用 `ops.swap_blocks`，与 PagedAttention 的 block table 机制兼容。但要确保：

1. CPU block size 是 GPU block size 的整数倍
2. offloading 不影响 GPU prefix caching 的 hash 计算

---

## 9. 与 LMCache 的对比

Native Offloading 与 LMCache Connector 在存储后端、缓存策略、适用场景等方面存在显著差异：

| 特性            | Native Offloading    | LMCache Connector              |
| --------------- | -------------------- | ------------------------------ |
| **存储后端**    | CPU 内存（pinned）   | CPU / Disk / Redis / 远程节点  |
| **缓存策略**    | LRU / ARC            | 可插拔（支持自定义）           |
| **地址空间**    | Block ID（物理地址） | Token Hash（逻辑地址）         |
| **多机支持**    | 不支持（单机）       | 支持（集群）                   |
| **内容感知**    | 无去重               | 全局去重                       |
| **Hybrid 支持** | 不支持               | 部分支持（需实现 SupportsHMA） |
| **配置复杂度**  | 低（单参数）         | 高（需配置 backend）           |

---

## 10. 未来发展方向

Native Offloading 未来有以下几个主要的发展方向。

### 10.1 Hybrid KV Cache Manager 支持

需要修改：

1. `BlockIDsLoadStoreSpec` 支持多 group 的 `block_ids: tuple[np.ndarray, ...]`
2. `OffloadingManager` 感知不同 attention type 的 eviction 策略
3. `CpuGpuOffloadingHandlers` 处理跨 group 的 block mapping

### 10.2 多层级缓存

结合 native offloading + LMCache：

```text
GPU KV Cache (热数据)
    ↓
CPU Memory (温数据，native offloading)
    ↓
Disk / Remote (冷数据，LMCache)
```

### 10.3 动态块大小

当前 CPU block size 是固定的。未来可以支持：

- Sliding Window 层使用小 block（细粒度）
- Full Attention 层使用大 block（减少元数据）

---

## 11. 总结

vLLM 的 **Native KV Cache Offloading** 是一个轻量级、高性能的 CPU offloading 解决方案：

- **简单易用**：单参数 `--kv-offloading-size` 即可启用
- **零侵入**：与现有 PagedAttention、Prefix Caching 完全兼容
- **灵活策略**：支持 LRU / ARC 两种缓存淘汰算法
- **异步传输**：多 CUDA stream 并行，隐藏 PCIe 传输延迟

**主要限制**：

- 暂不支持 Hybrid KV Cache Manager（Gemma-3、Qwen 3.5 MoE 等模型无法使用）
- 仅支持 CPU 内存，不支持 Disk / 远程存储
- 无内容感知去重，可能浪费存储空间

对于生产环境，如果需要更强大的 offloading 能力（如多机共享、磁盘缓存），建议使用 LMCache Connector；如果只需要简单的单机扩展，native offloading 是更轻量的选择。

---

## 参考文献

- [1] vLLM 项目仓库：[vllm](https://github.com/vllm-project/vllm)
- [2] vLLM Metrics 设计文档（含 KV Cache Offloading 历史背景说明）：[docs/design/metrics.md](https://github.com/vllm-project/vllm/blob/v0.18.0/docs/design/metrics.md#kv-cache-offloading)
- [3] LMCache 项目仓库：[LMCache](https://github.com/LMCache/LMCache)
- [4] ARC 算法论文：Nimrod Megiddo, Dharmendra S. Modha, "ARC: A Self-Tuning, Low Overhead Replacement Cache", FAST 2003
