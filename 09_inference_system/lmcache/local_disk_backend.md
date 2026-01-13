# LocalDiskBackend 源码分析

[< 返回存储架构概览](./lmcache_storage_overview.md)

`LocalDiskBackend` 构成了 LMCache 多级存储架构中的 **L3 本地磁盘缓存层**。它利用本地磁盘（强烈推荐 NVMe SSD）作为高容量、低成本的扩展缓冲池，主要用于接纳从 L1/L2 内存层溢出（Evict）的冷热数据。

与传统的文件存储不同，`LocalDiskBackend` 的设计目标并非数据的长期归档，而是作为 **Runtime Swap** 扩展 GPU/CPU 有限的内存容量。为了弥补磁盘 I/O 相对内存的延迟劣势，该模块深度集成了 **O_DIRECT 直通 I/O**、**异步优先级调度流水线** 以及 **精细的内存生命周期管理** 等高性能技术。

> **架构定位**: 在 `lmcache_storage_overview.md` 定义的层级中，它是 L3 层。需要特别注意的是，它**不具备断电持久化能力**（索引纯内存维护），且**强依赖** L1 层（`LocalCPUBackend`）作为内存分配器（Allocator）和数据传输中转站。

---

## 1. 存储架构与文件布局

针对 AI 推理场景中 KV Cache 数据的不可变性及“一次写入，多次读取”（WORM）的访问特征，LMCache 设计了极简的底层存储格式，旨在消除传统存储方案中的元数据开销以最大化 I/O 吞吐量。

### 1.1 扁平化文件结构

`LocalDiskBackend` 摒弃了复杂的目录树，采用扁平化的文件结构。每个 KV Cache Chunk 对应一个独立的磁盘文件。

- **存储路径**: 由配置项 `config.local_disk` 指定（如 `/mnt/nvme/lmcache`）。
- **文件名规则**: 在 `_key_to_path` 方法中实现，基于 `CacheEngineKey` 的字符串表示，将 `/` 替换为 `-`，并追加 `.pt` 后缀。
  - Key: `model/layer/chunk_hash`
  - Filename: `model-layer-chunk_hash.pt`

代码实现参考 [\_key_to_path](../lmcache/v1/storage_backend/local_disk_backend.py#L174-L178)：

```python
    def _key_to_path(
        self,
        key: CacheEngineKey,
    ) -> str:
        return os.path.join(self.path, key.to_string().replace("/", "-") + ".pt")
```

### 1.2 Raw Bytes 存储格式

为了极致的 I/O 性能，磁盘文件**不包含任何元数据头**（Header），仅存储 Tensor 的原始字节流（Raw Bytes）。这是通过 `write_file` 和 `read_file` 方法直接操作 `MemoryObj` 的 `byte_array` 来实现的。

- **写入**: 直接将内存中 `MemoryObj` 的 `byte_array` 写入磁盘。代码参考 [write_file](../lmcache/v1/storage_backend/local_disk_backend.py#L547-L561)。
- **读取**: 由于文件不自描述，读取时必须依赖内存中的**索引元数据**来获知数据的 Shape、Dtype 和 Format，才能正确地将字节流还原为 Tensor。代码参考 [read_file](../lmcache/v1/storage_backend/local_disk_backend.py#L563-L591)。

这种设计消除了序列化/反序列化的 CPU 开销，实现了 Memory-to-Disk 的零拷贝传输（配合 O_DIRECT）。

---

## 2. 索引机制与元数据管理

由于底层采用无 Header 的 Raw Bytes 格式存储，文件本身不具备自描述性，因此 `LocalDiskBackend` 必须在内存中维护一份完整的元数据索引，以实现数据的正确寻址与 Tensor 重构。

### 2.1 内存索引结构 (In-Memory Index)

为了在无元数据头的扁平文件中快速定位和还原 Tensor，`LocalDiskBackend` 在内存中维护了一个全局的元数据索引。这个索引是整个后端正常工作的核心，它弥补了文件系统无法自描述的缺陷。

- **索引结构**: `self.dict` (线程安全字典)。初始化时通过 `self.cache_policy.init_mutable_mapping()` 创建，参考 [\_\_init\_\_](../lmcache/v1/storage_backend/local_disk_backend.py#L113)。
- **映射关系**: `CacheEngineKey` -> `DiskCacheMetadata`。
- **DiskCacheMetadata 核心字段**: (定义于 [lmcache/utils.py](../lmcache/utils.py#L257-L266))
  - `path`: 文件绝对路径。
  - `size`: 数据字节大小。
  - `shape`: Tensor 形状 (关键，用于还原)。
  - `dtype`: 数据类型 (关键，用于还原)。
  - `fmt`: 内存格式 (如 `KV_BLOB` 或 `VLLM`)。
  - `cached_positions`: 缓存的位置信息。
  - `pin_count`: 引用计数，防止在使用时被驱逐。

### 2.2 并发控制与细粒度锁

`LocalDiskBackend` 使用 `self.disk_lock` 互斥锁来确保多线程环境下索引读写的一致性。

**关键设计：细粒度锁 (Fine-grained Locking)**：为了最大化并发性能，系统严格遵循**锁仅保护快速的内存元数据操作，而将耗时的磁盘 I/O 置于临界区（Critical Section）外执行**的原则。

以下面的 `get_blocking` 方法为例，我们可以清晰地看到这一模式。代码参考 [get_blocking](../lmcache/v1/storage_backend/local_disk_backend.py#L356-L384)：

```python
    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        # 1. 获取锁：进入临界区
        self.disk_lock.acquire()

        if key not in self.dict:
            self.disk_lock.release()
            return None

        # 2. 内存操作：更新 LRU 和拷贝元数据 (速度快，微秒级)
        self.cache_policy.update_on_hit(key, self.dict)
        disk_meta = self.dict[key]
        path = disk_meta.path
        # ... (获取 shape, dtype 等)

        # 3. 释放锁：离开临界区
        self.disk_lock.release()

        # 4. 磁盘 I/O：在锁外执行 (速度慢，毫秒级)
        # 此时其他线程可以继续访问 self.dict
        memory_obj = self.load_bytes_from_disk(
            key, path, dtype=dtype, shape=shape, fmt=fmt
        )

        return memory_obj
```

### 2.3 设计权衡：为何不持久化？ (Design Rationale)

深入分析源码可以发现，`self.dict` 是纯内存结构，且在 `LocalDiskBackend` 初始化时并没有从磁盘加载元数据的逻辑。这意味着一旦进程重启，之前的磁盘缓存将全部失效（即使文件还在）。

**设计考量**:

1. **定位为 Runtime Swap**: 目前 LMCache 将本地磁盘视为内存的扩展（类似 Swap 分区），而非持久化数据库。其生命周期通常与推理服务进程绑定。
2. **性能权衡**: 如果要支持持久化，必须在每次写入时同步更新元数据文件（如 SQLite 或 JSON），这会引入额外的 I/O 开销（尤其是小 IOPS），削弱 `O_DIRECT` 带来的高吞吐优势。
3. **数据时效性**: KV Cache 通常与特定的 Decoding Context 绑定，重启后之前的 Cache 往往不再适用（除非是 Prefix Caching 场景，但目前实现尚未针对此优化持久化），因此持久化的收益有限。

**潜在隐患**:

- **磁盘空间泄露**: 如果进程非正常退出（Crash），遗留的 `.pt` 文件不会被自动清理，长期运行可能导致磁盘占满。生产环境建议配合 `tmpwatch` 或启动脚本进行清理。

---

## 3. I/O 核心机制：O_DIRECT 与内存管理

本节深入剖析 `LocalDiskBackend` 的 I/O 加速机制，重点阐述其如何利用 `O_DIRECT` 技术消除操作系统页缓存带来的双重缓冲（Double Buffering）开销，并结合引用计数策略解决异步并发场景下的内存生命周期管理难题。

### 3.1 O_DIRECT 直通 I/O

Linux 的标准 I/O 会将数据缓存在 Page Cache 中，这在 KV Cache 场景下会导致双重缓存（LMCache 已有 L1 内存池）和额外的内存拷贝。

`LocalDiskBackend` 在 [write_file](../lmcache/v1/storage_backend/local_disk_backend.py#L547-L561) 和 [read_file](../lmcache/v1/storage_backend/local_disk_backend.py#L563-L591) 中实现了 `O_DIRECT` 模式。

以下是 `write_file` 的核心实现，清晰展示了 `O_DIRECT` 的启用条件与调用方式：

```python
    def write_file(self, buffer, path):
        # ...
        size = len(buffer)

        # 1. 条件检查：必须同时满足 对齐要求 (fblock_aligned) 和 配置开启 (use_odirect)
        # O_DIRECT 要求内存地址和大小通常需要对齐到 512 或 4096 字节
        if size % self.os_disk_bs != 0 or not self.use_odirect:
            # Fallback: 使用标准 Buffered I/O
            with open(path, "wb") as f:
                f.write(buffer)
        else:
            # 2. O_DIRECT 路径
            # os.O_DIRECT: 绕过 Page Cache
            # os.O_WRONLY | os.O_CREAT: 写模式，不存在则创建
            fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_DIRECT, 0o644)
            try:
                os.write(fd, buffer)
            finally:
                os.close(fd)
```

**关键点**:

- **对齐检查**: `size % self.os_disk_bs == 0`。这是 `O_DIRECT` 的硬性要求，否则内核会报错 `EINVAL`。
- **降级机制**: 如果条件不满足（例如数据块大小未对齐），代码会自动降级为普通的文件 I/O，保证了系统的鲁棒性。

### 3.2 完整写入流程：驱逐、锁定与异步 I/O

`LocalDiskBackend` 的写入流程（Put）不仅仅是简单的 I/O 提交，还涉及**同步驱逐**和**内存生命周期管理**。这是一个**混合了同步阻塞与异步执行**的复杂过程：

**写入流程 (Put) 详解**:

1. **任务去重 (Deduplication)**:

   - 首先检查该 Key 是否已有正在进行的写入任务。如果有，直接返回，避免重复 I/O。
   - 详见 [4.2 写入任务去重](#42-写入任务去重-deduplication)。

2. **空间检查与同步驱逐 (Synchronous Eviction)**:

   - **这是 Put 流程中唯一的阻塞点**。
   - 在提交任务前，检查磁盘缓存空间是否充足。如果不足，主线程会**阻塞**，循环调用 `cache_policy.get_evict_candidates` 并执行删除，直到腾出足够空间。
   - **源码位置**: [submit_put_task](../lmcache/v1/storage_backend/local_disk_backend.py#L310-L329)

   ```python
   # 伪代码逻辑
   while self.current_cache_size + required_size > self.max_cache_size:
       evict_keys = self.cache_policy.get_evict_candidates(...)
       # 同步删除文件，释放空间
       self.batched_remove(evict_keys, force=False)
   ```

3. **内存锁定 (Ref Count Up)**:

   - 通过显式引用计数机制，确保在异步 I/O 期间内存不被回收。
   - 主线程调用 `memory_obj.ref_count_up()`。

4. **异步提交**:

   - 将任务提交给 `LocalDiskWorker` 线程池（优先级为 2）。
   - 此时主线程返回，后续 I/O 由后台线程处理。

5. **物理写入与释放 (Worker Thread)**:
   - 后台线程执行 `O_DIRECT` 写入。
   - 写入完成后，调用 `memory_obj.ref_count_down()` 释放内存引用。
   - 更新 `self.dict` 索引，标记 Key 为可用。

**读取流程 (Get)**:

读取流程采用 **Caller-Allocated** 模式，相对简单：

1. **预分配**: 主线程查询索引获取 Shape/Dtype，直接调用 `self.local_cpu_backend.allocate(...)` 分配内存对象。
2. **直接读取**: 后台线程直接将磁盘数据读入该对象的 `byte_array` 缓冲区。引用计数由调用者（上层 Cache Engine）管理。

---

## 4. 任务调度：LocalDiskWorker

为避免磁盘 I/O 的长尾延迟阻塞关键的推理计算路径，`LocalDiskBackend` 采用全异步设计，将所有耗时的磁盘交互（写入、删除、预取）封装为独立任务，全权委托给内部专用的 `LocalDiskWorker` 代理执行，从而实现计算与存储的 pipeline 并行。

### 4.1 优先级线程池

底层使用 `AsyncPQThreadPoolExecutor` 实现了一个基于优先级的异步线程池。它结合了 `asyncio.PriorityQueue` 和线程池（`asyncio.to_thread`），确保关键路径上的 I/O 操作优先执行。

**源码位置**: [AsyncPQThreadPoolExecutor](../lmcache/v1/storage_backend/job_executor/pq_executor.py#L111)

**调度策略**:

1. **优先级队列**: 任务按 `(priority, order, task)` 元组入队。`priority` 数值越小，优先级越高。
2. **FCFS (First-Come-First-Served)**: 当优先级相同时，使用递增的计数器 (`itertools.count`) 确保先提交的任务先执行。

**优先级定义**:

| 任务类型     | 优先级       | 代码实现                       | 理由                                                                                               |
| :----------- | :----------- | :----------------------------- | :------------------------------------------------------------------------------------------------- |
| **Prefetch** | **0 (最高)** | `submit_task(..., priority=0)` | 预取操作通常发生在 decoding 阶段的 `retrieve` 环节，直接阻塞 GPU 计算流水线，必须最快响应。        |
| **Delete**   | **1 (中等)** | `submit_task(..., priority=1)` | 及时释放磁盘空间，防止因磁盘满导致新的写入失败或触发同步驱逐（Synchronous Eviction），阻塞主线程。 |
| **Put**      | **2 (最低)** | `submit_task(..., priority=2)` | 写入是后台持久化操作，可以容忍一定的延迟。让出带宽给 Prefetch 可以提升端到端延迟性能。             |

### 4.2 写入任务去重 (Deduplication)

在高并发场景下（如多个 Request 引用了相同的 Prefix），可能会对同一个 Key 多次触发 `put` 操作。为了避免重复 I/O 浪费带宽，`LocalDiskWorker` 维护了一个正在进行的任务列表 `put_tasks`。

**源码位置**: [LocalDiskWorker.put_tasks](../lmcache/v1/storage_backend/local_disk_backend.py#L40)

**工作流程**:

1. **提交检查**: 调用 `insert_put_task` 前，先检查 Key 是否已存在。
   - 如果存在，直接跳过并记录日志 (`Put task ... already in progress`)。
   - 代码: [submit_put_task](../lmcache/v1/storage_backend/local_disk_backend.py#L299-L301)
2. **任务注册**: 将 Key 加入 `put_tasks` 列表（由 `put_lock` 保护）。
3. **任务完成**: 磁盘写入完成后，调用 `remove_put_task` 将 Key 移除，允许后续再次写入。
   - 代码: [async_save_bytes_to_disk](../lmcache/v1/storage_backend/local_disk_backend.py#L492)

---

## 5. 配置项说明

`LocalDiskBackend` 的行为由 `LMCacheEngineConfig` 统一管理。以下是核心配置项及其对性能的影响：

- **源码位置**: [LMCacheEngineConfig](../lmcache/v1/config.py)

| 配置项                        | 类型    | 默认值        | 说明                                                                                                                                                                                                         |
| :---------------------------- | :------ | :------------ | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `local_disk`                  | `str`   | `None` (必填) | 本地磁盘缓存目录的绝对路径。如果目录不存在，`LocalDiskBackend` 会尝试自动创建。                                                                                                                              |
| `max_local_disk_size`         | `float` | `0.0`         | 允许使用的最大磁盘空间（单位：GB）。当缓存用量超过此阈值时，会触发**同步驱逐**策略。建议设置为磁盘容量的 80%-90%。                                                                                           |
| `extra_config["use_odirect"]` | `bool`  | `False`       | 是否开启 Linux O_DIRECT 直通 I/O 模式。**1）开启建议**: 使用 NVMe SSD 时强烈建议开启，可绕过 Page Cache 减少内存拷贝和 CPU 开销。**2）注意**: 开启后要求写入数据必须按 Block Size 对齐（通常 512B 或 4KB）。 |
| `cache_policy`                | `str`   | `"LRU"`       | 缓存淘汰策略。目前主要支持 LRU。该策略决定了当空间不足时哪些 Key 会被优先驱逐。                                                                                                                              |

**配置示例 (YAML)**:

```yaml
local_disk: "/mnt/nvme0/lmcache"
max_local_disk_size: 500 # 500 GB
cache_policy: "LRU"
extra_config:
  use_odirect: true
```
