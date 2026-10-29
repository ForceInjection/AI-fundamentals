# 第 4 讲：KV Cache 在 vLLM 里怎么调度与回收

> 120 分钟 | 课程第 4 讲 | 前置：第 3 讲的 nano-vllm

第 3 讲看的是最小实现，这一讲把同一组问题带到 vLLM V1：谁管理块、谁决定驱逐、prefix cache 怎么查、抢占之后怎么恢复。重点不是背类名，而是把 nano-vllm 的生命周期映射到 vLLM 的分层对象上。

## 先看对象关系

vLLM V1 的 KV 管理是分层委托，不是一个大 `BlockManager`：

```text
Scheduler（vllm/v1/core/sched/scheduler.py）
└── KVCacheManager（façade，vllm/v1/core/kv_cache_manager.py）
    └── KVCacheCoordinator（按 kv_cache_group 协调，kv_cache_coordinator.py）
        ├── SingleTypeKVCacheManager × N（single_type_kv_cache_manager.py）
        │   ├── FullAttentionManager
        │   ├── SlidingWindowManager
        │   └── MambaManager
        └── BlockPool（物理块池，block_pool.py）
```

Worker 侧是另一套表示：

```text
GPUModelRunner
└── InputBatch.block_table: MultiGroupBlockTable
    └── BlockTable × kv_cache_group（vllm/v1/worker/block_table.py）
```

Scheduler 侧每个请求的块表在 `SingleTypeKVCacheManager.req_to_blocks[request_id]`，类型是 `list[KVCacheBlock]`；Worker 侧的 block table 是把块号拷成张量，供 attention kernel 使用。两者是同一份映射的两种表示。

## 这一讲和 nano-vllm 最大的不同

基线 `8b365ff949` 的 V1 抢占只有 RECOMPUTE 一条路。`Scheduler._preempt_request`（`vllm/v1/core/sched/scheduler.py:1539`）做四件事：释放请求的块、把状态置为 `RequestStatus.PREEMPTED`、把 `num_computed_tokens` 清零、放回 `waiting` 队首。没有 scheduler 级的 swap，也没有 swapped 队列。

CPU offload 不是抢占路径，而是独立的 KV offloading connector，由 `kv_offloading_size` 开启（`vllm/config/cache.py:256`、`vllm/config/vllm.py:1123`）。它把已经算好的 KV 存到 CPU，后续命中时再取回，属于 KV transfer 层。不要把它映射成 nano-vllm 的 SWAP。

## BlockPool 的三种块状态

一个 `KVCacheCoordinator` 持有一个 `BlockPool`（`kv_cache_coordinator.py:96`），所有 `SingleTypeKVCacheManager` 共用。块的状态由三个字段决定：`ref_cnt`、`_block_hash`，以及是否在 `free_block_queue` 里。

| 状态 | 判据 | 在空闲队列里的位置 | 分配与回收行为 |
| --- | --- | --- | --- |
| 请求持有 | `ref_cnt > 0` | 不在队列 | 不能被分配；`free_blocks` 把 `ref_cnt` 减到 0 才入队 |
| 缓存可驱逐 | `ref_cnt == 0` 且有 `block_hash` | 队尾（`free_blocks` 后插） | 按 FIFO 复用，表现为 LRU 淘汰；`get_new_blocks` 取走时先 `_maybe_evict_cached_block` 清掉旧哈希 |
| 未缓存空闲 | `ref_cnt == 0` 且无 `block_hash` | 队首（`free_blocks` 前插） | 按 LIFO 复用，优先拿最近释放的块，改善 GPU 局部性 |

没有通用的 pin API。`touch` 只在命中时做 `ref_cnt += 1` 并把块从空闲队列移除；`unpin_blocks` 供“引用已释放、但内容还要被外部读取”的场景使用（例如 offload / 分层存储的读取侧），它不是请求生命周期里的通用 pin 接口，本课只在讲 offload 时提。

缓存块的 `ref_cnt` 归零不等于内容失效，它仍留在哈希表里，直到被 `get_new_blocks` 取走并驱逐。BlockPool 不区分“空闲”和“可驱逐”，两者都在空闲队列里，区别只是位置和是否带哈希。

## 学习目标

- 说清 `KVCacheManager`、`KVCacheCoordinator`、`SingleTypeKVCacheManager`、`BlockPool` 的分工。
- 沿 `get_computed_blocks` + `allocate_slots` 走完一次 prefill/admission，以及 decode 时 `allocate_slots(num_new_tokens=1)` 的块增长。
- 理解 prefix cache 的查询、命中、引用和驱逐，包括“最后 token 必重算”的边界。
- 区分 BlockPool 的三种块状态，以及非缓存块的 LIFO 复用和缓存块的 FIFO/LRU 淘汰。
- 区分 V1 的 recompute 抢占与 CPU offload connector，不把两者混成一条路径。
- 把 nano-vllm 的 `BlockManager` 操作按生命周期映射到 vLLM 的对应函数。

## 本讲产出

一张按生命周期成对写的“nano-vllm → vLLM”映射表，附一次请求的 block table 快照和 prefix cache 统计字段说明。

## 目录

- [课程总览](../README.md)
- [教学大纲](syllabus.md)
- [PPT 提纲](ppt-outline.md)
- [课堂实验](hands-on-exercise.md)
- [课后作业](homework.md)
- [实验环境](lab-environment.md)
- [实验代码与日志字段](code/README.md)

## 前后衔接

第 3 讲建立了最小实现的生命周期；本讲看生产引擎怎么补上缓存、驱逐和调度。第 5 讲继续往跨请求复用和分层存储走。
