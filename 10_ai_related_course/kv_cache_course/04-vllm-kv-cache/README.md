# 第 4 讲：KV Cache 在 vLLM 里怎么调度与回收

> 120 分钟 | 课程第 4 讲 | 前置：第 3 讲的 nano-vllm；了解 vLLM 的基本请求流程

第 3 讲看的是最小实现，这一讲把同一组问题带到 vLLM V1：谁管理块、谁决定驱逐、prefix cache 怎么查、抢占之后怎么恢复。重点不是背类名，而是把 nano-vllm 的生命周期映射到 vLLM 的分层对象上。

## 本讲回答三个问题

1. vLLM V1 把 KV 管理拆成哪几层？谁查前缀、谁分配、谁驱逐？
2. 一次请求从 prefill 到 decode，`allocate_slots` 做了什么？失败时为什么退回 recompute？
3. V1 抢占和 CPU offload 有什么区别？block table 在 Scheduler 和 Worker 两侧怎么表示？

## 一句话

vLLM V1 把 KV 生命周期拆成“查询、分配、登记、驱逐、抢占”五个动作：`KVCacheManager` 提供对外接口，`KVCacheCoordinator` 按 KV group 协调，`SingleTypeKVCacheManager` 维护每个请求的块表，`BlockPool` 管物理块和驱逐顺序。

## 学习目标

- 说清 `KVCacheManager`、`KVCacheCoordinator`、`SingleTypeKVCacheManager`、`BlockPool` 的分工。
- 沿 `get_computed_blocks` + `allocate_slots` 走完一次 prefill/admission，以及 decode 时 `allocate_slots(num_new_tokens=1)` 的块增长。
- 理解 prefix cache 的查询、命中、引用和驱逐，包括“最后 token 必重算”的边界。
- 区分 `BlockPool` 的三种块状态，以及非缓存块的 LIFO 复用和缓存块的 FIFO/LRU 淘汰。
- 区分 V1 的 recompute 抢占与 CPU offload connector，不把两者混成一条路径。
- 把 nano-vllm 的 `BlockManager` 操作按生命周期映射到 vLLM 的对应函数。

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

## `allocate_slots`：一次分配的主路径

prefill / admission 先从 prefix cache 查命中：

- `get_computed_blocks()` 调用 `find_longest_cache_hit()`，返回命中块、命中 token 数，以及混合模型用的 `shared_prefix_boundary`。
- `max_cache_hit_length = request.num_tokens - 1`，最后一个 token 必须重算，用于拿 logits。
- 因为 `num_computed_tokens` 要块对齐，最后 token 的重算可能扩展成整个块的重算。

然后进入 `allocate_slots()`：

1. `remove_skipped_blocks()` 先释放滑动窗口外不再需要的块。
2. `get_num_blocks_to_allocate()` 按 KV group 汇总还需要多少块。
3. 检查空闲块是否足够；等待或抢占过的请求还要预留 watermark。
4. `allocate_new_computed_blocks()` 先 touch 所有 group 的命中块，再处理外部命中块，避免前面的 group 分配时把后面 group 的命中块驱逐。
5. `allocate_new_blocks()` 从 `BlockPool` 取新块，写入 `req_to_blocks`。
6. `cache_blocks()` 把已经写满的块登记进 prefix cache。

任何一步发现块不够，`allocate_slots()` 返回 `None`。Scheduler 收到 `None` 后，会抢占 `running` 队尾的请求，再重试当前请求。

decode 阶段调用 `allocate_slots(num_new_tokens=1)`；只有跨块边界时才真正从 `BlockPool` 取新块，其他步只复用当前已分配的块。

## prefix 命中与末块边界

`Request` 在创建和追加 token 时计算块哈希。`find_longest_cache_hit()` 按块哈希链查最长前缀；命中后还要确认这个块属于同一个 KV group。

`cache_blocks()` 在分配完成后登记满块：

- `num_full_blocks = num_tokens // block_size`
- 只登记 `num_cached_blocks` 到 `num_full_blocks` 之间的新满块
- `BlockPool.cache_full_blocks()` 建立 `block_hash → block` 的映射

和 nano-vllm 的差别在末块：

- nano-vllm 的 `can_allocate()` 只查到 `num_blocks - 1`，满的末块也不复用。
- vLLM 用 `max_cache_hit_length = num_tokens - 1` 保证最后一个 token 重算；按块对齐后，可能整块重算。

## BlockPool：三种块状态

一个 `KVCacheCoordinator` 持有一个 `BlockPool`（`kv_cache_coordinator.py:96`），所有 `SingleTypeKVCacheManager` 共用。块的状态由 `ref_cnt`、`block_hash` 和是否在 `free_block_queue` 里决定。

| 状态 | 判据 | 在空闲队列里的位置 | 分配与回收行为 |
| --- | --- | --- | --- |
| 请求持有 | `ref_cnt > 0` | 不在队列 | 不能被分配；`free_blocks` 把 `ref_cnt` 减到 0 才入队 |
| 缓存可驱逐 | `ref_cnt == 0` 且有 `block_hash` | 队尾（`free_blocks` 后插） | 按 FIFO 复用，表现为 LRU 淘汰；`get_new_blocks` 取走时先 `_maybe_evict_cached_block` 清掉旧哈希 |
| 未缓存空闲 | `ref_cnt == 0` 且无 `block_hash` | 队首（`free_blocks` 前插） | 按 LIFO 复用，优先拿最近释放的块，改善 GPU 局部性 |

没有通用的 pin API。`touch` 只在命中时做 `ref_cnt += 1` 并把块从空闲队列移除；`unpin_blocks` 供“引用已释放、但内容还要被外部读取”的场景使用（例如 offload / 分层存储的读取侧），它不是请求生命周期里的通用 pin 接口。

## 抢占：V1 只有 RECOMPUTE

基线 `8b365ff949` 的 V1 抢占只有 RECOMPUTE 一条路。`Scheduler._preempt_request`（`vllm/v1/core/sched/scheduler.py:1539`）的核心是四步：

1. 释放请求的块；
2. 把状态置为 `RequestStatus.PREEMPTED`；
3. 把 `num_computed_tokens` 清零；
4. 把请求放回 `waiting` 队首。

没有 scheduler 级的 swap，也没有 `swapped` 队列。触发点是 `allocate_slots()` 返回 `None`：Scheduler 抢占 `running` 队尾（默认 FCFS）再重试当前请求；如果请求无法被抢占，就跳过本轮调度。

recompute 的代价是重新 prefill。prefix cache 可能仍然帮上忙，但最后 token 的重算和块对齐限制会让收益打折扣。

## CPU offload 不是抢占

CPU offload 是独立的 KV offloading connector，由 `kv_offloading_size` 开启（`vllm/config/cache.py:256`、`vllm/config/vllm.py:1123`）。它把已经算好的 KV 存到 CPU，后续命中时再取回，属于 KV transfer 层。

它不改变请求状态，也不负责腾出 `running` 请求占用的块。不要把 offload 映射成 nano-vllm 的 SWAP，也不要把 connector 的命中率当成 scheduler 的 swap 命中率。

## Block table 的两侧表示

Scheduler 侧：

- `SingleTypeKVCacheManager.req_to_blocks[request_id]` 是 `list[KVCacheBlock]`，按请求记录分配顺序。
- `free()` 用 `reversed(...)` 归还块，尾部块先释放，便于缓存淘汰。

Worker 侧：

- `InputBatch.block_table` 是 `MultiGroupBlockTable`，每个 KV group 一个 `BlockTable`。
- 单 group 模型取 `block_table[0]`；`get_numpy_array()` 返回 `(max_num_reqs, max_blocks_per_req)` 的预分配 buffer。
- `num_blocks_per_row[row]` 才是有效长度；未使用列是 0，0 号块是 null block，不是有效块。

## Prefix 统计

`KVCacheManager.make_prefix_cache_stats()`（`kv_cache_manager.py:236`）返回 `PrefixCacheStats`，包含 `requests`、`queries`、`hits` 等字段。命中率按 `hits / queries` 计算。

`log_stats=False` 时返回 `None`。`VLLM_LOG_STATS_INTERVAL=1` 只控制日志频率，不改变统计口径。示例日志和字段表见 [code/README.md](code/README.md)。

## 本讲边界

- V1 抢占只有 recompute，没有 scheduler 级 swap。
- CPU offload 是 connector，不改变请求状态，也不是抢占路径。
- 最后一个 token 必重算；按块对齐后，可能整块重算。
- `BlockPool` 的非缓存块走 LIFO，缓存块走 FIFO（表现为 LRU）；没有通用 pin API。
- Worker block table 的 0 是 null block，有效长度看 `num_blocks_per_row`。
- `make_prefix_cache_stats()` 会读取并重置统计；instrumentation 不能改变正常日志的调用顺序。

## 本讲产出

一张按生命周期成对写的映射表，附：

- 一次请求的 block table 快照，注明哪些列有效、哪些是 0 填充；
- prefix cache 的 `requests / queries / hits` 和命中率；
- 一句结论：两套实现最大的差异在“查询与分配的组织方式”“驱逐顺序”还是“抢占后 KV 是否保留”。

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
