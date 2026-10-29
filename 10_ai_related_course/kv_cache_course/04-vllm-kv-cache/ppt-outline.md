# 第 4 讲：KV Cache 在 vLLM 里怎么调度与回收 — PPT 提纲

每讲 10–12 页，只列页面主题和讲解落点；逐页讲稿待课程定稿后补。

1. **封面：从 nano-vllm 到 vLLM V1**
   落点：第 3 讲是一条最小路径，这一讲看生产引擎把同一生命周期拆成哪几层；基线 `8b365ff949`，V1 抢占只有 recompute。

2. **对象图：façade 与分层**
   落点：画出 `Scheduler → KVCacheManager → KVCacheCoordinator → SingleTypeKVCacheManager → BlockPool`；Worker 侧单独画 `BlockTable/MultiGroupBlockTable`；说明 per-request 块表在 `req_to_blocks`，不在 Scheduler 里。

3. **`allocate_slots`：一次分配的完整路径**
   落点：`get_computed_blocks` 查出命中块，`allocate_slots` 算 `get_num_blocks_to_allocate`，`allocate_new_computed_blocks` 加引用，`allocate_new_blocks` 取新块，最后 `cache_blocks` 登记；失败返回 `None`，调度器据此抢占或跳过。

4. **prefix 命中与末块边界**
   落点：`find_longest_cache_hit` 按块哈希链查；`max_cache_hit_length = request.num_tokens - 1`，最后 token 必重算；因为 `num_computed_tokens` 要块对齐，可能整块重算。对照 nano-vllm 连满的末块也不复用。

5. **驱逐：三种块状态与 LIFO/LRU**
   落点：`BlockPool` 一个池，块有三种状态：请求持有（`ref_cnt > 0`）、缓存可驱逐（`ref_cnt == 0` 且有 hash）、未缓存空闲（`ref_cnt == 0` 且无 hash）。`free_blocks` 把非缓存块前插（LIFO 复用），缓存块后插（FIFO 复用，表现为 LRU 淘汰）。没有通用 pin API。

6. **recompute：V1 抢占只有这一条路**
   落点：`_preempt_request` 释放块、置 `PREEMPTED`、`num_computed_tokens = 0`、放回 `waiting`；没有 swapped 队列、没有 scheduler 级 swap。默认 FCFS 策略下抢占对象是 running 队尾，触发点是 `allocate_slots` 返回 `None`。

7. **offload 边界：connector 不是抢占**
   落点：`kv_offloading_size` 打开独立 KV offloading connector（`vllm/config/cache.py:256`、`vllm/config/vllm.py:1123`），把已算好的 KV 存 CPU，命中时取回；它不改变请求状态，也不负责腾出 running 的块。不要映射成 nano-vllm 的 SWAP。

8. **实验：打印 block table 与 prefix 统计**
   落点：可选 GPU 路径打印 `input_batch.block_table[0].get_numpy_array()`（MultiGroup 取第 0 组），并读 `make_prefix_cache_stats()` 的 `requests/queries/hits`；`VLLM_LOG_STATS_INTERVAL=1` 让日志按秒输出。没有 GPU 时用 `code/README.md` 的示例日志和字段表。

9. **映射表：按生命周期成对写**
   落点：prefill 是 `can_allocate` + `allocate` ↔ `get_computed_blocks` + `allocate_slots`；decode 是 `can_append` + `may_append` ↔ `allocate_slots(num_new_tokens=1)`；释放是 `deallocate` ↔ `free`；登记是 `hash_blocks` ↔ `cache_blocks`。不要把 `BlockManager.allocate` 和 `allocate_slots` 当同层。

10. **小结：调度器决定 KV 的生死**
    落点：vLLM 把“查、算、分配、登记、回收”拆给不同层，BlockPool 只负责物理块和驱逐顺序；V1 的抢占是 recompute，offload 是另一条 connector 路径。为第 5 讲的跨请求复用和分层存储留接口。
