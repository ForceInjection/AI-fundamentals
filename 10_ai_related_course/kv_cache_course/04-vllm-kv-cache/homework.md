# 第 4 讲：课后作业

## 基础题

选一个 vLLM V1 的 KV 路径（分配、prefix 命中、驱逐、recompute 任选其一），写一份源码笔记：

- 入口函数与调用链；
- 关键数据结构；
- 一次请求经过该路径时，块的数量和状态如何变化；
- 与 nano-vllm 的对应实现差在哪。

## 源码题

读 `vllm/v1/core/kv_cache_coordinator.py` 和 `vllm/v1/core/single_type_kv_cache_manager.py`：

1. `KVCacheCoordinator.find_longest_cache_hit` 在单 group 和多 group 下分别走到哪个实现？`UnitaryKVCacheCoordinator` 和 `FullAttentionManager` 的分工是什么？
2. `allocate_new_computed_blocks` 为什么先对所有 group 的命中块 `add_local_computed_blocks`，再统一分配外部块？如果顺序反过来，可能出现什么驱逐问题？
3. `SingleTypeKVCacheManager.free` 为什么按 `reversed(...)` 归还块？`req_to_blocks` 和 `num_cached_block` 分别在什么时候被清理？

## 对比题

比较 V1 的 recompute 抢占和 CPU offload connector，不要把它们当成同一条路径：

- 触发条件：`_preempt_request` 由什么触发？`kv_offloading_size` 打开的是什么？
- 请求状态：recompute 后 `RequestStatus` 和 `num_computed_tokens` 怎么变？offload 是否改变请求状态？
- 块的所有权：recompute 释放的是哪些块？offload 搬运的是哪份 KV？
- 代价：各自消耗重算、显存还是 CPU-GPU 带宽？
- 适用场景：什么负载下 recompute 可接受？什么场景才需要 offload？

结论请落到源码：至少引用 `sched/scheduler.py:1539` 的 `_preempt_request`，以及 `vllm/config/cache.py:256` 和 `vllm/config/vllm.py:1123` 的 offload 配置入口。

## 巩固题

1. `max_cache_hit_length = request.num_tokens - 1` 保证了什么？如果 prompt 长度刚好是 `block_size` 的整数倍，命中长度和实际重算长度会差多少？
2. `SingleTypeKVCacheManager.free` 为什么用 `reversed(...)` 归还块？如果改成顺序归还，BlockPool 的 FIFO/LIFO 行为会怎样变化？
3. `BlockPool.touch` 和 `unpin_blocks` 分别在什么场景使用？如果把 `unpin_blocks` 当成请求生命周期的通用 pin 接口，会漏掉什么引用？
4. Worker 侧 `get_numpy_array()` 的未使用列是什么值？0 号块表示什么？怎样从 `num_blocks_per_row` 取出有效块号？
5. `make_prefix_cache_stats()` 返回哪些字段？为什么 `disable_log_stats=True` 时返回 `None`？连续调用两次会发生什么？
6. `allocate_slots()` 返回 `None` 后，Scheduler 的默认 FCFS 策略会抢占谁？如果被抢占请求的块不能被立即释放，调度器会怎么处理？

## 提交

源码笔记不超过两页，所有代码引用带 `file:line`。
