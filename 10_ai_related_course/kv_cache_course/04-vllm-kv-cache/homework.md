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

## 提交

源码笔记不超过两页，所有代码引用带 `file:line`。
