# 第 3 讲：KV Cache 在 nano-vllm 里怎么实现 — PPT 提纲

每讲 10–12 页，只列页面主题和讲解落点；逐页讲稿待课程定稿后补。

1. **封面：一条请求走完 KV 生命周期**
   落点：本讲只回答三件事，块从哪里来、前缀怎么命中、块不够时 KV 去哪里。

2. **对象图：谁记账，谁持有张量**
   落点：画出 `LLMEngine → Scheduler → BlockManager/Sequence` 和 `ModelRunner → GPU/CPU KV tensor`；强调 `block_table` 是账本与显存之间的索引，`BlockManager` 不碰张量。

3. **`step()` 的五步**
   落点：`schedule() → swap_out() → swap_in() → run() → postprocess()`；`add_request` 在 step 外；说明 `swap_out → swap_in → run` 为什么不能换序。

4. **`BlockManager`：分配、释放、登记**
   落点：`can_allocate` 返回可复用的前缀块数，`allocate` 建 `block_table`，`may_append` 只在跨块边界加块，`deallocate` 按引用计数回收；全程使用 `_allocate_block` / `_deallocate_block`，没有 `free` 方法。

5. **前缀哈希链与碰撞保护**
   落点：`hash_blocks` 只登记刚算满的块；哈希带前一块的种子形成链；命中后还要比对 `token_ids`，防止哈希碰撞误用。

6. **`ref_count`：共享块与回收条件**
   落点：共享前缀命中时 `ref_count += 1`；`ref_count > 0` 表示还有请求持有，不能重分配；归零后 `deallocate` 才调用 `_deallocate_block` 把块放回空闲表。共享块 `ref_count > 1` 也会让整条序列失去 swap 资格。

7. **`Scheduler`：三队列与抢占触发**
   落点：`waiting` / `running` / `swapped` 的分工；`can_append(seq)` 因空闲块不足返回 False 时，抢占 `running` 队尾，队尾为空则抢占当前序列自身；不存在“抢占阈值”这种参数。

8. **抢占的三条分支与前置条件**
   落点：按顺序判断。第一，序列的块在本步 `blocks_to_swap_in` 中，强制 RECOMPUTE；第二，否则 `can_swap_out(seq)` 为真则 `swap_out`；第三，否则 RECOMPUTE。`can_swap_out` 要求 CPU 空闲块不少于序列块数，且序列所有块 `ref_count == 1`。

9. **`kv_swap`：GPU 与 CPU 的同步拷贝**
   落点：块布局 `[2, L, num_blocks, block_size, H_kv, D]`，一次切片覆盖所有层和 K/V；实现用同步 `copy_`；swap-out 释放 GPU 块但保留 `num_cached_tokens`，swap-in 重建 `block_table` 和哈希链。

10. **实验：L03 场景 E 与 E2**
    落点：L03 只驱动 Scheduler/BlockManager 元数据，不生成 token；E 走 SWAP，E2 因 CPU 块为 0 走 RECOMPUTE；对照 `num_swapped_out_blocks`、`num_recompute_preemptions`、`num_cached_tokens`。

11. **三个边界与小结**
    落点：满的末块不复用、无 LRU 驱逐、CPU 块不参与前缀哈希；这三条是 nano-vllm 的简化，不是通用结论；为第 4 讲 vLLM 的完整实现留出对照点。
