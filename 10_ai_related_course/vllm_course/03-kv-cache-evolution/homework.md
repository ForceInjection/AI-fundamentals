# 第 3 讲课后作业

## 作业（必做两题）

**题 1（group 键为什么必须）**：nano 的哈希键是 `hash → block_id`，HEAD 是 `BlockHashWithGroupId`（哈希+group 后缀）。回答：

1. 用一个具体场景说明：滑窗组与 full 组共享同一段 prompt 时，若无 group 维度会发生什么错误（提示：两组的命中算法不同——`find_longest_cache_hit` 在 SlidingWindowManager 与 FullAttentionManager 里行为不同，`single_type_kv_cache_manager.py:1149` vs `:873`）。
2. HEAD 上「同一物理 block ID 被两组 overlay 使用」靠什么约束保证正确（`kv_cache_interface.py:638-641` 注释）？
3. 你的 fork 里 `swap_in` 重建哈希链（`block_manager.py:161-180`）——在 HEAD 的 `(hash, group)` 键下，这个重建逻辑需要怎么变？

**题 2（退役证据链独立复验）**：不依赖讲义，在基线 `2f5d8a2b1` 上独立验证三步证据链，每步给出你实际执行的命令与输出摘要：

1. `csrc/attention/` 下是否存在 PagedAttention CUDA kernel；
2. `vllm/v1/attention/ops/paged_attn.py` 的行数与剩余内容；
3. 任选一个 CUDA attention 后端（FlashAttention 或 Triton），找到它接收分页 KV 的参数位置（`file:line`）。
   结论：三步验证后，用一句话回答「PagedAttention 退役了什么、留下了什么」。

## 附加题（不计考核）

`prefix_caching.md` 的 Note 1「only cache full blocks」与 HEAD 的 `cache_partial_block`（`block_pool.py:459`）矛盾。找出 partial block 的目标场景（哪个 spec 类型需要它）与 partial→full 提升的触发条件（读 `block_pool.py:291-300` 与 `:580-593`），写 200 字以内的机制说明。

## 自测题（不提交）

1. 为什么「哈希前移到 `add_request`」是 cache-aware 准入（第 2 讲）的前置条件？
2. `free_blocks` 里无哈希块 prepend 队首(LIFO)、有哈希块 append 队尾(LRU)——两种策略各自优化什么？
3. 一个模型 45 层全是 full attention，`get_kv_cache_coordinator` 会走哪个分支？若其中 10 层是 sliding window 呢？
