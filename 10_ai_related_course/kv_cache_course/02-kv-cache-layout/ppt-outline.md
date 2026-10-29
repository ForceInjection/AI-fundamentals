# 第 2 讲：KV Cache 怎么放 — PPT 提纲

每讲 10–12 页，只列页面主题和讲解落点，不写完整讲稿。

1. 封面：从 token 到 block
   落点：本讲把第 1 讲的 token 账变成 page/slot 账，回答“存在哪里、怎么找”。

2. 逻辑序列与物理 KV 池
   落点：请求看到的 token 是连续的，池里的 block 可以离散分布；分页把两者解耦。

3. 术语表：token / block / slot
   落点：分清 physical block id 与 logical block index，slot 是池内全局位置，不是请求内偏移。

4. 张量形状：nano-vllm 与 vLLM 的差别
   落点：nano-vllm 是 `[2, L, num_blocks, block_size, H_kv, D]`；vLLM FlashAttention 后端每层是 `[num_blocks, num_kv_heads, block_size, 2 * head_size]`，层维度不在同一张量里。

5. `block_bytes` 与 `num_blocks`
   落点：`block_bytes = 2 × b_kv × L × block_size × H_kv × D`，`num_blocks = available_kv_bytes // block_bytes`；同时强调请求的 `num_blocks` 和池的 `num_blocks` 是两个含义。

6. slot mapping：token 写到哪里
   落点：`slot = physical_block_id × block_size + offset`；prefill 和 decode 都按 slot 列表散写；L02/L04 不打印 slot_mapping，本讲用 `block_table_demo.py` 补。

7. 三种浪费：reservation / internal / external
   落点：连续分配 `max_model_len` 要算 reservation waste 和 external fragmentation；固定 block 池只算 internal fragmentation，external 按构造为 0。

8. block table：decode 怎么找历史
   落点：`block_table[logical_block_index] = physical_block_id`，decode 按表项拼历史 K/V，表长随序列增长。

9. prefix 共享与 `ref_count`
   落点：共享以完整 block 为粒度；`block_size=4`、前 6 个 token 相同、第 7 个不同时只共享 block 0，`ref_count=2`。

10. 一个请求的 block table 示例
    落点：2 层、每层每 token 1 KB、`block_size=4`，则 `block_bytes=8 KB`；10 token 用 3 块共 24 KB，有效 20 KB，内部碎片 4 KB。

11. 实验：`block_table_demo.py`
    落点：打印 token → block → slot、block table、`ref_count` 和 A/B 碎片表，手算一行 slot 对答案。

12. 小结：分页解决的是“怎么放”，不是“存什么”
    落点：分页不减少 KV 总量，只减少浪费并让共享、换出、扩容成为可能；第 3 讲看这套结构在 nano-vllm 里怎么跑。
