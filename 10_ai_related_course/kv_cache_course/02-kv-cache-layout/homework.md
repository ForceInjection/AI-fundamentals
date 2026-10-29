# 第 2 讲：课后作业

## 基础题

先写清本讲的两个公式：

`block_bytes = 2 × b_kv × L × block_size × H_kv × D`

`num_blocks = available_kv_bytes // block_bytes`

玩具模型：2 层，`H_kv=2`，`D=128`，KV 按 FP16 存（`b_kv=2`），`block_size=4`。

1. 先验证每层每 token 的 K+V 是 1 KB：`2（K/V）× 2（b_kv）× 2（H_kv）× 128（D） = 1024 B`。再算每 token KV 和每 block 字节。
2. 一个 10-token 请求：画出 `block_table` 和每个 token 的 `slot`，算分配块数、分配字节、有效 KV 和内部碎片。
3. 第二个请求有 12 个 token，前 6 个 token 与第一个请求相同，第 7 个 token 起不同。`block_size=4`。问两个请求的 `block_table` 和 `ref_count`，并说明共享粒度。
4. 追问：如果第二个请求前 8 个 token 都与第一个请求相同，哪些 block 可以共享？注意第一个请求只有 10 个 token，最后一个 block 还没填满。

核对：第 3 题只共享 block 0（0-based index 0–3，共 4 个 token），block 0 的 `ref_count=2`；block 1 覆盖 0-based index 4–7，第 7 个 token（index 6）不同，整块不能共享。第 4 题共享 block 0 和 block 1；第一个请求的 block 2 只有 2 个 token（index 8–9），未填满，不参与共享。

## 源码题

读 `~/Project/study/nano-vllm` 的 `BlockManager`：

- 找出分配 block 的函数；
- 找出计算 block 哈希的函数；
- 说明“最后一个未填满的 block 为什么通常不参与前缀复用”。

## 巩固题

1. 写出 nano-vllm 和 vLLM FlashAttention 后端的 KV 池形状，说明层维度分别在哪里。
2. 固定 block 池的外部碎片为什么按构造为 0？连续分配 `max_model_len` 时，external fragmentation 又从哪里来？
3. 如果 `block_size` 从 4 改成 256，内部碎片和 block table 大小各会怎么变？

## 关于 block_size

本题的 `block_size=4` 只为手算，把映射看清楚。nano-vllm 默认 256（`nanovllm/config.py:17`，且要求是 256 的倍数），vLLM 默认 16，参考文章里的例子常用 16。block 越小，最后一个 block 的内部碎片越少，但同样序列需要更多 block，block table 更长，元数据和查表开销更大。

## 提交

一页图 + 一段源码说明，标注 `file:line`。
