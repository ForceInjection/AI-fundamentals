# 第 3 讲：课后作业

## 作业 2：给 nano-vllm 加一个 KV 观察点

在本地 nano-vllm 上选一个方向：

- 给 `BlockManager` 加 prefix hit / miss 计数；
- 给 `Scheduler` 加每步的 KV block 使用量；
- 实现一个最简单的 LRU 或 FIFO 驱逐策略，并补测试。

要求：

- 不改模型结构；
- 提交代码、测试命令、前后对比；
- 说明 nano-vllm 现有实现的一个限制（满的末块不复用 / 无 LRU / CPU 块无前缀哈希），以及你的改动是否绕过它；
- 如果改动涉及释放路径，确认走的是 `deallocate` → `_deallocate_block`，不要新增一个与现有命名冲突的 `free`。

## 巩固题

1. 为什么 `ref_count` 归零后才能 `_deallocate_block` / 被重分配？共享前缀时 `ref_count` 如何变化？
2. RECOMPUTE 与 SWAP 各自在什么负载下更划算？
3. 如果前缀哈希只按 block 内容计算，不做碰撞保护，可能出现什么问题？
4. `step()` 为什么必须保持 `swap_out → swap_in → run` 的顺序？把 `swap_in` 提前会破坏哪一步？
5. `block_size=4`，prompt 有 5 个 token（索引 0–4）。prefill 后采样出索引 5 的 token，使 `len(seq)=6`。之后 decode 到 `len(seq)=9` 时，下一步才需要新 block；用 `len(seq) % block_size` 解释为什么不是 `len=8`。
6. `can_allocate()` 为什么只扫描 `num_blocks - 1` 个块？如果把它改成扫描全部块，满的末块会发生什么？
7. `swap_in()` 为什么要按 `num_cached_tokens // block_size` 重建哈希链？如果不重建，前缀缓存和后续块的哈希会受什么影响？
8. `can_swap_out()` 为什么拒绝包含共享块的序列？如果强行把一个 `ref_count > 1` 的块换出，另一个请求会看到什么？

## 提交

代码放在独立分支或目录；报告不超过两页，附测试命令与输出摘要。
