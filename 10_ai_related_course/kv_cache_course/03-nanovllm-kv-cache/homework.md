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

## 提交

代码放在独立分支或目录；报告不超过两页，附测试命令与输出摘要。
