# 第 5 讲：课后作业

## 基础题：64 个会话的账

场景先钉死，算的时候不要换口径：

- 模型 Qwen3-0.6B，GQA，28 层，8 个 KV head，`head_dim=128`，FP16。每 token KV 为 `2 × 28 × 8 × 128 × 2 = 114,688 B`，约 112 KiB。
- KV 池 8 GiB，按上面的口径约 7.5 万 token；`page_size = 16`。
- 64 个会话，每个共享 2,000 token 的 system prompt，用户问题平均 200 token，输出 100 token。
- 到达方式分两种。A 冷启动：64 个请求同时到达，缓存为空。B 稳态：缓存已由一个预热请求建立，其余 63 个陆续到达。

1. 分别算 A、B 两种到达方式下，这一批请求的 prefill token 总量，以及相对“无 prefix caching”节省了多少。A 的答案要说明缓存命中率是多少，以及 KV 池容量对这批请求并发的影响。
2. 如果只共享 system prompt 的前 1,000 token，B 的 prefill 总量和聚合命中率是多少？
3. 用下面的简化模型估算三种情况的 TTFT 比例：无缓存、完整命中 2,000 token、只命中前 1,000 token。模型只计注意力矩阵乘：无缓存约 `(P+Q)²/2`，命中 `P_h` 个前缀 token 时约 `(P_u+Q)²/2 + (P_u+Q)·P_h`，其中 `P_u` 是未命中的前缀长度。
4. 验证“约 5.5×”这个数：P=2000、Q=200 时，完整命中的加速比用 `(P+Q)/(2Q)` 算是多少？为什么不能按命中 token 数直接算成 `2200/200 = 11×`？
5. 把“命中率”拆成两个口径：每请求命中长度（绝对 token 数，以及占该请求 prompt 的比例）和聚合命中率（命中 token 总和 / prompt token 总和）。构造一个例子，让聚合命中率与“各请求命中率的简单平均”明显不同，并说明哪个口径更适合回答“这套缓存值不值得开”。
6. `page_size = 16` 对第 2 问的命中长度有什么影响？如果 system prompt 是 2,005 token，完整命中长度是多少？

## 源码题

读 SGLang 的前缀匹配与驱逐路径：

- 入口：`UnifiedTreeCore.match_prefix`（`unified_tree_core.py:812`），经典实现是 `RadixCache.match_prefix`（`radix_cache.py:354`）。
- 分裂：`UnifiedTreeCore._split_node`（`:1390`），经典实现是 `RadixCache._split_node`（`:665`）。
- 驱逐：`UnifiedTreeCore.evict_device_leaf`（`:1610`）与 `demote`（`:1864`），经典实现是 `RadixCache.evict`（`:553`）。

回答三个问题：

1. 匹配落在节点中间时，split 之后新节点的 `lock_ref`、`value`、`host_value` 各怎么处理？
2. `lock_ref` 与第 3 讲 nano-vllm 的 `ref_count` 相比，粒度和累加方式有什么不同？为什么树节点需要沿路径累加，而 block 不需要？
3. 一个叶节点被删除后，它的父节点在什么条件下进入驱逐候选？把判断条件写成伪代码。

## 提交

一页计算（含公式和单位）+ 一段源码说明，源码引用标 `file:line`。计算部分先跑一遍 `python3 code/prefix_cache_sim.py --self-check`，确认模拟器的 split、`lock_ref`、驱逐行为和你的理解一致。
