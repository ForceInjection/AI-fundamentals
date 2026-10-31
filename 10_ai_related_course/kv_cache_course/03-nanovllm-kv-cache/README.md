# 第 3 讲：KV Cache 在 nano-vllm 里怎么实现

> 120 分钟 | 课程第 3 讲 | 前置：第 2 讲的 block/page；能读 Python

第 2 讲把 token 落到 block 和 slot，这一讲看这套映射在引擎里怎么动起来：请求进入、分配 block、前缀命中、decode 增长、抢占、换出、释放。nano-vllm 代码量小，正好把抽象结构落到具体函数。

## 本讲回答三个问题

1. 谁记账，谁持有 KV 张量？
2. 一个请求从 prefill 到结束，`block_table` 和 `ref_count` 怎么变？
3. block 不够时，Scheduler 什么时候把 KV 搬到 CPU，什么时候退回重算？

## 一句话

nano-vllm 把 KV 生命周期分成两层：`BlockManager` 管块账本，`ModelRunner` 管真实张量；`Scheduler` 用三个队列和引用计数，把分配、共享、抢占、换出串起来。

## 学习目标

- 说清 `BlockManager` 和 `ModelRunner` 各自负责什么，`block_table` 怎么把两边对上。
- 找到 `can_allocate` / `allocate` / `deallocate` / `may_append` 的调用关系，理解前缀哈希链和 `ref_count`。
- 看懂 `Scheduler` 的 prefill / decode 状态转换，以及 `can_append` 返回 False 后怎样触发抢占。
- 说清 SWAP 与 RECOMPUTE 两条出路的触发条件，以及本步刚 swap-in 时为什么必须强制 RECOMPUTE。
- 指出 nano-vllm 的实现边界：满的末块不复用、无 LRU、CPU 块不参与前缀哈希。

## 先看对象关系

先分开两类对象：一类只记账，一类持有张量。

```text
LLMEngine
├── Scheduler
│   ├── waiting / running / swapped 三个队列
│   ├── BlockManager（纯元数据）
│   │   ├── free_block_ids / used_block_ids
│   │   ├── hash_to_block_id：前缀哈希 → 物理块号
│   │   └── free_cpu_block_ids / swapped_block_tables：swap 用的 CPU 块账本
│   └── Sequence（token_ids、block_table、num_cached_tokens）
└── ModelRunner
    ├── kv_cache[2, L, num_blocks, block_size, H_kv, D]（GPU）
    └── cpu_kv_cache[2, L, num_cpu_blocks, ...]（CPU，pinned）
```

`BlockManager` 只改 `Block`、`free_block_ids`、`used_block_ids`、`hash_to_block_id` 这些元数据；真正的 K/V 在 `ModelRunner.kv_cache` 里。`Sequence.block_table` 是两者之间的索引，attention 用 `slot = block_id * block_size + offset` 找到具体位置。

`LLMEngine.add_request()` 在 `step()` 之外：它把 prompt 包成 `Sequence`，交给 `Scheduler.add()` 放进 `waiting`。`step()` 只负责把已经排队的请求推进一步。

## `step()` 的五步

`LLMEngine.step()`（`nanovllm/engine/llm_engine.py:52`）的顺序是固定的：

1. `Scheduler.schedule()`：从 `waiting` / `running` / `swapped` 选本步的序列，填 `blocks_to_swap_out` 和 `blocks_to_swap_in`。
2. `ModelRunner.swap_out()`：按映射把 GPU 块拷到 CPU。
3. `ModelRunner.swap_in()`：按映射把 CPU 块拷回 GPU。
4. `ModelRunner.run()`：prefill 或 decode，写本步的 K/V。
5. `Scheduler.postprocess()`：登记新满块的哈希，推进 `num_cached_tokens`，处理采样结果和释放。

`swap_out → swap_in → run` 的顺序是硬约束：`swap_out` 必须在源 GPU 块被本步 `run()` 覆盖之前读出；如果某个块同时是换出源和换入目标，也必须先 out 后 in。`add_request` 不属于这五步。

## BlockManager：块账本

`BlockManager` 在 `nanovllm/engine/block_manager.py`。它不持有 K/V，只维护块的状态、空闲队列和前缀哈希表。

### 分配

`Sequence.num_blocks = ceil(len(seq) / block_size)`。`can_allocate(seq)` 从第一个块开始扫描，但只看 `seq.num_blocks - 1` 个块：最后一个块即使写满，也不参与前缀命中。

扫描时用前缀哈希链找候选块，并比对 `block.token_ids == token_ids`，防止哈希碰撞。命中且已被别的请求占用的块不额外占用空闲块；最后如果空闲块不够 `num_new_blocks`，返回 `-1`，否则返回可复用的块数。

`allocate(seq, num_cached_blocks)` 分两段：

- 前 `num_cached_blocks` 个块直接挂到 `seq.block_table`；块已被其他请求持有时 `ref_count += 1`，否则把块从空闲队列取出并置 `ref_count = 1`。
- 剩余块用 `_allocate_block()` 新分配，依次追加到 `block_table`。

最后设置 `num_cached_tokens = num_cached_blocks * block_size`。

### 释放

`deallocate(seq)` 反向遍历 `block_table`，每块 `ref_count -= 1`；归零才调用 `_deallocate_block()` 放回 `free_block_ids`。随后把 `num_cached_tokens` 置 0，清空 `block_table`。

### 增长

`can_append(seq)` 判断下一步是否需要新块：

```python
len(self.free_block_ids) >= (len(seq) % self.block_size == 1)
```

`may_append(seq)` 只在 `len(seq) % block_size == 1` 时追加一个新块。以 `block_size=4`、prompt 5 个 token 为例：

| 时刻 | `len(seq)` | `len % 4` | 需要新块 | 下一个 token 写哪里 |
| --- | --- | --- | --- | --- |
| prefill 后采样 | 6 | 2 | 否 | block 1 的 offset 1 |
| decode 1 后 | 7 | 3 | 否 | block 1 的 offset 2 |
| decode 2 后 | 8 | 0 | 否 | block 1 的 offset 3 |
| decode 3 后 | 9 | 1 | 是 | block 2 的 offset 0 |

`len(seq)` 是已经产生的 token 数，下一步要算的是最后一个 token，所以块边界落在 `len % block_size == 1`，不是 `0`。

### 前缀哈希链

`compute_hash(token_ids, prefix)` 把上一块的哈希作为种子，再哈希当前块的 token；这样一串块形成链，前缀相同才会得到相同哈希。

`hash_blocks(seq)` 在 `postprocess` 里调用。它只登记本次真正写满的块：

```python
start = seq.num_cached_tokens // block_size
end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // block_size
```

`start == end` 时直接返回，说明这次没有写满任何新块。每个登记过的块会把 `hash` 和 `token_ids` 写进 `Block`，并更新 `hash_to_block_id`。

`can_allocate` 命中哈希后还会比对 `token_ids`，避免哈希碰撞把别的块当成前缀；`allocate` 直接使用 `can_allocate` 已经验证过的块。

### `ref_count`

共享前缀时，`allocate` 对已在 `used_block_ids` 里的块执行 `ref_count += 1`；每个请求结束时只释放自己的引用。`ref_count` 归零，块才回到空闲队列。

| 事件 | block 0 的 `ref_count` | 说明 |
| --- | --- | --- |
| A 分配 | 1 | A 持有 |
| B 命中同一前缀 | 2 | A、B 共享 |
| A 结束 | 1 | B 还在用 |
| B 结束 | 0 | 块回到空闲队列 |

`can_swap_out` 要求序列所有块的 `ref_count == 1`。只要有一个共享块，整条序列就不能 swap，退回 RECOMPUTE。

## Scheduler：三个队列

`Scheduler` 在 `nanovllm/engine/scheduler.py`，维护 `waiting`、`running`、`swapped` 三个队列。`schedule()` 的顺序是：

1. 先处理 `waiting`，做 prefill。只要本轮排到了 prefill，就直接返回，不再做 swap-in 和 decode。
2. 没有 prefill 时，从 `swapped` 尝试 swap-in，把 CPU 上的 KV 搬回 GPU。
3. 最后处理 `running`，做 decode。

prefill 阶段用 `max_num_batched_tokens` 控制本轮 token 预算。chunked prefill 只允许切分本轮的第一条序列；后面的序列如果放不下，就留到下一轮。

decode 阶段从 `running` 队首取序列，逐条调用 `can_append(seq)`。返回 False 时，抢占 `running` 队尾的序列，再重试；如果 `running` 已经空了，就抢占当前序列自身。这里没有“抢占阈值”参数，触发条件就是空闲块不够。

## 抢占的三条分支

`Scheduler.preempt()` 在 `nanovllm/engine/scheduler.py:97`，按顺序判断：

| 顺序 | 条件 | 动作 | 结果 |
| --- | --- | --- | --- |
| 1 | 序列的块出现在本步 `blocks_to_swap_in` 里 | 取消这次 swap-in，调用 `_recompute` | 强制 RECOMPUTE |
| 2 | `can_swap_out(seq)` 为真 | `swap_out`，序列进入 `swapped` | SWAP，KV 在 CPU |
| 3 | 其余情况 | `_recompute` | RECOMPUTE，KV 丢弃 |

第一条是 swap-in 的护栏：本步刚 swap-in 的块，CPU→GPU 的拷贝还没执行，GPU 块里是旧数据；这时再 swap-out 会把垃圾写回 CPU，覆盖真正的 KV，所以只能重算。

`can_swap_out(seq)` 有两个条件：CPU 空闲块不少于 `len(seq.block_table)`，并且序列所有块的 `ref_count == 1`。

## swap 数据路径

`kv_swap.swap_blocks(src, dst, mapping)` 的实现只有一行核心拷贝：

```python
dst[:, :, dst_id].copy_(src[:, :, src_id])
```

一次切片覆盖所有层和 K/V。方向由参数顺序决定：`swap_blocks(gpu, cpu, {gpu_id: cpu_id})` 是换出，`swap_blocks(cpu, gpu, {cpu_id: gpu_id})` 是换入。

`swap_out` 会释放 GPU 块、清空 `seq.block_table`，但保留 `num_cached_tokens`，这是回来断点续跑的凭证。`swap_in` 重新分配 GPU 块、重建 `block_table`、释放 CPU 块，并为 `num_cached_tokens // block_size` 个完整块重建哈希链。少了最后一步，这些块会从前缀缓存里消失，后续新块还会用错前缀种子。

在正常启动路径里，`cpu_offload_gb` 默认是 0，`ModelRunner` 会把 `num_cpu_kvcache_blocks` 设为 0，`can_swap_out` 永远为 False，抢占只有 RECOMPUTE 一条路。L03 为了演示 swap，会手动把 `num_cpu_kvcache_blocks` 设成非 0。

## 本讲边界

- `can_allocate` 不检查最后一个块，所以满的末块不会命中前缀缓存。
- `free_block_ids` 是普通 deque，按 `popleft` 分配，没有 LRU 或更复杂的驱逐策略。
- CPU 块只是搬运目的地，不参与前缀哈希，也没有自己的哈希表。
- 如果 `running` 只剩一条序列、又没有空闲块，`preempt(seq)` 之后本轮没有可调度序列，真实 `Scheduler` 会在 `assert scheduled_seqs` 处失败。L03 的模拟分支只用于展示抢占逻辑，不代表真实引擎能继续跑。
- swap 使用同步 `copy_`，拷贝期间整个 step 停住；nano-vllm 没有做拷贝和计算的重叠。

## 本讲产出

一张“请求状态 → block 操作 → KV 位置”的时序表，加一张 E/E2 对照表。表里至少写清：

- 请求状态（WAITING / RUNNING / SWAPPED / FINISHED）；
- 每一步的 block 操作（`allocate` / `may_append` / `swap_out` / `swap_in` / `deallocate`）；
- KV 在 GPU 还是 CPU；
- 触发抢占的断言（哪次 `can_append` 返回 False）；
- `blocks_to_swap_out` / `blocks_to_swap_in` 和三个计数器。

## 目录

- [课程总览](../README.md)
- [教学大纲](syllabus.md)
- [PPT 提纲](ppt-outline.md)
- [课堂实验](hands-on-exercise.md)
- [课后作业](homework.md)
- [实验环境](lab-environment.md)

## 前后衔接

第 2 讲讲了分页的静态结构；本讲看它在 nano-vllm 里怎么动起来。第 4 讲把同一套问题带到 vLLM V1，对比两套实现的分配、复用和回收。
