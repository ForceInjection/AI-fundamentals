# 第 3 讲：KV Cache 在 nano-vllm 里怎么实现

> 120 分钟 | 课程第 3 讲 | 前置：第 2 讲的 block/page

这一讲沿着 `LLMEngine.step()` 走一遍完整生命周期：请求进入、分配 block、写入 KV、前缀命中、抢占、换出、释放。nano-vllm 代码量小，适合把第 2 讲的抽象映射落到具体函数。

## 先看对象关系

在进入 `step()` 之前，先把两类对象分开：一类只记账，一类持有张量。

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

`Scheduler` 通过 `BlockManager` 改账本，`Sequence.block_table` 记录每个逻辑块落在哪个物理块。真正的 K/V 张量在 `ModelRunner` 里，attention 用 `slot = block_id * block_size + offset` 寻址。账本和显存靠 `block_table` 对齐。

`LLMEngine.add_request()` 在 `step()` 之外：它把 prompt 包成 `Sequence`，交给 `Scheduler.add()` 放进 `waiting`。`step()` 只负责把已经排队的请求推进一步。

## `step()` 的五步

`LLMEngine.step()`（`nanovllm/engine/llm_engine.py:52`）的顺序是固定的：

1. `Scheduler.schedule()`：从 `waiting` / `running` / `swapped` 选本步的序列，填 `blocks_to_swap_out` 和 `blocks_to_swap_in`。
2. `ModelRunner.swap_out()`：按映射把 GPU 块拷到 CPU。
3. `ModelRunner.swap_in()`：按映射把 CPU 块拷回 GPU。
4. `ModelRunner.run()`：prefill 或 decode，写本步的 K/V。
5. `Scheduler.postprocess()`：登记新满块的哈希，推进 `num_cached_tokens`，处理采样结果和释放。

`swap_out → swap_in → run` 的顺序是硬约束：`swap_out` 必须在源 GPU 块被本步 `run()` 覆盖之前读出；如果某个块同时是换出源和换入目标，也必须先 out 后 in。`add_request` 不属于这五步。

## 学习目标

- 找到 `_allocate_block` / `_deallocate_block` / `can_allocate` / `allocate` / `deallocate` / `hash_blocks` 的调用关系，理解前缀哈希链。
- 理解 `ref_count` 如何管理共享 block，什么时候才能回收。
- 看懂 `Scheduler` 的 prefill / decode 状态转换，以及 `can_append` 返回 False 时怎样触发抢占。
- 说清 RECOMPUTE 与 SWAP 两条抢占路径的触发条件和代价。
- 指出 nano-vllm 的实现边界：满的末块不复用、无 LRU、CPU 块不参与前缀哈希。

## 本讲产出

一张“请求状态 → block 操作 → KV 位置”的时序表（可画成图），附一次抢占前后的元数据断言。

## 目录

- [课程总览](../README.md)
- [教学大纲](syllabus.md)
- [PPT 提纲](ppt-outline.md)
- [课堂实验](hands-on-exercise.md)
- [课后作业](homework.md)
- [实验环境](lab-environment.md)

## 前后衔接

第 2 讲讲了分页的静态结构；本讲看它在引擎里怎么动起来。第 4 讲把同一套问题带到 vLLM V1，对比两套实现的分配、复用和回收。
