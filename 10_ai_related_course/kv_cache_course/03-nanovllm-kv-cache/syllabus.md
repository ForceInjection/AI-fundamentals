# 第 3 讲：KV Cache 在 nano-vllm 里怎么实现 — 教学大纲

## 基本信息

| 项目 | 内容 |
| --- | --- |
| 课时 | 120 分钟 |
| 先修 | 第 2 讲的 block/page；能读 Python |
| 代码基线 | 本地 nano-vllm，HEAD `b9dd55a` |
| 本讲定位 | 用最小实现串起 KV 的分配、复用、抢占和换出 |

## 教学目标

- **知识目标**：理解 BlockManager、前缀哈希、引用计数、调度和 swap 的协作方式，能说出 `step()` 五步的固定顺序。
- **能力目标**：能在 nano-vllm 里定位一次请求的 KV 生命周期，并解释每个状态转换。
- **素养目标**：区分“最小实现够用”与“生产引擎必须补”的边界。

## 时间安排

| 时间 | 环节 | 内容 |
| --- | --- | --- |
| 30 min | 概念 | 对象关系：`LLMEngine → Scheduler → BlockManager/Sequence`、`ModelRunner → GPU/CPU KV tensor`；`step()` 五步；block 分配与释放；前缀哈希链与碰撞保护；`ref_count`；prefill / decode；`can_append` 不足时触发抢占；RECOMPUTE 与 SWAP |
| 50 min | 源码 | 逐段读 `BlockManager`、`Scheduler`、`ModelRunner`、`Attention`；对照 `kv_swap` 的同步拷贝；指出满末块、无 LRU、CPU 块无前缀哈希三个边界 |
| 30 min | 实验 | 跑 CPU 测试和 `L03_scheduler.py`；核对场景 E/E2 的抢占路径、block 数和状态 |
| 10 min | 答疑 | 回答哈希链、引用计数、`can_append` 的返回条件、CPU 拷贝开销 |

## 重点与难点

| 类型 | 内容 | 处理方式 |
| --- | --- | --- |
| 重点 | 一次请求的 block 生命周期 | 从 `step()` 入口按调用顺序走一遍 |
| 重点 | 前缀哈希与 `ref_count` | 用共享前缀的两个请求画引用变化 |
| 难点 | 抢占的触发条件 | 强调 `can_append(seq)` 因空闲块不足返回 False，不是某个“抢占阈值” |
| 难点 | preempt 后的三条分支 | 先判本步已 swap-in，再判 `can_swap_out`，最后退回 RECOMPUTE |
| 难点 | 最小实现的边界 | 明确列出 nano-vllm 不做的事，不把“没实现”说成“不需要” |

## 关键调用顺序

`LLMEngine.step()` 的实际顺序：

```text
schedule() → swap_out() → swap_in() → run() → postprocess()
```

`add_request()` 在 `step()` 之外，只把 `Sequence` 放进 `waiting`。`swap_out → swap_in → run` 不可重排：换出源块可能在本步被换入目标或 `run()` 复用。

`Scheduler.schedule()` 内部按 `waiting`（prefill）→ `swapped`（swap-in）→ `running`（decode）的顺序推进。decode 阶段对每个序列调用 `BlockManager.can_append(seq)`；返回 False 时，抢占 `running` 队尾的序列再重试。若 `running` 已空，会抢占当前序列自身；但真实 Scheduler 随后会在 `assert scheduled_seqs` 处失败，这是 nano-vllm 的边界。L03 的模拟分支只用于展示抢占逻辑，不代表真实引擎能继续跑。

## 抢占的三条分支

`Scheduler.preempt()`（`nanovllm/engine/scheduler.py:97`）：

1. 序列的块出现在本步 `blocks_to_swap_in` 中：取消这次 swap-in，强制 RECOMPUTE。原因是 GPU 块里的 KV 还没拷回，换出会把无效数据写进 CPU 副本。
2. 否则调用 `BlockManager.can_swap_out(seq)`：CPU 空闲块足够，且序列所有块的 `ref_count == 1`，执行 SWAP。
3. 否则 RECOMPUTE：`_recompute()` 把状态置回 WAITING，`deallocate()` 释放块，序列放回 `waiting` 队首。

## 本讲边界

| 边界 | 代码位置 | 说明 |
| --- | --- | --- |
| 满的末块不复用 | `block_manager.py:66` | `can_allocate` 只扫描 `num_blocks - 1` 个块 |
| 无 LRU | `block_manager.py:47-55` | `free_block_ids` 用 `popleft` 分配，没有更复杂的驱逐策略 |
| CPU 块不参与前缀哈希 | `block_manager.py:34-37` | CPU 块只是 swap 的停放区，没有自己的哈希表 |
| 自身抢占可能空批 | `scheduler.py:82-93` | `preempt(seq)` 后本轮没有可调度序列，`assert scheduled_seqs` 会失败 |
| swap 同步拷贝 | `kv_swap.py:13-14` | `copy_` 执行期间整个 step 停住，没有和计算重叠 |

## 本讲产出

一张时序表，加一张 E/E2 对照表，至少标出：

- 请求状态（WAITING / RUNNING / SWAPPED / FINISHED）；
- 每一步的 block 操作（`allocate` / `may_append` / `swap_out` / `swap_in` / `deallocate`）；
- KV 在 GPU 或 CPU 的位置；
- 触发抢占的断言（哪次 `can_append` 返回 False）。

下表由学员填写，第一行是示例：

| 步骤 | 请求状态 | block 操作 | KV 位置 | 元数据/断言是否一致 |
| --- | --- | --- | --- | --- |
| prefill 完成 | A、B 均为 RUNNING | `allocate` 各分配 1 个 GPU 块 | GPU 2/2，空闲 0 | 一致：两条 `block_table` 长度均为 1 |
| 下一次 decode 调度 | 待填 | 待填 | 待填 | 待填 |
| A 命中 EOS 后 | 待填 | 待填 | 待填 | 待填 |

E/E2 对照表见[课堂实验](hands-on-exercise.md)的任务 2。

## 参考材料

- [nano-vllm](https://github.com/ForceInjection/nano-vllm)
- `~/Project/study/nano-vllm/nanovllm/engine/block_manager.py`：`_allocate_block` L47、`_deallocate_block` L57、`can_allocate` L62、`allocate` L79、`deallocate` L98、`can_append` L107、`may_append` L110、`hash_blocks` L114、`can_swap_out` L135、`swap_out` L142、`swap_in` L161
- `~/Project/study/nano-vllm/nanovllm/engine/scheduler.py`：`schedule` L33、`preempt` L97、`_recompute` L118、`postprocess` L125
- `~/Project/study/nano-vllm/nanovllm/engine/model_runner.py`：`allocate_kv_cache` L104、`swap_out` L132、`swap_in` L136
- `~/Project/study/nano-vllm/nanovllm/engine/kv_swap.py`：`swap_blocks` L1–14，整块切片拷贝
- `~/Project/study/nano-vllm/docs/llm-inference-visual/scripts/L03_scheduler.py`：本讲主实验
- `ai-infra-course/05-vllm-inference/code/trace_nanovllm.py`：pre-swap 参考，不跟踪 `swap_out/swap_in`，不作为主实验
