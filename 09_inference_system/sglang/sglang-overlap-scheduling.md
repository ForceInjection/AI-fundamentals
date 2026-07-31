# SGLang Overlap Scheduling 深度解析：CPU 与 GPU 的双流水线

> 2026-07-31 | 基于 SGLang v0.5.14 `scheduler.py` 源码分析

LLM 推理的一轮 iteration 包含两部分工作：GPU 上的模型 forward（矩阵乘法、attention）、CPU 上的结果处理（采样、更新 KV cache 元数据、组装下一轮输入）。标准调度器让它们串行：GPU 算完 → CPU 处理 → GPU 再算下一轮 → CPU 再处理。问题在于，GPU 在算的时候 CPU 在闲着，CPU 在处理的时候 GPU 在闲着。每次空闲都直接转化为延迟——对 decode 来说，每多等 1 ms，TTFT 和 TPOT 就多 1 ms。

SGLang 的 overlap scheduling 用一条额外的 output queue 打破了这个串行依赖。核心洞察一句话：**GPU 算 batch N 的时候，CPU 同时在处理 batch N-1 的结果。** 两个流水线非完全重叠——采样必须在 CPU 处理之后——但 forward 与 result processing 之间的重叠已经足以将 CPU 空闲时间从总延迟中剔除。

本文从 SGLang 源码出发，拆解 overlap scheduling 的四个关键机制：双流水线循环、FutureMap 输入 relay、DP attention 协调、以及不适合 overlap 的特殊场景。

---

## 一、为什么需要 overlap

### 1.1 标准调度器的串行瓶颈

标准的事件循环（`event_loop_normal`）中，每一步是串行的：

```text
Step N:
  recv_requests()          ← CPU
  get_next_batch_to_run()  ← CPU
  run_batch(batch)         ← GPU forward (compute-bound)
  process_batch_result()   ← CPU: sampling, metadata update, output
  launch_batch_sample()    ← CPU: grammar-guided sampling

Step N+1:
  recv_requests()          ← CPU
  ...
```

GPU forward 是这轮中最长的操作——decode 单 token 约 5-12 ms。CPU 处理约 0.5-2 ms。串行意味着每个 iteration 的总延迟 = GPU 时间 + CPU 时间，且 GPU 在算的时候 CPU 完全空闲。

对于高吞吐场景（大 batch），GPU 时间主导，CPU 空闲占比低，串行不是瓶颈。但对于延迟敏感场景（小 batch、低并发），CPU 空闲在端到端延迟中的占比可达 10-20%——每减少 1 ms 都对用户体验有可感知的影响。

### 1.2 核心洞察：本轮 GPU forward 与上轮 CPU 处理可以并行

两轮 iteration 之间存在天然的数据依赖分析：

```text
GPU(batch N)  依赖:  batch N 的输入 tensor（由 CPU 在 step N-1 组装）
CPU(batch N)  依赖:  GPU(batch N) 的结果，以及 CPU(batch N-1) 的部分状态
```

关键发现：**GPU(batch N) 不依赖 CPU(batch N-1) 的全部结果**——只需要 CPU(batch N-1) 组装的输入 tensor。CPU(batch N-1) 的其余工作（采样后处理、输出发送、元数据更新）可以与 GPU(batch N) 完全并行。

SGLang 的 overlap scheduling 正是利用这个依赖间隙：在 GPU 执行 batch N 的同时，CPU 处理 batch N-1 的结果（包括采样和元数据更新），然后 batch N 的采样在 batch N-1 的处理完成后立即启动——此时 GPU(batch N) 的结果已经返回或即将返回。

---

## 二、双流水线循环：result_queue 与延迟处理

上一节从依赖分析的角度解释了 overlap 为什么可行：`GPU(batch N)` 与 `CPU(batch N-1)` 可并行。本节看 SGLang 的代码如何实现——核心是一条 `result_queue`，将本应在上轮末尾执行的 CPU 处理推迟到本轮 GPU forward 启动之后。

### 2.1 event_loop_overlap 的整体结构

`event_loop_overlap`（`scheduler.py:1535`）是 overlap scheduling 的主循环。核心数据结构是 `result_queue`——一个 `(batch, result)` 的队列，用于延迟处理：

```python
# scheduler.py:1535-1658, simplified
def event_loop_overlap(self):
    self.result_queue: Deque[Tuple[ScheduleBatch, Result]] = deque()

    while True:
        recv_reqs = self.request_receiver.recv_requests()  # ① 收请求
        self.process_input_requests(recv_reqs)

        batch = self.get_next_batch_to_run(...)              # ② 组建 batch N

        if disable_overlap_for_batch:
            pop_and_process()                               # ← 不能 overlap，立即处理上轮

        if batch:
            batch_result = self.run_batch(batch)            # ③ 启动 GPU(batch N)
            #                                          ↑ PyTorch forward 异步启动 GPU kernel
            #                                            后立即返回，CPU 不等待 GPU 完成
            self.result_queue.append((batch.copy(), batch_result))
            # ↑ 此时 GPU 在执行 batch N，CPU 继续往下走

        if self.last_batch:
            if not disable_overlap_for_batch:
                pop_and_process()                           # ④ CPU 处理 batch N-1
        #                                                    （与 GPU(batch N) 并行！）

        if self.is_generation:
            self.launch_batch_sample_if_needed(batch_result)      # ⑤ 采样（单参数 API）
        self.last_batch = batch
```

将这个循环展开为时间线，overlap 的效果就很清晰了：

```text
非 overlap:
  │  GPU(batch0)  │CPU0│  GPU(batch1)  │CPU1│  GPU(batch2)  │CPU2│
  └──────────────────────────────────────────────────────────────→ 时间

overlap:
  │  GPU(batch0)  │     GPU(batch1)     │     GPU(batch2)     │
  │      │CPU0   ││        │CPU1       ││        │CPU2       ││
  └──────────────────────────────────────────────────────────────→ 时间
                ↑ GPU(batch1) 与 CPU0 并行
```

CPU 处理被"藏"在 GPU forward 的时间里——从串行变成并行，端到端延迟直观缩短。

### 2.2 pop_and_process：延迟处理的执行体

`pop_and_process` 从 `result_queue` 左侧弹出上一轮的 `(batch, result)`，调用 `process_batch_result`（`scheduler.py:3367`）——这条路径处理 decode 和 prefill 的结果，将 output token 发回 tokenizer、更新 metrics、发送健康检查信号。重要的是，这一切发生在 GPU 正在执行当前 batch forward 的同时。

### 2.3 采样时机的协调

采样（`launch_batch_sample_if_needed`，`scheduler.py:3335`）必须在 `pop_and_process` 之后调用——因为采样依赖上轮 CPU 处理更新了请求状态（如 grammar 约束）。但采样本身不阻塞 GPU——它发生在 CPU 侧，与下一轮的 GPU 准备（收请求、组 batch）并行。

---

## 三、FutureMap：跨 iteration 的输入 relay

`result_queue` 解决了 CPU-GPU 的时序重叠，但它引入了一个新问题：上一轮的采样结果如何传给下一轮的 GPU 输入？串行模式下直接读 `req.output_ids` 就行，但 overlap 模式中采样被推迟了——下游消费者拿不到数据。FutureMap 用一个 device-side relay 缓冲区解耦了这对生产者-消费者。

### 3.1 问题：采样结果如何传到下一轮 GPU 输入

`event_loop_normal` 中，采样结果直接写入 `req.output_ids`，下一轮 `get_next_batch_to_run` 可以直接读取——因为所有操作是串行的，不存在竞态。但 overlap 模式下，采样发生在 GPU(batch N-1) 的处理之后——而 GPU(batch N) 的输入 tensor 已经在 batch N 组建时确定。如果 batch N 需要 batch N-1 的采样结果（decode 场景），直接读取会拿到旧值。

### 3.2 FutureMap：采样结果的提前 relay

SGLang 用 `FutureMap` 解决这个问题。`FutureMap` 在本轮 batch 采样完成后，将刚刚产生的 output tokens（`r.output_ids[-1]`）通过 `stash` 推入一个 device-side relay 缓冲区（`scheduler.py:2668`）：

```python
last_tokens = torch.tensor(
    [r.output_ids[-1] for r in reqs], dtype=torch.int64, device=device
)
self.future_map.stash(
    batch.req_pool_indices, RelayPayload(bonus_tokens=last_tokens)
)
```

下一轮 batch 组建 GPU 输入时，通过 `resolve_forward_inputs(batch, self.future_map)`（`scheduler.py:3187`）从 relay 缓冲区中取出上一轮的 output tokens，拼入当前 batch 的 input tensor——这些值是在上一轮采样后立即写入的，不受本轮的 CPU-GPU 竞态影响。最后 `future_map.publish` 更新 seq_lens 供再下一轮使用。

`FutureMap` 本质是一个"双缓冲 relay"——上轮的采样输出在 CPU 处理阶段写入，下轮的 GPU 输入在 forward 前取出，两者通过 relay 解耦，不直接读写同一份状态。

---

## 四、DP Attention 中的 overlap 协调

`result_queue` 和 `FutureMap` 解决了单 rank 的时序和 relay 问题。但在多 rank 场景（DP Attention）中，overlap 决策本身也必须跨 rank 一致——否则 all-reduce 会挂死。本节拆解 `is_disable_overlap_for_batch` 中 DP 同步的机制。

### 4.1 问题：多 DP rank 的不一致

DP Attention 模式下，多个 DP rank 各自独立做 attention 计算，然后通过 all-reduce 聚合结果。如果不同 rank 在 overlap 模式下做出不同的调度决策（例如 rank 0 决定 overlap 而 rank 1 不 overlap），会导致 all-reduce 挂死——部分 rank 先发起通信，其他 rank 还在执行不同路径。

### 4.2 同步决策

DP Attention 模式下，`is_disable_overlap_for_batch`（`scheduler.py:1594`）通过 `require_mlp_sync` 标志自动切换到全局同步的判断逻辑：

```python
# scheduler.py:1602-1605
if self.require_mlp_sync:
    is_extend = lambda b: b and b.is_extend_in_batch  # 全局同步的 flag
else:
    is_extend = lambda b: b and b.forward_mode.is_extend()  # 本地的 forward mode
```

当 `require_mlp_sync=True` 时，判断"是否为 prefill"的依据从本地 `forward_mode` 切换为跨 rank 同步后的 `is_extend_in_batch`——这确保所有 DP rank 在 `is_disable_overlap_for_batch` 中得到相同的布尔值，从而在是否调用 `pop_and_process` 上保持步调一致，避免 all-reduce 死锁。

---

## 五、不适合 overlap 的场景

overlap 并非无代价。它引入了一层间接（result_queue）+ relay 开销（FutureMap），并且在连续 prefill 和 spec+grammar 两种场景下并行反而会伤害延迟或正确性。`is_disable_overlap_for_batch`（`scheduler.py:1594`）是这两个条件的集中判断：

### 5.1 连续 prefill batch

两轮 iteration 都是 prefill 时，overlap 被禁用。原因：prefill 的 TTFT 是最关键的用户感知指标。如果 GPU(prefill N) 与 CPU(prefill N-1) 重叠，prefill N-1 的采样延迟会推迟 prefill N 的 TTFT——与"降低 TTFT"的目标直接冲突。由环境变量控制（默认关闭，即默认允许 prefill 间 overlap）：

```python
# environ.py:315
SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP = EnvBool(False)
```

### 5.2 投机解码 + grammar 场景

当 decode batch 同时使用投机解码和 grammar 约束且 `result_queue` 非空时，overlap 被临时关闭。原因：grammar 约束下的采样依赖步骤间状态的一致性，overlap 延迟了上轮采样，可能导致 grammar 状态错位。源码中 `is_disable_overlap_for_batch` 的第二个条件（`scheduler.py:1614-1620`）：

```python
need_grammar_sync = (
    batch and not batch.spec_algorithm.is_none()
    and batch.has_grammar
    and batch.forward_mode.is_decode()
    and len(self.result_queue) > 0
)
```

### 5.3 last_batch 为空

没有上一轮的结果需要处理时，直接进入下一轮。

---

## 六、与非 overlap 模式的对比

| 维度         | `event_loop_normal`              | `event_loop_overlap`                          |
| ------------ | -------------------------------- | --------------------------------------------- |
| CPU-GPU 关系 | 串行：GPU 算完 → CPU 处理 → 循环 | 并行：GPU(batch N) 与 CPU(batch N-1) 同时执行 |
| 延迟结构     | 每轮 = GPU_time + CPU_time       | 每轮 ≈ max(GPU_time, CPU_time) + 少量同步开销 |
| 复杂度       | 低（单线顺序）                   | 中（result_queue + FutureMap relay）          |
| 适用场景     | 大 batch、GPU 时间 >> CPU 时间   | 小 batch、延迟敏感                            |
| Decode 收益  | 基准                             | 延迟降低 ~5-10%（典型小 batch 场景）          |
| Prefill 收益 | 基准                             | 默认禁用（连续 prefill 不 overlap）           |

---

## 七、总结

SGLang 的 overlap scheduling 解决的是一个简单但重要的问题：**标准推理循环中 CPU 在 GPU 计算期间空闲**。它的解决方案同样简洁：用一条 `result_queue` 将上轮的 CPU 处理推迟到本轮的 GPU forward 期间执行，用 `FutureMap` 打破 GPU 对 CPU 采样结果的数据依赖。

与 vLLM V1 的 token-level unified scheduling 不同，overlap 不是调度策略层面的创新——它不改变"谁被调度"，只改变"CPU 工作和 GPU 工作的时间关系"。但正是这种低层的时序优化，让 SGLang 在延迟敏感场景（低并发、小 batch、实时对话）中获得了相对 vLLM 的延迟优势。

---

## 延伸阅读

- [SGLang Scheduler 源码](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py) — `event_loop_overlap()` (L1590)，`process_batch_result()` (L3547)
- [SGLang 调度器：请求编排与 Batch 调度](sglang-scheduler.md)
- [Continuous Batching 深度解析：从静态批处理到迭代级调度](../prefill_decode/continuous_batching.md)
