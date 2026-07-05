# 投机解码如何与 KV Cache 交互：Placeholder、回滚与草稿 KV

投机解码（Speculative Decoding）让 draft model 一次预测 K 个 token，target model 批量验证后可能全部接受、部分接受、或全部拒绝。从 KV Cache 的视角看，这个"猜和验"的过程引入了三个传统自回归解码中不存在的操作：**为尚未生成的 token 预留 KV 槽位、猜错后回滚 `num_computed_tokens`、以及处理 draft model 和 target model 两组不同形状的 KV**。

本文以 vLLM V1 源码为基准，拆解这三个操作在 block 级 KV 管理中的实现细节及其工程代价。

> **前置阅读**：[投机解码图解](../../model_optimization/illustrated-speculative-decoding.md) — 投机解码的完整原理、算法家族与性能模型；[PagedAttention 原理介绍](../basic/paged_attention.md) — block table 与按需分配。

---

## 一、投机解码的 KV 写入：两次 forward，两组 KV

### 1.1 标准 decode vs 投机 decode 的 KV 增量

标准自回归 decode 每步只生成 1 个 token，KV Cache 追加非常规律：`num_computed_tokens` 每步 +1，block table 每 16 步（block_size=16）追加一个新 block。

投机解码的每一步则涉及两次 forward：

```text
Step N:
  Draft forward:  draft model 一次预测 K 个候选 token (如 K=5)
                 → 每个候选 token 都需要参与 attention 计算（读取 base model 各层的 K 和 V）
                 → Self-Speculation 下不产生独立 KV（详见 §1.2）

  Verify forward: target model 批量验证这 K 个 token
                 → 为每个 token 写入 target model 的 KV 到 block table
                 → 验证结果：可能接受前 3 个，拒绝后 2 个
                 → 被拒绝的 token 的 KV 已经写入了——怎么处理？
```

一次投机 step 中，target model 的 forward 会产生 K+1 个 token 的 KV 写入（K 个投机 token + 1 个 bonus token），均通过标准的 block table 路径存储。如果 acceptance rate = 80%、K=5，每步净增约 4 个有效 token 的 target KV——是标准 decode 的 ~4-5 倍。Self-Speculation 下 draft 阶段没有独立的 block 分配开销（详见 §1.2）；Dual-Model 场景下 draft model 的 KV 则通过独立的张量管理，见 §4.2。

### 1.2 Self-Speculation：draft 和 target 共用模型

vLLM 主要支持 Eagle-style self-speculation——draft 和 target 共享同一个 base model，draft head 是一个轻量级附加模块。这意味着 **draft 阶段不产生独立的 KV Cache**：draft forward 需要读取 base model 各层的 K 和 V 来做 attention，这些 K 和 V 正是 target model forward 过程中已写入 block table 的那些——draft forward 只读取，不额外写入。Draft head 自身的中间张量（如最后一层 hidden states）通过 `SpecDecodeWorker` 内部的 `extra_keys`/`extra_values` 独立管理，**不进入标准的 block table 分配路径**。

> 本文后续讨论聚焦于 Self-Speculation 场景（target KV 即为最终有效 KV），Draft-model-based speculation（两个独立模型）的 KV 对齐问题仅在 §4.2 中简要对比。

---

## 二、Placeholder：为"还没生成的 token"预留 KV 槽位

### 2.1 问题：异步调度下的时间差

在 vLLM V1 的异步调度模式（async scheduling）下，调度器在 GPU 还在执行上一步时就开始准备下一步的 batch。对于投机解码，这意味着：调度器需要在知道 draft model 会生成多少个 token **之前**，就为这些 token 预留 KV Cache 空间。

### 2.2 `num_output_placeholders` 的机制

vLLM 用一个计数器 `num_output_placeholders` 来表示"当前 step 预期会产生但尚未确认的 token 数"[^1]：

```python
# vllm/v1/core/sched/async_scheduler.py
# 每次调度投机 step 时：
cur_num_spec_tokens = len(spec_decode_tokens.get(req_id, ()))
request.num_output_placeholders += (
    self.num_sampled_tokens_per_step + cur_num_spec_tokens
)
```

`num_output_placeholders` 影响三个关键行为：

**Block 分配**：调度器在 `allocate_slots()` 中为 `num_output_placeholders` 预留 block 空间，确保 KV Cache 有足够的物理页来容纳即将到来的投机 token 的 KV。

**`num_computed_tokens` 的偏移**：在计算 `is_prefill_chunk` 条件时，`num_output_placeholders` 被加到 `num_tokens` 上——调度器把 placeholder 视为"即将计算但尚未完成的 token"，避免请求被错误地标记为 prefill 完成状态：

```python
# vllm/v1/core/sched/scheduler.py
is_prefill_chunk = (
    request.num_computed_tokens
    < request.num_tokens + request.num_output_placeholders
)
```

**Prefill 结束判定保护**：在检查 prefill 是否应该结束时，`num_output_placeholders` 防止调度器在投机 token 写入之前错误地认为 prefill 已完成：

```python
if (
    request.num_computed_tokens + 2 - request.num_output_placeholders
    < request.num_prompt_tokens
):
    return  # 还有 prompt token 未处理，不能结束 prefill
```

### 2.3 Placeholder 与 block 分配的关系

Placeholder 不直接触发 block 分配——它们是调度器在计算 block 需求时的**预期偏移量**。在 `schedule()` 阶段，`num_output_placeholders` 通过影响 `num_computed_tokens` 的计算偏移（见 §2.2 的 `is_prefill_chunk` 判定），间接改变了 `get_num_new_tokens()` 的返回值，进而影响本轮调度中 `allocate_slots()` 请求的 `num_new_tokens` 数量——这是一种间接的预留机制。实际的物理块分配仍由 `allocate_slots()` 在调度时执行。

---

## 三、回滚：猜错了怎么撤销

### 3.1 回滚的本质：倒退 `num_computed_tokens`

当 target model 验证发现只有前 A 个投机 token 正确（A < K），后 K-A 个被拒绝时，vLLM 不是去"删除"被拒绝 token 的 KV——而是**将 `num_computed_tokens` 回退到接受点**：

```python
# vllm/v1/core/sched/scheduler.py
# 投机验证后处理被拒绝的 token：
if request.num_computed_tokens > 0:
    request.num_computed_tokens -= num_rejected  # num_rejected = K - A

# 同步回退 placeholder 计数：
if request.num_output_placeholders > 0:
    request.num_output_placeholders -= num_rejected
```

被拒绝 token 的 KV 数据**仍然物理存在于 block 中**，但 `num_computed_tokens` 的倒退意味着：下一步 decode 时，模型会从接受点继续计算，新的 KV 会**覆盖**被拒绝 token 在 block 中的位置。不需要显式"删除"——PagedAttention 通过 `num_computed_tokens` 确定哪些位置的 KV 是有效的。

### 3.2 Block 粒度下的"部分回滚"

因为 KV 存储在 block 中（block_size=16），回滚可能跨越 block 边界，也可能不跨越：

```text
block_size = 16

回滚前：num_computed_tokens = 1024 (block 64 的第 0 个位置)
  step: draft=5, accepted=2, rejected=3
回滚后：num_computed_tokens = 1021 (block 63 的第 13 个位置)

block 63: [token 1008-1023] — 后 3 个位置的 KV 变为"垃圾数据"
                                下次 decode 到这些位置时会被覆写
```

**block 本身不会被释放**——只有整个 block 的所有 token 都被判定为无效时（如 Preemption）才会释放。投机解码的回滚只在 token 级别操作 `num_computed_tokens`，block 级别不做任何变动。

### 3.3 `async_tokens_to_discard`：处理飞行中的 stale token

异步调度引入了更复杂的情况：调度器可能在 GPU 还在执行时就发出了下一步的调度指令。如果此时触发了强制 Preemption 或 `_reset_prefix_cache()` 清空 prefix cache，已经"在飞行中"的投机 token 就变成了 stale data。vLLM 通过 `async_tokens_to_discard` 来丢弃这些 token：

```python
# 强制 preempt 时，标记所有在飞的 placeholder 为待丢弃
request.async_tokens_to_discard = request.num_output_placeholders
request.num_output_placeholders = 0

# 后续 step 中，逐个消费 discard 计数器，跳过 stale 帧
if request.async_tokens_to_discard > 0:
    request.async_tokens_to_discard -= 1
    return [], False  # 丢弃这一帧，不更新状态
```

这是 KV Cache 视角下投机解码最微妙的角落——不是在计算层回滚 KV，而是在**调度层回滚状态**。

---

## 四、草稿模型的 KV：存不存？怎么存？

### 4.1 Self-Speculation：几乎零额外开销

Eagle-style self-speculation 下，draft head 是 target model 的一个轻量级附加模块。整个 base model（包括所有 attention 层）的 K 和 V 与 target model 完全相同——draft forward 直接读取 target forward 已写入 block table 的这些 KV，不额外写入。Draft head 自身的中间张量（如最后一层 hidden states）通过 `extra_keys`/`extra_values` 独立管理，不进入 block table 分配路径，**不产生独立的 KV Cache**。

因此 Self-Speculation 的 KV 开销不是 "target KV + draft KV"，而仅仅是 "target KV + 额外 K 个被接受 token 的 KV"——整体 KV 增长与标准 decode 基本持平（每步净增 A 个 token 而非 1 个），但不会有双份存储的压力。

### 4.2 Dual-Model Speculation：形状对齐问题

如果 draft model 和 target model 是独立的两个模型（如 LLaMA-3 70B + LLaMA-3 8B），它们的 KV 形状可能不同：

- **head 数不同**：70B 有 64 Q heads / 8 KV heads，8B 可能只有 32 Q heads / 8 KV heads
- **head_dim 和层数不同**
- **GQA 分组比例不同**

此时 draft KV 和 target KV **不能放入同一个 block pool**——需要独立的 `KVCacheGroupSpec` 管理。Draft model 的 KV 通常在一个独立的小型 block pool 中分配，验证完成后立即释放，不跨 step 保留——这与 target model 的 KV 长期保留形成对比。当前 vLLM V1 以 Self-Speculation 为主路径，Dual-Model 场景更多通过 Eagle 的 draft head 机制实现（复用 base model hidden states，不产生独立 KV），完整 Dual-Model 的支持仍在演进中。

当前 vLLM V1 的 speculative decoding 以 Self-Speculation 为主路径，dual-model 的支持通过 Eagle 的 draft head 机制实现——draft head 复用 base model 的 hidden states，不产生独立 KV。

---

## 五、与 Prefix Caching 的冲突：投机 token 的 KV 无法缓存

投机解码和 Prefix Caching 在 vLLM V1 中可以同时启用[^2]，但**投机 token 产生的 KV 不能被 Prefix Caching 复用**。两者对 KV 确定性的假设存在根本冲突：

| 假设           |            Prefix Caching             |              Speculative Decoding              |
| -------------- | :-----------------------------------: | :--------------------------------------------: |
| KV 内容确定性  | 相同 token 序列 → 相同 KV → 相同 hash | 投机 token 的 KV 取决于 draft model 的随机采样 |
| Block 重用方式 |        基于 hash 的跨请求共享         |    投机 token 被拒绝后 KV 被覆写，hash 无效    |
| 有效范围       |       从 prompt 开始的连续前缀        | 仅 prompt 前缀部分（投机 token 开始前）可缓存  |

核心冲突在于：投机解码产生的 KV 序列**不是确定性的**——同一段 prompt，两次投机可能猜出不同的 token 序列，产生不同的 KV。但**Prompt 前缀部分（投机开始前）的 KV 仍然可以被 prefix cache 正常复用**——冲突只影响投机 token 自身产生的 KV block。

冲突是 **block 级的而非 token 级的**：如果一个 block 全部由投机 token 组成，它的 hash 对 prefix cache 无意义（因为下次投机可能产生不同的 token 序列）。如果一个 block 包含部分 prompt token 和部分投机 token，只有 prompt 部分的 hash 是确定性的——当前实现不支持这种 partial block hashing，因此整个 block 都无法被缓存。这意味着在多轮对话等 prefix-heavy 场景下，投机解码 + Prefix Caching 的组合仍然有价值：System Prompt 的 KV 被缓存（全部由 prompt token 组成的 block），后续请求的 prompt 前缀命中缓存，进入 decode 阶段后投机解码加速生成——两者各司其职，互不冲突。

---

## 六、取舍

### 6.1 三个额外开销

| 开销                 | 来源                                          |                量级                 |
| -------------------- | --------------------------------------------- | :---------------------------------: |
| **Placeholder 预留** | `num_output_placeholders` 让 block 分配更激进 | 每步多预留 K 个 token 的 block 空间 |
| **回滚的垃圾 KV**    | 被拒绝 token 的 KV 暂时占据 block             |  每步 K-A 个 token，被覆写前不释放  |
| **异步丢弃**         | `async_tokens_to_discard` 需要逐步清理        |       仅在 Preemption 时触发        |

### 6.2 一句话总结

**投机解码给 KV Cache 管理引入的核心操作是"先预留、再回滚"——`num_output_placeholders` 在调度层为尚未生成的 token 预留空间，`num_computed_tokens` 的倒退在逻辑层完成回滚，而被拒绝 token 的 KV 在 block 中物理保留直到被覆写。** 这种"乐观写入 + 逻辑回滚"的模式，与标准自回归解码"每步确定性地追加 1 个 token"的简单性形成了鲜明对比。理解和处理这两者之间的张力，是在推理引擎中正确集成投机解码的关键。

---

## 相关阅读

- [投机解码图解](../../model_optimization/illustrated-speculative-decoding.md) — 投机解码的完整原理、算法家族与性能模型
- [PagedAttention 原理介绍](../basic/paged_attention.md) — Block table、按需分配、碎片率
- [vLLM Chunked Prefill 与 KV Cache](vllm_chunked_prefill_kv_cache.md) — `num_computed_tokens` 在 Chunked Prefill 中的作用
- [Attention Sinks 与 KV Cache 淘汰策略](../eviction/attention_sinks_and_eviction.md) — KV Cache 满了怎么淘汰

[^1]: vLLM V1 异步调度器源码 [`vllm/v1/core/sched/async_scheduler.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/async_scheduler.py) — `num_output_placeholders` 在每次调度投机 step 时增加 `num_sampled_tokens_per_step + cur_num_spec_tokens`，在 reject 时减去 `num_rejected`。

[^2]: vLLM V1 调度器 [`vllm/v1/core/sched/scheduler.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py) — `num_computed_tokens -= num_rejected` 实现回滚；`async_tokens_to_discard` 在 preemption 时标记，在后续 `_update_request_with_output` 中逐帧丢弃。
