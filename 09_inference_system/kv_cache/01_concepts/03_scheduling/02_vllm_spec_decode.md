# 投机解码如何与 KV Cache 交互：Placeholder、回滚与草稿 KV

投机解码（Speculative Decoding）让 draft model 一次预测 K 个 token，target model 批量验证后可能全部接受、部分接受、或全部拒绝。从 KV Cache 的视角看，这个"猜和验"的过程引入了三个传统自回归解码中不存在的操作：**为尚未生成的 token 预留 KV 槽位、猜错后回滚 `num_computed_tokens`、以及处理 draft model 和 target model 两组不同形状的 KV**。

本文以 vLLM V1 源码为基准，拆解这三个操作在 block 级 KV 管理中的实现细节及其工程代价。

> **前置阅读**：[投机解码图解](../../../model_optimization/illustrated-speculative-decoding.md) — 投机解码的完整原理、算法家族与性能模型；[PagedAttention 原理介绍](../01_basic/02_paged_attention.md) — block table 与按需分配。

---

## 一、投机解码的 KV 写入：两次 forward，两组 KV

### 1.1 标准 decode vs 投机 decode 的 KV 增量

标准自回归 decode 每步只生成 1 个 token，KV Cache 追加非常规律：`num_computed_tokens` 每步 +1，block table 每 16 步（block_size=16）追加一个新 block。

投机解码的每一步则涉及两次 forward：

```text
Step N:
  Draft forward:  draft model 一次预测 K 个候选 token (如 K=5)
                 → 每个候选 token 都需要参与 attention 计算（读取 base model 各层的 K 和 V）
                 → drafter 层自身的 KV 写入独立的 draft KV cache group（详见 §1.2）

  Verify forward: target model 批量验证这 K 个 token
                 → 为每个 token 写入 target model 的 KV 到 block table
                 → 验证结果：可能接受前 3 个，拒绝后 2 个
                 → 被拒绝的 token 的 KV 已经写入了——怎么处理？
```

一次投机 step 中，target model 的 forward 会产生 K+1 个 token 的 KV 写入（K 个投机 token + 1 个 bonus token），均通过标准的 block table 路径存储。如果 acceptance rate = 80%、K=5，每步净增约 4 个有效 token 的 target KV——是标准 decode 的 ~4-5 倍。Draft 侧还会在独立的 draft KV cache group 中写入 drafter 层自身的 KV（详见 §1.2）；Dual-Model 场景下两组 KV 的形状对齐问题见 §4.2。

### 1.2 Self-Speculation：draft 层有独立的 KV cache group

vLLM 主要支持 Eagle-style self-speculation——draft 和 target 共享同一个 base model，draft head 是一个轻量级附加模块。但"共享 base model"不等于"共享 KV"：**drafter 的 attention 层在 KV cache 布局中构成独立的 group**。vLLM 在构建 KV cache 配置时会把 drafter attention 层聚到 `KVCacheGroupSpec.is_eagle_group=True` 的分组（字段定义见 `vllm/v1/kv_cache_interface.py:1420`，标注逻辑见 `_annotate_eagle_groups`，`vllm/v1/core/kv_cache_utils.py:2162-2218`），它有自己的 block table 和配额。

Draft group 与 target group 的关键差异在**生命周期**：draft KV 服务的是"本轮猜测"。`use_eagle_block_drop`（`vllm/config/speculative.py:1942-1944`）默认开启时，被拒绝的 draft trailing block 会被直接丢弃回收（EAGLE last-block drop，`vllm/v1/core/kv_cache_coordinator.py:105-107`），不会保留到下一轮。因此 draft group 的稳态占用近似"一轮猜测的 K 个 token × drafter 层数"，远小于 target 侧的全模型 KV，但不是零开销。

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

**调度跳过保护**：异步调度下，调度器需要判断"这个请求是否已经不用再调度"。由于 placeholder 计入 computed 计数，判定时要剔除它们——当剔除后已覆盖全部 prompt 与最大生成长度时，直接跳过调度步，避免为一个必然完成的请求多跑一步：

```python
# vllm/v1/core/sched/scheduler.py:636-640
if (
    request.num_output_placeholders > 0
    and request.num_computed_tokens + 2 - request.num_output_placeholders
    >= request.num_prompt_tokens + request.max_tokens
):
    # 确认上一步已抵达 max_tokens，跳过多余的调度步
    req_index += 1
    continue
```

### 2.3 Placeholder 与 block 分配的关系

Placeholder 不直接触发 block 分配——它们是调度器在计算 block 需求时的**预期偏移量**。在 `schedule()` 阶段，`num_output_placeholders` 通过影响 `num_computed_tokens` 的计算偏移（见 §2.2 的 `is_prefill_chunk` 判定），间接改变了 `get_num_new_tokens()` 的返回值，进而影响本轮调度中 `allocate_slots()` 请求的 `num_new_tokens` 数量——这是一种间接的预留机制。实际的物理块分配仍由 `allocate_slots()` 在调度时执行。

---

## 三、回滚：猜错了怎么撤销

### 3.1 回滚的本质：倒退 `num_computed_tokens`

当 target model 验证发现只有前 A 个投机 token 正确（A < K），后 K-A 个被拒绝时，vLLM 无需"删除"被拒绝 token 的 KV：**只需将 `num_computed_tokens` 回退到接受点**：

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

### 3.3 Preemption 与飞行中的 stale token

异步调度引入了更复杂的情况：调度器可能在 GPU 还在执行时就发出了下一步的调度指令。如果此时请求被强制 Preemption（或 `_reset_prefix_cache()` 清空 prefix cache），已经"在飞行中"的投机 token 就变成了 stale data。vLLM 的做法是在 preemption 时直接复位请求状态（`_preempt_request`，`vllm/v1/core/sched/scheduler.py:1570-1581`）：

```python
# vllm/v1/core/sched/scheduler.py（preemption 路径，节选）
request.num_stale_output_tokens = request.num_in_flight_tokens  # 在飞 token 记为 stale
request.num_output_placeholders = 0    # placeholder 立即清零，不再期待它们落地
request.drop_stale_output = drop_stale_output  # reset_prefix_cache 等场景传 True
```

后续 step 收到这批在飞输出时（`_update_request_with_output`，`vllm/v1/core/sched/async_scheduler.py:51-68`）分三种情况处理：

- stale 输出仍然送达（保证 spec-decode 的验收统计不被扰动），但**不再递减** `num_output_placeholders`——preemption 时它已被清零，再减会下溢；
- `drop_stale_output=True` 时（同 step 内 resume 的 reset 场景），整帧输出直接丢弃（`scheduler.py:2040-2041`）；
- 只有非 stale 的输出才走正常的 `cache_blocks()`，把确认后的 token 写入 prefix cache。

这是 KV Cache 视角下投机解码最微妙的角落：回滚的对象是调度层的状态，KV 数据本身留在 block 里等待覆写。

---

## 四、草稿模型的 KV：存不存？怎么存？

### 4.1 Self-Speculation：一层 draft 的短期 KV

Eagle-style self-speculation 下，draft head 是 target model 的一个轻量级附加模块（通常 1 层 decoder）。它的 KV 存放在 §1.2 介绍的独立 draft KV cache group 中：drafter attention 层有自己的 block table，每轮猜测的 K 个 token 写入这个 group，验证后被拒绝的 draft trailing block 由 block-drop 机制回收。

因此 Self-Speculation 的 KV 开销是 "target 全模型 KV + 一层 draft 的短期 KV"——draft 部分是每轮猜测的 K 个 token 的单层 KV，相对 target 侧是小头。整体 KV 增长与标准 decode 基本持平（每步净增 A 个有效 token 而非 1 个），不会有双份全模型存储的压力。

### 4.2 Dual-Model Speculation：形状对齐问题

如果 draft model 和 target model 是独立的两个模型（如 LLaMA-3 70B + LLaMA-3 8B），它们的 KV 形状可能不同：

- **head 数不同**：70B 有 64 Q heads / 8 KV heads，8B 可能只有 32 Q heads / 8 KV heads
- **head_dim 和层数不同**
- **GQA 分组比例不同**

此时 draft KV 和 target KV **不能放入同一个 block pool**——需要独立的 `KVCacheGroupSpec` 管理（vLLM 中 EAGLE/MTP 的 drafter 层正是以独立的 KV cache group 组织，见 §1.2）。Draft model 的 KV 通常在一个独立的小型 block pool 中分配，验证完成后立即释放，不跨 step 保留——这与 target model 的 KV 长期保留形成对比。

当前 vLLM V1 的 speculative decoding 以 Self-Speculation 为主路径：dual-model 的支持通过 Eagle 的 draft head 机制实现——draft head 复用 base model 的 hidden states，其 attention 层的 KV 存放在独立的 draft KV cache group 中；完整的独立 draft model（两个全模型各自持 KV）支持仍在演进中。

---

## 五、与 Prefix Caching 的交互：跨请求命中率低

投机解码和 Prefix Caching 在 vLLM V1 中可以同时启用[^2]，两者并非机制上互斥：token 一旦被确认（验证通过），其 KV 会走正常的 `cache_blocks()` 路径进入 prefix cache（`vllm/v1/core/sched/async_scheduler.py:66-68`）。真正的限制是**跨请求命中率低**，而非"无法缓存"：

| 维度           |            Prefix Caching             |                 Speculative Decoding                 |
| -------------- | :-----------------------------------: | :--------------------------------------------------: |
| KV 内容确定性  | 相同 token 序列 → 相同 KV → 相同 hash |  投机阶段 token 序列未定，验证确认后才固化入缓存     |
| Block 重用方式 |        基于 hash 的跨请求共享         | 确认后的 block 正常入缓存，被拒绝部分被覆写或丢弃    |
| 跨请求命中     |        Prompt 前缀命中率高            |      生成段序列因草稿/采样差异，跨请求命中很少       |

原因在于：投机解码产生的**生成段** token 序列依赖 draft model 的猜测与采样随机性——同一段 prompt，两次请求生成出的 token 序列通常不同，hash 链自然分叉。投机 token 对应的 block 对本请求的 decode 加速有效，但很难被下一个请求复用。而 **Prompt 前缀部分（投机开始前）的 KV 照常可以被 prefix cache 复用**。

限制是 **block 级的而非 token 级的**：如果一个 block 全部由投机 token 组成，它的 hash 对跨请求复用意义不大（下次请求很可能走出不同的序列）；如果一个 block 包含部分 prompt token 和部分投机 token，只有 prompt 部分的 hash 是确定性的——当前实现不支持这种 partial block hashing，因此整个 block 都难以跨请求命中。这意味着在多轮对话等 prefix-heavy 场景下，投机解码 + Prefix Caching 的组合仍然有价值：System Prompt 的 KV 被缓存（全部由 prompt token 组成的 block），后续请求的 prompt 前缀命中缓存，进入 decode 阶段后投机解码加速生成——两者各司其职。

---

## 六、取舍

### 6.1 三个额外开销

| 开销                 | 来源                                          |                量级                 |
| -------------------- | --------------------------------------------- | :---------------------------------: |
| **Placeholder 预留** | `num_output_placeholders` 让 block 分配更激进 | 每步多预留 K 个 token 的 block 空间 |
| **回滚的垃圾 KV**    | 被拒绝 token 的 KV 暂时占据 block             |  每步 K-A 个 token，被覆写前不释放  |
| **Preemption 复位**  | stale 在飞 token 由 `num_stale_output_tokens`/`drop_stale_output` 处理 |  仅在 Preemption / prefix cache reset 时触发  |

### 6.2 小结

**投机解码给 KV Cache 管理引入的核心操作是"先预留、再回滚"——`num_output_placeholders` 在调度层为尚未生成的 token 预留空间，`num_computed_tokens` 的倒退在逻辑层完成回滚，而被拒绝 token 的 KV 在 block 中物理保留直到被覆写。** 这种"乐观写入 + 逻辑回滚"的模式，与标准自回归解码"每步确定性地追加 1 个 token"的简单性形成了鲜明对比。理解和处理这两者之间的张力，是在推理引擎中正确集成投机解码的关键。

---

## 相关阅读

- [投机解码图解](../../../model_optimization/illustrated-speculative-decoding.md) — 投机解码的完整原理、算法家族与性能模型
- [PagedAttention 原理介绍](../01_basic/02_paged_attention.md) — Block table、按需分配、碎片率
- [vLLM Chunked Prefill 与 KV Cache](01_vllm_chunked_prefill.md) — `num_computed_tokens` 在 Chunked Prefill 中的作用
- [Attention Sinks 与 KV Cache 淘汰策略](../05_eviction/01_attention_sinks_and_eviction.md) — KV Cache 满了怎么淘汰

[^1]: vLLM V1 异步调度器源码 [`vllm/v1/core/sched/async_scheduler.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/async_scheduler.py) — `num_output_placeholders` 在每次调度投机 step 时增加 `num_sampled_tokens_per_step + cur_num_spec_tokens`，在 reject 时减去 `num_rejected`。

[^2]: vLLM V1 调度器 [`vllm/v1/core/sched/scheduler.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py) — `num_computed_tokens -= num_rejected` 实现回滚；preemption 时 `_preempt_request` 将 `num_output_placeholders` 清零并记录 `num_stale_output_tokens`（scheduler.py:1570-1581），stale 输出在 `_update_request_with_output`（async_scheduler.py:51-68）中按 `drop_stale_output` 丢弃或跳过计数。
