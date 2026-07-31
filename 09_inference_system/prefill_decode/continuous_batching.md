# Continuous Batching 深度解析：从静态批处理到迭代级调度

> 2026-07-29 | 基于 vLLM v0.27.0+ 与 SGLang v0.5.14 源码分析

LLM 推理引擎的调度器有一个根本矛盾：GPU 一次只算一个 batch，但请求是源源不断到达的，每条请求的输出长度也无法预知。如果 batch 是一条"刻在石头上"的固定名单——建好就不再改变——那么短请求必须等长请求完成才能释放 GPU，新请求必须等整个 batch 结束才能加入。这意味着 GPU 在大量时间里是空的——计算着 padding tokens，等待着一个遥遥无期的 batch 结束。

Continuous Batching 解决了这个问题。它的核心洞察只有一句话：**把 batch 从一个请求级的"静态容器"变成一个迭代级的"动态流体"**——每跑完一次 forward pass，重新决定 batch 里有谁。本文从静态 batching 的浪费出发，剖析 iteration-level scheduling 的核心原理，然后深入 vLLM V1 和 SGLang 的调度器源码，对比两种不同的实现哲学。

---

## 一、GPU 为什么需要 batching

Decode 阶段，每次 forward 只处理每个请求的 1 个新 token。以 7B 模型 FP16 为例：模型权重约 14 GB，每次 forward 必须完整读入计算单元。单 token 的计算量约为 2 × 7B = 14 GFLOPs。在一张 495 TFLOPS（FP16，不含稀疏性）的 H100 上，纯计算只需要 ~0.03 毫秒。但读取 14 GB 权重需要 ~4.2 毫秒（H100 HBM 带宽 ~3.35 TB/s）。

算一个 token：4.2 ms 等数据，0.03 ms 真正在算——**99% 以上的 GPU 时间在搬运权重，不到 1% 在计算。** 这就是 memory-bound 的本质。

batching 的动机由此而来：如果多个请求共享同一次权重读取，内存搬运的固定成本就被摊销了。一个 batch 里有 N 个请求，读取权重的开销仍然是 ~4.2 ms，但计算量增长到了 N × 14 GFLOPs。当 N 足够大时，计算时间追上内存时间，GPU 利用率显著提升。

```text
单请求 decode:  [读权重 4.2ms][算 token ~0.03ms] → GPU 利用率 < 1%
batch=64 decode: [读权重 4.2ms][算 64 token ~1.9ms] → GPU 利用率 ~31%
batch=256 prefill (每请求 4096 token): GEMM 计算密集 → GPU 利用率 ~60–70%
```

> 注：以上为简化的数量级示意。真实利用率受 kernel 融合、Tensor Core 效率、调度开销和多层 pipeline 影响。核心结论不变：decode 单 token 是极度 memory-bound 的，batching 是摊销内存搬运成本的唯一手段。Prefill 的矩阵-矩阵乘法（GEMM）天然更容易打满 GPU，这也是 prefill 阶段 GPU 利用率远高于 decode 的原因。

---

## 二、静态 batching 的三重浪费

### 2.1 Padding：短序列等长序列

静态 batching 要求 batch 中所有请求的序列对齐到同一长度。假设一个 batch 里有 4 个请求，目标是每个生成 10、50、100、500 个 token：

```text
Batch 组建（t=0）:
  Req A: 目标 10 tokens
  Req B: 目标 50 tokens
  Req C: 目标 100 tokens
  Req D: 目标 500 tokens
  输入 tensor 形状: (4, max_seq_len=500)

t=10: Req A 完成，但 batch 未结束 → GPU 继续算 A 的位置（padding = 0 向量）
t=50: Req B 完成 → A 已经空转了 40 步
t=100: Req C 完成 → A 空转 90 步，B 空转 50 步
t=500: Req D 完成 → batch 结束
```

Req A 的有效计算只占它总占用时间的 2%（10/500）。Req B 占 10%（50/500）。超过一半的 GPU 时间在计算 padding——这是字面意义上的"算空气"。

### 2.2 请求级锁定：batch 是铁笼子

静态 batching 的另一个问题：batch 组建后，在其生命周期内不能增减成员。

一个新请求在 t=20 到达。它必须等当前 batch 完全结束——也就是 t=500——才能加入下一批。这意味着它的 TTFT（首 token 延迟）至少是 480 步的 decode 时间，即使这个请求的 prompt 只有 10 个 token。

反方向同样浪费：Req A 在 t=10 就完成了，但它占用的 GPU 显存（KV cache）在 t=500 之前都无法释放。这些被锁定的显存本可以用来容纳更多请求，但现在只能空置。

### 2.3 输出长度的不可预测性

LLM 的输出长度是高度可变的。同一个 prompt "介绍一下 Transformer"，模型可能输出 50 个 token 的简短回答，也可能输出 2000 个 token 的详尽论述。静态 batching 无法预知哪个请求会最长，所以只能假设每个请求都生成到 `max_tokens` 上限——这又进一步放大了 padding 浪费。

那为什么早期推理系统还要用静态 batching？因为在 PagedAttention 之前，KV cache 是连续分配的——每个请求的 K、V 矩阵存储在预分配的连续内存中，大小在 batch 组建时就固定了。没有灵活的内存管理，就无法在运行时插入或删除请求。

---

## 三、核心洞察：从请求级到迭代级

### 3.1 语义跃迁：batch 不再是"容器"

静态 batching 中，"batch" 和"一组请求的整个生命周期"是绑定的——一个 batch 是一次完整的、不可分割的请求组合。

Continuous Batching 将这个绑定拆开。**调度粒度从"请求级"降到了"迭代级"**：每一次 forward pass（一个 iteration）就是一个独立的调度决策点。在这个决策点上，调度器问三个问题：

1. **谁完成了？** → 释放 KV cache，移出 batch
2. **谁在排队？** → 如果显存够，插入 batch
3. **谁该被抢占？** → 如果显存不够，换出或重算

用动态视角重写 2.1 的例子：

```text
t=0:   batch = {A, B, C, D}  → forward
t=10:  A 生成 EOS → 释放 A  → batch = {B, C, D}
       新请求 E 到达 → 显存够 → batch = {B, C, D, E}
t=50:  B 生成 EOS → 释放 B  → batch = {C, D, E}
       ...
```

每一步 forward 之后，batch 的组成都可以不同。Gone 是"batch 锁定"，取代的是"迭代级流体"。

### 3.2 为什么这件事以前做不了

Continuous Batching 在概念上并不复杂——每个 iteration 结束后重新决定 batch 成员。但工程上，这需要 KV cache 的内存管理满足两个条件：

1. **分配是细粒度的**：不能一次预分配整个请求的全部 KV cache（因为输出长度未知），必须按需逐块分配
2. **释放是离散的**：释放一个请求的 KV cache 不能要求释放一大片连续内存（因为那会留下碎片），必须是离散的、可以独立回收的单元

PagedAttention 的 block table 提供了这两个条件。它将 KV cache 切分为固定大小的 block（如 256 token/block），每个请求持有一张 block table——一个"逻辑位置 → 物理 block"的映射表。分配一个请求的 KV cache 就是分配一组空闲 block 并更新其 block table；释放就是把这些 block 标记为空闲。block 之间互不干扰，不存在碎片问题。

**PagedAttention 不是 Continuous Batching 的唯一前提，但它是让 Continuous Batching 可以工程实现的关键基础设施。** 没有 block table 的间接寻址，分配和释放的粒度就不够灵活，迭代级调度就无法在每一步 forward 之间完成。

### 3.3 调度循环的通用结构

无论 vLLM 还是 SGLang，Continuous Batching 的调度循环都遵循相同的骨架：

```text
while True:
    recv_requests()                     # 收新请求 → waiting queue
    batch = decide_what_to_run()        # 决策：谁进 batch
    run_batch(batch)                    # GPU forward
    process_results(batch)              # 处理输出，标记完成
```

三行伪代码，八千行源码——差异全在 `decide_what_to_run()` 里。vLLM 和 SGLang 在这个决策函数中做了截然不同的选择。

---

## 四、vLLM V1：token-level 的统一调度

### 4.1 取消 prefill/decode 的二元对立

vLLM V1 调度器最激进的设计选择在源码注释中表达得很清楚（`scheduler.py:398-407`）：

```python
# NOTE(woosuk) on the scheduling algorithm:
# There's no "decoding phase" nor "prefill phase" in the scheduler.
# Each request just has the num_computed_tokens and
# num_tokens_with_spec. num_tokens_with_spec =
# len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
# At each step, the scheduler tries to assign tokens to the requests
# so that each request's num_computed_tokens can catch up its
# num_tokens_with_spec.
```

传统调度器把请求分为"正在 prefill 的"和"正在 decode 的"两类，分别处理。vLLM V1 取消了这种分类。所有请求都是同一种东西：每个请求有一个 `num_computed_tokens`（已经算完的 token 数）和一个 `num_tokens_with_spec`（需要算到的目标 token 数，包括 prompt tokens、已生成的 output tokens 和投机解码的 spec tokens）。调度器的唯一工作是：**在 token budget 和显存的约束下，推进每个请求的 `num_computed_tokens` 追赶 `num_tokens_with_spec`。**

这个模型统一了所有场景：

| 场景                         | `num_computed_tokens` | `num_tokens_with_spec` | 差距（需要调度的 token 数） |
| ---------------------------- | --------------------- | ---------------------- | --------------------------- |
| 新请求（prompt 1000 tokens） | 0                     | 1000                   | 1000（chunked prefill）     |
| decode 请求                  | 1001                  | 1002                   | 1                           |
| 投机解码 decode              | 1002                  | 1009                   | 7                           |

Token-level scheduling 的最大优势在于：**chunked prefill 不再需要特殊处理**——长 prompt 被 `max_num_scheduled_tokens` 自动截断，不需要单独的"chunked prefill 阶段"或独立的调度路径。

### 4.2 调度循环：RUNNING → WAITING → 后处理

每次 forward pass 的调度 `schedule()`（`scheduler.py:396`）内部分为两个阶段，第三个阶段 `_update_after_schedule` 在 forward 完成后由引擎主循环调用。三者构成一个完整的调度循环：

**阶段一：调度 RUNNING 请求**（`schedule()` 内, `scheduler.py:441-634`）

遍历 `self.running` 列表中的每个请求，对每个请求计算 `num_new_tokens = num_tokens_with_spec - num_computed_tokens`。这个值：

- 对于 decode 请求 = 1（+ 投机 token 数）
- 对于 prefill 请求 = 剩余的 prompt token 数，受 `max_num_scheduled_tokens` 限制

然后尝试 `kv_cache_manager.allocate_slots(request, num_new_tokens)`——为这 `num_new_tokens` 个 token 分配 KV cache block。如果分配成功，请求被标记为 `scheduled`，消耗对应的 token budget。

如果分配失败（没有足够的空闲 block），触发 **preemption**：

- PRIORITY 模式下，选取优先级最低且到达时间最晚的 running 请求作为受害者
- FCFS 模式下，从 running 列表末尾弹出最近加入的请求
- 调用 `_preempt_request()` 释放其 block，重置状态，放回 waiting queue
- 然后重试 `allocate_slots`

**阶段二：调度 WAITING 请求**（`schedule()` 内, `scheduler.py:637-1050+`）

如果没有 preemption 在执行，调度器从 waiting queue 中依次取出请求，为新请求分配 KV cache block。同样受 token budget 和 KV cache 空间限制。新请求的 `num_new_tokens` 通常是整个 prompt（或其第一个 chunk）。

**阶段三：后处理**（`_update_after_schedule`, `scheduler.py:1164`，`schedule()` 返回后由引擎主循环调用）

推进每个被调度的请求的 `num_computed_tokens`，更新 `is_prefill_chunk` 标志（当 `num_computed_tokens < num_tokens_with_spec` 时为 True，表示该请求在下一个 iteration 仍需调度 prefill tokens）。

### 4.3 preemption：温和驱逐

vLLM 的 `_preempt_request()`（`scheduler.py:1140`）执行以下操作：

```python
def _preempt_request(self, request: Request, timestamp: float) -> None:
    self._free_request_blocks(request)       # 释放 KV cache block
    self.encoder_cache_manager.free(request) # 释放 encoder cache
    request.status = RequestStatus.PREEMPTED
    request.num_computed_tokens = 0          # 重置进度
    request.num_preemptions += 1
    self.waiting.prepend_request(request)    # 放回 waiting queue 头部
```

被抢占的请求的 `num_computed_tokens` 被重置为 0——这意味着恢复时需要重新 prefill 整个 prompt。但这不意味着所有计算都浪费了：如果该请求的 prompt 与其他请求共享 prefix，prefix cache 中已经存在的 block 不会被释放（引用计数 > 1），重新调度时可以直接命中。在 prefix 共享场景下，"重新 prefill" 的实际开销远小于从头计算。

### 4.4 完成与回收

请求完成后（生成 EOS 或达到 max_tokens），在 `update_from_output()`（`scheduler.py:1499`）中处理。`_free_request()` 将请求标记为 `FINISHED`，记录到 `finished_req_ids`，并立即释放 KV cache block（`scheduler.py:2108-2114`）。`defer_block_free` 在异步/重叠调度且启用 KV Connector（consumer 侧）时激活，此时 block 释放被延后到 GPU 侧的写入操作完成后——但在常见路径下，block 的回收是即时的。

---

## 五、SGLang：prefill 优先与主动驱逐

SGLang 的调度哲学与 vLLM V1 有一个根本差异：**SGLang 显式区分 prefill 和 decode，且 prefill 永远优先。**

### 5.1 get_next_batch_to_run()：prefill first

`get_next_batch_to_run()`（`scheduler.py:2684`）是 SGLang 的调度决策入口。其核心逻辑可以简化为：

```text
将上一个 prefill batch 中已完成的请求合并到 running_batch

new_batch = 尝试组建 prefill batch（PrefillAdder）

if new_batch is not None:
    return new_batch                    # ← prefill batch
elif running_batch is not empty:
    return update_running_batch()       # ← decode batch
else:
    return None                         # ← 空闲
```

**有 prefill 就做 prefill，没有 prefill 才做 decode。** 这带来了一个直接后果：在持续有请求到达的场景下，decode 请求会一直被推迟，直到没有新的 prefill 需要处理。这种"饥饿"是有意为之——SGLang 认为降低新请求的 TTFT 比维持现有请求的 TPOT 更重要。

### 5.2 PrefillAdder：五维预算

`PrefillAdder`（`schedule_policy.py:441`）是组建 prefill batch 的核心类。它像一个预算管理器，在五个维度上同时约束 prefill batch 的大小：

| 预算          | 变量/检查                    | 含义                                                              |
| ------------- | ---------------------------- | ----------------------------------------------------------------- |
| KV cache 空间 | `rem_total_tokens`           | L1 剩余空间。每次 admit 扣除 `extend + max_new + page`            |
| 计算量        | `rem_input_tokens`           | 本轮所有请求累计 prefill token 上限，防 GPU 中间激活值溢出        |
| 公平性        | `rem_chunk_tokens`           | 单个请求最多一次 prefill 这么多 token（= `chunked_prefill_size`） |
| 请求数        | `prefill_max_requests`       | 每轮 prefill batch 最多 admit 多少个请求                          |
| 请求槽位      | `get_num_allocatable_reqs()` | `req_to_token_pool` 剩余槽位数                                    |

任何一维预算耗尽，本轮 prefill batch 关闭。这种多维约束的精细化程度超过了 vLLM 的单一 `token_budget` 模型——SGLang 在公平性（`rem_chunk_tokens`）这个维度上做了显式的单请求上限控制。

但五维预算的复杂度也带来了一个问题：在某些极端参数组合下，某个维度会在其他维度之前频繁耗尽，导致其他预算始终未能充分利用。一个更灵活的预算仲裁机制（在各维度间动态分配权重）是可能的改进方向。

### 5.3 update_running_batch() 与 retract_decode

`update_running_batch()`（`scheduler.py:3135`）负责 decode batch 的维护：

```text
1. filter_batch()     → 移除已完成请求（EOS / abort）
2. if batch 空 → 返回 None
3. if KV cache 空间不够 → retract_decode()
4. prepare_for_decode() → 构建 decode 输入 tensor
```

**retract_decode** 是 SGLang 区别于 vLLM 最具标志性的机制。当 decode batch 自身的 KV cache 空间不足（通常是因为 prefill 持续 admit 新请求挤压了 decode 预算），`check_decode_mem` 失败，调度器主动驱逐正在 decode 的请求（`scheduler.py:3163`）：

```python
retracted_reqs, new_token_ratio, reqs_to_abort = batch.retract_decode(self.server_args)
```

被驱逐的请求回到 waiting_queue（`_add_request_to_queue(req, is_retracted=True)`），等待重新 prefill。这意味着一个请求可能在 decode 了一半的时候被突然"踢出"——它的已生成 token 不丢失（保存在请求对象中），但寄存在 GPU 上的 KV cache 全被丢弃，恢复时需要重新 prefill 整个 `prompt + 已生成 token` 序列。

**retract 和 preemption 的区别**：

- vLLM 的 preemption 是**保护性**的：显存不足时才被动驱逐，被驱逐的请求可能是任意 running 请求
- SGLang 的 retract 是**主动**的：只要有新 prefill 需要空间，就主动驱逐 decode 请求。新请求的 TTFT 优先于已在运行的请求的流畅性

### 5.4 SGLang 的请求状态机

SGLang 的请求生命周期比 vLLM 多了一条回退路径：

```text
Tokenizer 发来请求
       │
       ▼
  waiting_queue ────────→ prefill batch ────────→ running_batch (decode)
       ▲                        │                       │
       │                        │                       │
       └── retract_decode ──────┘                       │
         （被踢回 waiting_queue）                         │
                                                  生成 EOS → 完成
```

vLLM 的 preemption 也可能将请求从 running 移回 waiting，但在实际生产中，preemption 的触发频率远低于 SGLang 的 retract——因为 SGLang 的 prefill-first 策略天然更激进地消耗 KV cache，导致 decode 被驱逐的概率更高。

---

## 六、两种哲学的对比

### 6.1 时间复杂度：SGLang O(N) vs vLLM O(N²) 的误解

调度器复杂度常被用来比较两个框架，SGLang 需要逐请求越权检查导致 O(N²) 是一种常见批评。但两个框架的纯 decode 维护路径都是 O(N) 量级：filter 移除已完成请求（O(N)）、check_decode_mem 逐请求统计下轮页需求（也是 O(N)）、prepare_for_decode 逐请求分配 1 token 页（O(N)）——这些遍历虽然都是线性，但没有嵌套的"请求×请求"配对操作。真正的复杂度差异不在 decode 维护路径，而在 prefill 准入阶段——SGLang 的 PrefillAdder 对每个候选请求做前缀匹配（radix tree 查找），匹配开销与树深度和 KV cache 规模相关，这才是可能产生超线性的地方，但也不是简单 O(N²)。

### 6.2 调度优先级

| 维度          | vLLM V1                                     | SGLang                             |
| ------------- | ------------------------------------------- | ---------------------------------- |
| 优先级策略    | 无显式 prefill/decode 优先，统一 token 推进 | 显式 prefill > decode              |
| waiting 顺序  | FIFO（可通过 policy 配置）                  | FIFO（prefill batch 决定挑选顺序） |
| prefill 延迟  | 中等（decode 也可能在同一轮被调度）         | 低（prefill 优先抢资源）           |
| decode 稳定性 | 较高（preemption 触发条件保守）             | 较低（retract 触发条件激进）       |

### 6.3 preemption vs retract

| 维度                | vLLM preemption                     | SGLang retract                            |
| ------------------- | ----------------------------------- | ----------------------------------------- |
| 触发条件            | `allocate_slots` 失败（被动）       | `check_decode_mem` 失败（主动）           |
| 驱逐对象            | 优先级最低 / 队尾的 running 请求    | 正在 decode 的请求（由 retract 策略选择） |
| 状态保存            | 不保存（`num_computed_tokens=0`）   | 不保存（KV cache 丢弃）                   |
| 恢复方式            | 重新 prefill（prefix cache 可命中） | 重新 prefill（prefix cache 可命中）       |
| 触发频率            | 低（保守的显存使用策略）            | 中–高（激进的显存抢占策略）               |
| 已生成 token 丢失？ | 否（保存在 request.output_ids 中）  | 否（保存在 req.output_ids 中）            |

两种机制都依赖 prefix caching 来降低重新 prefill 的开销。如果被驱逐的请求的 prompt 或 prefix 仍在 L1 中（未被其他请求覆盖），重新 prefill 时这些 token 可以直接命中 cache——省掉了大量重复计算。但在 prefix hit rate 低的场景（如单轮对话、随机 prompt），retract 的代价是显著的：每个被驱逐的请求必须完整重算所有 token 的注意力。

### 6.4 chunked prefill 的集成方式

vLLM V1 的 token-level scheduling 将 chunked prefill 消解为 `max_num_scheduled_tokens` 的自然结果——不需要独立的 chunk 调度路径。SGLang 则在 `PrefillAdder` 中通过 `rem_chunk_tokens` 显式控制每个请求的单次 prefill 上限，且 chunked 请求有独立的 `add_chunked_req()` 续传路径。

两种方式在目的上等价（防止长 prefill 阻塞 decode），但在调度器的代码结构上差异很大：vLLM 把 chunking 当作 token budget 的副产品，SGLang 把它当作显式的调度策略。

### 6.5 TTFT 与 TPOT 的权衡

SGLang 的 prefill-first 策略在低负载下 TTFT 表现更好——新请求几乎不需要等待。但在高负载下，"永不饥饿的 prefill"意味着 decode 请求可能被频繁 retract，导致已被延迟的 decode 请求进一步被延迟。TPOT 尾部（P99）在这种场景下可能显著恶化。

vLLM V1 没有 prefill/decode 的显式优先级——两者通过共享的 token budget 竞争调度。这种"公平"设计在混合负载下通常提供更稳定的 TPOT，但代价是 TTFT 可能略高——尤其在大量长 prefill 堆积时。

这不是谁对谁错的问题。在线服务如果更关心新用户体验（如搜索、对话助手的首 token 感知延迟），SGLang 的 prefill-first 更合适。如果更关心已有用户的生成不被中断（如流式输出的一致性），vLLM 的温和 preemption 更合适。

---

## 七、与 KV Cache 管理的关系

### 7.1 block table 的间接寻址

PagedAttention 的 block table 为 continuous batching 提供了底层支撑，但其设计是围绕"灵活分配"而非"动态调度"展开的。Continuous batching 需要的不仅是灵活分配——它要求分配和释放可以在单个 iteration 内完成。

block table 满足了这个要求：分配一组 block（更新 page table entry）、释放一组 block（标记空闲）、移动一组 block（CPU↔GPU 拷贝 + 更新映射）——都是 O(block_count) 的操作。这保证了调度器可以在 `allocate_slots` 和 `_free_request_blocks` 中高效地操作显存。

### 7.2 prefix caching 的叠加效应

多个请求共享前缀时，它们的 block table 指向同一组物理 block。当一个请求完成或被抢占时，这些共享 block 不能直接释放——其他请求还在用。这需要引用计数机制：每个物理 block 维护被多少个请求引用，释放时减计数，计数归零才真正回收。

这给 continuous batching 增加了一层约束：**释放一个请求的 KV cache 不一定回收它所声称的全部显存。** 被共享的 prefix block 会"延迟解放"——直到最后一个引用它的请求也完成。这个约束在高 prefix hit rate 的场景（如多次相同 system prompt）下，可能导致 free block 数量低于预期，触发更频繁的 preemption 或 retract。

### 7.3 留给 decode 的空间

每次 admit 一个新请求时，KV cache 预算不仅要覆盖它当前需要 prefill 的 token（`extend_input_len`），还必须**预留给它未来要生成的所有 token**。在 SGLang 的 `PrefillAdder` 中，这个预留量 = `max_new × ratio - 已生成`（`max_new_tokens` 乘以一个预估的生成比率）。

一个生成 4096 个 token 的请求，它的"预期未来"远超它的"当前大小"。这个预留量直接决定了 running batch 能容纳多少个并发请求——预留太保守（ratio 大）则 batch size 小、吞吐上不去；预留太激进（ratio 小）则频繁 retract、体验退化。SGLang 的 `new_token_ratio` 跟踪机制（`new_token_ratio_tracker`）正是用来动态调整这个比率的——用实际观测到的 token 生成率来校准预估值。

---

## 八、总结

Continuous Batching 的本质是一个粒度跃迁：batch 从"请求级容器"变成了"迭代级流体"。这个跃迁之所以能实现，需要三个条件的成立：

1. **PagedAttention 的 block table** 提供了细粒度的分配和释放——没有这层间接寻址，迭代级调度无法在每个 forward pass 之间完成内存操作
2. **迭代级调度循环** 在每步 forward 之后重新评估 batch 组成——谁完成了、谁在排队、谁该被抢占，三个决策在一轮 iteration 内完成
3. **KV cache 预算管理** 在 admit 新请求时预留 decode 空间，在释放完成请求时回收 block，在空间不足时驱逐 running 请求

vLLM V1 和 SGLang 在这个共同框架下做了不同的选择：

- vLLM V1 取消了 prefill/decode 的二元对立，将所有请求统一为 `num_computed_tokens` → `num_tokens_with_spec` 的追赶过程。preemption 是保护性的，保守地驱逐 running 请求。chunked prefill 是 token budget 的自然结果，不需要独立调度路径。

- SGLang 显式维护 prefill > decode 的优先级。retract 是主动的，激进地为新请求抢出 KV cache 空间。chunked prefill 通过 PrefillAdder 的五维预算控制，chunked 请求有独立的续传路径。

两个框架谁更好？这取决于你更关心新用户的等待（TTFT）还是现有用户的流畅（TPOT）。SGLang 选择了前者，vLLM 选择了公平——这两种选择反过来塑造了各自框架的调度气质。

---

## 延伸阅读

- [vLLM V1 Scheduler 源码](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py) — `schedule()` 方法（L396），`_preempt_request()`（L1140），`update_from_output()`（L1499）
- [SGLang Scheduler 源码](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py) — `get_next_batch_to_run()`（L2684），`update_running_batch()`（L3135）
- [SGLang Schedule Policy 源码](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/schedule_policy.py) — `PrefillAdder`（L441），`add_one_req()`（L815）
- [SGLang 调度器：请求编排与 Batch 调度](../sglang/sglang-scheduler.md) — SGLang 调度器的完整源码分析
- [SGLang Chunked Prefill — 原理与代码实现](../sglang/chunked_prefill.md)
- [PagedAttention 退役的技术原因](../vllm/module_analysis/pagedattention_retirement.md)
- [SGLang KV Cache Pool 三层管理模型](../sglang/sglang-kv-pool-management.md)
