# SGLang 调度器：请求编排与 Batch 调度

> 2026-07-24 | 基于 sglang v0.5.14 源码分析

sglang 调度器是推理引擎的中央决策者：每一轮 GPU 计算执行什么任务、处理哪些请求，都由它决定。核心矛盾在于，新请求的 prefill（算出第一个 token）和已有请求的 decode（继续生成后续 token）竞争同一块 GPU——优先 prefill 能降低首 token 延迟（TTFT），但会推迟 decode 导致吞吐（TPOT）下降；优先 decode 则新请求永远排不上队。调度器的所有复杂性，都源于在这两者之间找到平衡。

---

## 一、核心循环与调度优先级

在深入调度逻辑之前，先明确两个基本概念：

| 概念        | 是什么                                 | 生命周期                                       |
| ----------- | -------------------------------------- | ---------------------------------------------- |
| **Request** | 一个用户请求（一次 conversation turn） | 到达 → waiting_queue → prefill → decode → 完成 |
| **Batch**   | 一次 GPU forward 中同时处理的请求集合  | 组建 → forward → 结果处理 → 完成               |

### 1.1 事件循环

`scheduler.py` 的 `event_loop_normal()` 是一个无限循环，每次迭代跑一次 GPU forward：

```text
while True:
    recv_requests()              ← 收新请求 → waiting_queue
    batch = get_next_batch_to_run()  ← 决策：prefill 还是 decode
    run_batch(batch)             ← GPU forward + 处理结果
```

**每步只跑一种 batch。batch 之间串行，batch 内部并行。**

| 维度           | 串行/并行 | 原因                               |
| -------------- | --------- | ---------------------------------- |
| batch **之间** | 串行      | GPU 一次只 forward 一个 batch      |
| batch **内部** | 并行      | 多请求的 tokens 拼成 tensor 同时算 |

### 1.2 chunked prefill 的影响

长 prompt 无法一次 prefill 完（受 GPU 显存和公平性限制），需要拆成多个 chunk 分步执行——这就是 chunked prefill。`chunked_prefill_size` 控制每个 chunk 的最大 token 数。

没有 chunked prefill 时，长 prompt 会长时间阻塞 decode：

```text
时间 →
[========== 150K prefill ==========] → [decode] → [decode]
              ↑ 这段时间其他请求完全无 token 输出，ITL 飙升
```

有 chunked prefill 时（size=8192），长 prompt 被拆分，穿插 decode：

```text
[prefill 8K] → [decode] → [decode] → [prefill 8K] → [decode] → [decode] → ...
                ↑ 其他请求每 ~300ms 就能插队生成一个 token
```

### 1.3 调度优先级：Prefill > Decode

`get_next_batch_to_run()` 的决策逻辑（`scheduler.py:2684`）：

```text
new_batch = get_new_batch_prefill()   # 尝试组建 prefill batch

if new_batch:
    return new_batch                   # ← 有 prefill 就做 prefill
elif running_batch:
    return update_running_batch()      # ← 没 prefill 才做 decode
else:
    return None                        # ← 空闲
```

**Prefill 永远优先。** 原因：

- Prefill 是延迟敏感操作——新请求等着出第一个 token（用户感知的 TTFT）
- Decode 已在线上运行，多等一步影响不大
- 如果 decode 优先，新请求永远排不上（饥饿）

---

## 二、Request 与 Batch

### 2.1 Request 与 Batch 的关系

一个 Request **穿过**多个 Batch：

- Prompt 处理阶段：可能被拆成多个 **prefill batch**（chunking）
- Token 生成阶段：持续待在 **running batch** 中，每步产出 1 个 token

**约束**：一个 Request 不能同时出现在两个 Batch 中。如果请求正在 prefill batch 中被处理，它就不在 decode batch 中——反之亦然。这保证了 KV cache 一致性。

### 2.2 完整状态机

```text
Tokenizer 发来请求
       │
       ▼
  waiting_queue ──────→ prefill batch (一次 GPU prefill forward)
       ▲                        │
       │                   is_chunked > 0?
       │                   ┌─Yes→ 保留在调度器，下一轮继续 prefill
       │                   │
       │                   No
       │                   │
       │                   ▼
       │              running_batch (decode forward，每步 1 token)
       │                   │
       │              output_ids.append(next_token)
       │                   │
       │          ┌─ finished? → 返回 Tokenizer
       │          │
       └── retract_decode ─┘（L1 满了，被踢回 waiting_queue）
```

### 2.3 调度示意

```text
Request A ──┐
Request B ──┤  waiting_queue
Request C ──┘       │
                    ▼
            Prefill Batch #1
            ┌─────────────────────┐
            │ A (完整 prefill)     │  ← extend_input_len 小，一次完成
            │ B (chunk 1/3)       │  ← extend_input_len 大，截断
            └─────────────────────┘
                    │ A 完成 prefill → 进入 decode
                    │ B is_chunked=2 → 继续等待

            Prefill Batch #2
            ┌─────────────────────┐
            │ C (完整 prefill)     │
            │ B (chunk 2/3)       │  ← 继续处理 B 的剩余 tokens
            └─────────────────────┘
                    │ C 完成 prefill → 进入 decode
                    │ B is_chunked=1 → 继续等待

            Prefill Batch #3
            ┌─────────────────────┐
            │ B (chunk 3/3)       │  ← B 的最后一个 chunk
            └─────────────────────┘
                    │ B 完成 prefill → 进入 decode

            Decode Batch（持续运行，每步产出 1 token/请求）
            ┌──────────────────────────────┐
            │ A (生成 token_1)              │
            │ B (生成 token_1)              │
            │ C (生成 token_1)              │
            │ ...更多请求...                 │
            └──────────────────────────────┘
                    │ 重复，直到各请求 finished
                    ▼
                 返回 Tokenizer
```

> 浏览器打开 [scheduler-visual.html](assets/scheduler-visual.html) 查看带颜色标注的交互版本。

---

## 三、Prefill Batch 组建

核心是 `PrefillAdder`（`schedule_policy.py:441`）。它像一个预算管理器：

```text
PrefillAdder 参数:
    rem_input_tokens   ← 整个 batch 最多 prefill 多少 tokens（由 GPU 显存决定）
    rem_chunk_tokens   ← 单个请求最多 prefill 多少 tokens（= chunked_prefill_size）
    can_run_list       ← 本轮被 admit 的请求列表

从 waiting_queue 逐个取请求:
    1. init_next_round_input()
       ├─ 重建 fill_ids
       ├─ tree_cache.match_prefix() → prefix_indices（缓存命中部分）
       └─ extend_input_len = fill_ids长度 - prefix_indices长度

    2. add_one_req()
       ├─ extend_input_len ≤ rem_chunk_tokens → 完整 admit
       └─ extend_input_len > rem_chunk_tokens → 截断为 chunk，标记 is_chunked

    3. 预算用完或 waiting_queue 空 → batch 关闭
```

### 3.1 五种准入预算

| 预算          | 变量/检查                    | 级别       | 含义                                                                                        |
| ------------- | ---------------------------- | ---------- | ------------------------------------------------------------------------------------------- |
| KV cache 空间 | `rem_total_tokens`           | batch 总量 | `available_size + evictable_size - offset`。每个请求 admit 后扣减 `extend + max_new + page` |
| 计算量        | `rem_input_tokens`           | batch 总量 | 所有请求累计 prefill token 上限，防 GPU 中间激活值溢出                                      |
| 公平性        | `rem_chunk_tokens`           | 单请求上限 | 单个请求最多一次 prefill 这么多 token（= `chunked_prefill_size`）                           |
| 请求数        | `prefill_max_requests`       | 单批上限   | 每轮 prefill batch 最多 admit 多少个请求                                                    |
| 请求槽位      | `get_num_allocatable_reqs()` | batch 总量 | `req_to_token_pool` 剩余槽位数——已满则设置 `batch_is_full`，循环 break                      |

检查失败时的行为：

| 预算          | 失败返回   | 效果                                        |
| ------------- | ---------- | ------------------------------------------- |
| KV cache 空间 | `NO_TOKEN` | 设置 `batch_is_full`，本轮不再 admit 新请求 |
| 计算量        | `NO_TOKEN` | 同上                                        |
| 公平性        | `OTHER`    | 直接 break，跳过当前请求                    |
| 请求数        | `OTHER`    | 同上                                        |
| 请求槽位      | —          | 在循环体中直接设置 `batch_is_full` 后 break |

任何 break 都终止本轮循环，排在后面的请求被跳过，等待时间累计到 `idle_in_queue_ms`。

### 3.2 KV cache 空间预算

```text
total_tokens = extend_input_len + max_new + page_size
```

- `extend_input_len`：需要新 prefill 的 token（不含 prefix cache 已命中部分）
- `max_new`：预估输出 token 数（`max_new_tokens × ratio - 已生成`）——**预留 decode 阶段 KV cache 空间**
- `page_size`：页对齐开销（最多一页）

已有 prefix cache（`prefix_indices`）**不重复扣预算**——它已占用 L1，`_req_inc_lock_ref` 加锁防止被驱逐。

### 3.3 Prefill 日志解读

每条 Prefill 日志是一次 GPU forward，处理一组请求的 extend tokens 并行计算。decode 日志有采样机制（`scheduler_components/metrics_reporter.py:714`，`forward_ct_decode % decode_log_interval != 0` 时跳过，默认间隔 40），prefill 无此限制——每次 EXTEND forward 都输出日志。

关键字段：

| 字段            | 含义                                                                                                                                       |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `#new-seq`      | 本轮 batch 中的请求总数（含新 admit + 续 chunked 请求）。字段名有误导性——代码中 `can_run_list` 同时包含 `add_one_req` 和 `add_chunked_req` |
| `#new-token`    | 本轮需 GPU 计算的 extend tokens 总量                                                                                                       |
| `#cached-token` | 命中前缀缓存的 tokens（不重算）                                                                                                            |
| `#running-req`  | batch 中总请求数（= `#new-seq` + 已在 batch 中的 decode 转入请求）                                                                         |

---

## 四、Decode Batch 管理

已在生成的请求组成 `running_batch`。每次 decode step（`update_running_batch`, `scheduler.py:3135`）：

```text
update_running_batch():
    1. filter_batch() → 移除已完成请求
    2. if batch 空 → 返回 None
    3. if KV cache 空间不够 → retract_decode()
       └─ 驱逐部分 decode 请求，让出空间给 prefill，被驱逐请求回到 waiting_queue
    4. 返回更新后的 running_batch（后续 run_batch 调用 prepare_for_decode 构建输入 tensor）
```

**retract_decode** 是内存压力保护：L1 满了但还有新 prefill 请求 → 主动驱逐 decode 请求的 KV cache → 被驱逐的请求回到 waiting_queue 等待重新 prefill。极端情况下出现"请求被中断"就是这个机制导致的。

---

## 五、关键时序

单请求的 TTFT 可以从 scheduler 日志中拆解：

```text
queue_ms    = wait_queue_entry_time - scheduler_recv_time      ← Tokenizer → 入队
schedule_ms = forward_entry_time    - wait_queue_entry_time    ← 入队 → 被调度
forward_ms  = prefill_finished_time - forward_entry_time       ← GPU prefill 计算
─────────────────────────────────────────────────────────────
TTFT_ms     = queue_ms + schedule_ms + forward_ms
```

| 阶段          | 含义                                               | 说明                                     |
| ------------- | -------------------------------------------------- | ---------------------------------------- |
| `queue_ms`    | Tokenizer 到入队                                   | 受网络传输和 tokenization 影响，通常很小 |
| `schedule_ms` | 入队到被调度（含 waiting_queue 排队 + chunk 等待） | 主要瓶颈，受并发和 chunk 机制影响        |
| `forward_ms`  | GPU prefill 前向计算                               | 受 `extend_input_len` 和 batch 大小影响  |
| `TTFT_ms`     | 端到端首 token 延迟                                | `queue_ms + schedule_ms + forward_ms`    |

---

## 六、相关代码位置

| 文件                                                 | 方法/类                          | 作用                             |
| ---------------------------------------------------- | -------------------------------- | -------------------------------- |
| `scheduler.py:2684`                                  | `get_next_batch_to_run()`        | 调度决策入口                     |
| `scheduler.py:2844`                                  | `_get_new_batch_prefill_raw()`   | 组建 prefill batch               |
| `scheduler_components/batch_result_processor.py:180` | `process_batch_result_prefill()` | prefill 结果处理                 |
| `schedule_policy.py:441`                             | `PrefillAdder`                   | chunk 截断决策                   |
| `schedule_policy.py:976`                             | `add_one_req()`                  | 单请求 admit 逻辑                |
| `schedule_batch.py:1162`                             | `init_next_round_input()`        | 前缀匹配 + extend_input_len 计算 |
