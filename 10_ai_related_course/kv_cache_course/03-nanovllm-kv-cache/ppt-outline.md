# 第 3 讲：KV Cache 在 nano-vllm 里怎么实现 — PPT 大纲

## 设计说明

- **总页数**：20 页
- **演讲时长**：120 分钟（30 min 概念 + 50 min 源码/演示 + 30 min 实验 + 10 min 答疑）
- **建议风格**：深蓝 + 暖灰；元数据用蓝色，GPU 张量用橙色，CPU 副本与危险状态用红色
- **字体**：标题 32–36 pt，正文 18–22 pt，代码块用等宽字体
- **教学策略**：先画“账本与张量”两层对象图，再沿一次请求走完 `step()`，最后用 E/E2 对照 SWAP 与 RECOMPUTE
- **可视化策略**：
  - 对象图固定左右两栏：左边 `Scheduler` / `BlockManager`，右边 `ModelRunner` / GPU/CPU 张量
  - `block_table`、`ref_count`、`num_cached_tokens` 用同一组数字贯穿全讲
  - 抢占用一张状态图：`RUNNING → SWAPPED → RUNNING` 与 `RUNNING → WAITING → RUNNING`
- **动手环节**：第 15–17 页，5 min 环境检查 + 25 min 任务

---

## 开场（第 1–2 页）

### 第 1 页：封面

- **标题**：KV Cache 在 nano-vllm 里怎么实现
- **副标题**：一条请求走完分配、共享、抢占与回收
- **课程信息**：KV Cache 八讲 · 第 3 讲 · 120 分钟
- 日期、讲师信息
- **一张图**：左边 `block_table`，右边 GPU / CPU 两块 KV 池，中间一条箭头标 `slot`

### 第 2 页：本讲议程

- **第一部分**：对象与生命周期（30 min）— 第 3–8 页
  - 两类对象 / `step()` 五步 / BlockManager / 前缀哈希 / `ref_count` / 小结
- **第二部分**：调度与抢占（50 min）— 第 9–14 页
  - 三队列 / prefill / decode / 抢占触发 / 三条分支 / swap 路径
- **第三部分**：动手实验（30 min）— 第 15–17 页
  - 5 min 环境检查 + 25 min L03 任务
- **总结与答疑**：10 min — 第 18–20 页
- 本讲只回答三个问题：谁记账；一次请求的 `block_table` 和 `ref_count` 怎么变；块不够时 KV 去哪里

---

## 第一部分：对象与生命周期（30 min，第 3–8 页）

### 第 3 页：两类对象：元数据与张量

- 左栏：`Scheduler` → `BlockManager` → `Sequence.block_table`
- 右栏：`ModelRunner` → `kv_cache[2, L, num_blocks, block_size, H_kv, D]`

  ```text
  Scheduler / BlockManager          ModelRunner
  ┌──────────────────────┐          ┌─────────────────────────────┐
  │ free / used blocks   │          │ GPU kv_cache                │
  │ hash → block id      │  ──────▶ │ [2, L, num_blocks, ...]     │
  │ Sequence.block_table │          │ CPU cpu_kv_cache（pinned）  │
  └──────────────────────┘          └─────────────────────────────┘
  ```

- `BlockManager` 不碰 K/V 张量，只改块账本
- `block_table` 是账本和显存之间的索引；attention 用 `slot = block_id * block_size + offset`
- **一句话**：先分清谁记账、谁持有张量，后面的状态变化才不会混

### 第 4 页：`step()` 的五步

- `LLMEngine.step()`（`llm_engine.py:52`）固定顺序：

  ```text
  schedule() → swap_out() → swap_in() → run() → postprocess()
  ```

- `schedule()` 选本步序列，填 `blocks_to_swap_out` / `blocks_to_swap_in`
- `swap_out` 先 GPU→CPU，`swap_in` 再 CPU→GPU，最后 `run()` 写本步 K/V
- `postprocess()` 登记哈希、推进 `num_cached_tokens`、处理采样与释放
- `add_request()` 在 `step()` 外，只把 `Sequence` 放进 `waiting`
- **一句话**：`swap_out → swap_in → run` 不能换序，换出源块可能在本步被覆盖

### 第 5 页：BlockManager 的分配与释放

- `Sequence.num_blocks = ceil(len(seq) / block_size)`
- `can_allocate(seq)` 扫描 `seq.num_blocks - 1` 个块，返回可复用前缀块数；空闲块不够返回 `-1`
- `allocate(seq, num_cached_blocks)`：
  - 命中块挂到 `block_table`，已被引用则 `ref_count += 1`
  - 剩余块用 `_allocate_block()` 新分配
  - 最后设置 `num_cached_tokens = num_cached_blocks * block_size`
- `deallocate(seq)` 反向遍历 `block_table`，`ref_count` 归零才放回空闲队列
- **一句话**：分配先复用前缀，再补新块；释放只看引用计数

### 第 6 页：前缀哈希链与碰撞保护

- 哈希带前一块的种子：

  ```text
  h0 = H(tokens 0–3, seed = -1)
  h1 = H(tokens 4–7, seed = h0)
  h2 = H(tokens 8–11, seed = h1)
  ```

- `hash_blocks()` 只登记本次写满的块；`start == end` 时没有新满块
- `can_allocate()` 命中后还要比对 `block.token_ids == token_ids`，防止哈希碰撞
- `can_allocate()` 不检查最后一个块：满的末块不复用
- **一句话**：哈希负责找候选，`token_ids` 负责确认；最后一块不参与前缀复用

### 第 7 页：`ref_count` 与共享前缀

- 两个请求共享前缀块 0：

  | 事件 | block 0 的 `ref_count` | 说明 |
  | --- | --- | --- |
  | A 分配 | 1 | A 持有 |
  | B 命中 | 2 | A、B 共享 |
  | A 结束 | 1 | B 还在用 |
  | B 结束 | 0 | 块回到空闲队列 |

- `ref_count` 归零才能 `_deallocate_block()`
- `can_swap_out()` 要求所有块 `ref_count == 1`；有共享块就整条序列退回 RECOMPUTE
- **一句话**：共享块的生命周期由引用计数管理，不能按请求直接覆盖

### 第 8 页：第一部分小结

- 三个账本：
  - `block_table`：逻辑块 → 物理块
  - `num_cached_tokens`：已经算过多少 token 的 KV
  - `ref_count`：一个物理块被几个请求持有
- 一条请求的生命周期：`WAITING → RUNNING → FINISHED`，中间可能进 `SWAPPED`
- 下一部分：看 Scheduler 怎么用这三个账本调度和抢占
- **一句话**：块账本、token 进度、引用计数，三者必须一起看

---

## 第二部分：调度与抢占（50 min，第 9–14 页）

### 第 9 页：Scheduler 的三个队列

- `waiting`：还没做 prefill 或等待重算的请求
- `running`：已经进入 decode 的请求
- `swapped`：KV 在 CPU、等待搬回 GPU 的请求
- `schedule()` 顺序：prefill → 有 prefill 就返回；否则 swap-in → decode
- **一句话**：先喂新请求，再搬回旧请求，最后推进 decode

### 第 10 页：Prefill 与 chunked prefill

- `can_allocate(seq)` 决定能否放入本步
- `max_num_batched_tokens` 控制本轮 token 预算
- chunked prefill 只切分本轮第一条序列；后面的序列放不下就留到下一轮
- 命中前缀时，`num_tokens = seq.num_tokens - num_cached_blocks * block_size`
- **一句话**：prefill 先看块够不够，再看 token 预算够不够

### 第 11 页：Decode 与 block 增长

- `can_append(seq)`：

  ```python
  len(self.free_block_ids) >= (len(seq) % self.block_size == 1)
  ```

- `may_append(seq)` 只在 `len(seq) % block_size == 1` 时追加新块
- 以 `block_size=4`、prompt 5 token 为例：

  | 时刻 | `len(seq)` | `len % 4` | 需要新块 |
  | --- | --- | --- | --- |
  | prefill 后采样 | 6 | 2 | 否 |
  | decode 1 后 | 7 | 3 | 否 |
  | decode 2 后 | 8 | 0 | 否 |
  | decode 3 后 | 9 | 1 | 是 |

- **一句话**：块边界落在 `len % block_size == 1`，不是 `0`

### 第 12 页：抢占触发：`can_append` 返回 False

- decode 时逐条检查 `can_append(seq)`
- 返回 False 说明下一步需要新块，但空闲块不够
- 先抢占 `running` 队尾；队尾为空时抢占当前序列自身
- 没有“抢占阈值”这种参数，触发条件就是空闲块不足
- **一句话**：抢占不是定时器，是块分配失败后的让路

### 第 13 页：抢占的三条分支

- `Scheduler.preempt()`（`scheduler.py:97`）按顺序判断：

  | 顺序 | 条件 | 动作 | 结果 |
  | --- | --- | --- | --- |
  | 1 | 序列的块在本步 `blocks_to_swap_in` 里 | 取消 swap-in，`_recompute` | 强制 RECOMPUTE |
  | 2 | `can_swap_out(seq)` 为真 | `swap_out`，进 `swapped` | SWAP，KV 在 CPU |
  | 3 | 其余情况 | `_recompute` | RECOMPUTE，KV 丢弃 |

- `can_swap_out()`：CPU 空闲块够，且所有块 `ref_count == 1`
- 边界：只剩一条 running 序列且无空闲块时，`preempt(seq)` 后本轮没有可调度序列，真实 `Scheduler` 会在 `assert scheduled_seqs` 处失败；L03 的模拟分支只展示逻辑
- **一句话**：先看 swap-in 护栏，再看能不能 swap，最后才重算

### 第 14 页：swap 数据路径与哈希重建

- `kv_swap.swap_blocks(src, dst, mapping)`：

  ```python
  dst[:, :, dst_id].copy_(src[:, :, src_id])
  ```

- 一次切片覆盖所有层和 K/V；方向由参数顺序决定
- `swap_out`：GPU→CPU，释放 GPU 块，清空 `block_table`，保留 `num_cached_tokens`
- `swap_in`：CPU→GPU，重建 `block_table`，释放 CPU 块，并重建完整块的哈希链
- 少了哈希重建，这些块会从前缀缓存消失，后续新块还会用错前缀种子
- **一句话**：swap 搬的是块，保留的是 token 进度，重建的是前缀索引

---

## 第三部分：动手实验（30 min，第 15–17 页）

### 第 15 页：[动手] 环境与基线

- 确认 nano-vllm 在 `b9dd55a`
- 准备 Qwen3-0.6B 的 config 目录，设置 `NANOVLLM_MODEL_PATH`
- 先跑 CPU 测试：

  ```bash
  cd ~/Project/study/nano-vllm
  git rev-parse --short HEAD
  python -m pytest tests/ -v
  ```

- 再跑 L03：

  ```bash
  PYTHONPATH=. python docs/llm-inference-visual/scripts/L03_scheduler.py "$NANOVLLM_MODEL_PATH"
  ```

- L03 §1–§6 纯元数据；§7 读 config、构造真实 `Scheduler`，不加载权重
- **完成标准**：测试全绿，L03 七段都跑完

### 第 16 页：[动手] 跑 L03：E/E2 对照

- 场景 E：GPU 块 2、CPU 块 8；A、B 各 256 token，prefill 后各采样 1 个 token（`len=257`，`num_cached_tokens=256`）
- 场景 E2：同 E，但 CPU 块为 0

  | 项目 | E（SWAP） | E2（RECOMPUTE） |
  | --- | --- | --- |
  | 抢占路径 | `can_swap_out` 为真 → `swap_out` | CPU 块不足 → `_recompute` |
  | 被抢占请求状态 | SWAPPED | WAITING |
  | `block_table` | 清空 | 清空 |
  | `num_cached_tokens` | 保留 256 | 归零 |
  | 计数器 | `num_swapped_out_blocks=1` | `num_recompute_preemptions=1` |

- 记录 `blocks_to_swap_out` / `blocks_to_swap_in` 和三个计数器
- **一句话**：E 保留 KV，E2 丢弃 KV；差别落在 `num_cached_tokens`

### 第 17 页：[动手] 结果讨论与边界

- 为什么被抢占的是 B，不是 A？
- 为什么 `swap_out` 后 `num_cached_tokens` 还保留？
- 为什么 E2 只能重算？
- 为什么 `swap_in` 要重建哈希链？
- 可选：跑 `tests/verify_swap.py` 检查 token 一致性（需要 GPU 和权重）
- **过渡**：第 4 讲看 vLLM V1 怎么用 `KVCacheManager`、`BlockPool` 做同一件事

---

## 总结与答疑（10 min，第 18–20 页）

### 第 18 页：本讲小结

- 一条请求的生命周期：`WAITING → RUNNING → FINISHED`，中间可能 `SWAPPED`
- 三个账本：`block_table`、`num_cached_tokens`、`ref_count`
- 三个队列：`waiting`、`running`、`swapped`
- 三条抢占分支：swap-in 护栏、SWAP、RECOMPUTE
- **一句话**：nano-vllm 用最小实现把分配、共享、抢占和换出串成了一条完整路径

### 第 19 页：六个常见错误

- 以为满的末块也能命中前缀缓存
- 把 `ref_count` 归零当成“内容失效”，实际上还要等 `_deallocate_block`
- 把 `can_append` 当成“每步都要新块”
- 认为共享块也可以直接 swap
- 把 `swap_out` / `swap_in` / `run` 的顺序调换
- 以为 CPU 块也参与前缀哈希

### 第 20 页：下次预告 + 参考资料

- 下次：第 4 讲《KV Cache 在 vLLM 里怎么调度与回收》— `KVCacheManager`、`BlockPool`、prefix cache、recompute 抢占
- 课后练习：给一次请求画“状态 → block 操作 → KV 位置”的时序表，并标出 E/E2 的分叉
- 参考资料：
  - [第 3 讲实验](hands-on-exercise.md)
  - [nano-vllm](https://github.com/ForceInjection/nano-vllm)
  - `nanovllm/engine/block_manager.py`
  - `nanovllm/engine/scheduler.py`
  - `nanovllm/engine/model_runner.py`
  - `docs/llm-inference-visual/scripts/L03_scheduler.py`
- Q&A：留 3 分钟，重点检查 `ref_count`、`can_append`、swap 护栏和 E/E2 差异
