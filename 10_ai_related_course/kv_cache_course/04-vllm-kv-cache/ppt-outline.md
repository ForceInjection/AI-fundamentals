# 第 4 讲：KV Cache 在 vLLM 里怎么调度与回收 — PPT 大纲

## 设计说明

- **总页数**：20 页
- **演讲时长**：120 分钟（30 min 概念 + 50 min 源码/演示 + 30 min 实验 + 10 min 答疑）
- **建议风格**：深蓝 + 暖灰；Scheduler 侧用蓝色，Worker 与 BlockPool 用橙色，抢占与危险状态用红色
- **字体**：标题 32–36 pt，正文 18–22 pt，代码和函数名用等宽字体
- **教学策略**：先画分层对象图，再沿 `allocate_slots` 走一次分配，最后用 mapping 表对照 nano-vllm
- **可视化策略**：
  - 对象图固定五层：`Scheduler → KVCacheManager → KVCacheCoordinator → SingleTypeKVCacheManager → BlockPool`
  - BlockPool 三态用一张队列图，左端是未缓存空闲块，右端是缓存可驱逐块
  - Worker block table 用矩阵示意，标出有效列和 0 填充
- **动手环节**：第 15–17 页，5 min 准备 + 25 min 任务

---

## 开场（第 1–2 页）

### 第 1 页：封面

- **标题**：KV Cache 在 vLLM 里怎么调度与回收
- **副标题**：从 nano-vllm 的一条路径，到生产引擎的分层管理
- **课程信息**：KV Cache 八讲 · 第 4 讲 · 120 分钟
- 日期、讲师信息
- **一张图**：左边 Scheduler 的请求块表，右边 BlockPool 的空闲队列，中间一条 `allocate_slots` 箭头

### 第 2 页：本讲议程

- **第一部分**：对象与分配（30 min）— 第 3–8 页
  - 分层对象图 / `allocate_slots` / prefix 命中 / BlockPool 三态 / 驱逐顺序 / 小结
- **第二部分**：抢占、offload 与观测（50 min）— 第 9–14 页
  - V1 抢占 / recompute 代价 / offload 边界 / block table 两侧 / prefix 统计 / 生命周期
- **第三部分**：动手实验（30 min）— 第 15–17 页
  - 5 min 准备 + 25 min 源码、日志与映射表任务
- **总结与答疑**：10 min — 第 18–20 页
- 本讲只回答三个问题：谁查前缀、谁分配、谁驱逐；`allocate_slots` 做了什么；recompute 和 offload 怎么区分

---

## 第一部分：对象与分配（30 min，第 3–8 页）

### 第 3 页：分层对象图：谁负责哪一段

- Scheduler 只负责调度决策，不直接持有 Worker 的张量
- `KVCacheManager` 是对外 façade；`KVCacheCoordinator` 按 KV group 协调
- `SingleTypeKVCacheManager` 维护每个请求的 `req_to_blocks`
- `BlockPool` 管物理块、哈希映射和空闲队列
- Worker 侧 `InputBatch.block_table: MultiGroupBlockTable` 是同一份映射的张量表示
- **一句话**：先分清 Scheduler、Worker、BlockPool 三层，后面的调用才不会串

### 第 4 页：`allocate_slots` 的主路径

- prefill/admission：`get_computed_blocks` 先查前缀
- `allocate_slots` 内部顺序：

  ```text
  remove_skipped_blocks
  → get_num_blocks_to_allocate
  → 检查空闲块与 watermark
  → allocate_new_computed_blocks
  → allocate_new_blocks
  → cache_blocks
  ```

- 命中块先 touch 所有 group，再分配外部块，避免前面 group 的分配驱逐后面 group 的命中块
- 块不够时返回 `None`，Scheduler 抢占后重试
- **一句话**：查询、计算、分配、登记分四段；块不够时统一返回 `None`

### 第 5 页：prefix 命中与末块边界

- 请求创建时计算块哈希，`find_longest_cache_hit()` 查最长前缀
- `max_cache_hit_length = request.num_tokens - 1`
- 最后一个 token 必须重算，用于产生 logits
- `num_computed_tokens` 要块对齐，所以最后 token 的重算可能扩展成整块重算
- nano-vllm 是满的末块也不复用；vLLM 是最后 token 必重算
- **一句话**：vLLM 不是“末块不复用”，而是“最后 token 必须再算一次”

### 第 6 页：BlockPool 的三种块状态

| 状态 | 判据 | 空闲队列位置 | 行为 |
| --- | --- | --- | --- |
| 请求持有 | `ref_cnt > 0` | 不在队列 | 不能被分配；引用归零才入队 |
| 缓存可驱逐 | `ref_cnt == 0` 且有 `block_hash` | 队尾 | FIFO 复用，表现为 LRU 淘汰 |
| 未缓存空闲 | `ref_cnt == 0` 且无 `block_hash` | 队首 | LIFO 复用，优先拿最近释放的块 |

- `get_new_blocks()` 从队首取块，取走前先清理旧哈希
- `touch()` 命中时 `ref_cnt += 1`，并把块从空闲队列移除
- **一句话**：一个 BlockPool，块的状态由引用计数和哈希共同决定

### 第 7 页：驱逐顺序与 pin 边界

- `free_blocks()` 按 `ref_cnt` 归零处理：
  - 非缓存块前插：LIFO，改善 GPU 局部性
  - 缓存块后插：FIFO，表现为 LRU 淘汰
- `get_new_blocks()` 从队首取；缓存块被取走时清掉旧哈希
- 没有通用 pin API；`unpin_blocks()` 只服务“引用已释放、内容还要被外部读取”的场景
- **一句话**：LIFO 管性能，FIFO 管复用；pin 不是请求生命周期的常规操作

### 第 8 页：第一部分小结

- 三个视角：
  - Scheduler 侧：`req_to_blocks[request_id]` 记录请求的块列表
  - Worker 侧：`MultiGroupBlockTable` 把块号拷成张量
  - BlockPool：物理块、哈希和空闲队列
- 一次分配串起查询、块数计算、分配、缓存登记
- 下一部分：看分配失败后的抢占，以及 offload 为什么不是 swap
- **一句话**：同一份 KV 映射，Scheduler 用对象看，Worker 用张量看

---

## 第二部分：抢占、offload 与观测（50 min，第 9–14 页）

### 第 9 页：V1 抢占只有 recompute

- 触发点：`allocate_slots()` 返回 `None`
- Scheduler 抢占 `running` 队尾（默认 FCFS），再重试当前请求
- `_preempt_request()`（`scheduler.py:1539`）：
  - 释放请求的块
  - 状态置为 `PREEMPTED`
  - `num_computed_tokens = 0`
  - 放回 `waiting` 队首
- 没有 scheduler 级 swap，也没有 `swapped` 队列
- **一句话**：V1 的抢占是“丢 KV、回 waiting、重新 prefill”

### 第 10 页：recompute 的代价与 prefix cache

- 被抢占请求的 `num_computed_tokens` 归零，下一次从 prefill 重新开始
- prefix cache 可能仍然命中一部分前缀，但不是“断点续跑”
- 最后 token 必重算、块对齐限制仍在，命中收益会被削弱
- 对照 nano-vllm：SWAP 保留 `num_cached_tokens`，回来接着 decode
- **一句话**：recompute 省显存，不省计算；SWAP 省计算，不省拷贝

### 第 11 页：CPU offload 不是抢占

- `kv_offloading_size` 开启 KV offloading connector（`config/cache.py:256`）
- 已算好的 KV 存 CPU，后续命中时取回，属于 KV transfer 层
- 不改变请求状态，也不负责腾出 `running` 请求的块
- 不要把 offload 映射成 nano-vllm 的 SWAP
- **一句话**：recompute 是调度器的让路，offload 是搬运层的缓存

### 第 12 页：block table 的两侧表示

- Scheduler 侧：`req_to_blocks[request_id]` 是 `list[KVCacheBlock]`
- Worker 侧：`InputBatch.block_table` 是 `MultiGroupBlockTable`
- 单 group 模型取 `block_table[0]`
- `get_numpy_array()` 返回 `(max_num_reqs, max_blocks_per_req)`
- 有效长度看 `num_blocks_per_row[row]`；其他列是 0 填充，0 号块是 null block
- **一句话**：对象侧看块列表，张量侧看有效列；不要把 0 当成有效块

### 第 13 页：prefix 统计与日志

- `make_prefix_cache_stats()` 返回 `PrefixCacheStats`
- 字段：`requests` / `queries` / `hits`，命中率 = `hits / queries`
- `log_stats=False` 时返回 `None`
- `VLLM_LOG_STATS_INTERVAL=1` 控制日志频率，不改变统计口径
- 示例日志和 `parse_stats_log.py --demo` 见 `code/README.md`
- **一句话**：统计会读取并重置；instrumentation 不能改变正常日志的调用顺序

### 第 14 页：一次请求的块生命周期

| 阶段 | 调用 | 请求块表 | BlockPool 变化 |
| --- | --- | --- | --- |
| prefill / admission | `get_computed_blocks` + `allocate_slots` | 命中块 + 新块 | 命中块 `ref_cnt += 1`，新块入表 |
| decode 未跨块 | `allocate_slots(num_new_tokens=1)` | 不变 | 不取新块 |
| decode 跨块 | 同上 | 增加一块 | 从队首取新块 |
| 请求结束 | `free` | 清空 | `ref_cnt` 归零，按 LIFO/FIFO 入队 |

- **一句话**：请求块表按块增长，BlockPool 按引用和哈希回收

---

## 第三部分：动手实验（30 min，第 15–17 页）

### 第 15 页：[动手] 读源码与对象图

- 只读 vLLM 源码，不要求安装、不要求 GPU
- 按对象图读入口：
  - `KVCacheManager.get_computed_blocks`（`kv_cache_manager.py:264`）
  - `KVCacheManager.allocate_slots`（L371）
  - `KVCacheCoordinator.get_num_blocks_to_allocate`（L169）
  - `SingleTypeKVCacheManager.allocate_new_blocks`（L366）
  - `BlockPool.get_new_blocks` / `free_blocks`（L668 / L776）
- 目标：填出 prefill 阶段的 mapping 行
- **完成标准**：能说出查询、块数计算、分配、登记各在哪一层

### 第 16 页：[动手] block table 与 prefix 日志

- 无 GPU：跑 `python3 code/parse_stats_log.py --demo`，解释 `requests / queries / hits`
- 可选 GPU：
  - 打印 `input_batch.block_table[0].get_numpy_array()`
  - 打印 `make_prefix_cache_stats()`
  - 设置 `VLLM_LOG_STATS_INTERVAL=1`
- 记录有效列、0 填充和命中率
- **一句话**：日志字段能对上源码，才算真正读懂统计

### 第 17 页：[动手] 映射表与讨论

- 填完 mapping 表：prefill / decode / 释放 / 缓存登记 / 末块
- 讨论：
  - 为什么 vLLM 的最后一个 token 必重算？
  - 为什么 BlockPool 的缓存块走 FIFO，非缓存块走 LIFO？
  - V1 抢占和 CPU offload 的边界在哪？
- **过渡**：第 5 讲看跨请求复用、radix tree 和分层存储

---

## 总结与答疑（10 min，第 18–20 页）

### 第 18 页：本讲小结：一组映射表

| 生命周期 | nano-vllm | vLLM V1 |
| --- | --- | --- |
| prefill / admission | `can_allocate` + `allocate` | `get_computed_blocks` + `allocate_slots` |
| decode 增长 | `can_append` + `may_append` | `allocate_slots(num_new_tokens=1)` |
| 释放 | `deallocate` | `free` |
| 缓存登记 | `hash_blocks` | `cache_blocks` |
| 抢占 | SWAP / RECOMPUTE | RECOMPUTE |

- **一句话**：同一生命周期，vLLM 把它拆给了更多层

### 第 19 页：六个常见错误

- 把 `KVCacheManager` 当成一个大 `BlockManager`
- 以为 vLLM V1 抢占会 swap KV 到 CPU
- 把 CPU offload connector 当成抢占路径
- 认为 `max_cache_hit_length` 能命中最后一个 token
- 把 Worker block table 的 0 当成有效块
- 把 `make_prefix_cache_stats()` 当成只读查询，忘了它会重置统计

### 第 20 页：下次预告 + 参考资料

- 下次：第 5 讲《KV Cache 怎么复用与分层》— radix tree、HiCache、跨请求复用
- 课后练习：完成 mapping 表，并写一句“两套实现最大的差异”
- 参考资料：
  - [第 4 讲实验](hands-on-exercise.md)
  - [实验代码与日志字段](code/README.md)
  - [vLLM](https://github.com/vllm-project/vllm)
  - `vllm/v1/core/kv_cache_manager.py`
  - `vllm/v1/core/kv_cache_coordinator.py`
  - `vllm/v1/core/single_type_kv_cache_manager.py`
  - `vllm/v1/core/block_pool.py`
  - `vllm/v1/worker/block_table.py`
- Q&A：留 3 分钟，重点检查 `allocate_slots`、BlockPool 三态、recompute 与 offload 的边界
