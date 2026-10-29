# 第 4 讲：instrumentation 片段与日志字段

本目录放第 4 讲实验用的 instrumentation 片段和示例日志说明，不需要真的跑 vLLM 也能完成字段解读部分。

- `parse_stats_log.py`：解析 vLLM 统计日志的纯 Python 脚本，可用内置示例验证；
- 下面的片段用于本地 vLLM checkout 的临时修改，实验后还原，不要把改动带进课程仓库。

## 一、不跑模型：示例日志与字段

`VLLM_LOG_STATS_INTERVAL=1` 时，vLLM 大约每秒打印一行统计。示例（数值仅用于说明字段，不是实测）：

```text
Engine 000: Avg prompt throughput: 128.4 tokens/s, Avg generation throughput: 36.2 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 2.7%, Prefix cache hit rate: 75.0%
Engine 000: Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 41.8 tokens/s, Running: 2 reqs, Waiting: 1 reqs, Preemptions: 2, GPU KV cache usage: 61.4%, Prefix cache hit rate: 40.0%
```

| 字段 | 源码位置 | 含义 | 实验里看什么 |
| --- | --- | --- | --- |
| `Avg prompt throughput` | `vllm/v1/metrics/loggers.py` 的 `log()` | 本统计窗口的 prefill token 吞吐 | 短请求下通常很小，不能拿它判断 KV 效率 |
| `Avg generation throughput` | 同上 | 本统计窗口的 decode token 吞吐 | 抢占增多时通常下降 |
| `Running` | `SchedulerStats.num_running_reqs` | running 队列里的请求数 | 与 `Waiting` 一起看排队和并发 |
| `Waiting` | waiting + skipped waiting | 等待调度的请求数 | 持续大于 0 说明块或 token 预算不足 |
| `Preemptions` | `num_preemptions` 累计值 | 本进程发生的抢占次数 | 大于 0 说明 `allocate_slots` 曾返回 `None` 并触发 `_preempt_request` |
| `GPU KV cache usage` | `BlockPool.get_usage()` | 已用块 / 总块 | 接近 100% 时下一次分配容易触发抢占 |
| `Prefix cache hit rate` | `CachingMetrics.hit_rate` | 最近请求的 `hits / queries` | 共享前缀是否被复用；注意它按 token 计，不按请求数计 |
| `External prefix cache hit rate` | connector 统计 | offload / 外部 KV 的命中率 | 只有启用 connector 时出现 |

`make_prefix_cache_stats()` 返回 `PrefixCacheStats`（`vllm/v1/metrics/stats.py:117`），字段包括：

| 字段 | 含义 |
| --- | --- |
| `requests` | 本窗口内新请求数 |
| `queries` | 查询的 token 数 |
| `hits` | 命中的 token 数 |
| `preempted_requests` / `preempted_queries` / `preempted_hits` | 其中经历过抢占的请求及其查询、命中 |
| `reset` | 是否调用过 `reset_prefix_cache` |

命中率 = `hits / queries`。`log_stats` 关闭时该方法返回 `None`。

没有 GPU 时可以直接跑解析脚本看字段：

```bash
python3 parse_stats_log.py --demo
python3 parse_stats_log.py /path/to/vllm.log
```

## 二、可选 GPU：打印 block table

在本地 vLLM checkout 的 `vllm/v1/worker/gpu_model_runner.py` 里，找到 `execute_model()` 中 `commit_block_table` 之后的位置（基线约 L1971），临时插入：

```python
self.input_batch.block_table.commit_block_table(num_reqs)

# --- 临时 instrumentation ---
bt = self.input_batch.block_table[0]          # MultiGroupBlockTable 取第 0 组
arr = bt.get_numpy_array()                    # shape: (max_num_reqs, max_blocks_per_req)
for row in range(min(num_reqs, 2)):
    n = int(bt.num_blocks_per_row[row])
    print(f"[block_table] row={row} n={n} blocks={arr[row, :n].tolist()}")
# --- end ---
```

读法：

- `row` 是请求在 batch 里的槽位，不是 request id；
- `n` 是该请求当前有效的块数，`arr[row, :n]` 才是块号；
- 其余列是 0 填充，0 号块是 null block，不能当成有效块；
- 混合模型（多个 KV cache group）要对每个 group 分别打印 `self.input_batch.block_table[g]`。

## 三、可选 GPU：打印 prefix 统计

在 `vllm/v1/core/sched/scheduler.py` 的 `make_stats()` 里，`make_prefix_cache_stats()` 调用之后（基线约 L2852）临时插入：

```python
prefix_cache_stats = self.kv_cache_manager.make_prefix_cache_stats()
if prefix_cache_stats is not None:
    print(
        f"[prefix] requests={prefix_cache_stats.requests} "
        f"queries={prefix_cache_stats.queries} hits={prefix_cache_stats.hits}"
    )
```

注意：

- `make_prefix_cache_stats()` 会读取并重置统计，正常日志路径也调用它；插入 print 不能改变调用顺序；
- 需要 `LLM(..., disable_log_stats=False)`，否则 `log_stats` 为假，返回 `None`；
- `VLLM_LOG_STATS_INTERVAL=1` 只影响日志频率，不影响统计口径。

## 四、可选 GPU：用公开指标读 prefix 命中

不想改源码时，可以用 `LLM.get_metrics()`（V1 接口，要求 `disable_log_stats=False`）：

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-0.6B",
    enforce_eager=True,
    disable_log_stats=False,
    max_model_len=2048,
)
prefix = "KV cache 的第 4 讲示例前缀。" * 20
outs = llm.generate(
    [prefix, prefix + "不同后缀"],
    SamplingParams(temperature=0.0, max_tokens=8),
)

for metric in llm.get_metrics():
    if metric.name in ("vllm:prefix_cache_queries", "vllm:prefix_cache_hits"):
        print(metric.name, metric.labels, metric.value)
```

两次请求共享同一段前缀，第二次应当命中；如果第二个请求的 `num_tokens` 不足一个块，命中会是 0。

## 五、和实验的对应关系

| 实验步骤 | 用哪段 |
| --- | --- |
| 读对象图、函数级路径 | 不需要本目录，直接读源码 |
| 解释统计日志 | 第一节目录 + `parse_stats_log.py` |
| 打印一次请求的 block table | 第二节 |
| 打印 prefix 统计 | 第三节或第四节 |
| 填映射表 | 回到 [hands-on-exercise.md](../hands-on-exercise.md) |
