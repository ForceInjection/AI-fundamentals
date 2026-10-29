# 第 7 讲：DeepSeek-V4.1-Flash 的 890 B/token

40 分钟。目标是把 890 B/token 拆开，并算清一次 prefix 命中的节省与付出。实验不要求 GPU，只用 Python 和源码。

## 步骤

1. 复算 890（10 分钟）。按 `09_inference_system/kv_compression/02-deepseek-v41-flash.md` 与 README 的算式，写出：2.5 个 entry 从哪四层来、main KV 288 B 怎么组成、indexer K 68 B 怎么组成。
2. 读源码（12 分钟）。在本地 vLLM 里找到：
   - `vllm/config/cache.py` 的 `swa_bounded_replay` 字段；
   - `vllm/v1/core/sched/scheduler.py` 的 `_mark_prefix_replay`，看它返回什么、把 `request.replay_start` 设成什么；
   - `vllm/v1/attention/backends/mla/sparse_swa.py` 的 `DeepseekV4SWACache`；
   - `vllm/v1/kv_cache_interface.py` 的 `prefix_cacheable`，确认它和 `bounded_replay` 的关系。
3. 画时间线（10 分钟）。取 H = 32,768（32K 命中），写出 replay_start、replay 区间，以及区间内第 i 个 query 的可见窗口。
4. 算节省与付出（8 分钟）。假设 prefill 吞吐 P = 20,000 token/s，replay 的每 token 代价是全量 prefill 的 k 倍（先取 k = 1 做保守上界），填下面的算例表。

## 算例

符号公式：

```text
replay_tokens = min(128, H)
replay_start  = H − replay_tokens
saved_tokens  = H − replay_tokens
T_saved       = saved_tokens / P
T_replay      = replay_tokens × k / P
T_net         = T_saved − T_replay
saved : replay = saved_tokens : (replay_tokens × k)
```

H = 32K、P = 20,000 token/s、k = 1 的算例：

| 项 | 数值 | 算式 |
| --- | --- | --- |
| replay_tokens | 128 | `min(128, 32768)` |
| replay_start | 32,640 | `32768 − 128` |
| saved_tokens | 32,640 | `32768 − 128` |
| T_saved | 1.632 s | `32,640 / 20,000` |
| T_replay | 6.4 ms | `128 / 20,000` |
| T_net | 1.626 s | `1.632 − 0.0064` |
| saved : replay | 255 : 1 | `32,640 : 128` |

两个观察：

- 比值只由 H 和窗口 128 决定，与 P 无关。P 变，绝对时间变，比值不变。
- k = 1 已经是很保守的上界。decoder 侧 layers 21–39 只重放最后 128 个 prompt token，replay 的实际每 token 代价低于全量 prefill，k 通常小于 1。

## 产出

一张分层表，按 global main KV、global indexer K、SWA KV 三行列，附：

- 890 B/token 的逐项加法；
- replay 窗口 `[H−128, H)` 的示意，标出 replay_start；
- 上面的符号公式和一个算例；
- 一句结论：这次命中节省了多少 prefill，又额外付出了多少 replay，比值是多少。

## 核对

改 H 重算时，可用 `code/kv_account.py` 对账：

```bash
python3 code/kv_account.py --seq 32768 --prefill-tps 20000
```

脚本会打印 890 的分解、三档配置口径、SWA 占用和 replay 算例。脚本只做算术，不替代源码阅读。
