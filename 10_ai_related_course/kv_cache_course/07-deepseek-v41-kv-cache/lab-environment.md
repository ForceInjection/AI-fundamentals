# 第 7 讲：实验环境

本讲以读源码和复算为主，不要求 GPU。共用环境、依赖版本和代码基线见 [课程实验环境](../lab-environment.md)。

## 本讲额外要求

- vLLM 位于 `~/Project/ai-infra/vLLM`，HEAD `8b365ff949`；
- 预读文章：`09_inference_system/kv_compression/02-deepseek-v41-flash.md`；
- 只用 Python 标准库，不需要安装 vLLM，也不需要模型权重；
- 不把 V4-Pro / V4-Flash / V4.1-Flash 的数字混用，每个容量数字注明变体和统计范围。

## 源码阅读清单

| 文件 | 看什么 |
| --- | --- |
| `vllm/config/cache.py` | `swa_bounded_replay` 字段与默认值 |
| `vllm/engine/arg_utils.py` | `--swa-bounded-replay` flag 怎么映射到配置字段 |
| `vllm/v1/core/sched/scheduler.py` | `_mark_prefix_replay` 的调用点与 `replay_start` |
| `vllm/v1/attention/backends/mla/sparse_swa.py` | `DeepseekV4SWACache`、`block_size` |
| `vllm/v1/kv_cache_interface.py` | `prefix_cacheable` 与 `bounded_replay` 的关系 |
| `vllm/models/deepseek_v41/attention.py` | `swa_bytes_per_token`、`compressed_bytes_per_token` 的取值 |

行号会随 commit 漂移，按符号名搜索，不按行号硬找。

## 复算脚本

```bash
cd 10_ai_related_course/kv_cache_course/07-deepseek-v41-kv-cache
python3 code/kv_account.py --seq 32768 --prefill-tps 20000
```

脚本输出：

- 890 B/token 的 entry 与字节分解；
- 890 / 1,050 / 1,650 三档配置口径；
- 8K / 32K / 128K 的总 KV 与 SWA 占比；
- H = 32K 的 replay 窗口与节省/付出算例。

脚本里内置了 `2.5 × (288 + 68) = 890` 等断言，改参数后先看断言是否仍然成立。

## 提示

- `swa_bytes_per_token` 在 V4.1 record 下是 528，V4 record 下是 584。本讲统一按 V4.1。
- `890` 是 global-only 口径，不含 SWA。任何把 890 和 2.7 MB 混成一个每 token 数字的写法都要打回。
- 源码里 `prefix_cacheable` 返回 `not bounded_replay`，这是条件结论的直接依据。
