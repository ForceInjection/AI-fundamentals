# 第 8 讲：KV Cache 怎么算容量、怎么选引擎

> 120 分钟 | 课程第 8 讲 | 前置：第 1–7 讲

前七讲分别处理了 token KV、page/slot 口径、KDA state、SWA/global KV。这一讲先纠正一个说法：这些不是三个独立池。真正要同时求解的是两层账本，它们共享同一份 HBM 预算。

## 从三本账到两层账本

第 6 讲把 GLM-5.3-Flash 的缓存拆成 token KV、indexer、KDA state 三本账。这个拆法用来理解对象是对的，用来做容量规划不够：三本账容易被误读成三个可以分别定容的池。

实际只有两层：

| 账本 | 计量对象 | 单位 | 增长方式 | 典型条目 |
| --- | --- | --- | --- | --- |
| token 池 | per-token KV + indexer | B/token，按 page/block 对齐 | 随序列长度线性增长 | GLM 7,260 B/token（容量预留）；DeepSeek 890 B/token（global） |
| request state 池 | 每请求的固定状态 | MB/slot 或 MB/request | 随并发请求数增长 | GLM KDA 147.6 MB/slot；DeepSeek SWA 2.7 MB/request |

这两层共享 HBM 预算：

- SGLang 先由 `mem_fraction_static` 切出权重和静态池，剩下的内存再按 `mamba_full_memory_ratio` 分给 mamba 池和 token 池（自动定容路径；显式给 `max_mamba_cache_size` 时走另一条分支）；
- vLLM 由 `gpu_memory_utilization`（默认 0.92）做显存剖析，剩下的给 KV 页（mamba state 也按页计）；显式给 `kv_cache_memory_bytes` 时，它覆盖 `gpu_memory_utilization` 的结果。

page/slot 只是 token 池的分配粒度，不是第三个池。DRAM 副本是 host cache，影响命中率和驱逐，不参与“可用并发”上限。把三者并列为三个独立池，会算出一个不存在的加法。

## 两个算例：同样的两层，不同的瓶颈

算例 A：GLM-5.3-Flash + SGLang，scaled FP8，单 rank、attn_tp = 1。

| 上下文 | token 池（7,260 B/token） | KDA state（5 slots × 147.6 MB） | 先爆的账 |
| --- | --- | --- | --- |
| 8K | 59.5 MB | 738.1 MB | KDA，约为 token 的 12 倍 |
| 32K | 237.9 MB | 738.1 MB | KDA，约为 token 的 3.1 倍 |
| 128K | 951.6 MB | 738.1 MB | token，约为 KDA 的 1.3 倍 |

算例 B：DeepSeek-V4.1-Flash + vLLM，890 B/token global，SWA 2.7 MB/request。

| 上下文 | global token 池 | SWA state | 合计 | SWA 占比 |
| --- | --- | --- | --- | --- |
| 8K | 7.29 MB | 2.70 MB | 9.99 MB | 27.0% |
| 32K | 29.16 MB | 2.70 MB | 31.87 MB | 8.5% |
| 128K | 116.65 MB | 2.70 MB | 119.36 MB | 2.3% |

同一个“两层账本”，GLM 的瓶颈随长度在 KDA 和 token 之间切换，DeepSeek 的 SWA 在长上下文下几乎可以忽略。拿 GLM 的 KDA 账去套 DeepSeek，或者反过来，都会得到错误的“先爆”结论。

## 参数怎么映射

| 概念 | SGLang | vLLM | nano-vllm |
| --- | --- | --- | --- |
| 并发请求上限 | `--max-running-requests` | `--max-num-seqs` | `Config.max_num_seqs` |
| token 池容量 | `--max-total-tokens`（上限，超过 profiled 值会被压回） | `--gpu-memory-utilization` / `--kv-cache-memory-bytes` 推导 | block 池，由 `max_num_batched_tokens`、`max_model_len`、`kvcache_block_size` 与显存决定 |
| 每步 token 预算 | `--max-prefill-tokens` / `--chunked-prefill-size` | `--max-num-batched-tokens` | `max_num_batched_tokens` |
| request state 池 | `--max-mamba-cache-size` + `--mamba-full-memory-ratio` | 无对应 CLI，由 `--max-num-seqs` 与显存自动定容 | 无 |
| 分配粒度 | `--page-size`（GLM 物理页 64） | `--block-size`（默认 16） | `kvcache_block_size`（默认 256） |

`max_running_requests` 不是想设多少就设多少。SGLang 里它同时受两个上限约束：

```text
max_running_requests ≤ token_capacity // 2
max_running_requests ≤ max_mamba_cache_size // ratio
```

`max_mamba_cache_size` 是 state slot 数，不是请求数。`ratio` 的基础值是 3，GLM 默认走 `extra_buffer` + overlap 的 5 slots/request 策略，所以可用请求数按 5 除。`mamba_full_memory_ratio`（默认 0.9）是 token 池和 state 池的双预算旋钮：调大它，mamba 池变大，token 池变小。

完整参数表、page size 表和公式见[教学大纲](syllabus.md)。

## DRAM 按对象算

DRAM 侧不是 HBM 的等比例镜像。按对象列：

| 模型 | 对象 | 能否进 DRAM 缓存 | 说明 |
| --- | --- | --- | --- |
| GLM-5.3-Flash | 11 层 KV | 能，HiCache | `PoolName.KV` |
| GLM-5.3-Flash | 34 层 KDA state | 能，HiCache | `PoolName.MAMBA` |
| GLM-5.3-Flash | indexer | 不能（CUDA hybrid 路径） | 无独立 `host_value` |
| GLM-5.3-Flash | compress tail | 不能 | 不在 RadixTree 上 |
| DeepSeek-V4.1-Flash | SWA KV | 独立 host 池 | 每机 10% 主机内存，分钟级 TTL |
| DeepSeek-V4.1-Flash | global KV | 能，持久层 | SSD + 主机内存，至少 72 小时 |

所以 GLM 的 DRAM 容量按 KV + KDA 两类对象估；DeepSeek 的 DRAM 要分成两个池，SWA 池按 TTL 周转，global 池按前缀复用率估。把 HBM 全量乘个系数当 DRAM 预算，两边都不对。

## 选型判断

PD 分离、offload、prefix cache 都不是默认选项。判断依据是测量值，不是固定阈值：

- PD：看 prefill 与 decode 的时长比，以及 KV 传输带宽和延迟。decode 足够长、传输能被打包进流水线时才划算。
- offload：看 host 带宽、换出/换入频率、命中率和 SLO。换出频繁、命中率低时，offload 只增加尾延迟。
- prefix cache：看前缀复用率和 host 容量。唯一前缀占多数时，缓存本身的开销不划算。

没有“prefill 占比超过 40% 就上 PD”这类通用阈值。选型理由里必须出现你测到的数，否则只能写成空话。

## 学习目标

- 把 token 池账和 request state 池账算到同一张容量表里。
- 分清 `max_total_tokens`、`max_running_requests`、`max_mamba_cache_size` 各自约束什么。
- 判断 PD 分离、offload、prefix cache 在这个负载下是否值得。
- 对照 SGLang、vLLM、nano-vllm 的参数映射，写出选型理由。
- 给 Agent 长会话负载做一份可执行的容量规划。

## 本讲产出

一页容量规划 + 一页引擎选型理由，所有数字带公式或命令。实验只要求选一个模型加一个引擎，不要求三个引擎都填同一套账。

## 目录

- [课程总览](../README.md)
- [教学大纲](syllabus.md)
- [PPT 提纲](ppt-outline.md)
- [课堂实验](hands-on-exercise.md)
- [课后作业](homework.md)
- [实验环境](lab-environment.md)

## 前后衔接

这是课程收尾。第 1–7 讲的账本在这里合流；课堂实验同时作为结课实验，结课作业是作业 3，占总评 40%，包含一页决策备忘。
