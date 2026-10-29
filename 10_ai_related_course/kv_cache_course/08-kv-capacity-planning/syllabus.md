# 第 8 讲：KV Cache 怎么算容量、怎么选引擎 — 教学大纲

## 基本信息

| 项目 | 内容 |
| --- | --- |
| 课时 | 120 分钟（30 概念 + 40 源码与演示 + 40 实验 + 10 答疑） |
| 先修 | 第 1–7 讲 |
| 本讲定位 | 把两层账本合成容量规划，并给出引擎选型方法 |

## 教学目标

- **知识目标**：掌握 token 池账和 request state 池账的算法，以及两者共享 HBM 预算的求解方式。
- **能力目标**：给定 Agent 负载，算出每 rank HBM、DRAM 对象容量、先爆的账本和建议配置。
- **素养目标**：先算账，再选型；不把“某引擎更快”当成没有负载前提的结论。

## 时间安排

| 时间 | 环节 | 内容 |
| --- | --- | --- |
| 30 min | 概念 | 两层账本；token 池的页对齐；request state 池的两种形态（GLM KDA / DeepSeek SWA）；`mem_fraction_static` + `mamba_full_memory_ratio` 的联合求解；page size 与碎片 |
| 40 min | 源码与演示 | SGLang `resolve_max_num_reqs` 的双重上限；vLLM `gpu_memory_utilization` / `kv_cache_memory_bytes` 的池推导；nano-vllm 的 block 池；对照示例启动日志的字段；运行 `code/capacity_plan.py` |
| 40 min | 实验 | 选一个模型加一个引擎，算 token 池、request state 池、page 开销；给出先爆的账本和建议配置 |
| 10 min | 答疑 | 回答双预算、`max_total_tokens` 的上限语义、DRAM 按对象计量、PD/offload 的判据 |

## 账本定义

### token 池账

token 池按 token 计，按 page/block 对齐：

```text
aligned_len = ceil(seq / page) × page
request_token_bytes = aligned_len × bytes_per_token
C_token = pool_tokens // avg_aligned_len
```

GLM-5.3-Flash 的 `bytes_per_token` 是 7,260 B（scaled FP8 容量预留，含 11 层 KV 与 11 层 indexer）。DeepSeek-V4.1-Flash 的 global 部分是 890 B/token，SWA 另记在 request state 池。

### request state 池账

两种模型形态不同：

```text
GLM KDA：
  KDA/request = slots_per_request × 147.6 MB     预留上限，默认 5 slots，约 738 MB
  C_state = max_mamba_cache_size // slots_per_request

DeepSeek SWA：
  SWA/request = 40 层 × 128 位置 × 528 B ≈ 2.7 MB
```

GLM 的 state 按 slot 计费，slot 数是池容量；DeepSeek 的 SWA 按请求固定大小，池按请求上限定容。两者都不能折算成“每 token 字节”。

### 联合求解

SGLang 的双预算（自动定容时；显式给 `max_mamba_cache_size` 走另一条分支）：

```text
mamba_budget = total_rest_memory × r / (1 + r)
token_budget = total_rest_memory / (1 + r)
r = mamba_full_memory_ratio（默认 0.9）
```

`total_rest_memory` 是 `mem_fraction_static` 划走权重和静态区之后剩下的 HBM。调大 `r`，mamba 池变大、token 池变小。

vLLM 没有 mamba 专用 ratio：

```text
kv_pool_bytes ≈ gpu_memory_utilization × HBM − 权重 − 激活峰值
```

显式指定 `kv_cache_memory_bytes` 时，它覆盖 `gpu_memory_utilization`，直接定 KV 池大小。mamba state 也按 KV 页管理，没有单独的 `max_mamba_cache_size` CLI，由 `--max-num-seqs` 与显存自动定容。

## 参数映射

| 概念 | SGLang | vLLM | nano-vllm |
| --- | --- | --- | --- |
| 并发请求上限 | `--max-running-requests`（全局值按 attn DP 分片） | `--max-num-seqs` | `Config.max_num_seqs` |
| token 池容量 | `--max-total-tokens`：token 池上限，超过 profiled 值会被压回，主要用于开发/调试 | 由 `--gpu-memory-utilization` 或 `--kv-cache-memory-bytes` 推导 | block 池，由 `max_num_batched_tokens`、`max_model_len`、`kvcache_block_size` 与显存决定 |
| 每步 token 预算 | `--max-prefill-tokens` / `--chunked-prefill-size` | `--max-num-batched-tokens` | `max_num_batched_tokens` |
| request state 池 | `--max-mamba-cache-size` + `--mamba-full-memory-ratio` | 无对应 CLI，由 `--max-num-seqs` 与显存自动定容 | 无 |
| 分配粒度 | `--page-size`（GLM 物理页 64） | `--block-size`（默认 16） | `kvcache_block_size`（默认 256） |

三个容易混的点：

- SGLang 的 `max_total_tokens` 不是“我要多少就有多少”。源码里 `_apply_token_constraints` 取 `min(token_capacity, user_limit)`，超过 profiled 值会打 warning 并压回。
- vLLM 的 `--max-num-batched-tokens` 是每步预算，不是池容量。池容量由显存剖析决定。
- nano-vllm 只有 block 池，没有 KDA/SWA 的 request state 账。它适合演示 token/block，不适合演示 GLM 的 KDA 账。

### `max_running_requests` 的双重上限

```text
max_running_requests ≤ token_capacity // 2
max_running_requests ≤ max_mamba_cache_size // ratio
```

`max_mamba_cache_size` 是 state slot 数，不是请求数。`ratio` 的基础值是 3；GLM 默认 `extra_buffer` + overlap 策略按 5 slots/request 预留，所以是 `max_mamba_cache_size // 5`。关掉 overlap 是 4，`no_buffer` 或非 triton 后端是 3。

## page size 与碎片

| 引擎 / 模型 | 默认分配粒度 | 说明 |
| --- | --- | --- |
| nano-vllm | 256 token | `kvcache_block_size=256`，且必须是 256 的倍数 |
| vLLM | 16 token | `--block-size` 默认 16；混合模型可能按模型另定 |
| GLM-5.3-Flash + SGLang | 64 token | DSA KV 与 indexer 的物理页都是 64；`index_kpool=4` 只改选择分组，不改物理页 |

碎片公式：

```text
碎片 = ceil(len / page) × page − len
```

以 len = 1,000 为例：page 16 浪费 8 token，page 64 浪费 24 token，page 256 浪费 24 token。长度随机时，平均浪费约 page/2，最坏 page−1。

## DRAM 账

| 模型 | 对象 | 是否进 DRAM | 口径 |
| --- | --- | --- | --- |
| GLM-5.3-Flash | 11 层 KV | 是 | HiCache `PoolName.KV` |
| GLM-5.3-Flash | 34 层 KDA state | 是 | HiCache `PoolName.MAMBA` |
| GLM-5.3-Flash | indexer | 否（CUDA hybrid 路径） | 无独立 `host_value` |
| GLM-5.3-Flash | compress tail | 否 | 不在 RadixTree 上 |
| DeepSeek-V4.1-Flash | SWA KV | 是，独立 host 池 | 每机 10% 主机内存，分钟级 TTL |
| DeepSeek-V4.1-Flash | global KV | 是，持久层 | SSD + 主机内存，至少 72 小时 |

DRAM 容量按对象和复用率算，不按 HBM 全量镜像算。

## 重点与难点

| 类型 | 内容 | 处理方式 |
| --- | --- | --- |
| 重点 | 两层账本 | 先定 token 池和 state 池，再求两者共享的 HBM 预算 |
| 重点 | 参数映射 | 每个参数标注它约束的是池容量、每步预算还是并发上限 |
| 难点 | 双预算求解 | 用 `mamba_full_memory_ratio` 的公式手算一次，观察 token 池怎么被挤小 |
| 难点 | 先爆的账本 | 按“可用并发 = 池容量 / 单请求占用”逐本比较，不凭直觉 |
| 难点 | 选型判断 | 先写测量值和约束，再比较引擎；没有阈值，只有负载前提 |

## 本讲产出

一页容量表。下表由学员填写，第一行是示例：

| 项目 | 数值 | 公式 / 命令 |
| --- | --- | --- |
| 示例：GLM token 池（128K，单请求） | 951.6 MB | 7,260 × 131,072 |
| token 池（你的负载） | 待填 | 待填 |
| request state 池 | 待填 | 待填 |
| page / block 开销 | 待填 | 待填 |
| 每 rank HBM | 待填 | 待填 |
| DRAM 按对象 | 待填 | 待填 |
| 先爆的账本 | 待填 | 待填 |

选型理由写清：负载形态、瓶颈、PD/offload/prefix cache 的取舍，以及验证用的日志或命令。

## 考核说明

本课程统一三处称呼：

- **课堂实验**：每讲 40 分钟的随堂练习；第 8 讲的课堂实验同时作为**结课实验**。
- **结课作业**：作业 3，占总评 40%。
- **结课项目**：一页决策备忘，是结课作业的必做部分，不单独计分。

作业 3 要求分别算“平均驻留”和“峰值 64×128K”两种场景，并说明推荐配置按哪一个场景定。

## 参考材料

- [AI-fundamentals KV Cache 技术体系](../../../09_inference_system/kv_cache/README.md)
- [GLM-5.3-Flash KV 布局](../../../09_inference_system/sglang/glm53-flash-kv-layout.md)
- [DeepSeek-V4.1-Flash 精读](../../../09_inference_system/kv_compression/02-deepseek-v41-flash.md)
- [nano-vllm](https://github.com/ForceInjection/nano-vllm)
- [vLLM](https://github.com/vllm-project/vllm)
- [SGLang](https://github.com/sgl-project/sglang)
- 本地源码：`~/Project/ai-infra/sglang/python/sglang/srt/mem_cache/kv_cache_configurator.py`（`_apply_token_constraints`、`resolve_max_num_reqs`）
