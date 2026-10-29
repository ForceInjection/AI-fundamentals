# 第 7 讲：KV Cache 在 DeepSeek-V4.1-Flash 里怎么压缩

> 120 分钟 | 课程第 7 讲 | 前置：第 1、5、6 讲

第 6 讲把 GLM-5.3-Flash 的缓存拆成 token KV、indexer、KDA state 三种对象。这一讲换一个模型，看另一种拆法：DeepSeek-V4.1-Flash 只把 global KV 压到 890 B/token，SWA KV 退出持久层，前缀命中时重放最后 128 个 token。

## 先看问题

V4-Flash 的 global KV 约 3,514 B/token（报告 Figure 1(b)）。持久层里还要放 SWA KV，后者占了近一半容量。

Agent 负载是 input-heavy：上下文越长，prefill 的计算和 KV 的存储越是大头。但 SWA KV 的访问模式和长期保留并不匹配。它只在活跃会话的分钟级窗口里被复用，却和 global KV 一起按 72 小时保留。

V4.1 把两件事分开处理：

- global KV 用 CSA2 和 FP4 压到 V4-Flash 的 1/4；
- SWA KV 移出持久层，改放每机 10% 主机内存的专用池，TTL 分钟级；未命中时用 bounded replay 重算窗口。

两个因子相乘，持久化 KV 降到 V4-Flash 的 1/8。先有问题感，再看 FP4 和层间共享怎么把 890 这个数做出来。

## 5 分钟回顾：混合注意力为什么把缓存拆成多种对象

第 6 讲的对象划分来自注意力类型：KDA 层按 request 维护 state，full 层按 token 记 KV，indexer 另记一份 K。V4.1 的划分来自另一组机制：global 分支覆盖全序列，SWA 分支只看最近 128 个位置。两讲的共同点是，缓存不能用一个“每 token 多少字节”概括；区别在于 V4.1 的多个对象里，只有 global KV 随序列长度线性增长。

## 学习目标

- 说清 global KV 与 SWA KV 的分工、精度和生命周期。
- 复算 890 B/token，并指出 2.5 这个系数从哪四层来。
- 解释 CED、CSA2、层级 indexer 在缓存上的位置，能各画一张简图。
- 判断 prefix caching 什么时候只缓存 global KV，什么时候会反过来缓存 SWA。
- 能把 V4-Pro / V4-Flash / V4.1-Flash 的数字按模型层数和统计范围分开，不混用。

## 890 B/token 的算式

global KV = main KV + indexer K，两者都按 entry 计。

```text
entry 数（每 token）
  3 个 encoder Full 层 × 1/2 = 1.5     encoder 的 CSA2 压缩率 m = 2
  1 个 decoder Full 层 × 1   = 1       decoder 的 CSA2 压缩率 m = 1
  合计 2.5 entry/token

main KV（NVFP4，每 entry）
  512 个 e2m1 值 / 2 = 256 B
  32 个 E4M3 scale    = 32 B
  合计 288 B

indexer K（MXFP4，每 entry）
  128 个 e2m1 值 / 2 = 64 B
  4 个 UE8M0 scale   = 4 B
  合计 68 B

global KV = 2.5 × (288 + 68) = 890 B/token
```

四个 Full 层是 2、8、14、20。encoder 的三个 Full 层各把 2 个 token 压成 1 个 entry，decoder 的 Full 层按 1 个 token 1 个 entry，所以是 1.5 + 1 = 2.5。

两条容易写错的地方：

- 890 只算 global KV，不含 SWA。
- indexer K 已经含在 890 里，不是第三层。把 68 B 再加一遍会得到 1,060 B/token 这种不存在的口径。

## 配置口径：890 不是 vLLM 的默认值

| 路径 | main KV / entry | indexer K / entry | 每 token | 说明 |
| --- | --- | --- | --- | --- |
| NVFP4 main + MXFP4 indexer | 288 B | 68 B | **890 B** | 报告与参考实现口径，需要 SM100 MegaAttention |
| NVFP4 main + FP8 indexer | 288 B | 132 B | **1,050 B** | 只把 main 换成 FP4，indexer 留在 FP8 |
| FP8 main + FP8 indexer | 528 B | 132 B | **1,650 B** | vLLM 不启用 FP4 KV 时的默认路径 |

890 只在 NVFP4 main、MXFP4 indexer、SM100 MegaAttention 三者同时成立时才是 890。vLLM 里显式指定 `nvfp4_ds_mla` 需要 V4.1 的 KV record，而 FlashMLA 只在 SM100 上解码 FP4；条件不满足会报错。自动选后端时，只有 SM100 且拓扑允许才走 MegaAttention，否则回落 FlashMLA/FP8，容量表要按实际路径重算。

所以容量规划的第一步不是套 890，而是确认部署走哪条路径。拿 890 去算 FP8 部署，会把容量低估 46%；反过来拿 1,650 去套 FP4 部署，会高估 85%。

## SWA 与总 KV

每请求的 SWA KV 不随上下文增长，只覆盖窗口：

```text
SWA/request = 40 层 × 128 位置 × 528 B = 2,703,360 B ≈ 2.7 MB

总 KV(seq) = 890 × seq + 2.7 MB      (seq ≥ 128)
```

40 层里，前两层是纯 SWA，其余 38 层是 CSA2，但每层都有自己的 SWA 分支。528 B 是 V4.1 的 FP8 SWA 记录；源码里 V4.1 record 下 `swa_bytes_per_token = 528`，V4 record 下是 584。

短序列要换个写法。seq < 128 时 SWA 只覆盖 seq 个位置：

```text
总 KV(seq) = seq × (890 + 40 × 528) = seq × 22,010 B      (seq < 128)
```

这时 SWA 才是大头，890 只占 4%。这也是为什么不能把 890 当成“一个 token 的全部缓存”。

2.7 MB 是内容量级。实际分配按 32-token block 做页对齐，会有少量对齐损耗，做精确容量表时按页算。

## Prefix caching 与 replay

默认 `swa_bounded_replay=True`，并且同时满足下面三个条件时，prefix cache 只保留 global KV，SWA KV 不参与：

- 使用 V2 model runner（`use_v2_model_runner=True`）；
- prefill context parallel size = 1；
- 不在 ROCm 上。

任一条件不满足，vLLM 会把 bounded replay 关掉并打 warning，SWA cache 反过来参与 prefix caching。源码里的对应关系是 `SlidingWindowMLASpec.prefix_cacheable = not bounded_replay`。所以“prefix cache 只缓存 global KV”是条件结论，不是无条件事实。在非 V2 runner、prefill CP > 1 或 ROCm 上，容量表要把 SWA 也算进 prefix cache。

命中后重建 SWA 的范围由 H 决定，H 是当前序列长度：

```text
replay_start = H − min(128, H)

H ≥ 128:  replay [H−128, H)
H < 128:  replay [0, H)
```

从 s 开始的 replay 段里，位置 i 的 query 只能看到 `[max(s, i−W+1), i]`，W = 128。decoder 侧 layers 21–39 只处理每个请求最后 128 个 prompt token，因为 layer 20 是最后一个压缩 KV 源，21 层往后都在复用它。

replay 是近似重建。报告写明重放得到的 SWA KV 与完整 prefill 不是数学等价，不同命中点之间也不保证逐位相同；训练阶段会模拟同样的 replay 来补偿。

## 三个结构，一张图一句话

CED：

```text
第 0–19 层  causal encoder
  0–1       纯 SWA
  2–19      CSA2 (m=2)，Full 在 2 / 8 / 14
                │ H_20（encoder 输出）
                ▼
第 20–39 层  decoder
  20        Full：从 H_20 投影 main KV / indexer K
  24/28/32/36  Reindex：复用 KV，用自己的 indexer Q 重打分
  其余 15 层   Reuse：KV 与 Top-K 索引都复用（加 encoder 的 15 层，共 30 层）
```

CED = 20 encoder + 20 decoder，decoder 的 global KV 从 encoder 输出投影，prefill 的 global 部分只算前一半层。

CSA2：

```text
38 个 CSA2 层
├─ Full      4 层 (2/8/14/20)     自己算 main KV + indexer K
├─ Reindex   4 层 (24/28/32/36)   复用 KV，用自己的 indexer Q 重打分
└─ Reuse    30 层                 复用 KV 和 Top-K 索引
```

CSA2 的 38 层 = 4 Full + 4 Reindex + 30 Reuse；只有 4 个 Full 层产 main/indexer K，2.5 entry/token 就是从这 4 层来的。

层级 indexer：

```text
layer 20（Full）
  扫全量上下文
    ├─ 选 Top-512
    └─ 块级候选：2048 块 × 8 位置 = 16,384 个位置
                    │ 候选池
                    ▼
layer 24 / 28 / 32 / 36（Reindex）
  只在 16,384 个候选内打分，输出本层 Top-512
```

首层扫全量建候选池，后续层池内 top-k 512，深层 indexer 的成本从随上下文线性变成常数。

## 源码位置

| 对象 | 位置 | 说明 |
| --- | --- | --- |
| `CacheConfig.swa_bounded_replay` | `vllm/config/cache.py:242` | 字段，默认 True；CLI 是 `--swa-bounded-replay`（`vllm/engine/arg_utils.py:1337`） |
| `_mark_prefix_replay` | `vllm/v1/core/sched/scheduler.py:3018` | 调度器里的真正触发点；调用点在 `:1081`（新请求准入）和 `:3070`（异步 KV 加载完成） |
| `DeepseekV4SWACache` | `vllm/v1/attention/backends/mla/sparse_swa.py:72` | SWA KV 的 paged cache，`block_size=32` |
| `SlidingWindowMLASpec.prefix_cacheable` | `vllm/v1/kv_cache_interface.py:933` | 返回 `not bounded_replay` |

`swa_bounded_replay` 不是函数名，也不是某个 replay 函数的开关。它是配置字段，调度器读到之后在 `_mark_prefix_replay` 里把命中尾部标成待重放。

## 本讲产出

一张分层表，列出 global main KV、global indexer K、SWA KV 的精度、覆盖范围、每 token 字节，以及是否进 prefix cache。表后附 890 的逐项加法和一个 H = 32K 的 replay 算例。

## 目录

- [课程总览](../README.md)
- [教学大纲](syllabus.md)
- [PPT 提纲](ppt-outline.md)
- [课堂实验](hands-on-exercise.md)
- [课后作业](homework.md)
- [实验环境](lab-environment.md)

## 前后衔接

第 6 讲看的是“同一模型内多种缓存形态”，本讲看“用压缩和层间共享把 token 账做小”。第 8 讲把两讲的账本合流到容量规划：token 池怎么算、request state 池怎么算、两者怎么共享同一份 HBM 预算。
