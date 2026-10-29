# 第 7 讲：KV Cache 在 DeepSeek-V4.1-Flash 里怎么压缩 — 教学大纲

## 基本信息

| 项目 | 内容 |
| --- | --- |
| 课时 | 120 分钟（30 概念 + 40 源码与演示 + 40 实验 + 10 答疑） |
| 先修 | 第 1 讲的 KV 公式；第 5 讲的 prefix caching；第 6 讲的混合注意力缓存对象 |
| 代码基线 | 本地 vLLM，HEAD `8b365ff949` |
| 本讲定位 | 理解 global KV + SWA KV 的压缩路线，以及命中后的 replay 语义 |

开场用 5 分钟回顾第 6 讲：混合注意力为什么把缓存拆成多种对象。回顾要落在一句话上：缓存对象由注意力机制决定，不由层数决定；第 6 讲按 KDA / full 拆，本讲按 global / SWA 拆。

## 教学目标

- **知识目标**：掌握 global KV、SWA KV、indexer K 的边界与精度；掌握 CED、CSA2、层级 indexer 的分工。
- **能力目标**：能从 config 复算 890 B/token，并追踪一次 prefix 命中后的 replay 窗口。
- **素养目标**：报容量时同时说明模型变体、模型层数、统计范围（是否含 SWA/indexer、是否只算 global）。

## 时间安排

| 时间 | 环节 | 内容 |
| --- | --- | --- |
| 30 min | 概念 | 5 分钟回顾第 6 讲；V4-Flash 的 global KV 约 3,514 B/token，加上占近一半的 SWA 持久化；global（main KV + indexer K）与 SWA 两条分支；CED、CSA2、层级 indexer；890 的算式；890 / 1,050 / 1,650 三档配置口径 |
| 40 min | 源码与演示 | `CacheConfig.swa_bounded_replay` 字段与 CLI；scheduler `_mark_prefix_replay` 的触发路径；`DeepseekV4SWACache` 的 paged cache；`prefix_cacheable` 的条件；现场运行 `code/kv_account.py` 对账 |
| 40 min | 实验 | 复算 890；按 H = 32K 画 replay 时间线；用假设的 prefill 吞吐算节省与付出；填分层表 |
| 10 min | 答疑 | 回答 FP4 口径、层间共享范围、128 位置窗口、V4 各变体为什么不能相除 |

## 重点与难点

| 类型 | 内容 | 处理方式 |
| --- | --- | --- |
| 重点 | 890 B/token 的构成 | 先拆 entry 数 2.5，再拆 main 288 与 indexer 68，最后乘回去 |
| 重点 | prefix 命中的 replay | 画 `[H−128, H)` 窗口，区分全局复用与窗口重建 |
| 难点 | 层间共享 | 先讲 4 个 Full 层产 KV、4 个 Reindex 层重选、30 个 Reuse 层搭便车，再落到 2.5 |
| 难点 | 变体口径 | 建 V4-Pro / V4-Flash / V4.1-Flash 对照表，列模型层数与统计范围，禁止跨行相除 |
| 难点 | 条件化的 prefix cache | 把“只缓存 global KV”写成条件句，列出三种关闭 bounded replay 的配置 |

## 本讲产出

一张分层表。下表由学员填写，第一行是示例：

| 层 | 内容 | 精度 | 覆盖范围 | 每 token 字节 | 是否进 prefix cache |
| --- | --- | --- | --- | --- | --- |
| 示例：global main KV | 4 个 Full 层产出，跨层复用 | NVFP4 | 全序列 | 720 B（288 × 2.5） | 是 |
| global main KV | 待填 | 待填 | 待填 | 待填 | 待填 |
| global indexer K | 待填 | 待填 | 待填 | 待填 | 待填 |
| SWA KV | 待填 | 待填 | 待填 | 待填 | 待填 |

表后写一句：890 是 global-only 口径，SWA 按请求单独记 2.7 MB；两者不能合并成一个每 token 数字。

## 参考材料

- [DeepSeek-V4.1-Flash 技术报告精读](../../../09_inference_system/kv_compression/02-deepseek-v41-flash.md)
- [vLLM DeepSeek-V4.1-Flash](https://vllm-project.github.io/2026/10/07/deepseek-v41-flash.html)
- `~/Project/ai-infra/vLLM/vllm/config/cache.py`（`swa_bounded_replay`）
- `~/Project/ai-infra/vLLM/vllm/v1/core/sched/scheduler.py`（`_mark_prefix_replay`）
- `~/Project/ai-infra/vLLM/vllm/v1/attention/backends/mla/sparse_swa.py`（`DeepseekV4SWACache`）
- `~/Project/ai-infra/vLLM/vllm/v1/kv_cache_interface.py`（`prefix_cacheable`）

## 考核衔接

本讲的课堂实验产出是第 8 讲容量表的输入。第 8 讲的课堂实验同时作为结课实验，结课作业是作业 3，占总评 40%。三处称呼在第 8 讲统一说明。
