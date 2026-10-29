# KV Cache 八讲：从 token 账本到生产引擎

> 8 讲 × 120 分钟 | 2026-10 | 源码基线见下文

面向有 Python / PyTorch 基础、了解 Transformer 基本结构的工程师。不要求 CUDA、Kubernetes 经验；建议先完成 `ai-infra-course` 模块 5/6，第 4 讲前至少了解 vLLM 的基本请求流程。

课程只围绕一个问题：**KV Cache 到底存了什么、放在哪、怎么复用、什么时候不够用。**

## 八讲一览

| # | 主题 | 这一讲回答的问题 | 目录 |
| --- | --- | --- | --- |
| 1 | **KV Cache 为什么存在** | decode 为什么必须缓存 K/V，一个 token 的账怎么算 | [01-kv-cache-basics](01-kv-cache-basics/README.md) |
| 2 | **KV Cache 怎么放** | token slot、block、page、block table 怎么对应 | [02-kv-cache-layout](02-kv-cache-layout/README.md) |
| 3 | **KV Cache 在 nano-vllm 里怎么实现** | block manager、前缀哈希、ref_count、preempt/swap 怎么串起来 | [03-nanovllm-kv-cache](03-nanovllm-kv-cache/README.md) |
| 4 | **KV Cache 在 vLLM 里怎么调度与回收** | `KVCacheManager`、`BlockPool`、前缀缓存与 nano-vllm 差在哪 | [04-vllm-kv-cache](04-vllm-kv-cache/README.md) |
| 5 | **KV Cache 怎么复用与分层** | radix tree 怎么命中，HiCache 什么时候换出到 DRAM | [05-kv-cache-reuse](05-kv-cache-reuse/README.md) |
| 6 | **KV Cache 在 GLM-5.3-Flash 里怎么变成三本账** | 45 层文本层里 34 层 KDA + 11 层 full，token KV、indexer、KDA state 各怎么存 | [06-glm53-flash-kv-cache](06-glm53-flash-kv-cache/README.md) |
| 7 | **KV Cache 在 DeepSeek-V4.1-Flash 里怎么压缩** | 890 B/token 的 global KV 和 SWA KV 怎么分工 | [07-deepseek-v41-kv-cache](07-deepseek-v41-kv-cache/README.md) |
| 8 | **KV Cache 怎么算容量、怎么选引擎** | token 池账、request state 池账与 page/slot 口径怎么一起算 | [08-kv-capacity-planning](08-kv-capacity-planning/README.md) |

## 每讲目录

每一讲一个目录，目录内固定六份 Markdown 文件；需要时另有 `code/` 目录放可执行脚本。

| 文件 | 内容 |
| --- | --- |
| `README.md` | 本章导读、学习目标、产出与前后衔接 |
| `syllabus.md` | 120 分钟教学安排、重点难点、参考材料 |
| `ppt-outline.md` | 10–20 页的页面主题与讲解落点；第 1 讲为参考式 20 页 |
| `hands-on-exercise.md` | 课堂实验（30–40 分钟；第 1 讲 35 分钟） |
| `homework.md` | 课后作业与巩固题 |
| `lab-environment.md` | 本讲环境说明；共用部分指向根目录 |

## 时间节奏

| 讲次 | 概念 | 源码/演示 | 实验 | 答疑 | 说明 |
| --- | --- | --- | --- | --- | --- |
| 第 1–2 讲 | 30 min | 45 min | 35 min | 10 min | 公式和分页需要较多手算 |
| 第 3–4 讲 | 30 min | 50 min | 30 min | 10 min | 概念与源码交替推进 |
| 第 5–8 讲 | 30 min | 40 min | 40 min | 10 min | 源码锚点课前预读，课堂留时间做实验 |

## 课程定位

- **不是 vLLM 入门课**。vLLM 的调度、PagedAttention、KV 优化基础，建议先看 `ai-infra-course` 模块 5/6；本课程把时间留给 KV Cache 的结构、源码和容量判断。
- **不是论文串讲**。每个模型只读与 KV Cache 有关的类、池、树节点和启动参数，论文只补背景。
- **以本地代码为准**。nano-vllm、vLLM、SGLang 各钉一个 commit；课堂上的 `file:line` 都按这些版本讲。

## 源码与材料基线

| 材料 | 版本 / 路径 | 用途 |
| --- | --- | --- |
| nano-vllm | 本地 `~/Project/study/nano-vllm`，HEAD `b9dd55a` | 第 2–3 讲实验；第 4 讲用 `BlockManager` 做对照阅读 |
| vLLM | 本地 `~/Project/ai-infra/vLLM`，HEAD `8b365ff949` | 第 4、7 讲源码 |
| SGLang | 本地 `~/Project/ai-infra/sglang`，HEAD `f4de6abee6` | 第 5、6 讲源码 |
| AI-fundamentals | 本仓库 `09_inference_system/` | 概念、GLM-5.3-Flash、DeepSeek-V4.1-Flash 预读 |
| ai-infra-course | 本地 `~/Project/study/ai-infra-course` | 模块 5/6 作为前置；复用公式图、计算器和 trace 脚本 |

## 口径约定

- 容量数字必须同时写清：模型变体、精度、统计范围、是否单 rank、`attn_tp` 取值。
- 文章里的 `MB` 按 `10^6 B` 计，不用 MiB；如果换算成 MiB，要另标单位。
- `page/slot` 是 token 池的分配粒度，不是独立的容量池；DRAM 副本属于 host cache，不参与“可用并发”上限。
- 课程中出现的 `nanovllm` 只作为 Python 包名或路径；展示名统一写 `nano-vllm`。

## 学习产出

完成八讲后，应能独立回答：

1. 给定模型 config 和部署参数，算出 token 池账、request state 池账，以及 page/slot 对齐后的容量。
2. 看懂 nano-vllm / vLLM / SGLang 里 KV Cache 的分配、复用、驱逐和换出路径。
3. 分别判断 GLM-5.3-Flash 内部 KDA 子池与 KV 子池谁先成为瓶颈，以及 DeepSeek-V4.1-Flash 内部 SWA KV 与 global KV 谁先成为瓶颈。
4. 为 Agent 长会话负载写一份容量规划与引擎选型说明。

## 共享文件

- [课程总纲](syllabus.md)
- [共用实验环境](lab-environment.md)
- [术语表](glossary.md)
- [可复用配图](visuals.md)

## 参考材料

- [AI-fundamentals KV Cache 技术体系](../../09_inference_system/kv_cache/README.md)
- [三层状态：图解 GLM-5.3-Flash 的 KV Cache 布局](../../09_inference_system/sglang/glm53-flash-kv-layout.md)
- [DeepSeek-V4.1-Flash 技术报告精读](../../09_inference_system/kv_compression/02-deepseek-v41-flash.md)
- [nano-vllm](https://github.com/ForceInjection/nano-vllm)
- [vLLM DeepSeek-V4.1-Flash](https://vllm-project.github.io/2026/10/07/deepseek-v41-flash.html)
- [SGLang GLM-5.3-Flash cookbook](https://github.com/sgl-project/sglang/blob/0bcd822377da7b5718e674eaf9c870d349424dd1/docs/cookbook/autoregressive/GLM/GLM-5.3-Flash.mdx)
