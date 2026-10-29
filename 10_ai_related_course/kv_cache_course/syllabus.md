# KV Cache 八讲 — 课程总纲

## 基本信息

| 项目        | 内容                                                                                         |
| ----------- | -------------------------------------------------------------------------------------------- |
| 课程名称    | KV Cache 八讲：从 token 账本到生产引擎                                                       |
| 课时        | 8 讲 × 120 分钟                                                                              |
| 授课对象    | 有 Python / PyTorch 基础、了解 Transformer 基本结构的工程师                                  |
| 先修        | 矩阵乘法、注意力机制、PyTorch 基础；建议先完成 `ai-infra-course` 模块 5/6；不要求 CUDA / K8s |
| 第 4 讲前置 | 了解 vLLM 的基本请求流程和 PagedAttention 概念；没有用过 vLLM 也能听，但需要课前补模块 5     |
| 主线        | token 账 → page/slot 口径 → request/state 账 → 引擎与容量                                    |

## 教学目标

- **知识目标**：掌握 KV Cache 的形状、分页、复用、分层与混合注意力下的状态分类。
- **能力目标**：能读 nano-vllm / vLLM / SGLang 的 KV 路径，能算 token 池账与 request state 池账。
- **素养目标**：每个容量数字都带“模型变体 + 精度 + 统计范围”，不把不同代际、不同并行方式的数字混在一起。

## 课程边界

| 内容                                      | 放在哪里                                           |
| ----------------------------------------- | -------------------------------------------------- |
| Transformer、QKV、prefill/decode 入门     | 第 1 讲快速建立；细节参考 `ai-infra-course` 模块 5 |
| vLLM 部署、benchmark、基础 PagedAttention | 前置材料；第 4 讲只读 KV 生命周期源码              |
| LMCache、Mooncake、量化、安全             | 不铺开；第 5 讲解 L3 时提及 Mooncake/HF3FS，LMCache 作为课后延伸 |
| GLM-5.3-Flash / DeepSeek-V4.1-Flash       | 第 6、7 讲，只读 KV 相关代码与配置                 |

## 八讲安排

第 1–2 讲采用 **30 min 概念 + 45 min 源码/演示 + 35 min 实验 + 10 min 答疑**；第 3–4 讲采用 **30 min 概念 + 50 min 源码/演示 + 30 min 实验 + 10 min 答疑**；第 5–8 讲采用 **30 min 概念 + 40 min 源码/演示 + 40 min 实验 + 10 min 答疑**，源码锚点要求课前预读。各讲的详细时间表、重点难点和参考材料放在对应章节的 `syllabus.md`。

| #   | 主题                                       | 核心问题                                                                    | 章节教学大纲                                                                 |
| --- | ------------------------------------------ | --------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| 1   | KV Cache 为什么存在                        | decode 为什么必须缓存 K/V，一个 token 的账怎么算                            | [01-kv-cache-basics/syllabus.md](01-kv-cache-basics/syllabus.md)             |
| 2   | KV Cache 怎么放                            | token slot、block、page、block table 怎么对应                               | [02-kv-cache-layout/syllabus.md](02-kv-cache-layout/syllabus.md)             |
| 3   | KV Cache 在 nano-vllm 里怎么实现           | block manager、前缀哈希、ref_count、preempt/swap 怎么串起来                 | [03-nanovllm-kv-cache/syllabus.md](03-nanovllm-kv-cache/syllabus.md)         |
| 4   | KV Cache 在 vLLM 里怎么调度与回收          | `KVCacheManager`、`BlockPool`、前缀缓存与 nano-vllm 差在哪                  | [04-vllm-kv-cache/syllabus.md](04-vllm-kv-cache/syllabus.md)                 |
| 5   | KV Cache 怎么复用与分层                    | radix tree 怎么命中，HiCache 什么时候换出到 DRAM                            | [05-kv-cache-reuse/syllabus.md](05-kv-cache-reuse/syllabus.md)               |
| 6   | KV Cache 在 GLM-5.3-Flash 里怎么变成三本账 | 45 层文本层里 34 层 KDA + 11 层 full，token KV、indexer、KDA state 各怎么存 | [06-glm53-flash-kv-cache/syllabus.md](06-glm53-flash-kv-cache/syllabus.md)   |
| 7   | KV Cache 在 DeepSeek-V4.1-Flash 里怎么压缩 | 890 B/token 的 global KV 和 SWA KV 怎么分工                                 | [07-deepseek-v41-kv-cache/syllabus.md](07-deepseek-v41-kv-cache/syllabus.md) |
| 8   | KV Cache 怎么算容量、怎么选引擎            | token 池账、request state 池账与 page/slot 口径怎么一起算                   | [08-kv-capacity-planning/syllabus.md](08-kv-capacity-planning/syllabus.md)   |

## 学习产出与支撑章节

| 产出                                     | 支撑章节         | 验收物                                   |
| ---------------------------------------- | ---------------- | ---------------------------------------- |
| token 池账（每 token KV + indexer）      | 第 1、2、6、7 讲 | 一张带模型、精度、序列长度和并发的容量表 |
| request state 池账（KDA state / SWA KV） | 第 6、7、8 讲    | 按 request 或 state slot 计的容量表      |
| page/slot 对齐与碎片                     | 第 2、3、4、8 讲 | 一个请求的 block table 与内部碎片计算    |
| 引擎映射与选型                           | 第 3–8 讲        | nano-vllm / vLLM / SGLang 的类与参数对照 |
| Agent 长会话规划                         | 第 8 讲          | 一页容量规划 + 一页选型理由              |

## 作业地图

| 编号   | 落点    | 内容                                                         | 计入                    |
| ------ | ------- | ------------------------------------------------------------ | ----------------------- |
| 作业 1 | 第 1 讲 | 通用 KV 估算器（Qwen3-0.6B / Qwen2.5-7B，MHA/GQA，FP16/FP8） | 源码笔记 30% 的基础部分 |
| 作业 2 | 第 3 讲 | 给 nano-vllm 加 KV 观察点或简单驱逐策略                      | 源码笔记 30% 的源码部分 |
| 作业 3 | 第 8 讲 | Agent 长会话容量规划与引擎选型                               | 结课作业 40%            |

## 考核建议

| 项目     | 权重 | 形式                                                       |
| -------- | ---- | ---------------------------------------------------------- |
| 课堂实验 | 30%  | 每讲一个短实验，提交截图、脚本输出或结论                   |
| 源码笔记 | 30%  | 作业 1 + 作业 2；选一条 KV 路径，写清类、池、树节点和参数  |
| 结课作业 | 40%  | 作业 3：Agent 负载的容量规划与引擎选型；一页决策备忘为必做 |

## 课程资料

- [课程 README](README.md)
- [共用实验环境](lab-environment.md)
- [术语表](glossary.md)
- [AI-fundamentals KV Cache 技术体系](../../09_inference_system/kv_cache/README.md)
- [GLM-5.3-Flash KV 布局](../../09_inference_system/sglang/glm53-flash-kv-layout.md)
- [DeepSeek-V4.1-Flash 精读](../../09_inference_system/kv_compression/02-deepseek-v41-flash.md)
- [nano-vllm](https://github.com/ForceInjection/nano-vllm)
- [vLLM DeepSeek-V4.1-Flash](https://vllm-project.github.io/2026/10/07/deepseek-v41-flash.html)
- [SGLang GLM-5.3-Flash cookbook](https://github.com/sgl-project/sglang/blob/0bcd822377da7b5718e674eaf9c870d349424dd1/docs/cookbook/autoregressive/GLM/GLM-5.3-Flash.mdx)
