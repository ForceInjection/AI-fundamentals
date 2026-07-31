# 推理系统技术体系

一个 70B 模型在 H100 上跑 128K 上下文的推理——KV Cache 吃掉 320 GB 显存，attention kernel 的 GPU 利用率不到 25%，Prefill 的一次长 prompt 能把 Decode 阻塞数秒。**推理系统优化的本质，就是解决这三个数字背后的核心矛盾：显存不够、算力空转、延迟抖动。**

本目录从底层机制到上层系统，覆盖 LLM 推理优化的完整纵深。建议按 §1→§4 顺序阅读——每章建立在前一章的基础之上，最后以成本分析和工程实践收尾。

---

## 1. 基础原理

在讨论任何优化之前，先理解推理的两阶段（Prefill/Decode）、KV Cache 为什么存在、以及并行策略如何把大模型塞进多张 GPU。

- **[KV Cache 技术体系](kv_cache/README.md)** — 25 篇文章，从 KV Cache 基础到分布式管理的完整导航。
  - 基础：KV Cache 原理、PagedAttention、五种注意力存储格式
  - 优化：Prefix Caching、压缩量化、淘汰策略、Chunked Prefill、PD 分离传输、Prefetching、CUDA Graph
  - 系统：LMCache、Mooncake、KVBM、HiCache、Tair KVCache
  - 容量：GLM-5 推演、ROI 评估
- **[Prefill 与 Decode 深度拆解](prefill_decode/prefill_decode_qkv_calculation.md)**（[交互可视化](prefill_decode/prefill_decode_visual.html) · [校验脚本](prefill_decode/prefill_decode_validate.py)） — 从一个具体例子出发，标注每一步的矩阵形状与计算量，从 compute-bound vs memory-bound 的根本差异推导出所有优化方向的必然性。
- **[Continuous Batching 深度解析](prefill_decode/continuous_batching.md)** — 将 batch 从请求级"静态容器"变为迭代级"动态流体"的核心技术。从静态 batching 的三重浪费出发，深入 vLLM V1 的 token-level 统一调度与 SGLang 的 prefill-first 主动驱逐，对比两种调度哲学的 TTFT/TPOT 权衡。
- **[大模型推理并行策略](parallelism/README.md)**（[交互可视化](parallelism/parallelism_visual.html)） — DP、TP、PP、EP、SP 五种策略的维度拆解与混合部署案例。
  - 入门：[并行策略总览](parallelism/parallelism_strategies.md)
  - 深度：[专家并行（EP）深度解析](parallelism/expert_parallelism_deep_dive.md)

---

## 2. 推理引擎

理解了 KV Cache 和并行策略之后，看两个主流引擎如何将这些概念落地为工程实现。

### 2.1 vLLM

- **[vLLM 推理系统](vllm/README.md)** — 模块分析、路由调度、硬件优化的完整导航。
  - 注意力架构：MHA→MLA→NSA 演进、DeepSeek V4 支持、MLA→CSA/HCA 进化、[DeepSeek-V3 / V4 端到端 Pipeline 走读](vllm/module_analysis/deepseek_v3_inference_pipeline.md)
  - 系统机制：CUDA Graph、Hybrid KV Cache Manager、投机解码方法全景、原生 KV Offloading
  - 路由：Router 架构、Semantic Router
  - 硬件：WideEP、Blackwell/GB200 优化

### 2.2 SGLang

- **[SGLang 推理引擎](sglang/README.md)** — RadixAttention 前缀缓存与 HiCache 分层存储。
  - [HiCache 深入详解](sglang/hicache_deep_dive.md)：L1–L3 三级缓存架构、HiRadixTree 元数据拓扑、预取与写回策略
  - [KV Pool 管理](sglang/sglang-kv-pool-management.md)：物理存储、Radix Tree 索引与请求视图，lock_ref 正确性保证、Pool/Allocator 类型体系
  - [调度器](sglang/sglang-scheduler.md)：Prefill > Decode 优先级、PrefillAdder 准入预算、retract_decode 内存保护、TTFT 时序拆解
  - [Overlap Scheduling](sglang/sglang-overlap-scheduling.md)：CPU-GPU 双流水线，将上轮 CPU 处理与本轮 GPU forward 并行执行
  - [Chunked Prefill 原理与实现](sglang/chunked_prefill.md)：长 prompt 切分为 chunk 与 decode 交替调度的源码级分析
  - [超大规模推理调优案例](sglang/sglang_scaling_case_study.md)：PD 分离架构下的 KV Cache 竞态与时序缺陷定位
  - [HiSparse 深度解析](sglang/hisparse_deep_dive.md)：DSA 稀疏选择从 attention kernel 提升到系统 coordinator，page 级选择性加载
  - 可视化：[推理流水线](sglang/assets/inference-pipeline.html)、[调度器](sglang/assets/scheduler-visual.html)、[KV Pool 三层关系](sglang/assets/sglang-kv-pool-three-relation.html)

---

## 3. 模型优化

推理引擎提供了执行框架，模型层面的优化则进一步压缩算力和显存开销。

- **[图解投机解码](model_optimization/illustrated-speculative-decoding.md)** — draft model 草拟 K 个候选、target model 批量验证，"猜和验"的核心机制。
- **[MTP 深度解析](model_optimization/mtp-multi-token-prediction.md)** — 训练时植入多 token 预测能力，推理时以 self-speculation 消除独立 draft forward，与投机解码并行的另一条加速路径。
- **[NVIDIA 模型优化器](model_optimization/nvidia_model_optimizer.md)** — 工具链详解与优化实践。
- 相关：KV Cache 层面的压缩与量化见 §1 中 [KV Cache 技术体系](kv_cache/README.md) 的压缩章节。

---

## 4. 显存分析

把模型部署到具体硬件之前，先算清楚显存账。

- **[显存估算](memory_calc/README.md)** — LLM 推理显存估算的理论、脚本与配置。
- **[LLM 显存占用分析](memory_calc/memory_analysis.md)** ([配套 PPT](memory_calc/llm_memory_analysis.pptx)) — 模型参数、KV Cache 与激活值的全量估算方法。
- 计算脚本：`memory_calc/calculate_qwen3_memory.py`、`memory_calc/calculate_deepseek_v4_memory.py`

---

## 5. 参考设计与部署

企业级落地需要参考架构与运维方案。

- **[推理优化参考设计](reference_design/README.md)** — 14 篇系列文章：背景目标、集群规模分类、技术选型、架构设计、性能评估、实施检查清单。
- **[模型部署](deployment/README.md)** — DeepSeek-V3 H20、Qwen2-VL 昇腾的方案与 SLO 验证。

---

## 6. 推理成本分析

所有优化的最终落脚点。API 按量计费与 Coding Plan 订阅的双线分析。

- **[API 定价分析](cost_analysis/llm_api_pricing_analysis.md)** — 基于 OpenRouter 的多模型成本测算，配套 [抓取脚本](cost_analysis/fetch_pricing.py)。
- **[Coding Plan 对比](cost_analysis/coding_plan/coding_plan_report.md)** — 11 款 AI 编程工具订阅成本与隐藏条款解析。

---

## 7. 推理框架学习资料

- **[nano-vllm](https://github.com/ForceInjection/nano-vllm)** — 极简版 vLLM 实现，约 1400 行代码中保留 PagedAttention、连续批处理、TP 与 CUDA Graph。在线 PPT：[nano-vllm 实战课程](https://forceinjection.github.io/nano-vllm/)。
