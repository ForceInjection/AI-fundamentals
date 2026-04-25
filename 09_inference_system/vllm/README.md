# vLLM 推理系统优化与分析

本目录主要收录了关于 vLLM（一个高效的大型语言模型推理引擎）的深入分析、相关模块的研究以及在特定硬件架构上的性能优化实践。通过对 vLLM 底层机制和系统架构的解构，旨在为 AI 基础设施开发者和研究人员提供高价值的技术参考。

---

## 1. 核心模块分析 (module_analysis)

本小节包含了对 vLLM 核心运行时模块的深度解析，重点探讨其内存管理与调度机制。

- [Native KV Offloading 解析](./module_analysis/vllm_native_kv_offloading.md)：详细分析了 vLLM 原生的 KV Cache 卸载机制，探讨其如何在 GPU 显存受限的情况下，利用主机内存提升吞吐量。
- [Hybrid KV Cache Manager 深度解析](./module_analysis/vllm_hybrid_kv_cache_manager_deep_dive.md)：探讨混合 KV 缓存管理器的设计原理与实现，分析其如何优化多层级存储资源分配。
- [CUDA Graphs 深度解析](./module_analysis/vllm_cuda_graph_deep_dive.md)：探讨 vLLM 在解码阶段如何利用 CUDA Graphs 技术大幅降低 CPU 调度开销及其底层内存固化机制。
- [DeepSeek V4 长上下文注意力支持解析](./module_analysis/vllm_deepseek_v4.md)：深入探讨 vLLM 对 DeepSeek V4 模型高效注意力机制的底层实现与算子优化。

---

## 2. 关联组件分析 (related_module)

本小节整理了与 vLLM 配合使用的外部路由与请求调度组件的分析。

- [vLLM Router 概述](./related_module/vllm_router.md)：介绍 vLLM 请求路由器的基础架构与功能。
- [Semantic Router 深度解析](./related_module/vllm_semantic_router_deep_dive.md)：深入探讨基于语义的路由分发策略，及其在复杂推理场景下如何提高缓存命中率和整体吞吐量。

---

## 3. 硬件架构优化 (hardware_optimization)

本小节收录了 vLLM 在前沿硬件平台上的部署策略、扩展性测试及性能调优案例。

- [DeepSeek 与 Blackwell 架构扩展性分析](./hardware_optimization/scaling_deepseek_blackwell.pptx)：关于如何在 NVIDIA Blackwell 架构上扩展 DeepSeek 模型推理的演示文稿。
- [DeepSeek Blackwell Wide EP 优化](./hardware_optimization/vllm_deepseek_blackwell_wide_ep.md)：探讨针对 DeepSeek 模型在 Blackwell 架构下利用宽泛的专家并行（Expert Parallelism）进行的特定优化策略。
- [GB200 性能优化](./hardware_optimization/vllm_gb200_optimization.pptx)：针对 NVIDIA GB200 超级芯片的 vLLM 推理优化实践及性能评估演示。
