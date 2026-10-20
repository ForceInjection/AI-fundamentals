# DeepInfer

DeepSeek V4.1-Flash 的生产推理引擎，2026-10-20 开源（Apache-2.0）。本目录收录源码级拆解。

## 文章

- **[DeepInfer 源码拆解：155 万亿 tokens/天背后的引擎，开源了哪一半](deepinfer-source-deep-dive.md)** — 按代码路径拆 ZOS 异步流水线、DSpark 变长验证、YOCO/SWA 截断与 Wide-EP；同时标出开源边界：引擎核心在仓库里，3FS KV 存储、Entrypoint 前缀匹配、请求级 LB 不在。

## 相关

- [DeepSeek 推理集群单卡经济学](../cost_analysis/deepseek_inference_cluster_unit_economics.md) — 算的是 155 万亿 tokens/天背后的单卡账本。
- [DeepSeek-V4 端到端推理 Pipeline 走读](../vllm/module_analysis/deepseek_v4_inference_pipeline.md) — V4 模型侧的逐层走读。
- [Engram：条件记忆与主机内存参数](../engram/01-engram-deep-dive.md) — V4.1 的 N-gram 条件记忆。
- [vLLM WideEP 架构](../vllm/hardware_optimization/deepseek_blackwell_wide_ep.md) — Wide-EP 的另一条实现路径。
