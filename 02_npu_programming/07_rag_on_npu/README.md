# RAG on NPU 实战

本模块聚焦于在 Ascend NPU 上搭建 RAG (Retrieval-Augmented Generation) pipeline。Embedding 模型在 NPU 本地推理做语义向量编码，LLM 生成由外部 API 提供——**计算密集的编码放在本地加速器，需要超大模型的生成放在云端**。

核心挑战：CANN 8.0.1 + torch_npu 2.1.0 的版本组合需要精确定位兼容依赖（transformers ≤ 4.38.2、sentence-transformers ≤ 2.7.0）。NPU 编码 115 条文本耗时 **0.8s (153 条/s)**，CPU 编码耗时 337.8s，加速 **~422×**。

## 1. [RAG Pipeline on NPU](01_rag_pipeline_on_npu.md)

从 RAG 的离线索引和在线查询两个阶段入手，覆盖 BGE 模型的 NPU 推理适配、FAISS 向量检索、OpenAI 兼容 API 的 prompt 构建。包含完整的版本兼容性说明和性能对比数据。
