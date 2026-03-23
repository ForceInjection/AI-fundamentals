# 模型训练与微调

本目录包含模型训练、微调及后训练（Post-Training）相关的技术文档和框架。

---

## 1. 训练框架与评估 (ai_ops_design)

本节收录了 AIOps 领域的后训练技术、基准测试及模型评估框架的设计文档。

- [AIOps 后训练技术](ai_ops_design/aiops_post_training.md)
- [Kubernetes 模型评估框架](ai_ops_design/kubernetes_model_evaluation_framework.md)
- [Kubernetes AIOps 基准测试生成框架](ai_ops_design/kubernetes_aiops_benchmark_generation_framework.md)

---

## 2. LLM 微调实战 (sft_example)

本节提供了针对大语言模型的监督微调（SFT）的理论讲解及代码实战示例。

- [SFT 微调实战指南](sft_example/README.md) - SFT 实战项目的全景指南与资源总览。
- [Qwen 2 大模型指令微调实战](sft_example/train_qwen2.ipynb) - 基于 Qwen2 模型的指令微调 Jupyter Notebook 实战。
- [一文入门垂域模型 SFT 微调](sft_example/一文入门垂域模型SFT微调.md) - 深入解析垂域模型 SFT 微调的理论与全流程。
