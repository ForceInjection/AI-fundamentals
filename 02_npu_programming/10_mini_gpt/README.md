# Mini-GPT：手写 Transformer 训练

本模块从零实现 GPT-2 风格的 decoder-only Transformer（~11M 参数，字符级编码），在单张 NPU 上完成训练和文本生成。核心价值不在模型本身，而在于**亲手写出 self-attention 的每一行代码**——从 Q·Kᵀ / √dₖ 到 causal mask 到 online softmax，全部手写、不做封装。

2000 次迭代训练耗时 43 秒，loss 从 5.43 降至 0.14。生成的文本能正确使用"达芬奇架构""HCCS 全互联"等训练数据中的术语。

## 1. [Mini-GPT 训练详解](01_mini_gpt_training.md)

从语言模型的基本概念出发，逐步拆解 Transformer 六大核心机制（Self-Attention、Multi-Head、FFN、残差连接、LayerNorm、Position Embedding），再到模型架构、训练过程、文本生成策略和实测数据。全文理论线与实践线交织，建议对照 `train_gpt.py` 源码阅读。
