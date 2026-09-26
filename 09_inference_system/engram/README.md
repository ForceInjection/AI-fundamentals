# Engram：条件记忆与主机内存参数

围绕 DeepSeek「条件记忆」路线（Engram / N-gram Embedding）的专题：参数不住 HBM、住主机内存的那根稀疏轴。论文出处 [arXiv:2601.07372](https://arxiv.org/abs/2601.07372)，官方实现 [deepseek-ai/Engram](https://github.com/deepseek-ai/Engram)。

## 文章

- **[条件记忆：DeepSeek V4.1 Engram 如何用 O(1) 查表换掉一层计算](01-engram-deep-dive.md)** — 源码深读：论文机制四件套（tokenizer 压缩、多头 mul-XOR 哈希素数表、上下文门控单 GEMM、确定性寻址）、官方 422 行 demo、SGLang 推理侧的巨页手术与三种存储一张 kernel；LPDDR 换 HBM 的优势与代价逐项算账。

## 相关

- [KV 压缩推到极限三部曲](../kv_compression/README.md) — Engram 在 V4.1-Flash 上的载体：CED/CSA2/FP4 KV 与多级缓存
- LMSYS 博客（2026-09-10）— SGLang 侧 Engram 的 189 GiB 钉住内存与两种 host 布局部署取舍
