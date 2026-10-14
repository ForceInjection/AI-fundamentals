# Prefill 与 Decode：两阶段的推理机制

推理系统的所有优化都从这两个阶段的差异出发：Prefill 是 compute-bound 的并行大矩阵乘，Decode 是 memory-bound 的逐 token 串行读取。本目录从一个具体例子把两阶段的矩阵形状与计算量算清楚，并给出批处理调度从静态到迭代级演进的完整脉络。

## 内容导航

- **[为什么 GPU 生成每个 token 时利用率不到 5%？——Prefill 与 Decode 深度拆解](prefill_decode_qkv_calculation.md)**（[交互可视化](prefill_decode_visual.html) · [校验脚本](prefill_decode_validate.py)）：逐步标注 Prefill 与 Decode 每一步的矩阵形状与计算量，从 compute-bound vs memory-bound 的根本差异推导出 GQA、量化、PagedAttention、Offloading、PD 分离等优化方向的必然性。
- **[Continuous Batching 深度解析：从静态批处理到迭代级调度](continuous_batching.md)**：静态 batching 的三重浪费、vLLM V1 的 token-level 统一调度与 SGLang 的 prefill-first 主动驱逐，两种调度哲学的 TTFT/TPOT 权衡。

## 相关链接

- [KV Cache 技术体系](../kv_cache/README.md)：KV Cache 的存储、复用与压缩机制
- [大模型推理并行策略](../parallelism/README.md)：多卡拆分与通信模式
