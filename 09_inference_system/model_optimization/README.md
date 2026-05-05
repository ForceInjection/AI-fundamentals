# 模型优化技术

在推理成本结构中，**decode 阶段通常占端到端时间的 70–90%**，其瓶颈既来自模型本身的参数量与计算量，也来自自回归生成每步只出 1 个 token 的串行约束。模型优化因此沿两条正交路径展开：一是**从模型下手**——通过量化、稀疏、剪枝、蒸馏压缩权重与激活，降低单步成本；二是**从解码协议下手**——通过投机解码、Medusa、EAGLE 等机制让一次前向输出多个 token，放大每步收益。本目录覆盖这两条路径上具代表性的技术方案与工具链。

## 1. 投机解码：突破自回归串行瓶颈

投机解码（Speculative Decoding）以"小模型草拟、大模型验证"的方式，把 decode 阶段从"一次一个 token"变成"一次 K 个 token"，在接受率 α 足够高时可将 decode 延迟降低 2–3 倍。

- **[图解投机解码](illustrated-speculative-decoding.md)**：核心机制（草拟-验证状态机）、算法家族（vanilla / Medusa / EAGLE / Lookahead）、系统实现（vLLM / TensorRT-LLM 集成）、接受率 α 与小大模型耗时比 ρ 的性能模型，以及接受率评测方法

关键符号速查：$K$（草拟步长）/ $E[A]$（接受前缀期望长度）/ $\alpha = E[A]/K$（接受率）/ $\rho = t_\text{small}/t_\text{large}$（耗时比）。

## 2. 模型压缩工具链：NVIDIA ModelOpt

面向生产部署的一站式压缩工具包，覆盖从 Hugging Face / PyTorch / ONNX 模型导入到 TensorRT-LLM / vLLM 部署的完整链路。

- **[NVIDIA Model Optimizer 技术详解](nvidia_model_optimizer.md)**：量化（FP8 / INT8 / INT4 / Blackwell NVFP4，内置 SmoothQuant / AWQ / SVDQuant / AutoQuantize）、2:4 结构化稀疏（SparseGPT / ASP）、剪枝、蒸馏，以及与 TensorRT-LLM / vLLM 的部署集成

## 3. 选型思路

两条路径并非互斥，实际落地常组合使用：

| 目标                 | 优先路径                    | 可叠加技术                     |
| -------------------- | --------------------------- | ------------------------------ |
| 降低单卡显存占用     | 量化（INT8 / INT4 / FP8）   | + 2:4 稀疏、权重蒸馏           |
| 提升吞吐（tokens/s） | 投机解码（vanilla / EAGLE） | + Continuous Batching、PD 分离 |
| 降低端到端延迟       | 投机解码 + FP8 量化         | + CUDA Graphs、TensorRT-LLM    |
| 压缩模型部署体积     | 蒸馏 + INT4 量化            | + 结构化剪枝                   |

> 相关阅读：KV Cache 层面的优化参见 [KV Cache 压缩技术](../kv_cache/01_concepts/compression/kv_cache_compression.md)；CUDA Graph 层面的解码优化参见 [vLLM CUDA Graphs 深度解析](../vllm/module_analysis/vllm_cuda_graph_deep_dive.md)。
>
> 联动导航：结合 [`../memory_calc/`](../memory_calc/README.md) 做压缩后的显存重算，用 [`../cost_analysis/`](../cost_analysis/README.md) 验证量化 / 投机带来的单位 token 成本变化，在 [`../reference_design/`](../reference_design/README.md) 下进行整体架构权衡。
