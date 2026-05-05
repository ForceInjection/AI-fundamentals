# LLM 推理显存估算

显存是 LLM 推理中**最先触达的系统性瓶颈**：它同时约束了单卡可加载的模型规模、可服务的并发请求数（Batch Size）以及可支持的上下文长度（Context Window）。在自回归生成场景中，KV Cache 随序列长度线性增长，一个看似够用的配置可能在长对话第 10 轮时 OOM。以 **Llama-3-8B（GQA：32 层 × 8 KV heads × 128 head\_dim，FP16）** 为例：每 token 的 KV Cache 约 **128 KiB**，累积 32K tokens 后单个会话已占用 **4 GiB**；若 batch=8 并发，仅 KV 就咬掉 **32 GiB**，足以在一张 A100-40G 上触发 OOM。本目录把"模型权重 + KV Cache + 中间激活"三部分拆开估算，并针对 **MHA/GQA/MLA 与 DeepSeek V4 新型注意力** 给出差异化公式，配套可直接运行的估算脚本。

## 1. 理论分析

- **[LLM 模型推理显存占用深度分析](memory_analysis.md)**：权重 / KV Cache / 中间激活三部分的分解估算方法，覆盖 FP16/BF16/INT8/INT4 精度、GQA 与 MLA 分摊策略、以及 DeepSeek V4 的 K=V 共享 + 分层压缩（c4a / c128a）+ Indexer Cache 等非典型结构
- **[`llm_memory_analysis.pptx`](llm_memory_analysis.pptx)**：配套讲稿（可用于内部培训或评审汇报）

## 2. 估算脚本与配置

脚本均为零依赖的独立 Python 文件，输出显存总量、按组件拆解的占比、以及不同并发 × 序列长度下的敏感性表格。

- **[`calculate_qwen3_memory.py`](calculate_qwen3_memory.py)**：Qwen3-0.6B 级别通用 MHA/GQA 模型的显存估算器（适用多数开源 Dense 模型）
- **[`qwen3_06b_config.json`](qwen3_06b_config.json)**：Qwen3-0.6B 的 HuggingFace `config.json` 样例，供脚本读取
- **[`calculate_deepseek_v4_memory.py`](calculate_deepseek_v4_memory.py)**：**DeepSeek V4 专用估算器**——处理 K=V 共享（无 2×）+ 反向 RoPE、异构层（30 层 c4a + 31 层 c128a）、Indexer Cache、MoE（384 routed + 1 shared，top-6）、FP8 / FP4 混合精度等特殊结构
- **[`deepseek_v4_pro_config.json`](deepseek_v4_pro_config.json)**：DeepSeek V4 Pro 的配置样例，与脚本的 `--config` 参数对接

> **为什么需要 V4 专用脚本**：通用 MHA/GQA/MLA 公式在 DeepSeek V4 上会**严重低估** KV Cache 占用，因为 c4a/c128a 层压缩比不同、Indexer Cache 仅在 c4a 层存在，且 FP8 KV + FP4 Indexer 的混合精度使简单乘 2（FP16）不再成立。详见 [`vllm/module_analysis/vllm_deepseek_v4.md`](../vllm/module_analysis/vllm_deepseek_v4.md)。

## 3. 使用流程

1. **选择脚本**：通用 Dense / GQA 模型 → `calculate_qwen3_memory.py`；DeepSeek V4 系列 → `calculate_deepseek_v4_memory.py`
2. **准备 config**：从 HuggingFace 模型仓库下载目标模型的 `config.json`，或复用目录中的样例
3. **运行估算**：`python calculate_<model>_memory.py --config path/to/config.json [--seq-len N --batch-size B]`
4. **规划部署**：结合 [KV Cache 容量规划](../kv_cache/01_concepts/capacity_planning/glm5_kv_cache_capacity_planning.md) 与 [性能评估指标体系](../reference_design/05-性能评估指标体系.md) 转化为并行策略与硬件清单
