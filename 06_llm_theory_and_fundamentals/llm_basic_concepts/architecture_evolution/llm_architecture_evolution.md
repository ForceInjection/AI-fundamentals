# LLM 架构演进史——从 GPT-1 到 DeepSeek-V3 的七个拐点

2017 年 Transformer 论文发表时，它的设计目标是机器翻译——Encoder 读源语言，Decoder 写目标语言。七年后的 2024 年，几乎所有前沿 LLM（GPT-4o、Claude 4、DeepSeek-V3、LLaMA 3、Qwen 2.5）都是纯 Decoder 架构，共享一套高度收敛的「标准配方」：RMSNorm + RoPE + SwiGLU + 无 bias。

中间发生了什么？为什么 Encoder-Decoder 被抛弃了？Billion 参数级别的模型怎么变成了 Trillion 级别？MoE 为什么在 2024 年集体爆发？本文梳理七个关键拐点，每个拐点回答一个问题：**当时的主流做法遇到了什么瓶颈，新方案用什么代价换来了什么收益。**

---

## 一、拐点 1：Decoder-Only 路线胜出（2017-2018）

原始 Transformer 是完整的 Encoder-Decoder。Google 的 BERT 选择了 Encoder-Only（双向理解），OpenAI 的 GPT-1 选择了 Decoder-Only（单向生成）。

双向理解在分类、抽取、问答等判别任务上占据绝对优势——BERT 一度横扫 11 项 NLP 基准。但生成任务的权重在 2019 年后急剧上升：对话、写作、代码补全、翻译——这些场景天然需要自回归生成。Decoder-Only 用更简单的架构覆盖了更广的任务面，最终胜出。

Encoder 被抛弃的代价是丧失了深度双向理解——这可能部分解释了为什么现代 LLM 在需要精准信息提取的任务上反而不如小模型。

---

## 二、拐点 2：规模效应被发现——越大越好（2019-2020）

GPT-2 (1.5B) 和 GPT-3 (175B) 的架构与 GPT-1 几乎一样，只是参数和数据成倍放大。但规模带来的不仅是量变——GPT-3 证明了 **In-Context Learning**：无需微调，只需在 Prompt 中给几个示例，模型就能完成新任务。

这个发现彻底改变了 NLP 的研究范式：从「为每个任务训练一个模型」变成「训练一个足够大的模型，用 Prompt 适配所有任务」。GPT-3 的架构没有任何创新——Pre-LN 归一层、GELU 激活、可学习位置编码、96 层——但 175B 的规模本身就是最大的创新。

这一阶段的代价是训练成本失控。GPT-3 的训练约需 3.14 × 10²³ FLOPs，按 2020 年的硬件价格估算约 1200 万美元。只有少数头部实验室能参与这个游戏。

---

## 三、拐点 3：Chinchilla 修正——从「堆参数」到「堆数据」（2022）

DeepMind 用一个简单实验颠覆了 GPT-3 路线的底层假设：70B 的 Chinchilla（配 1.4T tokens）在几乎所有 Benchmark 上击败了 280B 的 Gopher。参数量小了 4 倍，数据量多了 4.7 倍——反而更强。

这个结论直接催生了以数据效率为核心的设计哲学。LLaMA 1 (2023) 彻底贯彻了这一思路：65B 参数配 1.4T tokens（token/param = 21.5），使用的全是公开可用数据，却在 7B-65B 的规模区间全面超越 GPT-3。

> 详见 [Scaling Laws：参数、数据、算力的三角博弈](../scaling_laws/scaling_laws.md) 对数据效率问题的完整推导。

---

## 四、拐点 4：LLaMA 的「标准配方」形成（2023）

LLaMA 1 不仅改变了数据策略，还重新定义了开源 LLM 的架构标准。它引入的三项改动后来被 Qwen、DeepSeek、Mistral 等几乎所有开源模型沿用：

| LLaMA 的改动      | 取代的旧方案          | 收益                           |
| ----------------- | --------------------- | ------------------------------ |
| Pre-LN → RMSNorm  | LayerNorm             | 相同效果，每次归一化省一次归约 |
| GELU → SwiGLU     | GELU                  | 门控机制，更好的训练收敛       |
| Sinusoidal → RoPE | 绝对位置编码          | 相对位置 + 上下文外推能力      |
| 移除 bias 项      | W_Q/W_K/W_V 中的 bias | 简化模型，精度无明显损失       |

LLaMA 2 (2023.07) 架构与 LLaMA 1 完全一致——唯一的改动是数据翻倍 + RLHF。当 Meta 不再改架构、只在数据和对齐上发力时，说明 Decoder-Only 的架构设计已经趋于成熟。

---

## 五、拐点 5：上下文窗口的军备竞赛（2023-2024）

训练长度的物理瓶颈被 RoPE + YaRN 突破。原本 4K 训练的模型，通过调整 RoPE 的旋转频率，可以在不重新训练的前提下外推到 32K-128K。

| 模型                  | 上下文长度 | 关键方法              |
| --------------------- | ---------- | --------------------- |
| GPT-3 (2020)          | 2K         | —                     |
| GPT-4 (2023)          | 8K-128K    | 推测为稀疏注意力      |
| Claude 3 (2024)       | 200K       | 未公开                |
| Gemini 1.5 Pro (2024) | 1M-2M      | 环形注意力 / 混合架构 |
| LLaMA 3 (2024)        | 128K+      | RoPE + YaRN 外推      |
| Qwen 2.5 (2024)       | 128K       | RoPE + YaRN 外推      |

长上下文扩展的代价是推理成本。上下文翻倍，自注意力的计算量翻四倍（O(n²)）。FlashAttention 把 O(n²) 的内存复杂度降到了 O(n)，使得 128K 上下文在实际硬件上变得可运行——这是算法优化与硬件限制之间的精妙平衡。

---

## 六、拐点 6：MoE 的工程胜利（2023-2024）

Mixtral 8×7B (2023.12) 证明了 MoE 可以在开源场景下工程化落地：8 个专家中每次激活 2 个，推理成本约等于 12B 模型，能力超 70B 模型。

DeepSeek-V3 (2024.12) 将 MoE 推向了极致：671B 总参数（256 个路由专家 + 1 个共享专家），每次只激活 37B（Top-8 routing）。训练仅需 ~2.7M H800 GPU hours——不到 GPT-4 训练成本的十分之一——能力达到 GPT-4o 水平。配合 Multi-Head Latent Attention（MLA）大幅压缩 KV Cache，以及 Multi-Token Prediction（MTP）提升训练效率，DeepSeek-V3 代表了 2024 年 LLM 效率工程的最高水平。

MoE 的引入带来了新的工程挑战：专家负载不均衡会导致部分 GPU 空闲等待、路由崩塌（token 涌向少数热门专家）、以及跨专家通信开销。但这些都是可解决的工程问题，而非不可逾越的架构限制。

---

## 七、拐点 7：推理 Scaling——新的增长轴（2024-2025）

前面六个拐点都是训练阶段的创新。OpenAI o1 (2024.09) 和 DeepSeek-R1 (2025.01) 开辟了一条新路：**不改变训练方式，改变推理方式。**

通过强化学习让模型学会在推理时生成长 Chain-of-Thought——有时数千 token——然后从中提取最终答案。这相当于把「思考时间」当作可伸缩的计算资源：给更多时间，就得到更好的答案。这个方向不受训练数据和参数规模的硬约束，是目前最被看好的下一代 Scaling 方向。

代价是延迟——o1 的推理延迟是 GPT-4o 的 5-50 倍。对于需要毫秒级响应的场景（如代码补全），推理 Scaling 不适用；但对于需要深度分析的场景（数学证明、法律分析、科研推理），用时间换质量是完全值得的。

---

## 八、七年演进的三条主线

```text
            2017 ────────────────────────────────── 2025

架构:    Encoder-Decoder → Decoder-Only (不可逆转的收敛)
规模:    117M → 1.5B → 175B → 671B (MoE) → ?
效率:    Dense → MoE → MoE + MLA + MTP → 推理 Scaling
范式:    为每个任务微调 → Few-shot → Zero-shot → Agent
上下文:  512 → 2K → 4K → 128K → 1M+
```

三条主线的交汇点是：**架构在收敛，效率在分化，推理在成为新的增长轴。** 架构层面的创新空间越来越集中在注意力机制的变体（如 MLA）和 MoE 的路由策略上；效率提升则从训练端（Chinchilla、MoE、MTP）向推理端（o1/R1）转移。

---

## 九、相关资源

- [Transformer 架构详解](../transformer/transformer_architecture.md) — 标准配方的逐组件拆解。
- [位置编码：从 Sinusoidal 到 RoPE](../positional_encoding/positional_encoding.md) — 上下文扩展的数学基础。
- [Scaling Laws](../scaling_laws/scaling_laws.md) — 拐点 2、3、6 背后的数据效率逻辑。
- [混合专家 (MoE)](../moe/mixture_of_experts_moe_visual_guide.zh-CN.md) — MoE 的架构细节与负载均衡。
- LLaMA: Open and Efficient Foundation Language Models (2023)
- DeepSeek-V3 Technical Report (2024)
