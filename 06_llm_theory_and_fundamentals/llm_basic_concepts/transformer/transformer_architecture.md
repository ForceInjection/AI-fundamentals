# Transformer 架构详解——从自注意力到完整解码器

2017 年之前，处理序列数据的主流架构是 RNN 和 LSTM。它们在机器翻译、语音识别等任务上统治了近十年，但有一个结构性的硬伤：**第 t 个 token 必须等第 t-1 个算完才能开始**。这个串行依赖带来了两个无法绕过的问题——序列越长训练越慢（GPU 的并行能力被浪费），以及跨长距离的信息会随着梯度传播逐步衰减甚至消失。

同年，Vaswani 等人的《Attention Is All You Need》用**自注意力（Self-Attention）**一次性解决了这两个问题：每个 token 直接关注序列中所有其他 token，没有先后顺序，只有相关度权重。这个简单的想法催生了 Transformer 架构，进而演化为今天所有大语言模型（GPT、LLaMA、Qwen、DeepSeek、Claude）的共同基础。

但切换到全并行架构并非没有代价。去掉串行顺序意味着位置信息丢失——要让模型区分「我打你」和「你打我」，靠权重表是不够的。同时，全注意力在长序列上的计算量是 O(n²)，当上下文从 2K 扩展到 128K 时，这个代价变得不可忽视。

本文从零拆解 Transformer 的核心组件。阅读目标是：**读完能看懂一份 `modeling_llama.py` 级别的源码**，并理解每个组件解决了什么问题、又带来了什么新问题。

---

## 一、问题：RNN 的串行瓶颈

RNN/LSTM 处理序列的方式是逐步迭代——隐藏状态 $h_t$ 由 $h_{t-1}$ 和当前输入 $x_t$ 共同决定。

```text
RNN:  token1 → token2 → token3 → ... → tokenN  （串行，O(n) 步）
```

这个设计有两个致命缺陷：

- **无法并行训练**：计算 token_N 必须先算完前面 N-1 步。序列越长，一轮前向/反向传播的时间线性增长。
- **长距离依赖衰减**：梯度和信息在反向传播时穿过几十个时间步后，会因连乘效应指数级衰减（梯度消失）或放大（梯度爆炸）。

这两个问题在 2014-2016 年间催生了大量的变体方案——LSTM 的门控机制、注意力作为 RNN 的辅助模块——但本质上仍在串行框架内打补丁。

## 二、方案：自注意力——让每个 token 同时看见所有 token

Transformer 的答案是放弃串行假设：**不再通过隐藏状态逐步传递信息，而是让每个 token 直接查询序列中的每个 token。**

```text
Transformer: 每个 token ⇄ 所有 token（并行，O(1) 步）
```

这个「直接查询」机制就是自注意力。

### 2.1 Q / K / V：信息检索的类比

自注意力可以用一个信息检索的类比来理解：

```text
Q (Query)  = x · W_Q    —— "我在找什么？"
K (Key)    = x · W_K    —— "我能提供什么？"
V (Value)  = x · W_V    —— "如果被选中，我给出什么信息？"
```

每个 token 的 Embedding 向量通过三个不同的线性变换，分别投影为查询向量 Q、键向量 K、值向量 V。$W_Q$、$W_K$、$W_V$ 是可学习的权重矩阵，维度为 $d_{\text{model}} \times d_k$（通常 $d_k = d_{\text{model}} / \text{num\_heads}$）。

这个三元组设计来自信息检索：Query 代表用户的搜索意图，Key 代表文档的索引标签，Value 代表文档的实际内容。把每个 token 既当作「提问者」又当作「被检索的文档」，就得到了自注意力。

### 2.2 注意力分数：四步计算

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q \cdot K^\top}{\sqrt{d_k}}\right) \cdot V
$$

步骤拆解：

1. $Q \cdot K^\top$ → 得分矩阵 S（token i 对 token j 的原始相关度）
2. $S / \sqrt{d_k}$ → 缩放，防止大点积值导致 softmax 梯度消失
3. $\text{softmax}$ → 归一化为概率分布（每个 token 对所有 token 的注意力权重之和 = 1）
4. $\times V$ → 加权求和，得到每个 token 的「上下文感知」表示

### 2.3 为什么要除以 √d_k？

自注意力有一个容易被忽视的数学细节。当 $d_k$ 较大时，$Q \cdot K$ 的点积结果的方差会增大（约等于 $d_k$）。大的输入值让 softmax 进入饱和区——输出几乎变成 one-hot 向量，梯度趋近于零。除以 $\sqrt{d_k}$ 将方差压回 1，让梯度在训练中保持健康。

```python
# PyTorch 实现骨架
def self_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attn_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, V)
```

这个简单的缩放操作解决了梯度健康问题，但它不解决自注意力的另一个根本限制——一组 Q/K/V 只能表达一种「相关性的定义」。

---

## 三、方案延伸：多头注意力——从单视角到多视角

单组 Q/K/V 只能捕捉一种关系模式。比如在处理「苹果很好吃」时，一组注意力可能关注「苹果→好」的语义关联，但它无法同时关注「苹果→名词主语」的语法角色和「苹果→吃」的动作搭配。

多头注意力（Multi-Head Attention）的解法是**并行运行多个独立的注意力头**，每个头有自己的 W_Q、W_K、W_V，在各自的低维子空间中计算：

$$
\begin{aligned}
\text{MultiHead}(Q, K, V) &= \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \cdot W_O \\
\text{head}_i &= \text{Attention}(Q \cdot W_{Qi}, K \cdot W_{Ki}, V \cdot W_{Vi})
\end{aligned}
$$

- 每个 head 的维度为 d_k = d_model / h（如 512 / 8 = 64）
- 所有 head 的输出拼接后经 W_O 投影回 d_model
- 总计算量与等效的单头注意力相同（并行度换表达力）

**代价**：多头增加了模型的理解能力，但也让注意力的可解释性变得更困难——面对 32 个头、每层一张注意力热力图，人类几乎无法直观理解模型究竟在「关注」什么。此外，多头之间的信息完全独立，直到拼接步骤才交汇——这意味着每个头可能学习到冗余或退化的注意力模式。

---

## 四、前馈网络：注意力之后的逐 token 变换

自注意力让 token 之间交换信息，但交换完后，每个 token 的表示还需要一次独立的非线性变换。这个任务由前馈网络（FFN）承担：

$$
\text{FFN}(x) = W_2 \cdot \sigma(W_1 \cdot x) \qquad\text{（现代 LLM 通常去掉偏置项 ）}
$$

- 先将 d_model 维投影到 d_ff 维（通常 d_ff = 4 × d_model），再投影回 d_model
- FFN 是 Transformer block 中参数量最大的部分，约占 2/3
- σ 是激活函数：原始论文用 ReLU，现代 LLM 几乎全部换为 SwiGLU

### 4.1 SwiGLU：门控机制的引入

从 ReLU 到 GELU 再到 SwiGLU 的演进，反映了 LLM 架构中的一个核心发现：**让网络自己学会「哪些信息该通过、哪些该抑制」，比用一个固定的激活函数更有效。**

$$
\begin{aligned}
\text{SwiGLU}(x) &= (x \cdot W_{\text{gate}} \odot \text{SiLU}(x \cdot W_1)) \cdot W_2 \\
\text{SiLU}(x) &= x \cdot \sigma(x)
\end{aligned}
$$

SwiGLU 引入了一个可学习的「门」——`x · W_gate`。这个门与经 SiLU 激活的主通路做逐元素乘法，让网络在每个 token、每个维度上独立决定信息通过量。代价是参数量增加约 33%（多了一个 W_gate 矩阵），但训练稳定性和收敛速度的提升足以抵消这个开销。

| 模型系列         | FFN 激活 | d_ff 比例     |
| ---------------- | -------- | ------------- |
| 原始 Transformer | ReLU     | 4×            |
| GPT-2/3          | GELU     | 4×            |
| LLaMA 1/2/3      | SwiGLU   | 8/3× (≈2.67×) |
| Qwen 2           | SwiGLU   | 8/3×          |

---

## 五、位置编码：解决注意力「看不到顺序」的问题

自注意力有一个根本性的盲区：**它对 token 的位置完全不敏感。** 打乱整个输入序列，计算出的注意力分数不变。

但语言是有严格顺序的。「我打你」和「你打我」共享完全相同的 token 集合，含义却相反。RNN 通过串行计算自然地捕获了顺序——Transformer 的并行注意力牺牲了这一点。因此必须通过额外的机制向模型注入位置信息。

### 5.1 两种注入策略

**加到输入端**：将位置编码 p_i 直接加到 token embedding x_i 上——input_i = x_i + p_i。原始 Transformer 使用固定的正弦/余弦函数生成 p_i，GPT-2/3 则用可学习的 Embedding 表。

**融入注意力计算**：不修改 token 表示本身，而是根据位置对 Q 和 K 施加旋转变换。RoPE（旋转位置编码）是这一策略的代表——它将位置信息编码为 Q·K 点积中的相对位置项，天然支持比训练时更长的上下文。

> 详见本目录的 [位置编码：从 Sinusoidal 到 RoPE](../positional_encoding/positional_encoding.md) 对 RoPE 数学原理与 NTK/YaRN 外推技术的完整推导。

---

## 六、层归一化与残差连接：让深网络能训练

自注意力和 FFN 解决了信息交换和变换的问题，但直接堆叠几十层会导致训练崩溃——早期的 Transformer 训练到 6 层以上就开始不稳定。

### 6.1 问题：内部协变量漂移

深度网络训练中，每层参数的更新会改变该层输出的分布。这种漂移逐层累积，使得底层接收到的梯度信号变得极其不稳定。层归一化（LayerNorm）是解决这一问题的标准手段：

$$
\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \varepsilon}} + \beta
$$

其中 $\gamma$ 和 $\beta$ 是可学习的参数，$\varepsilon$ 防止除零。

### 6.2 归一化位置的演化：Post-LN → Pre-LN → RMSNorm

| 变体    | 归一化位置                     | 使用情况              | 原因                       |
| ------- | ------------------------------ | --------------------- | -------------------------- |
| Post-LN | 残差连接之后                   | 原始 Transformer      | 训练不稳定，梯度在深层衰减 |
| Pre-LN  | 残差连接之前（先归一化再计算） | GPT-2/3               | 训练更稳定，梯度流更通畅   |
| RMSNorm | 同 Pre-LN，但只做缩放          | LLaMA、Qwen、DeepSeek | 速度更快，精度无明显损失   |

RMSNorm 去掉了均值居中（μ），仅保留均方根缩放。实验证明去掉居中操作对精度影响微乎其微，但省去了一次归约计算——在大规模训练中，这个微观优化累积为可观的加速：

$$
\text{RMSNorm}(x) = \gamma \cdot \frac{x}{\sqrt{\text{mean}(x^2) + \varepsilon}}
$$

### 6.3 残差连接：让梯度绕过子层

有了归一化还不够。堆叠 32 层以上的网络时，梯度在反向传播中经过每一层都会衰减。残差连接提供了一个「快捷通道」——将子层的输入直接加到输出上，让梯度可以绕过子层的反向传播直达输入：

$$
\text{output} = x + \text{Sublayer}(\text{LayerNorm}(x)) \qquad\text{（Pre-LN：先归一化再进子层，现代 LLM 的标准做法）}
$$

梯度可以直接通过「+」操作跳过 Sublayer 的反向传播——这相当于给每一层都提供了一条直达输入的梯度高速公路。没有残差连接，32 层以上的 Transformer 几乎无法训练。

---

## 七、完整数据流：一个 Decoder Block 的标准配方

将以上所有组件串接起来，以 LLaMA 为例——这是 2024 年主流开源 LLM 共同遵循的「标准配方」。交互可视化版本见 [Decoder Block 数据流可视化](transformer_block_visual.html)。

```text
输入: token embeddings (batch, seq_len, d_model)
  │
  ├─ 1. RMSNorm ──→ 2. Self-Attention (with RoPE + Causal Mask)
  │                                          │
  │       ┌──────────────────────────────────┘
  │       ▼
  │    3. 残差相加 (+)
  │       │
  │       ▼
  ├─ 4. RMSNorm ──→ 5. FFN (SwiGLU)
  │                    │
  │       ┌───────────┘
  │       ▼
  │    6. 残差相加 (+)
  │       │
  │       ▼
  输出 → 送入下一个 Block（重复 24-80 层，视模型规模而定）
```

**因果掩码**是 Decoder 独有的约束：生成第 t 个 token 时，模型只能看到位置 1 到 t-1，不能「偷看」未来。通过一个上三角为 -∞ 的矩阵实现：

```text
因果掩码（seq_len=4）：
  [  0, -∞, -∞, -∞ ]
  [  0,  0, -∞, -∞ ]
  [  0,  0,  0, -∞ ]
  [  0,  0,  0,  0 ]
```

---

## 八、权衡：Transformer 得到了什么，牺牲了什么

```text
     得到                                    牺牲
  ──────────────────────                ──────────────────
  训练完全并行化                       推理仍是自回归串行的
  O(1) 步内跨越任意距离                 每步的注意力计算是 O(n²)
  统一架构替代 RNN/CNN                 位置信息需额外编码（RoPE）
  易于扩展到千亿参数                    显存占用与序列长度平方成正比
```

这些权衡驱动了 Transformer 后续 7 年的演化方向——FlashAttention 缓解了 O(n²) 的内存瓶颈，RoPE+YaRN 让 4K 训练的模型外推到 128K，MoE 在维持推理成本的同时成倍扩大参数容量。这些正是 [LLM 架构演进史](../architecture_evolution/llm_architecture_evolution.md) 所覆盖的内容。

### 8.1 关键设计决策对照

| 决策       | 原始 Transformer (2017) | 现代 LLM (2023+) | 原因                                           |
| ---------- | ----------------------- | ---------------- | ---------------------------------------------- |
| 架构       | Encoder-Decoder         | Decoder-only     | 生成任务更自然，且在足够大规模下理解与生成统一 |
| 激活函数   | ReLU                    | SwiGLU           | 门控机制提供更好的训练动态                     |
| 归一化位置 | Post-LN                 | Pre-LN / RMSNorm | 训练稳定性，RMSNorm 进一步节省计算             |
| 位置编码   | Sinusoidal              | RoPE             | 编码相对位置，且支持上下文外推                 |
| 偏置项     | 有                      | 无               | 简化模型，精度无明显损失                       |
| 注意力头数 | 8                       | 32-64            | 细粒度匹配硬件并行度                           |

---

## 九、相关资源

- [Decoder Block 数据流可视化](transformer_block_visual.html) — 本文 §7 的交互可视化，可点击组件查看详情，支持完整流/Attention/FFN 三种视角。
- [位置编码：从 Sinusoidal 到 RoPE](../positional_encoding/positional_encoding.md) — RoPE 的数学原理与 NTK/YaRN 外推技术。
- [LLM 架构演进史](../architecture_evolution/llm_architecture_evolution.md) — Transformer 诞生后 7 年的关键拐点与「标准配方」的形成。
- [混合专家 (MoE)](../moe/mixture_of_experts_moe_visual_guide.zh-CN.md) — 如何用稀疏激活进一步扩展参数规模。
- 《Attention Is All You Need》(2017) — 原始论文。
- Jay Alammar, [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — 经典图解教程。
- Georgia Tech, [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) — 交互式可视化，可逐 token 观察注意力计算。
- [bbycroft/llm](https://bbycroft.net/llm) — 3D 可视化 GPT 类模型的完整推理过程。
