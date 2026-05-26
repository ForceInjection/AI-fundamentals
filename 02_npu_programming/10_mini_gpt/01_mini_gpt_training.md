# Mini-GPT：从零手写 Transformer 在 NPU 上训练

本文从零实现一个 GPT-2 风格的 decoder-only Transformer，在单张 Ascend 910B3 NPU 上完成训练和文本生成。目标不是做一个可用的 LLM，而是通过**手写每一行代码**来理解 Transformer 的内部机制——从 self-attention 的数学公式，到 causal mask 为什么要用上三角矩阵，再到 AdamW 优化器为什么比 SGD 更好。模型 ~11M 参数，字符级编码，2000 次迭代训练耗时 43 秒。

全文分两条线交织展开：**理论线**解释每个组件"为什么这样设计"（背景、动机、数学过程），**实践线**展示每个组件"怎么用代码实现"（PyTorch 代码、NPU 训练技巧、生成策略）。建议阅读顺序：先通读 §2（Transformer 核心机制），再对照 `train_gpt.py` 源码看 §3（模型架构），最后结合训练结果看 §7（loss 曲线与生成效果分析）。

---

## 1. 背景

在开始写代码之前，先厘清三个问题：为什么要手写而不用现成框架、语言模型究竟在做什么任务、字符级编码和子词编码各有什么取舍。这三个问题决定了后续所有设计决策的出发点。

### 1.1 为什么要手写

前面 7 个 phase 都是"用框架跑模型"——调用 `torchvision.models.resnet50()`、`SentenceTransformer(...)`、`torch.nn.Conv2d(...)`。这些方式能快速验证 NPU 是否可用，但跳过了一个关键环节：**理解模型内部的每一行代码在做什么**。

手写 GPT 的目的不是做一个可用的 LLM，而是回答以下问题：

- Self-attention 中的 Q、K、V 到底是什么？它们是怎样从输入变换而来的？
- "Scaled dot-product attention" 为什么要除以 √dₖ？
- Causal mask 为什么是上三角矩阵？不用 mask 会怎样？
- Multi-head 到底"多"在哪里？每个 head 在学什么？
- Position embedding 为什么需要？去掉会有什么后果？
- LayerNorm 放在 Attention 之前还是之后？为什么 GPT-2 选择"之前"？
- 训练时 forward 和 backward 各干了什么？loss 为什么能下降？

当我们亲手写出 `Q @ K^T / sqrt(d_k)`、`masked_fill(-inf)`、`F.softmax(att, dim=-1)` 时，对 Transformer 的理解会从"知道原理"变成"能写出来"。这两者之间有本质区别。

### 1.2 什么是语言模型

语言模型的任务很简单：**给定前面的文本，预测下一个 token**。

```text
输入:  "NPU 是华为"
目标:  预测下一个字符应该是什么？

模型计算: P(下个字符 | "NPU 是华为")
→ "昇" (概率 0.23), "的" (概率 0.15), "一" (概率 0.08), ...
→ 正确答案最可能是 "昇"
→ 模型输出 "昇" 的概率越高，loss 越低
```

训练过程就是不断把文本切成 (输入, 目标) 对，让模型反复练习"猜下一个字"。猜得越准，loss 越低。训练完成后，给一个开头（prompt），让模型一个字一个字地往后"续写"，就得到了文本生成。

### 1.3 为什么是字符级

通常 LLM 使用子词（subword）编码：

```text
BPE (GPT-2):  "NPU 加速计算" → [374, 249, 11865, 287]  (4 tokens)
字符级:       "NPU 加速计算" → [N, P, U,  , 加, 速, 计, 算]  (8 tokens)
```

字符级编码有两个教育优势：

- **无需外部依赖**：不用下载 tokenizer 配置文件，代码中一个 Python dict 就搞定。子词编码需要 BPE 合并规则文件（通常几 MB）、预处理脚本，增加了理解负担
- **更直观**：每个字符就是一个 token，encode/decode 完全透明。我们可以直接"看到"模型输入了什么、输出了什么，不需要在 token ID 和文本之间来回查表

代价是序列更长——同样的 block_size=128 下，字符级模型只能"看到"约 128 个字符（2-3 句话），而 BPE 模型可以看到约 2-3 倍的内容。但对于学习目的，这完全不是问题——我们追求的是"理解"，不是"好用"。

---

## 2. Transformer 核心机制

在展示代码之前，先理解每个组件在数学上做了什么。这是手写实现的理论基础。

### 2.1 Self-Attention：让每个 token "看到"其他 token

Self-attention 的核心思想是：**序列中的每个位置都可以直接访问所有其他位置的信息**。这是 Transformer 区别于 RNN 的根本特征——RNN 需要一步步传递信息（token1→token2→token3），而 Transformer 中 token1 可以直接"看到"token100。

**数学过程：**

```text
输入: X [B, T, C]  (batch_size, seq_len, n_embd)

Step 1: 投影到 Q、K、V
  Q = X @ W_Q    Query:  "我在找什么？"
  K = X @ W_K    Key:    "我是什么？"
  V = X @ W_V    Value:  "我的内容是什么？"

Step 2: 计算注意力分数
  scores = Q @ K^T / √dₖ    [B, T, T]
  矩阵中 (i, j) 位置表示 token_i 对 token_j 的"关注程度"

Step 3: Softmax 归一化
  weights = softmax(scores, dim=-1)
  每行加起来 = 1，每个值表示"应该花多少注意力在对应 token 上"

Step 4: 加权求和
  output = weights @ V    [B, T, C]
  每个位置的输出 = 所有位置 V 的加权平均
```

**为什么要除以 √dₖ？**

当 dₖ（每个 head 的维度）较大时，Q·Kᵀ 的点积值会很大（因为很多项相加）。大值送入 softmax 后，梯度会变得非常小（softmax 饱和区），训练几乎停滞。除以 √dₖ 将方差控制在 1 左右，保持在 softmax 的敏感区域。这不是理论猜想，而是原始 Transformer 论文的实验发现——不加这个缩放因子，大模型根本训不动。

**Causal Mask：让模型不能"偷看未来"**：

语言模型的任务是预测**下一个** token。如果模型能看到未来的 token，它就直接"抄答案"了，永远学不会预测。Causal mask 是一个上三角为 -∞ 的矩阵：

```text
         t0   t1   t2   t3
  t0  [  0   -∞   -∞   -∞  ]    ← t0 只能看到自己
  t1  [  0    0   -∞   -∞  ]    ← t1 能看到 t0, t1
  t2  [  0    0    0   -∞  ]    ← t2 能看到 t0, t1, t2
  t3  [  0    0    0    0  ]    ← t3 能看到所有前面的 token

softmax(-∞) = 0 → 未来位置的注意力权重为 0，模型无法利用未来信息
```

### 2.2 Multi-Head：多个"视角"同时关注

单头注意力的局限性：每个 token 只能以一种方式聚合上下文。但实际语言中，一个词可能需要同时关注"语法搭配"（下一个词是什么时态？）和"语义关联"（主语是谁？）。

Multi-head 的做法是：**将 Q/K/V 拆成多个更小的 head，每个 head 独立计算注意力，最后拼起来**。

```text
单头: Q/K/V [T, 384] → Attention → [T, 384]
多头: Q/K/V [T, 384] → split 6 heads → 6 × [T, 64] → 6 个独立 Attention → concat → [T, 384]
```

每个 head 只处理 1/6 的维度（64 而不是 384），但由于它们是独立计算的，不同 head 可以学会关注不同的模式：

- Head 1：关注相邻词的语法关系
- Head 2：关注远距离的主语-谓语搭配
- Head 3：关注标点符号和句子边界

实际训练中，head 的分工不会这么清晰，往往是混合的。但多个 head 确实提供了比单头更丰富的表示能力。

### 2.3 Feed-Forward Network (FFN)：对每个位置独立做非线性变换

Attention 负责"沟通"——让不同位置的 token 交换信息。但它的运算是完全线性的（矩阵乘法 + 加权求和），需要 FFN 引入非线性。

```text
FFN(x) = GELU(x @ W1) @ W2

W1: [n_embd, 4*n_embd]  → 先升维 4 倍 (384→1536)
W2: [4*n_embd, n_embd]  → 再降回来 (1536→384)
```

为什么中间维度是 4 倍？这是一个工程上的经验值（原始 Transformer 论文的选择）。更大的中间维度意味着更强的表示能力（可以在高维空间中做更复杂的变换），代价是更多的参数和计算量。4 倍是 GPT-2 所有尺寸都保持的比例。

FFN 对序列中的**每个位置独立**做相同的变换——它不管 token 之间的顺序关系。顺序关系完全由 Attention 和 Position Embedding 处理。

### 2.4 Residual Connection + LayerNorm：让深层网络可以训练

```text
标准的 Transformer Block:
  x = x + Attention(LayerNorm(x))     ← 残差：输入直接加回输出
  x = x + FFN(LayerNorm(x))

100 层的网络，梯度要从第 100 层传到第 1 层。没有残差连接时，梯度经过 100 次乘法后会指数级衰减（梯度消失）。残差连接的 "x + ..." 提供了一条"高速公路"，梯度可以直接从第 100 层流到第 1 层。

LayerNorm 的作用：将每层的输入标准化为均值 0、方差 1 的分布。
  - 没有 LayerNorm：每层的输出分布不断漂移，深层网络很难收敛
  - 有 LayerNorm：每层都面对"干净"的输入，训练更稳定
```

**Pre-norm vs Post-norm：** GPT-2 使用 pre-norm（先 LayerNorm 再 Attention/FFN），而不是 post-norm（先 Attention/FFN 再 LayerNorm）。pre-norm 的梯度流更顺畅（残差路径上没有 LayerNorm 挡路），训练更稳定，尤其适合深层网络。代价是最终输出可能需要额外一个 LayerNorm（`ln_f`）。

### 2.5 Position Embedding：让模型知道顺序

Attention 机制本身**没有顺序概念**——它对所有位置一视同仁。"我 爱 你"和"你 爱 我"在 Attention 看来只是 token 不同，不知道谁在前谁在后。

Position embedding 的解决方案：**给每个位置一个唯一的向量**，直接加到 token embedding 上。

```text
位置 0: [0.01, -0.03,  0.02, ...]  (384 维的向量，可训练)
位置 1: [0.02,  0.01, -0.04, ...]
位置 2: [-0.01, 0.03,  0.01, ...]
...

输入 = TokenEmbedding("NPU") + PositionEmbedding(0)
     = 这个词的语义向量      + 它在第 0 个位置的向量
```

这样，同一个词在不同位置会有不同的表示，模型可以学会"第一个词和第二个词的语法角色不同"。

---

## 3. 模型架构

有了理论基础后，本节看具体实现——先整体结构（数据从输入到输出的流向），再参数配置（每个超参数的选择理由），最后参数量拆解（每一层到底占了多少参数）。建议对照 `train_gpt.py` 源码阅读。

### 3.1 整体结构

```text
Input Tokens [B, T]   (batch=32, seq_len=128)
    │
    ├── Token Embedding  [855, 384]    ← 每个字符的语义向量
    ├── Position Embedding [128, 384]  ← 每个位置的位置向量
    │   └── x = tok_emb + pos_emb + Dropout
    │
    ▼
    ┌──────────────────────────────────────────┐
    │  TransformerBlock × 6                    │  (可以调 n_layer)
    │  ┌──────────────────────────────────────┐│
    │  │ 1. x = x + Attention(LayerNorm(x))   ││  LN → Attention → Residual
    │  │ 2. x = x + FFN(LayerNorm(x))         ││  LN → FFN → Residual
    │  └──────────────────────────────────────┘│
    │     ... 重复 6 次 ...                     │
    └──────────────────────────────────────────┘
    │
    ▼
LayerNorm → Linear [384, 855] → logits [B, 128, 855]
    │                                   ↑
    └───────────────────────────────────┘
       每个位置输出 855 个分数，对应 855 个字符的概率

loss = CrossEntropyLoss(logits, targets)
     = -log(P(正确字符))
     初始: -ln(1/855) ≈ 6.75
     训练后: loss → 0.14 表示模型对正确字符非常有信心
```

### 3.2 参数配置

Transformer 的超参数之间有一个设计约束：**n_embd 必须能被 n_head 整除**，因为每个 head 的维度 = n_embd / n_head。常见的做法是选 2 的幂次（64, 128, 256），这样在 NPU 上的内存对齐最友好。以下参数是以"11M 参数、单卡训练、30 分钟内收敛"为目标选定的：

| 参数       | 值                       | 为什么选这个值                                 |
| ---------- | ------------------------ | ---------------------------------------------- |
| vocab_size | 动态（训练集唯一字符数） | 字符级编码，无需预设词表大小                   |
| block_size | 128                      | 中文约 128 字符 = 2-3 句话，够看到基本的上下文 |
| n_layer    | 6                        | 6 层足以学习基本的语言模式，不会太浅也不会太深 |
| n_head     | 6                        | 每头 64 维 (384/6)，是 2 的幂，硬件友好        |
| n_embd     | 384                      | 6×64=384，总参数量 ~11M，单卡训练合适          |
| batch_size | 32                       | 32×128 = 4096 tokens/batch，NPU 利用率高       |
| lr         | 3e-4                     | AdamW 常用学习率，不过大也不太小               |
| dropout    | 0.1                      | 轻微正则化，防止过拟合（对于小数据尤其重要）   |

这些参数都可以通过命令行覆盖（`--n-layer 8 --n-embd 512`），方便快速实验不同配置。

### 3.3 参数量拆解

```text
Token Embedding:    855 × 384 = 328,320
Position Embedding: 128 × 384 =  49,152

每个 Block:
  CausalSelfAttention:
    c_attn (QKV):  384 × (3×384) = 442,368
    c_proj:         384 × 384     = 147,456
  FFN:
    fc1:            384 × (4×384) = 589,824
    fc2:            (4×384) × 384 = 589,824
  LayerNorm×2:      384 × 4       =   1,536
  ─────────────────────────────────────────
  每个 Block 合计:                1,771,008

6 个 Block:        6 × 1,771,008 = 10,626,048
Final LayerNorm:                   768
LM Head:           (共享 Token Embedding 权重, 0 额外参数)
───────────────────────────────────────────
总计:                            ≈ 11,004,288 (~11M)
```

可以看到，FFN 是参数量的绝对主力（每个 block 的 1.77M 参数中，FFN 占了 1.18M，约 67%）。这也是为什么扩大 FFN 的中间维度（4×→8×）会比增加层数更显著地增加参数量。

---

## 4. 训练过程

数据准备好、模型定义好后，训练就是把这两者对接起来：反复取 batch → forward 算 loss → backward 算梯度 → optimizer 更新参数。本节拆解这个循环中的每一步。

### 4.1 数据准备

字符级编码的流程很简单，在代码中由 `CharTokenizer` 类封装（`fit()` 建立映射、`encode()` 编码、`decode()` 还原）：

```python
# CharTokenizer.fit(text) — 建立字符↔ID 映射
chars = sorted(set(text))  # 855 个唯一字符
char_to_id = {c: i for i, c in enumerate(chars)}

# CharTokenizer.encode(text) — 编码全部文本
data = [char_to_id[c] for c in text]  # 57,013 个整数
```

### 4.2 训练循环

每个 iteration 做了什么：

```text
1. get_batch():
   随机选取 32 个起点，各取 128 个连续 token 作为输入 x
   对应的 y = 每个输入右移 1 位（语言模型的标准做法：
   用 token[0:127] 预测 token[1:128]）

2. forward():
   logits = model(x)              # [32, 128, 855]
   loss = CrossEntropy(logits, y) # 比较预测和真实值

3. backward():
   loss.backward()                # 反向传播，计算所有参数的梯度

4. optimizer.step():
   用梯度更新参数，使 loss 减小
   AdamW 的更新规则比普通 SGD 更复杂，引入了动量和自适应学习率
```

### 4.3 Loss 函数

Cross-Entropy Loss，语言模型的标准选择：

```text
对每个位置 i 和 batch 中的每个样本 b:

loss_i,b = -log(P_model(正确答案_i,b))

例如: 正确答案是字符 '昇' (ID=234)
      模型预测概率: '昇'=0.23, '的'=0.15, ...
      loss = -log(0.23) = 1.47

如果模型预测概率提升到 0.9:
      loss = -log(0.9) = 0.105

loss 越低 = 模型越有信心预测正确
初始随机: loss ≈ -ln(1/855) = 6.75
```

### 4.4 优化器：AdamW

SGD 只有一个全局学习率。AdamW 给每个参数独立的学习率，基于该参数的历史梯度：

- **Momentum（动量）**：如果某个参数一直往同一个方向更新，就加速它——像滚雪球
- **Adaptive LR（自适应学习率）**：如果某个参数梯度很大（影响大），就减小它的学习率（谨慎更新）；梯度小则增大学习率
- **Weight Decay（权重衰减）**：每步将所有参数向 0 拉一点点（0.01 倍），防止参数值过大导致过拟合

AdamW 是 Adam 的改进版，把 weight decay 从自适应学习率中解耦出来——一个小但重要的修正，使训练更稳定。

---

## 5. 文本生成

训练完成后，如何让模型"写"出新文本？

### 5.1 自回归生成

```text
while len(output) < max_tokens:
    logits = model(current_sequence)    # forward pass
    next_token_logits = logits[-1]      # 只取最后一个位置的预测
    probs = softmax(next_token_logits)  # 转为概率
    next_token = sample(probs)          # 按概率采样
    current_sequence.append(next_token) # 拼回去，下一轮继续
```

每一步都依赖之前生成的所有 token——这就是"自回归"（autoregressive）。

### 5.2 Temperature：控制"创造性"

```text
probs = softmax(logits / temperature)

temperature = 0.2:  分布更尖锐 → 高概率词更高 → 输出保守、重复
temperature = 0.8:  分布适中     → 保留一定随机性 → 输出合理但有变化
temperature = 2.0:  分布更平坦   → 低概率词被放大   → 输出随机、不连贯
```

temperature 不会改变哪个 token 得分最高，只改变概率分布的"陡峭程度"。GPT-2 论文中发现 temperature=0.8-1.0 是较好的默认范围。

### 5.3 Top-K Sampling：截断低概率词

如果直接按完整 855 个词的概率采样，那些概率极低的 token（比如生僻字、标点）偶尔被选中，会破坏生成质量。Top-K 的做法是：

```text
1. 选出概率最高的 K 个 token（如 K=40）
2. 将其余 token 的概率设为 0
3. 只在 Top-K 中重新归一化并采样
```

K 越小，生成越保守；K 越大，生成越多样。K=40 是 GPT-2 论文中使用的默认值，在我们的实验中也表现良好。

---

## 6. NPU 训练的特殊考虑

同样的 PyTorch 代码在 NPU 和 GPU 上运行，行为有三个值得注意的差异：首次运行的编译延迟、小模型的内存占用特征、以及为什么本实验不开启 AMP。

### 6.1 图编译延迟

NPU 第一次执行模型时，CANN 的图编译器会对计算图进行优化（算子融合、内存复用、数据布局转换）。这会导致第一个 iteration 耗时远大于后续：

```text
iter 1:   ~2000ms  (含图编译)
iter 2+:  ~22ms    (正常速度)
```

我们的训练脚本没有显式 warmup，但第一个 iter 的实际耗时被后续 iter 平均了。如果用 profiler 观测，第一个 iter 的 trace 会包含大量编译相关的 kernel。需要注意：loss 值通过 `estimate_loss()` 在 `model.eval()` 模式下独立计算，不受图编译延迟影响——iter=1 的 loss=5.43 是准确的随机初始状态评估。

### 6.2 内存占用

```text
11M 参数 × 4 bytes (FP32) = 44 MB (模型权重)
+ 优化器状态 (AdamW 需要 2 个 momentum buffer): ~88 MB
+ 中间激活 (batch=32, seq=128): ~50 MB
+ CANN 运行时开销: ~50 MB
─────────────────────────────────────────────
总计: ~230 MB HBM
```

远低于 910B3 的 64 GB HBM，资源利用率很低——这是小模型的典型特征。实际训练大模型时，HBM 带宽利用率才能真正体现 NPU 的价值。

### 6.3 为什么不用 AMP

通常训练会开启 AMP（FP16 混合精度）提升吞吐。但在本实验中：

- 模型太小（11M），AMP 的加速效果不明显
- 字符级 vocab（855）太小，FP16 的精度优势无法体现
- 教学目的：FP32 训练更简单，不需要解释 loss scaling 和 gradient overflow

---

## 7. 训练结果

以下数据来自在 NPU 7 上实际运行 `train_gpt.py` 的输出。训练语料为项目中的 7 篇 Ascend 学习文档——选择这些文档而非通用语料，是为了验证模型能否学会其中的技术术语和文档格式。

### 7.1 实验配置

| 参数                                   | 值                                         |
| -------------------------------------- | ------------------------------------------ |
| 训练语料                               | 7 篇 Ascend 学习文档，57,013 字符          |
| 字符词表大小                           | 855（含中英文、数字、标点、Markdown 符号） |
| 参数量                                 | 11.00M                                     |
| block_size / n_layer / n_head / n_embd | 128 / 6 / 6 / 384                          |
| batch_size / lr / max_iters            | 32 / 3e-4 (AdamW) / 2000                   |
| 硬件                                   | Ascend 910B3 × 1 (NPU 7)                   |

### 7.2 Loss 曲线

| iter | loss     | 阶段分析                                                                                                                     |
| ---- | -------- | ---------------------------------------------------------------------------------------------------------------------------- |
| 1    | 5.43     | 初始随机权重。随机猜测 855 个字符，理论 baseline = -ln(1/855) ≈ 6.75。模型还没学到任何东西，但权重初始化让它比纯随机稍好一点 |
| 200  | 2.52     | 模型开始学会最频繁的字符组合（空格、换行、常见标点）。这些模式简单且出现频率高，最快被掌握                                   |
| 400  | 1.56     | 开始学习常见词和短语（"NPU""CANN""PyTorch"）。这些词的字符序列是固定的，模型只需记忆                                         |
| 600  | 0.87     | 句子级模式开始涌现。Markdown 的标题标记和表格分隔符等格式字符的使用方式被学会                                                |
| 800  | 0.47     | 开始记忆训练数据中的具体段落。loss 快速下降但泛化能力下降                                                                    |
| 1000 | 0.28     | 已进入过拟合区间。loss < 1 通常意味着模型在"背诵"训练集                                                                      |
| 2000 | **0.14** | 严重过拟合——57K 字符 vs 11M 参数，参数数量是数据量的 ~200 倍。模型几乎记住了整个训练集                                       |

**总训练时间：43 秒。** 2000 iters，平均每 iter ~22ms（forward + backward + optimizer step 全部在 NPU 上完成）。

### 7.3 为什么过拟合是预期结果

数据量和模型规模之间的健康比例通常要求 **训练 token 数 >> 参数量**。例如 GPT-3 的训练数据约 300B tokens，参数量 175B，比例约 1.7:1。

我们的情况：

```text
57,013 tokens / 11,000,000 params ≈ 0.005:1
```

每个参数平均只见过 0.005 个 token——这是严重的"数据不足"。打个比方：这就像一个学生背了 57,000 个字的课文，但脑容量能记 1100 万个字的细节——他不需要理解课文意思，直接逐字背诵就行。模型也是如此：过多的参数给了它"死记硬背"的能力，而不是被迫学习可以泛化的语言规则。

要改善过拟合，需要：

- **增加数据量**：≥ 10MB 文本（约 1000 万字符），比例提升到 ~1:1
- **减小模型**：n_layer=4, n_embd=256 → ~3M 参数，对 57K tokens 仍然偏多但会改善
- **增大 dropout**：0.1→0.3，更强的正则化
- **early stopping**：在 loss ~1.5 时停止训练，得到更有泛化能力的模型

### 7.4 生成效果与分析

```text
Prompt: NPU 是

生成结果:
NPU 是华为昇腾系列 AI 处理器，采用 **达芬奇 (Da Vinci)**：设计算架构，
7 nm 制程。本服务器配备 8 张 910B3 卡，通过 HCCS (Huawei Cache
Coherence System) 全互联。

关键硬件参数 (实测数据)：

| 参数                    | 值
```

**分析：**

这个生成结果虽然是"背诵"训练数据，但展现了一些有趣的行为：

- 正确组织了"NPU 是...AI 处理器"的定义句式
- 准确使用了"达芬奇(Da Vinci)"的技术术语，并保留了 Markdown 强调标记 `**`
- "7 nm 制程"是训练数据中确实存在的硬件参数
- `HCCS (Huawei Cache Coherence System)` 完整保留了缩写+全称的格式
- 自动生成了 Markdown 表格的表头，格式正确

不足之处：

- 表格没有完整填充数据（`| 参数 | 值` 后直接截断）
- "设计算架构"多了一个"计"字（生成的小瑕疵）
- 完全在训练数据的知识范围内，没有"创造性"——这是过拟合的直接体现

换一个训练数据中不常见的 prompt：

```text
Prompt: MindSpore

生成结果:
MindSpore                                   |
| ------------ | -------------------------------------------------------------------
```

这是因为 "MindSpore" 在训练数据中主要出现在对比表格中（MindSpore vs PyTorch），模型学会了：当看到 "MindSpore" 时，接下来应该画表格。但由于过拟合，它只会画表格，不会生成关于 MindSpore 的描述性文字。

---

## 8. 代码结构

`train_gpt.py` 约 310 行，所有组件放在一个文件中，方便对照理论章节阅读。以下按数据流顺序（token 从输入到输出经过的路径）列出各模块：

```text
10_mini_gpt/
└── train_gpt.py       # Mini-GPT 完整实现（~310 行）
    ├── CharTokenizer           — 字符↔ID 映射 (encode/decode)
    ├── CausalSelfAttention     — 多头 causal self-attention
    │   ├── Q/K/V 合并投影       (一次 Linear 替代三次)
    │   ├── Split/Merge heads   (reshape + transpose)
    │   ├── Scaled dot-product  (Q·Kᵀ / √dₖ + causal mask)
    │   └── Output projection   (c_proj + dropout)
    ├── TransformerBlock        — Attention + FFN + 残差
    │   ├── pre-norm LayerNorm  (GPT-2 风格)
    │   ├── FFN (4× expand)     (GELU 激活)
    │   └── 2 个残差连接        (梯度高速公路)
    ├── MiniGPT                 — 完整模型组装
    │   ├── Token + Position Embedding
    │   ├── N × TransformerBlock
    │   ├── Final LayerNorm + LM Head
    │   ├── Weight Tying        (LM head 共享 token embedding 权重)
    │   └── generate()          (自回归 + top-k sampling)
    ├── Trainer                 — 训练循环
    │   ├── get_batch()         (随机采样 x/y 对)
    │   ├── estimate_loss()     (多 batch 平均 loss)
    │   ├── train()             (主循环 + 进度打印)
    │   └── save_checkpoint()   (模型 + tokenizer + 配置)
    └── main()                  — CLI (train / generate 两种模式)
```

---

## 9. 与之前 phase 的联系

本 phase 的手写 Transformer 不是孤立的工作——它的每个组件都可以在之前的实验中找到对应的基础：

| Phase               | 关联                                                                                                                                              |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| Phase 1 (Hello NPU) | 矩阵乘法是 Attention 的计算核心。Q·Kᵀ 本质就是两个矩阵的乘法——和 `torch.matmul(A, B)` 完全一样                                                    |
| Phase 3 (ResNet-50) | 训练循环（forward→loss→backward→optimizer.step）的模式完全相同，只是模型从 CNN 换成了 Transformer                                                 |
| Phase 7 (Profiling) | 可以对本脚本做 profiling，观察 Attention（矩阵乘法 bound）和 FFN（也是矩阵乘法 bound）各占多少 NPU 时间                                           |
| Phase 6 (RAG)       | RAG 中的 embedding 模型是 BERT（encoder-only），而 GPT 是 decoder-only。两者的 Attention 机制几乎一样，区别在于 BERT 是双向、GPT 是单向（causal） |

---

## 10. 后续扩展

本实验的 Mini-GPT 是一个刻意简化的起点——11M 参数、57K 字符、字符级编码，每一项都在"压制"模型的能力，以便把焦点放在理解而非性能上。在此基础上，后续可以按以下优先级逐步放开约束：

**建议顺序**：先换子词编码（立即改善"视野"），再增大数据（让模型真正学会语言规则），然后上 AMP 和多卡（把训练速度提上来），最后尝试 FlashAttention 等进阶优化。

| 方向           | 做法                        | 预期效果                                       |
| -------------- | --------------------------- | ---------------------------------------------- |
| 更大的数据     | 中文维基百科（~1.5GB 文本） | loss 不会降到 0.1，但生成质量显著提升          |
| 子词编码       | BPE tokenizer，vocab=5000   | 序列长度缩短 ~2×，同样 block_size 下"视野"翻倍 |
| AMP 混合精度   | `torch.npu.amp.autocast()`  | 吞吐提升 1.5-2×                                |
| 多卡 DDP       | 8 张 NPU 数据并行           | 等效 batch_size 增大 8×，大模型训练成为可能    |
| FlashAttention | 减少 HBM 读写               | 对长序列（block_size ≥ 512）加速明显           |
| 学习率调度     | Cosine decay / warmup       | 训练更稳定，最终 loss 更低                     |

---

## 参考链接

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Language Models are Unsupervised Multitask Learners (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [nanoGPT (Andrej Karpathy)](https://github.com/karpathy/nanoGPT)
- [Let's build GPT: from scratch, in code, spelled out (Karpathy)](https://www.youtube.com/watch?v=kCc8FmEb1nY)
