# 从 GPT-2 到 Kimi K3：注意力机制的演进史

> 本文基于 Baseten 博客 "[22,580: GPT-2 to Kimi K3, explained](https://www.baseten.co/blog/22580-gpt-2-to-kimi-k3-explained/)"（作者 Ali Taha，2026.07.30）翻译并深度注释。在保留原文叙事线的基础上，补充了完整的 PyTorch 示例代码、对照 vLLM/SGLang 源码的验证注释、以及与本站已有文章的交叉引用。文中以 `> **注**：` 标记的块为本站补充内容。

---

两万两千五百八十。这是能塞进一个 Kimi K3（2026）里的 GPT-2（2019）的数量——2.8 万亿参数 ÷ 1.24 亿参数 = 22,580。

但这不仅仅是规模的故事。每一次架构迭代，都在解决前一代的一个具体缺陷。本文追踪一条完整的技术演进线索——从 softmax 注意力到线性注意力，从 DeltaNet 到 Gated DeltaNet，从 Kimi Linear 到 Kimi K3 的混合架构——每一步回答一个问题：**前一步卡在哪里，新方案用什么代价换来了什么收益。**

---

## 一、GPT-2（2019）：起点

GPT-2 是最纯粹的 Decoder-Only Transformer。输入经过 12 层完全相同的 block，最后通过 LayerNorm 和 lm_head 输出词表上的概率分布。

### 1.1 完整架构

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GPT2Attention(nn.Module):
    """GPT-2 的多头自注意力（含因果 mask）。"""
    def __init__(self, n_embd=768, n_head=12, dropout=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.d_head = n_embd // n_head   # 64

        # Q、K、V 在一次矩阵乘法中完成投影
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=True)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape  # batch, seq_len, n_embd

        # 一次投影得到 Q、K、V，然后拆成多头
        qkv = self.c_attn(x)                     # [B, T, 3*C]
        q, k, v = qkv.split(C, dim=-1)           # 各 [B, T, C]
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)  # [B, nh, T, dh]
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        # Softmax 注意力
        att = (q @ k.transpose(-2, -1)) * (1.0 / self.d_head ** 0.5)  # [B, nh, T, T]
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        att = att.masked_fill(causal_mask, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v                               # [B, nh, T, dh]

        # 合并多头并输出投影
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class GPT2Block(nn.Module):
    """GPT-2 的一个 Transformer Block。"""
    def __init__(self, n_embd=768, n_head=12, dropout=0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = GPT2Attention(n_embd, n_head, dropout)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),   # GELU 激活
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))   # 残差 + 注意力
        x = x + self.mlp(self.ln_2(x))    # 残差 + FFN
        return x


class GPT2(nn.Module):
    """GPT-2 完整模型（124M 参数）。"""
    def __init__(self, vocab_size=50304, n_layer=12, n_embd=768, n_head=12, block_size=1024):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([GPT2Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(pos)  # [B, T, C]
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # [B, T, vocab_size]
        return logits
```

### 1.2 自回归解码的低效

问题出在生成环节。自回归解码时，每生成一个 token 都需要把**整个序列**重新送进模型：

```python
def generate_naive(model, prompt, max_new_tokens=50):
    """朴素生成：每步重算所有 token 的投影——O(T²) 计算量。"""
    tokens = prompt.clone()
    for _ in range(max_new_tokens):
        logits = model(tokens)            # 对所有 T 个位置计算注意力
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # 只用最后一个位置
        tokens = torch.cat([tokens, next_token], dim=1)
    return tokens
```

每一步都对**所有前置 token** 计算 Q、K、V 投影和注意力分数，但最终只消费最后一个位置的 logits。序列长度 T 时，第 t 步的计算量与 t² 成正比——这是平方级增长的浪费。

---

## 二、KV Cache：用空间换时间

自回归解码时，已生成 token 的 K 和 V 在后续步骤中不会改变（因果 mask 确保当前 token 看不到未来 token）。因此可以将它们的 K、V **存储**下来，新步骤只计算新 token 的 Q、K、V，Q 只与缓存中的全部 K 做注意力。

### 2.1 带 KV Cache 的注意力

```python
class GPT2AttentionWithCache(GPT2Attention):
    """在 GPT-2 注意力上加入 KV Cache 支持。"""
    def forward(self, x, kv_cache=None):
        B, T, C = x.shape

        qkv = self.c_attn(x).split(C, dim=-1)  # 只投影当前序列（可能只有 1 个 token）
        q, k, v = [t.view(B, T, self.n_head, self.d_head).transpose(1, 2) for t in qkv]

        if kv_cache is not None:
            # 将新 token 的 K、V 追加到缓存
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)  # [B, nh, past+T, dh]
            v = torch.cat([v_cache, v], dim=2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / self.d_head ** 0.5)
        att = F.softmax(att, dim=-1)
        y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y, (k, v)  # 返回更新后的缓存


def generate_with_kv_cache(model, prompt, max_new_tokens=50):
    """使用 KV Cache 的生成：每步只计算新 token。"""
    tokens = prompt.clone()

    # Prefill：一次性计算 prompt 的全部 K、V 并缓存
    logits, kv_caches = model(tokens, use_cache=True)  # 返回每层的 cache
    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    tokens = torch.cat([tokens, next_token], dim=1)

    for _ in range(max_new_tokens - 1):
        # Decode：每次只送入 1 个新 token，复用缓存的 K、V
        logits, kv_caches = model(next_token, kv_caches=kv_caches, use_cache=True)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tokens = torch.cat([tokens, next_token], dim=1)
    return tokens
```

### 2.2 KV Cache 的代价

KV Cache 将每步解码的计算量从 O(T²·D) 降到 O(T·D)，但存储随序列长度**线性增长**：

$$\text{KV Cache 大小} = 2 \times n_{\text{layer}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times \text{seq\_len} \times \text{dtype\_bytes}$$

对于 70B 模型、128K 上下文的标准 GQA8 配置，仅 KV Cache 就吃掉了约 320 GB 显存。

更关键的是，解码时的每步都要从 HBM 中**流式读出全部 KV Cache**。即使计算量已经减小，内存带宽仍是瓶颈：每步 decode 的计算强度（FLOP/Byte）极低，属于 **memory-bound** 操作。

> **注**：KV Cache 的存储、压缩、量化、淘汰策略等，详见本站 **[KV Cache 技术体系](../../../09_inference_system/kv_cache/README.md)**（42 篇文章）。PagedAttention 如何将 KV Cache 从连续内存中"打碎"以消除碎片，See **[vLLM PagedAttention 源码分析](../../../09_inference_system/vllm/README.md)**。

---

## 三、线性注意力（2020）：固定大小的状态矩阵

KV Cache 随序列长度线性增长，解码时反复流式读取——内存带宽成为制约因素。线性注意力的核心洞察是：**如果对 Q 和 K 的非线性变换发生在点积之前，乘法就可以重新结合。**

### 3.1 核心思想：重新结合（Reassociation）

标准 softmax 注意力：

$$\text{softmax}\left(\frac{Q K^\top}{\sqrt{d}}\right) \cdot V$$

softmax 在 Q·K 乘积**之后**施加非线性（指数 + 归一化），导致无法改变计算顺序——必须显式计算并存储 N×N 的注意力矩阵。

线性注意力的变通：对 Q 和 K **各自**施加一个非线性特征映射 φ（如 ELU+1），使它们的乘积可以重新结合：

$$\phi(Q) \cdot \left(\phi(K)^\top \cdot V\right) = \left(\phi(Q) \cdot \phi(K)^\top\right) \cdot V$$

前者先算 φ(K)ᵀ·V（D×D 矩阵），再与 φ(Q) 相乘——复杂度 O(ND²)，与序列长度 N **线性**相关。后者先算 φ(Q)·φ(K)ᵀ（N×N 矩阵）——复杂度 O(N²D)。

```python
import torch
import torch.nn.functional as F

class LinearAttention(nn.Module):
    """线性注意力：用 ELU+1 特征映射替代 softmax，固定大小状态矩阵。

    关键公式：
        S = S + k^T @ v       # [D, D] 状态矩阵，不随序列长度增长
        o = q @ S / (q @ z)   # z 是归一化项
    """
    def __init__(self, d_model=512, d_head=64):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    @staticmethod
    def elu_feature_map(x):
        """ELU + 1：将输入映射到非负空间，作为 softmax 的近似替代。"""
        return F.elu(x) + 1.0

    def forward_recurrent(self, x):
        """循环模式（decode）：逐 token 更新状态矩阵。

        这是线性注意力的核心优势 —— 每步解码只需 O(D²) 而非 O(T·D)。
        """
        B, T, C = x.shape
        q = self.W_q(x).view(B, T, -1, self.d_head)  # 简化为单头
        k = self.W_k(x).view(B, T, -1, self.d_head)
        v = self.W_v(x).view(B, T, -1, self.d_head)

        q = self.elu_feature_map(q)  # φ(q)
        k = self.elu_feature_map(k)  # φ(k)

        S = torch.zeros(B, self.d_head, self.d_head, device=x.device)  # 状态矩阵 [B, D, D]
        z = torch.zeros(B, self.d_head, device=x.device)               # 归一化项 [B, D]
        outputs = []

        for t in range(T):
            qt, kt, vt = q[:, t], k[:, t], v[:, t]          # [B, D]

            # 固定大小的状态更新（不随 t 增长！）
            S = S + torch.einsum('bd,bv->bdv', kt, vt)      # k^T @ v → [B, D, D]
            z = z + kt                                       # 归一化分母累加

            # 读出
            o = torch.einsum('bd,bdv->bv', qt, S)           # q @ S → [B, D]
            o = o / (torch.einsum('bd,bd->b', qt, z).unsqueeze(-1) + 1e-8)
            outputs.append(o)

        return torch.stack(outputs, dim=1)  # [B, T, D]
```

### 3.2 与标准注意力的对比

| 机制 | 状态大小 | 每步解码复杂度 | 瓶颈 |
|---|---|---|---|
| Softmax + KV Cache | O(N·D) 随序列增长 | O(N·D) 从 HBM 读取全部 cache | 内存带宽 |
| 线性注意力（循环） | O(D²) **固定不变** | O(D²) 纯矩阵乘法 | 状态容量（D² 有限） |

### 3.3 代价：表达力的损失

ELU+1 只是 softmax 核的一个粗糙近似。softmax 通过指数函数对相似度做**尖锐的差异化**（最重要的 key 获得压倒性权重），而 ELU+1 对所有正值对给予线性级别的权重。论文中常提到的"千倍加速"，通常对比的是**无 cache 的旧基线**，相对现代 KV-cached 实现的收益更温和。

> **注**：ELU+1 特征映射选取得非常巧妙——ELU 的负值部分趋近于 -1，加 1 后确保非负（类似 softmax 的 exp 始终为正），同时保留了近似线性的梯度。但这种"一刀切"的非线性无法捕捉 softmax 中 token 间的差异化竞争。

---

## 四、DeltaNet：用 Delta 规则解决信息干扰

### 4.1 线性注意力的信息干扰问题

线性注意力的状态矩阵 S 通过**纯加性更新**积累信息：

$$S_t = S_{t-1} + k_t^\top \otimes v_t$$

一旦状态矩阵的容量（由 D×D 维度决定）被填满，继续写入新关联就会覆盖旧信息。Schlag 等人精确地描述了这一现象：

> "当序列长度超过存储容量时，模型会进入超容量状态——新的关联与已有关联产生干扰，且没有任何机制让旧信息离开缓存。"

这就是「信息干扰（interference）」问题：纯加性更新是一个只进不出的记忆体。

### 4.2 Delta 规则：只写入真正的新信息

Delta 规则的核心思想来自神经科学中的**联想记忆更新法则**：在写入新关联之前，先用当前 key 读回已存储的值，只将**新旧值之差（delta）**写入存储。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeltaNetAttention(nn.Module):
    """DeltaNet：用 Delta 规则替代纯加性更新。

    三步走：
        1. v_old = k @ S      —— 用当前 key 读回已存储的值
        2. u = beta * (v - v_old)  —— 只取真正的新信息（delta）
        3. S = S + k^T @ u    —— 写入修正后的值

    关键洞察：如果 k 是单位向量，k @ (k^T @ v) = v，读回的值就是精确的旧关联。
    因此对 Q 和 K 做 F.normalize 使读回精确，擦除项成为真正的投影。
    """
    def __init__(self, d_model=512, d_head=64):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_beta = nn.Linear(d_model, 1, bias=False)      # 逐 token 写入强度（标量）
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward_recurrent(self, x):
        """DeltaNet 的循环（decode）实现：逐 token 做 delta 更新。

        线性的 k @ S 操作先读出旧关联，再通过 (v - v_old) 只写入差异部分。
        """
        B, T, C = x.shape
        q = self.W_q(x).view(B, T, -1, self.d_head)
        k = self.W_k(x).view(B, T, -1, self.d_head)
        v = self.W_v(x).view(B, T, -1, self.d_head)

        # SiLU + 归一化：SiLU 提供非线性，normalize 使读回精确
        q = F.normalize(F.silu(q), dim=-1)
        k = F.normalize(F.silu(k), dim=-1)

        beta = torch.sigmoid(self.W_beta(x)).view(B, T, 1)    # [B, T, 1] 每 token 标量

        S = torch.zeros(B, self.d_head, self.d_head, device=x.device)  # 状态矩阵 [B, D, D]
        outputs = []

        for t in range(T):
            qt, kt, vt = q[:, t], k[:, t], v[:, t]    # [B, D]
            beta_t = beta[:, t]                        # [B, 1]

            # 步骤 1：读回当前 key 对应的旧值
            v_old = torch.einsum('bd,bdv->bv', kt, S)  # k @ S → [B, D]

            # 步骤 2：只写入差值（delta）
            u = beta_t * (vt - v_old)                   # [B, D]

            # 步骤 3：外积写入
            S = S + torch.einsum('bd,bv->bdv', kt, u)   # S += k^T @ u

            # 读出（无需归一化分母——delta 规则自带平衡）
            o = torch.einsum('bd,bdv->bv', qt, S)
            outputs.append(o)

        return torch.stack(outputs, dim=1)
```

### 4.3 为什么归一化是关键

单个关联 S = kᵀ ⊗ v 用同一 key 读回时，得到 (k·kᵀ)·v = ‖k‖²·v（key 的平方范数乘以 v）。通过 `F.normalize(F.silu(k), dim=-1)` 先对 key 施加非线性激活再归一化到单位长度，使 ‖k‖² = 1，读回的值就是**精确的旧关联**。这使得擦除项成为真正的投影——旧信息被**减法移除**，而非叠加覆盖。

Q 是可学习的「指针」：Wq 和 Wk 读取同一残差流，查询向量指向该事实被写入时的 key 方向。更新流程是——先问当前 key 从 cache 里取回了什么旧信息，将其从要存的值中减去，再乘以 key 加回状态矩阵。旧信息被移除，新信息在它的位置上写入。

---

## 五、分块并行化 DeltaNet：让 Prefill 也能高效

### 5.1 串行瓶颈

DeltaNet 的循环实现有一个致命的低效：prefill 阶段必须串行逐 token 计算——每个 token 都需要前一个 token 的状态来计算要减去的修正项。纯加性线性注意力虽然也是循环的，但至少可以通过结合律把一次 prefill 的前缀折叠计算。Delta 规则的「读回-修正-写入」步骤天然串行。

解决办法是**分块（chunking）**：将序列切成大小为 C 的块，块内做真正的因果注意力（可并行），跨块用状态矩阵（串行但只需 C 次而非 N 次）。

### 5.2 DeltaNet 的矩阵形式

Delta 规则的原始循环形式：

$$S_t = S_{t-1} - \beta_t \cdot (k_t^\top \otimes (k_t \cdot S_{t-1})) + \beta_t \cdot (k_t^\top \otimes v_t)$$

可重新参数化为一个更紧凑的形式：

$$S_t = S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top$$

这一重新参数化是将 delta 更新**向量化到块级别**的关键——它允许我们在一个块内一次性计算全部 C 个 delta。

### 5.3 分块实现

```python
def chunked_delta_net(q, k, v, beta, chunk_size=64):
    """DeltaNet 的分块并行实现。

    块内：做真正的因果注意力（可并行）
    跨块：用状态矩阵折叠（串行，但只需 ceil(T/C) 步）

    Args:
        q, k, v: [B, T, D] — 已归一化的 Q、K（q, k 为单位向量）
        beta: [B, T, D] — 逐 token 写入强度
        chunk_size: 块大小（C=64 或 128，匹配 GPU 张量核指令粒度）

    Returns:
        o: [B, T, D] — 输出
    """
    B, T, D = q.shape
    C = chunk_size
    n_chunks = (T + C - 1) // C

    S = torch.zeros(B, D, D, device=q.device)  # 状态矩阵 [B, D, D]
    outputs = []

    for chunk_idx in range(n_chunks):
        start = chunk_idx * C
        end = min(start + C, T)
        actual_C = end - start

        # 取出当前块
        q_c = q[:, start:end]      # [B, c, D]
        k_c = k[:, start:end]
        v_c = v[:, start:end]
        beta_c = beta[:, start:end]

        # ---- 块内：因果注意力（可并行矩阵乘法） ----
        # 注意：线性注意力/DeltaNet 在块内做因果掩码的 qk^T·v，
        # 不使用 softmax——特征映射（ELU+1 或 SiLU+normalize）已提供非线性。
        # 这是与标准 softmax 注意力的关键区别。
        attn_intra = torch.einsum('btd,bsd->bts', q_c, k_c)  # [B, c, c]
        attn_intra = torch.tril(attn_intra, diagonal=-1)     # 因果掩码（不含对角线）
        o_intra = torch.einsum('bts,bsd->btd', attn_intra, v_c)  # [B, c, D]

        # ---- 跨块：从状态矩阵读出已有信息 ----
        o_inter = torch.einsum('btd,bdv->btv', q_c, S)   # [B, c, D]
        # 注释：这里为简化省略了 delta 修正的完整前向代入步骤，
        # 完整实现见下文 5.4 节。

        # 合并块内和跨块，更新状态矩阵
        o_c = o_intra + o_inter
        outputs.append(o_c)

        # 用整个块的 K、V 更新状态矩阵（加性折叠）
        S = S + torch.einsum('btd,btv->bdv', k_c, v_c)

    return torch.cat(outputs, dim=1)  # [B, T, D]
```

### 5.4 Delta 规则的完整块内修正（前向代入）

上述简化实现使用纯加性更新跨块折叠。真正的 DeltaNet 分块实现需要在块内做 **delta 修正**，这是通过前向代入（forward substitution）求快速逆来实现的：

```python
def delta_correction_within_chunk(k_c, v_c, beta_c, S):
    """块内的 delta 修正：构建三角修正矩阵 T，用前向代入求逆。

    核心公式：
        T = -(K_beta @ K^T).tril(-1)   # 严格下三角修正矩阵
        对 T 做前向代入（for i in range(1, C): T[i,:i] += T[i,:,None] * T[:,:i]）
        T += I                           # 加单位矩阵
        W = T @ K_beta                   # 修正后的 key 权重
        U = T @ V_beta                   # 修正后的 value
        然后：u_i = U[i] - w_i @ S       # 这个块的全部修正量

    这使 delta 规则在块内可以被向量化计算，而不丢失"读回旧值再写入差值"的语义。
    """
    B, c, D = k_c.shape
    device = k_c.device

    # K_beta = beta * k：将写入强度与 key 合并
    K_beta = beta_c.unsqueeze(-1) * k_c          # [B, c, D]
    V_beta = beta_c.unsqueeze(-1) * v_c          # [B, c, D]

    # 构建严格下三角修正矩阵（不含对角线）
    T = -torch.einsum('btd,bsd->bts', K_beta, k_c)  # [B, c, c]
    T = torch.tril(T, diagonal=-1)                    # 只保留下三角

    # 前向代入：将三角矩阵转换为全矩阵形式
    for i in range(1, c):
        T_i = T[:, i, :i]                            # [B, i]
        T_col = T[:, :i, :i]                          # [B, i, i]
        correction = torch.einsum('bi,bij->bj', T_i, T_col)  # [B, i]
        T[:, i, :i] = T_i + correction

    # 加上单位矩阵
    T = T + torch.eye(c, device=device).unsqueeze(0)  # [B, c, c]

    # 将修正应用到 K 和 V 的权重上
    W = torch.einsum('bts,bsd->btd', T, K_beta)       # [B, c, D]
    U = torch.einsum('bts,bsd->btd', T, V_beta)       # [B, c, D]

    return W, U
```

### 5.5 C 的选择：FLOP 最少 vs 墙钟最快

```
C = 1   → 退化为纯循环线性注意力（最少 FLOP，但 GPU 利用率极低）
C = T   → 恢复完整 O(T²) softmax 注意力（最多 FLOP，但 GPU 矩阵乘法硬件充分利用）
C = 64  → 当前 GPU 张量核（如 UMMA 指令）的最佳粒度
```

> **注**：Kimi K3 的 KDA 实际使用 `FLA_CHUNK_SIZE = 64`（`vllm/third_party/flash_linear_attention/ops/utils.py:31`）。对于 100 万 token 的上下文，这需要 15,625 次串行跨块步骤——这是我们在 [post-kv-cache-era-challenges.md](../../../09_inference_system/post-kv-cache-era-challenges.md) §3 中分析的 KDA chunkwise serial 约束的来源。

### 5.6 对比：MHA vs DeltaNet

| 维度 | MHA + KV Cache | DeltaNet（循环模式） |
|---|---|---|
| 每步解码 FLOP | O(T·D) | O(D²) |
| 显存占用（每层） | O(T·D) 随序列增长 | O(D²) **固定** |
| 信息更新方式 | 每次重读全部上下文 | Delta 规则——增量修正 |
| 旧信息移除 | KV Cache 淘汰策略（LRU 等） | Delta 规则自动擦除 |
| Prefill 效率 | O(T²) 但可并行 | O(T·D²) 需要分块并行 |

---

## 六、Gated DeltaNet（GDN）：合并门控与 Delta 规则

### 6.1 Delta 规则的局限：只能替换，不能衰减

Delta 规则有一个隐含假设：记忆中的旧关联可以被精确地「擦除并替换」。但实际使用中，有时需要的是**全局衰减**——上下文切换时批量清除多条关联，或者让旧记忆随时间自然减弱。

Mamba-2 早已展示了一种简单有效的方案：**加性更新 + 门控衰减**：

$$S_t = \alpha_t \cdot S_{t-1} + k_t^\top \otimes v_t, \quad \alpha_t \in [0, 1]$$

问题是，这种均匀衰减"不区分不同 key-value 关联的重要性"——需要忘记某个特定事实时，所有关联都被同等削弱。

### 6.2 GDN：门控衰减 × Delta 修正

Gated DeltaNet 将两种机制合并：

$$S_t = \alpha_t \cdot S_{t-1} - \beta_t \cdot (k_t^\top \otimes (k_t \cdot (\alpha_t \cdot S_{t-1}))) + \beta_t \cdot (k_t^\top \otimes v_t)$$

两个参数各司其职：
- **α（衰减门）**：控制旧状态的全局保留程度。α=1 时完全保留，α=0 时清空记忆。
- **β（写入强度）**：控制新信息的写入力度。β=0 时跳过更新。

α=1 时退化为纯 DeltaNet；α=0 时清空记忆体。

```python
class GatedDeltaNet(nn.Module):
    """Gated DeltaNet：在 Delta 规则基础上增加逐 token 衰减门。

    状态更新公式：
        S_t = α_t · S_{t-1}  +  Delta更新(k_t, v_t, α_t · S_{t-1})

    其中 Delta 更新在衰减后的旧状态上执行，确保「先遗忘，再写入」。
    """
    def __init__(self, d_model=512, d_head=64, d_state=None):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.d_state = d_state or d_head   # 状态矩阵维度

        self.W_q = nn.Linear(d_model, d_head, bias=False)
        self.W_k = nn.Linear(d_model, d_head, bias=False)
        self.W_v = nn.Linear(d_model, d_head, bias=False)
        # 双门控参数（均为 per-token 标量；per-channel 是 KDA 的改进）
        self.W_alpha = nn.Linear(d_model, 1, bias=False)    # 衰减门（标量）
        self.W_beta = nn.Linear(d_model, 1, bias=False)     # 写入强度（标量）
        self.W_o = nn.Linear(d_head, d_model, bias=False)

    def forward_recurrent(self, x):
        B, T, C = x.shape

        q = F.normalize(F.silu(self.W_q(x).view(B, T, -1)), dim=-1)
        k = F.normalize(F.silu(self.W_k(x).view(B, T, -1)), dim=-1)
        v = self.W_v(x).view(B, T, -1)

        alpha = torch.sigmoid(self.W_alpha(x)).view(B, T, 1)  # [B, T, 1] 每 token 标量
        beta = torch.sigmoid(self.W_beta(x)).view(B, T, 1)

        S = torch.zeros(B, self.d_head, self.d_state, device=x.device)
        outputs = []

        for t in range(T):
            qt, kt, vt = q[:, t], k[:, t], v[:, t]
            alpha_t = alpha[:, t]
            beta_t = beta[:, t]

            # 步骤 1：先衰减旧状态
            S = alpha_t.unsqueeze(-1) * S

            # 步骤 2：在衰减后的状态上执行 Delta 更新
            v_old = torch.einsum('bd,bdv->bv', kt, S)
            u = beta_t * (vt - v_old)
            S = S + torch.einsum('bd,bv->bdv', kt, u)

            # 读出
            o = torch.einsum('bd,bdv->bv', qt, S)
            outputs.append(o)

        return torch.stack(outputs, dim=1)
```

### 6.3 架构演进至此

```
加性线性注意力 (2020)     →  固定大小状态，但信息只进不出
      ↓
DeltaNet                   →  用 Delta 规则移除旧关联再写入，但无法全局衰减
      ↓
Gated DeltaNet (GDN)       →  合并门控衰减 + Delta 修正，记忆管理完整了
```

> **注**：这一演进线（加性 → Delta → Gated Delta）独立于 Transformer 的 MHA→MQA→GQA→MLA 主线。两条线在 Kimi Linear / Kimi K3 中交汇：KDA 提供恒定大小的循环记忆，周期性的 MLA 层提供完整上下文的 softmax 检索。详见 [post-kv-cache-era-challenges.md](../../../09_inference_system/post-kv-cache-era-challenges.md) §3。

---

## 七、KDA / Kimi Linear：精细门控 × 混合架构

### 7.1 GDN 的最后一个短板

Gated DeltaNet 用一个标量 α 控制每 token 的全局衰减力度。但序列中的 token 有不同的信息密度——一个"的"字和一个实体名词的衰减需求完全不同。**逐通道（per-channel）门控**给状态矩阵的每个维度独立的衰减速率，这是 Kimi Linear 论文最重要的贡献。

### 7.2 逐通道门控

```python
class KimiLinearAttention(nn.Module):
    """KDA（Kimi Delta Attention）：在 GDN 基础上的三项关键改进。

    改进 1：逐通道门控（per-channel gating）
        将单一标量 α 替换为每通道独立的值 α ∈ R^D。
        不同通道可以有不同的衰减速率——高频通道快速遗忘，
        低频通道保留长程信息。

    改进 2：MLA 层混合
        在 KDA 层之间交错插入 Multi-head Latent Attention (MLA) 层，
        周期性提供完整上下文的 softmax 检索能力。

    改进 3：MoE FFN
        将密集 FFN 替换为混合专家层，以固定推理成本获得更大模型容量。
    """
    def __init__(self, d_model=512, d_head=64):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head

        self.W_q = nn.Linear(d_model, d_head, bias=False)
        self.W_k = nn.Linear(d_model, d_head, bias=False)
        self.W_v = nn.Linear(d_model, d_head, bias=False)
        # 关键改进：每个通道独立的门控参数
        self.W_alpha = nn.Linear(d_model, d_head, bias=False)   # [B, T, D] 每通道衰减
        self.W_beta = nn.Linear(d_model, d_head, bias=False)    # [B, T, D] 每通道写入强度
        self.W_o = nn.Linear(d_head, d_model, bias=False)

    def forward_recurrent(self, x):
        B, T, C = x.shape

        q = F.normalize(F.silu(self.W_q(x).view(B, T, -1)), dim=-1)
        k = F.normalize(F.silu(self.W_k(x).view(B, T, -1)), dim=-1)
        v = self.W_v(x).view(B, T, -1)

        # 逐通道门控：alpha 不再是标量，而是 [B, T, D] 的向量
        alpha = torch.sigmoid(self.W_alpha(x)).view(B, T, -1)  # [B, T, D]
        beta = torch.sigmoid(self.W_beta(x)).view(B, T, -1)

        S = torch.zeros(B, self.d_head, self.d_head, device=x.device)
        outputs = []

        for t in range(T):
            qt, kt, vt = q[:, t], k[:, t], v[:, t]
            alpha_t = alpha[:, t]  # [B, D] — 每通道独立衰减！
            beta_t = beta[:, t]

            # 逐通道衰减（element-wise 乘法在 D 维度上）
            S = alpha_t.unsqueeze(-1) * S           # [B, D, D]，每行独立衰减

            # Delta 更新（与 GDN 相同）
            v_old = torch.einsum('bd,bdv->bv', kt, S)
            u = beta_t * (vt - v_old)
            S = S + torch.einsum('bd,bv->bdv', kt, u)

            o = torch.einsum('bd,bdv->bv', qt, S)
            outputs.append(o)

        return torch.stack(outputs, dim=1)
```

逐通道门控不是简单的参数量增加——它有精确的数学作用：让不同通道学会不同的衰减时间尺度。高频通道快速遗忘旧 token 的局部语法细节，低频通道保留跨句子的实体和主题信息。

> **注**：逐通道衰减本质上是一组可学习的指数移动平均（EMA）滤波器。每个通道的 α_d 决定了该维度的有效记忆长度：记忆半衰期 = log(0.5) / log(α_d)。当 α_d → 1 时，该通道成为事实上的「持久记忆槽」。

### 7.3 Kimi Linear 的核心主张

Kimi Linear 论文在受控对比下（相同参数量、相同训练数据）展示：其混合架构的表现**超过了全注意力 Transformer**。作者将其定位为"drop-in architectural replacement"——可替换标准注意力的即插即用组件。

Kimi Linear 的解码吞吐比全注意力最高提升 6 倍——这个提升来自用 O(D²) 的固定大小状态操作替代 O(T·D) 的 KV Cache 读取。

---

## 八、Kimi K3：工业化混合架构

Kimi K3 在 Kimi Linear 的基础上做了规模化升级。其核心架构是一个**23 次循环的宏结构**：

```
每个宏循环（共 23 次）：
  ├── Layer_i+0: KDA + 稠密 FFN + SiTU 激活
  ├── Layer_i+1: KDA + LatentMoE + SiTU 激活
  ├── Layer_i+2: KDA + LatentMoE + SiTU 激活
  └── Layer_i+3: MLA + LatentMoE + SiTU 激活  ← Gated MLA + 周期性 softmax 检索

每 3 个宏循环（12 层）做一次 AttnRes
```

设计哲学："KDA 提供恒定状态循环记忆，周期性 MLA 层保留对完整上下文的 softmax 检索。"

### 8.1 Gated MLA

在 Kimi K3 中，MLA 层不再是一个标准的 softmax 注意力——它在 MLA 的输出上增加了一个**门控向量**，对检索到的每个特征做逐元素乘法，控制多少特征流入残差流：

```python
class GatedMLA(nn.Module):
    """Gated Multi-head Latent Attention：Kimi K3 的 MLA 变体。

    标准 MLA 做 softmax 检索并将结果写入残差流。
    Gated MLA 增加了一个从输入学习的门控向量，控制检索到的特征中
    哪些应该进入残差流，哪些应该被抑制。
    """
    def __init__(self, d_model=512, d_latent=128, n_head=8):
        super().__init__()
        self.d_model = d_model
        self.d_latent = d_latent
        self.n_head = n_head

        # MLA 的 latent 投影
        self.W_dkv = nn.Linear(d_model, d_latent, bias=False)   # 压缩到 latent 空间
        self.W_uk = nn.Linear(d_latent, d_model, bias=False)    # 上投影到 K
        self.W_uv = nn.Linear(d_latent, d_model, bias=False)    # 上投影到 V
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # Gated MLA 特有：门控向量
        self.W_gate = nn.Linear(d_model, d_model, bias=False)   # 门控投影

    def forward(self, x, kv_cache=None):
        B, T, C = x.shape

        # 标准 MLA：压缩 KV 到 latent 空间，再上投影
        c_kv = self.W_dkv(x)                             # [B, T, d_latent]
        k = self.W_uk(c_kv)                               # [B, T, C]
        v = self.W_uv(c_kv)                               # [B, T, C]
        q = self.W_q(x)

        # 多头拆分 + softmax 注意力
        q = q.view(B, T, self.n_head, -1).transpose(1, 2)
        k = k.view(B, T, self.n_head, -1).transpose(1, 2)
        v = v.view(B, T, self.n_head, -1).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        att = att.masked_fill(causal_mask, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)

        # 门控：控制多少 MLA 检索结果进入残差流
        gate = torch.sigmoid(self.W_gate(x))              # [B, T, C]
        y = gate * y

        return self.W_o(y)
```

> **注**：NoPE（不使用位置编码）是 Kimi K3 MLA 的重要设计选择。标准 RoPE 需要在 Q、K 投影后施加旋转变换，破坏了 MLA 的 latent 压缩优势——如果 K 需要携带位置信息，latent 空间就必须保留足够维度来编码位置。K3 的 Gated MLA 完全舍弃 RoPE，依赖门控机制弥补无位置编码带来的上下文理解损失。这与 DeepSeek-V4 的 CSA/HCA 中保留 RoPE 的路线形成对比。

### 8.2 LatentMoE：在压缩空间中路由

Kimi K3 共 898 个专家。其中 2 个是**共享专家**（每个 token 都经过它们），其余 896 个通过可学习的路由器为每个 token 选择 16 个激活。

关键创新是**潜在空间专家**：专家运算发生在压缩的潜在空间中，前向传播速度更快，FLOP 几乎减半。

```python
class LatentMoE(nn.Module):
    """Kimi K3 的潜在空间 MoE。

    传统 MoE：W_gate→专家→输出（在 d_model 维度上直接运算）
    LatentMoE：下投影→专家→上投影（在压缩空间中运算，FLOP 约减半）

    898 个专家 = 2 个共享专家 + 896 个路由专家（选 16 个）
    """
    def __init__(self, d_model=512, d_latent=256, n_routed_experts=896,
                 n_shared_experts=2, top_k=16):
        super().__init__()
        self.d_model = d_model
        self.d_latent = d_latent
        self.top_k = top_k

        # 路由器：将 token 分配到专家
        self.router = nn.Linear(d_model, n_routed_experts, bias=False)

        # 共享专家（每个 token 都过）
        self.shared_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_latent, bias=False),   # 下投影到潜在空间
                nn.SiLU(),
                nn.Linear(d_latent, d_model, bias=False),   # 上投影回原空间
            ) for _ in range(n_shared_experts)
        ])

    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.view(-1, C)  # [B*T, C]

        # 路由
        router_logits = self.router(x_flat)                 # [B*T, n_experts]
        routing_weights, selected_experts = torch.topk(router_logits, self.top_k, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=-1)

        # 共享专家（简化示意——实际实现涉及高效的 sparse matmul）
        shared_out = sum(expert(x_flat) for expert in self.shared_experts)

        # 路由专家（简化示意：顺序执行，生产中会做 batched sparse matmul）
        routed_out = torch.zeros_like(x_flat)
        for i in range(self.top_k):
            expert_idx = selected_experts[:, i]             # [B*T]
            weight = routing_weights[:, i].unsqueeze(-1)    # [B*T, 1]

            # 示意：实际需要按专家分组做 batched matmul
            expert = self._get_expert(expert_idx)
            routed_out += weight * expert(x_flat)

        return (shared_out + routed_out).view(B, T, C)

    def _get_expert(self, idx):
        """示意：根据 expert_id 获取对应的专家权重。

        生产环境中，896 个路由专家的权重按 expert_id 组织为参数矩阵，
        通过 sparse matmul（如 Megablocks 的 block-sparse 乘法）高效计算，
        避免显式循环 16 次。此处简化为概念示意。
        """
        pass
```

> **注**：LatentMoE 的 latent 维度约为 d_model 的一半，因此每个专家的计算量约等于原始 MoE 的 50%。898 选 16 的 top-k 设计在 MoE 中属于非常高的稀疏比——只有约 1.8% 的参数被激活。这提供了很大的容量空间，但也对路由器的负载均衡提出更高要求。

### 8.3 SiTU 激活：替代 SiLU

SiLU（Sigmoid Linear Unit，也叫 SwiSH）是 2023 年后 LLM 的标准激活函数。Kimi K3 引入 SiTU（Sigmoid-Tanh Unit）：

$$\text{SiTU}(x) = \beta \cdot \tanh\left(\frac{x}{\beta}\right) \cdot \sigma(x)$$

其中 β 是可学习参数。当 β → ∞ 时，tanh(x/β) → 0，SiTU 退化为 SiLU。当 β 较小时，tanh 项提供额外的非线性约束，使激活值有界。

```python
class SiTU(nn.Module):
    """Sigmoid-Tanh Unit：Kimi K3 使用的 SiLU 替代品。

    gate = x[:, :d].to(float32)    — 门控部分（前 d 维）
    up   = x[:, d:].to(float32)    — 上投影部分（后 d 维）

    situ_a = beta * tanh(gate / beta) * sigmoid(gate)
             ↑ 这部分替代了 SiLU 的 gate * sigmoid(gate)
    """
    def __init__(self, d_model=512, beta_init=1.0, linear_beta_init=None):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(beta_init))
        # 可选的线性项 β：对 up 投影也做 tanh 约束
        self.linear_beta = nn.Parameter(torch.tensor(linear_beta_init)) if linear_beta_init is not None else None

    def forward(self, x):
        # SiLU 风格的门控分离
        gate, up = x.chunk(2, dim=-1)

        # SiTU：tanh(gate/β) 替代了 SiLU 中的 gate 本身
        gate_fp32 = gate.to(torch.float32)
        up_fp32 = up.to(torch.float32)

        situ_a = self.beta * torch.tanh(gate_fp32 / self.beta) * torch.sigmoid(gate_fp32)
        if self.linear_beta is not None:
            up_fp32 = self.linear_beta * torch.tanh(up_fp32 / self.linear_beta)
        return (situ_a * up_fp32).to(x.dtype)
```

**工程代价**：没有融合内核时，SiTU 比原始 SiLU 路径慢近 3 倍。LatentMoE 中专家前向的加速部分抵消了这一成本——这体现了推理系统中的一个常见矛盾：更好的数学特性需要融合内核才能在工程上落地。

---

## 九、AttnRes：选择性深度残差访问

### 9.1 残差稀释问题

标准残差网络中，第 l 层的输入是所有前层输出的**等权求和**：

$$h_l = h_1 + \sum_{i=1}^{l-1} f_i(h_i)$$

这个设计的缺陷在于**缺乏选择性**：不同类型层收到相同的聚合状态，尽管它们可能需要不同的加权比例。更深的问题是——纯加性循环迫使后层学习越来越大的输出值，才能对累积残差产生可感知的影响，这可能破坏训练稳定性。

### 9.2 AttnRes：学习式中继

AttnRes（Attention Residual）的解决方案简洁而优雅：对求和中的每一项乘一个专门的权重，权重通过查询-键注意力学习：

$$h_l = \alpha_0 \cdot h_1 + \sum_{i=1}^{l-1} \alpha_i \cdot f_i(h_i)$$

其中每个 αᵢ 由**查询-键点积**计算：每层学习一个查询向量，键和值来自更早的残差流状态，分数归一化到和为 1。

```python
class AttnRes(nn.Module):
    """AttnRes：块级深度残差注意力。

    问题：标准残差流中，第 l 层的输入是所有前层输出的等权求和。
         后层无法对早期层做选择性访问。

    方案：对残差项的求和用学习到的注意力权重替代等权求和。
         每层学习一个查询向量，键和值来自块级残差表征。

    在 Kimi K3 中，AttnRes 以块粒度应用（12 层一个块），
    平衡了「更多选择性」与「更低开销」。
    """
    def __init__(self, d_model=512, n_blocks=8):
        super().__init__()
        self.d_model = d_model
        self.n_blocks = n_blocks

        # 每个块的可学习查询（用于计算对历史块的注意力权重）
        self.proj = nn.Linear(d_model, 1, bias=False)  # 将 [B, T, D] 映射到标量分数
        self.norm = nn.LayerNorm(d_model)

    def forward(self, blocks, partial_block):
        """
        Args:
            blocks: List[Tensor]，每个 [B, T, D] — 历史块的残差表征（已完成）
            partial_block: [B, T, D] — 当前正在构建的块（尚未完成）

        Returns:
            h: [B, T, D] — 加权求和后的残差，作为当前块的输入

        Kimi K3 中的实际用法：
            V = torch.stack(blocks + [partial_block])              # [N+1, B, T, D]
            K = norm(V)                                            # 键 = 值（自注意力风格）
            logits = einsum('d, n b t d -> n b t', proj.weight.squeeze(), K)  # 查询-键点积
            h = einsum('n b t, n b t d -> b t d', logits.softmax(0), V)       # 加权求和
        """
        # 将所有块表征堆叠在一起（包括当前的半成品块）
        all_blocks = blocks + [partial_block]             # List of [B, T, D]
        V = torch.stack(all_blocks, dim=0)                # [N+1, B, T, D]

        # 键 = 归一化后的值（自注意力风格：查询来自当前上下文，键来自历史表征）
        K = self.norm(V)                                  # [N+1, B, T, D]

        # 查询-键点积：每个块得到一个标量分数
        weight = self.proj.weight.squeeze()               # [D]
        logits = torch.einsum('d, n b t d -> n b t', weight, K)  # [N+1, B, T]

        # Softmax 归一化（在块维度上）
        alpha = F.softmax(logits, dim=0)                  # [N+1, B, T]

        # 加权求和
        h = torch.einsum('n b t, n b t d -> b t d', alpha, V)  # [B, T, D]
        return h
```

### 9.3 AttnRes 与 MLA 的分工

Ali Taha 给出了一个精辟的总结：

> "AttnRes and MLA address the same underlying limitation from different directions. KDA layers operate with constant-size state and must inevitably discard information. MLA retrieves from the token context, while AttnRes retrieves from earlier depth-wise representations."

KDA 的恒定大小状态**不可避免会丢失信息**。MLA 从 **token 维度**（横向——完整上下文）检索被丢弃的信息，AttnRes 从**深度维度**（纵向——早期层表征）回收被残差稀释的信号。两者互补。

### 9.4 工程权衡

- **推理开销**：约增加 2% 的推理延迟
- **收益**：1.25 倍计算效率提升（推理速度更快）、缓解残差稀释与隐藏状态增长
- **粒度选择**：每 12 层做一次（而非每层），"在块边界做固定间隔的 AttnRes，以更低的成本捕获大部分收益"

> **注**：AttnRes 在 vLLM 中通过 Triton kernel 实现（`vllm/models/kimi_k3/nvidia/ops/attn_res.py`），使用 `torch.einsum` 和向量化操作来高效计算。12 层一次的块级策略是工程上精妙的「足够好」折中——每层都做 AttnRes 训练和推理成本太高，但块级已经在很大程度上恢复了选择性访问能力。

---

## 十、总结：什么变了？

从 GPT-2 到 Kimi K3，注意力机制的演进不是简单的规模放大，而是每一次架构迭代都针对前一代的具体缺陷：

| 阶段 | 解决了什么 | 新引入的代价 |
|---|---|---|
| **GPT-2** (2019) | — 基础 softmax 注意力 | 自回归解码 O(N²) 计算，每一步重算所有前置 token |
| **KV Cache** | 已生成 token 的 K/V 不需要重算 | Cache 随序列长度线性增长，解码受限于内存带宽 |
| **线性注意力** (2020) | 用固定大小状态矩阵替代不断增长的 cache | ELU+1 近似 softmax，表达力下降 |
| **DeltaNet** | Delta 规则解决加性更新的信息干扰 | 只能替换不能衰减，prefill 需要分块并行化 |
| **Gated DeltaNet** | 合并门控衰减 + Delta 修正 | 标量门控太粗粒度 |
| **KDA / Kimi Linear** | 逐通道门控 + MLA 混合 + MoE | 混合架构的复杂性 |
| **Kimi K3** | 工业规模混合 + Gated MLA + LatentMoE + SiTU | 新激活函数需要融合内核，架构极其复杂 |
| **AttnRes** | 选择性深度残差访问，解耦横向与纵向信息检索 | +2% 推理延迟 |

根本洞见可以归结为一句话：**固定容量的联想记忆需要逐出策略**。纯加性线性操作一旦达到容量就会引入干扰——这就是为什么每一代架构都在增加「选择性」：门控是选择性的写入衰减，Delta 规则是选择性的值替换，MLA 是选择性的 token 检索，AttnRes 是选择性的深度检索。

而**注意力仍然是迄今为止最有效的选择性读取机制**——这正是 Kimi K3 即使在 KDA 为主力的架构中，依然保留周期性 MLA 层和 AttnRes 的原因。每一分额外的容量，都花在了具有特定功能作用的地方。

---

> **参考与延伸阅读**
>
> - 原文：[22,580: GPT-2 to Kimi K3, explained](https://www.baseten.co/blog/22580-gpt-2-to-kimi-k3-explained/) — Ali Taha, Baseten (2026.07.30)
> - 源码验证：[post-kv-cache-era-challenges.md](../../../09_inference_system/post-kv-cache-era-challenges.md) — 39 处对照 vLLM/SGLang 源码的机制验证
> - 架构主线：[LLM 架构演进史](llm_architecture_evolution.md) — GPT-1 到 DeepSeek-V3 的七个拐点
> - KV Cache：[KV Cache 技术体系](../../../09_inference_system/kv_cache/README.md) — 42 篇文章，从原理到分布式管理
> - 基础概念：[Transformer 架构详解](../transformer/transformer_architecture.md) — 从自注意力到完整 Decoder Block

