# DeepSeek-V3 端到端推理 Pipeline 走读：一个 Token 的 61 层旅程

> 2026-07-29 | 基于 vLLM v0.27.0+ `deepseek_v2.py` (1933行) 与 SGLang v0.5.14 `deepseek_v2.py` (3010行) 源码分析

DeepSeek-V3 的推理链路涉及四组技术的交叉：MLA 压缩 KV cache、MoE 稀疏激活、MTP 多 token 预测、FP8 低精度量化。现有文章分别拆解了各组件——MLA 如何通过低秩压缩将 KV cache 缩小到标准 MHA 的 ~1.8%、MoE 如何通过 256 选 8 的路由将 671B 参数的激活量控制在 37B、MTP 如何通过 self-speculation 让一个 forward 处理 2 个 token——但缺少一篇"串起来"的文章。

**读者知道各部件如何工作，却不知道一个 token 从 embedding 进入、从 lm_head 出来的 61 层旅程中，这些部件以什么顺序被调用、之间传递什么形状的 tensor、数据依赖关系是怎样的。**

这就是本文的目的：以 DeepSeek-V3 的标准配置为舞台，以 **一个 decode token** 为主线，跟踪它从输入到输出的完整前向传播。每个阶段回答三个问题：做什么、为什么这样设计、代码如何实现。沿路已有专文深入的部分（MLA 原理、MTP 机制、EP 切分策略）只做"简要回顾 + 链接"，不重复展开。

> **为什么只追踪 decode token？** 因为 decode 是推理的核心瓶颈——GPU 利用率低、latency 敏感、KV cache 持续增长。Prefill 是 decode 的并行化版本：输入形状从 `(1, hidden)` 变成 `(N, hidden)`，Q/K/V 的序列维度同步扩展，但阶段的调用顺序完全相同。文章以 decode 为主线，在关键分叉处标注 prefill 的差异。

---

## 一、舞台设定：DeepSeek-V3 的模型拓扑

在进入逐层走读之前，先建立空间坐标系。单个 decode token 穿越的全部路径是固定的：

```text
token_id: (1,)
  │
  ▼
Embedding: VocabParallelEmbedding        → hidden_states: (1, 7168)
  │
  ▼
┌─ Layer 0 ─────────────────────────────┐
│  input_layernorm  →  MLA Attention    │  ← dense MLP（前 3 层）
│  post_attention_layernorm  →  MLP     │
└───────────────────────────────────────┘
  │
  ▼
   ... Layer 1, Layer 2 (dense MLP) ...
  │
  ▼
┌─ Layer 3 ─────────────────────────────┐
│  input_layernorm  →  MLA Attention    │  ← MoE（后 58 层）
│  post_attention_layernorm  →  MoE     │
└───────────────────────────────────────┘
  │
  ▼
   ... Layers 4–60 (MoE) ...
  │
  ▼
Final RMSNorm                              → hidden_states: (1, 7168)
  │
  ▼
LM Head: ParallelLMHead                   → logits: (1, 129280)
  │
  ▼
Sampler: temperature + top-p + top-k     → next_token_id: (1,)
  │
  ▼
下一轮 decode 的输入
```

关键配置（来自 `transformers` 的 `DeepseekV3Config`）：

| 参数                       | 值    | 含义                                       |
| -------------------------- | ----- | ------------------------------------------ |
| `hidden_size`              | 7168  | 隐藏维度                                   |
| `num_hidden_layers`        | 61    | 总层数                                     |
| `num_attention_heads`      | 128   | 注意力头数                                 |
| `q_lora_rank`              | 1536  | Q 的低秩投影维度                           |
| `kv_lora_rank`             | 512   | KV 的低秩投影维度（压缩瓶颈）              |
| `qk_nope_head_dim`         | 128   | 每头 non-RoPE 维度                         |
| `qk_rope_head_dim`         | 64    | 每头 RoPE 维度                             |
| `v_head_dim`               | 128   | 每头 V 维度                                |
| `first_k_dense_replace`    | 3     | 前 3 层用 dense MLP，之后用 MoE            |
| `n_routed_experts`         | 256   | 路由专家总数                               |
| `n_shared_experts`         | 1     | 共享专家数                                 |
| `num_experts_per_tok`      | 8     | 每个 token 激活的专家数                    |
| `n_group`                  | 8     | 专家分组数（先选组再选专家）               |
| `topk_group`               | 4     | 每 token 选的组数                          |
| `routed_scaling_factor`    | 2.5   | 路由权重缩放因子（也是 FP16 溢出修复系数） |
| `moe_intermediate_size`    | 2048  | 每个专家的 FFN 中间维度                    |
| `intermediate_size`        | 18432 | Dense MLP 的 FFN 中间维度                  |
| `num_nextn_predict_layers` | 1     | MTP 头数量                                 |

> **两个框架，同一套逻辑。** vLLM 的 `DeepseekV2ForCausalLM.forward()`（`deepseek_v2.py:1870`）和 SGLang 的 `DeepseekV2ForCausalLM.forward()`（`deepseek_v2.py:2842`）遵循完全相同的调用链。本文主要引用 vLLM 源码（行号精确），SGLang 源码仅在实现有显著差异时标注。

---

## 二、第一阶段：Embedding — 从 token ID 到 7168 维向量

一个 decode token 是上一步采样产生的整数——比如 `token_id = 15234`。

把 token ID 想象成图书馆的索书号。每个索书号不是一个孤立的数字——它指向书本在书架上的物理位置。Embedding 层就是这本"书架地图"：129280 行的巨大对照表，每一行是一个 7168 维的向量。这 7168 个数字不直接描述词的含义，但它们构成了一套坐标——意义相近的词在这个空间中位置相近，"国王"和"女王"的向量内积远大于"国王"和"桌子"。后续的 61 层 transformer，就是在这套坐标上不断做线性变换和非线性激活，逐层提炼语义。

### 2.1 VocabParallelEmbedding

`DeepseekV2Model` 的第一行有效代码是 embedding 查表（`deepseek_v2.py:1365-1371`）：

```python
# deepseek_v2.py:1404
def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
    return self.embed_tokens(input_ids)
```

`VocabParallelEmbedding` 在 TP>1 时会按词表维度切分。129280 词表在 TP=8 下，每个 rank 持有 129280/8 = 16160 行的 embedding 子矩阵。查表后得到 `(1, 7168/tp)` = `(1, 896)` 的部分向量——需要通过一次 all-reduce 才能还原完整的 `(1, 7168)` 向量。这个 all-reduce 是 TP 在 embedding 层的唯一通信开销，在 decode 阶段（单 token）的量级可忽略。

### 2.2 进入第一层

`hidden_states: (1, 7168)` 传入 `DeepseekV2DecoderLayer.forward()`（`deepseek_v2.py:1266`）。第一层的 `residual=None`，走的是非融合的 RMSNorm 路径：

```python
# deepseek_v2.py:1281-1283
if residual is None:
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
```

这是第一层独有的路径——建立了 `residual = 原始 hidden_states` 的初始值。从第二层开始，`residual` 不为 None，走 fused 路径：

```python
# deepseek_v2.py:1285
hidden_states, residual = self.input_layernorm(hidden_states, residual)
```

一次 kernel 完成 normalization 和 residual add——减少两次 GPU kernel launch 的开销。

---

## 三、第二阶段：MLA Attention — 7168 维的压缩与展开

MLA 的原理已有专文（[MLA 的 TP 切分：为什么 8 张 GPU 存了同一份 KV cache](mla_tp_kv_redundancy.md)），这里聚焦于**一个 token 在 attention forward 中的实时数据流**。

MLA 的思路可以用"写摘要"来理解。假设你每读一个词，要写一份信息卡片供后续查阅——标准 MHA 的做法是把词的完整信息（每个头 192 维的检索标签 K + 128 维的内容摘要 V，共 ~80 KB）一字不漏地抄下来。MLA 的发现是：每份卡片的信息冗余度很高，其实可以用极短的笔记（512 维的压缩状态 kv_c）概括全部内容，再配合一个不能省略的位置戳（64 维 k_rope，类似"第 37 页第 2 段"这样的坐标）。需要查阅时，把压缩笔记展开回完整的标签和摘要——恢复质量几乎无损耗，但卡片盒的厚度减少了 98%。

### 3.1 fused_qkv_a_proj：联合低秩投影

DeepSeek-V3 的 Q 和 KV 共享同一个输入，各自需要先做低秩投影。与其发两次 GEMM kernel，不如合并为一次。这就是 `DeepSeekV2FusedQkvAProjLinear`（`deepseek_v2.py:906`）的动机——**两个投影共享输入，合并为一次矩阵乘法**：

```python
# deepseek_v2.py:1004-1009
self.fused_qkv_a_proj = DeepSeekV2FusedQkvAProjLinear(
    proj_input_size,                                          # 7168
    [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
    # = [1536, 512 + 64] = [1536, 576]
)
```

输入 `(1, 7168)`，输出 `(1, 1536 + 576) = (1, 2112)`。前半段 1536 维是 Q 的低秩表示，后半段 576 维中前 512 维是 kv_c（KV 的压缩表示）、后 64 维是 k_rope（RoPE 的 key 分量，不压缩）。

> **为什么 k_rope 不能被压缩？** RoPE 是位置编码的旋转变换，要求 key 的维度与 query 的 rope 部分维度匹配（都是 64 维/head × 128 heads）。低秩压缩会破坏旋转不变性。这就是 MLA 将 KV 拆为"可压缩的 kv_c（512 维）"和"不可压缩的 k_rope（64 维）"两部分的原因。

### 3.2 Q 的两步展开

```text
fused_qkv_a_proj 输出
  ├── q_latent: (1, 1536)  ──→ q_a_layernorm ──→ q_b_proj ──→ q_nope: (1, 128×128)
  │                                                                   +
  │                            q_rope: (1, 128×64) ← 从输入 hidden_states 单独投影
  │                                                                   │
  │                                                                   ▼
  │                                                             Q: (1, 128 heads, 128+64=192)
  └── (kv_latent + k_rope, 见 3.3)
```

`q_b_proj`（`deepseek_v2.py:1021-1027`）是 `ColumnParallelLinear`——输入 1536 维，输出 `num_heads × qk_head_dim = 128 × 192 = 24576` 维。在 TP=8 时每个 rank 持有 `1536 → 24576/8 = 3072` 的部分权重，输出需要 all-gather 才能得到完整 Q。

### 3.3 KV 的压缩路径

```text
fused_qkv_a_proj 输出
  ├── (q 部分，见 3.2)
  └── kv_latent: (1, 512)  ──→ kv_a_layernorm ──→ kv_b_proj ──→ k_nope: (1, 128*128)
                            │                                     v:      (1, 128*128)
                            │
                            └── k_rope: (1, 64) ──────────────────────────────┘
                                                                               │
                                                                               ▼
                                                                         K: (1, 128, 128+64=192)
                                                                         V: (1, 128, 128)
```

`kv_b_proj` 的输出是 `num_heads * (qk_nope_head_dim + v_head_dim) = 128 * (128 + 128) = 32768` 维——同时产生 K 的 nope 部分和完整的 V。这又是一次"合并投影"：k_nope 和 v 共享同一个低秩输入 `kv_c`，在同一层 `kv_b_proj` 中展开，省掉一次独立的 V 投影。

### 3.4 Attention 计算与 KV Cache 写入

MLA 的 attention 计算本身与标准 MHA **完全一致**：

```text
Q: (1, 128, 192)           ←── 128 heads, 192 维 (128 nope + 64 rope)
K: (seq_len, 128, 192)     ←── 所有历史 token 的 key（含当前 token 刚写入的）
V: (seq_len, 128, 128)     ←── 所有历史 token 的 value
score = softmax(QK^T / √192)      → (1, 128, 1, seq_len)
output = score × V                 → (1, 128, 1, 128)
attn_out = output.reshape(1, 16384) → 经 O 投影回到 7168
```

关键差异不在计算，而在**写入 KV cache 的内容**——不是完整的 K `(128 heads × 192 dim)` 和 V `(128 heads × 128 dim)`，而是压缩后的 `kv_c: (1, 512)` 和 `k_rope: (1, 64)`。这两个加起来 **576 维**，而标准 MHA 的 KV cache per token 是 `128 heads × (192 + 128) dim × 2 bytes = ~80 KB`。MLA 将其压缩到 `(512 + 64) × 2 bytes = ~1.15 KB`——不到标准 MHA 的 1.5%。

> **Prefill 的差异**：输入形状从 `(1, 7168)` 变为 `(N, 7168)`，所有投影的 batch 维度同步扩展。KV cache 写入从"追加 1 个 token"变为"追加 N 个 token"。Attention 的 QK^T 从 `(1, seq_len)` 变为 `(N, seq_len)` 的矩阵——这是 prefill 阶段计算密集的核心（GEMM）vs decode 阶段 memory-bound 的核心（GEMV）的分界线。

---

## 四、第三阶段：从 Attention 到 FFN — 两个容易被忽略的细节

Attention 和 FFN 之间有两次"交接"——一次数值修正和一次 kernel 融合——虽然各占不到 1% 的延迟，但 FP16 溢出修复若缺失会导致 NaN，fused RMSNorm 若不用则每层白白多两次 kernel launch。对 61 层而言，这些"容易忽略"的细节累积起来决定模型能否稳定运行。

### 4.1 FP16 溢出修复

FP16 能表示的最大值是 65504。DeepSeek-V3 的 MoE 层在路由后会除以 `routed_scaling_factor=2.5` 来归一化路由权重——但 attention 输出不受这个缩放因子保护。如果 attention 输出本身数值偏大，与 MoE 的 `2.5` 倍路由权重相乘，结果可能超过 65504，变成 NaN。解决方案很简单：attention 输出提前乘以 `1/2.5`，让两个缩放互相抵消——就像出门前先脱掉外套，进屋后自然不嫌热。

MLA attention 输出之后、进入 `post_attention_layernorm` 之前，有一段特殊的数值处理（`deepseek_v2.py:1296-1307`）：

```python
if (
    not isinstance(self.self_attn, DeepseekAttention)
    and hidden_states.dtype == torch.float16
):
    # Fix FP16 overflow
    hidden_states *= 1.0 / self.routed_scaling_factor
    if self.layer_idx == 0:
        residual *= 1.0 / self.routed_scaling_factor
```

关键在于 **residual 的处理**。因为 residual 在所有层之间共享且持续累加，只需要在第 0 层对它做一次缩放（`routed_scaling_factor=2.5`），后续层只缩放当层的 hidden_states——residual 的量级从第 0 层起就稳定下来了。如果第 1 层之后还继续缩放 residual，等于反复除以 2.5，残差信号会被逐渐冲淡。

### 4.2 residual 的 Fused RMSNorm

从第二层开始，`input_layernorm` 和 `post_attention_layernorm` 都是 fused 版本。以 `post_attention_layernorm` 为例（`deepseek_v2.py:1320`）：

```python
hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
```

这一步等价于：`residual = residual + hidden_states; hidden_states = rms_norm(residual); return hidden_states, residual`。融合为一个 kernel 后，避免了中间 residual sum 写入 HBM 再读回的往返——residual 和 hidden_states 各读一次，norm 结果和更新后的 residual 直接写入，相比分开执行的两次 kernel 节省了一次 HBM 往返。

---

## 五、第四阶段：MoE FFN — 激活 37B 而非 671B 的关键

MoE 的思路是"专家会诊"。想象一家超大型医院，有 256 位专科医生（routed experts）。病人来了不需要所有医生都看——分诊台（gate）读一下症状描述（hidden_states），从 256 人中选出最对口的 8 位。这 8 位各自独立诊断（gate_proj → up_proj → SiLU → down_proj），给出加权意见（路由权重 × 诊断结果），汇总成最终方案。还有 1 位全科医生（shared expert）每人都必看——负责基本的体征检查，不依赖分诊台的决策。这样一来，医院虽然储备了 256 人的知识总量（对应 671B 参数），但每接待一个病人只动用了 9 人（激活 ~37B）。

### 5.1 为什么需要 MoE

DeepSeek-V3 有 671B 总参数。如果每一层都像 dense 模型那样激活全部参数，一个 forward 需要把 671B 权重都读一遍——decode 的 memory-bound 瓶颈会被放大到不可接受的程度。MoE 的解决方案：**每层只激活 256 个专家中的 8 个**（+ 1 个共享专家），激活参数约 37B，不到总参数的 6%。

### 5.2 前 3 层不路由：first_k_dense_replace

DeepSeek-V3 的前 3 层不使用 MoE，而是标准的 `DeepseekV2MLP`（dense FFN，`deepseek_v2.py:229`, `intermediate_size=18432`）。在 `DeepseekV2DecoderLayer.__init__()` 中判断（`deepseek_v2.py:1215-1218`）：

```python
is_moe_layer = (
    config.n_routed_experts is not None
    and layer_idx >= config.first_k_dense_replace  # ≥ 3
    and layer_idx % moe_layer_freq == 0            # 每层都是 MoE
)
```

如果 `is_moe_layer=False`，用 `DeepseekV2MLP`；否则用 `DeepseekV2MoE`。

**为什么前 3 层用 dense？** 浅层的 hidden states 尚未充分分化语义特征。如果第 0 层的 token 就开始路由到不同的专家，路由信号缺乏足够的语义信息来支撑 256 选 8 的决策。前 3 层 dense 给 token 提供了一致的早期特征提取，第 4 层开始 token 的表示已经足够丰富，可以可靠地路由到不同的专家。

### 5.3 门控路由：从 7168 维到 8 个专家 ID

从第 4 层起，`DeepseekV2MoE.forward()`（`deepseek_v2.py:398`）成为 FFN 的计算主体。路由分两步：

**Step 1：Gate 投影**。`self.gate(hidden_states)` 输出 `(1, 256)` 的 logits——256 个专家，每个一个分数。

```python
# deepseek_v2.py:416-419
router_logits, _ = self.gate(hidden_states)
final_hidden_states = self.experts(hidden_states=hidden_states, router_logits=router_logits)
```

**Step 2：分组路由 + top-k**。`FusedMoE` 内部的路由逻辑（由 `FusedMoE` 的 `top_k` 和 `topk_group` 参数控制）：

```text
256 个专家 → 分为 8 组（n_group=8），每组 32 个专家
  → 对每组计算组内专家 logits 之和，选 top-4 组（topk_group=4）
    → 在选中的 4 组内（共 128 个专家），选 logits 最高的 8 个（num_experts_per_tok=8）
      → 返回 8 个专家索引 + softmax(logits[选中的8个]) 作为路由权重
```

分组路由的设计动机：256 个专家全量计算 logits 后直接 top-8 选出的专家可能集中在少数几组。先选组再选专家的两级路由迫使 token 分散到不同组，提升专家利用率——这是一种间接的负载均衡策略。

### 5.4 Shared Expert：每个 token 必定激活

除了 8 个 routed experts，每个 token 还必定经过 1 个 shared expert（`n_shared_experts=1`）。shared expert 是一个独立的小型 MLP（`moe_intermediate_size=2048`），其输出**不受路由权重影响**，直接加到最终输出上：

```text
final = sum(weight_i × expert_i(hidden)) + shared_expert(hidden)
```

shared expert 提供所有 token 共享的"通识"能力（语法、常识），routed experts 提供"专识"（领域知识）。这种"共享 + 路由"的设计避免了将宝贵的专家容量浪费在所有 token 都需要的基础运算上。

### 5.5 专家计算与并行

选中的 8 个专家各自完成 `gate_proj → up_proj → activation(SiLU) → down_proj`。每个专家的中间维度是 2048（`moe_intermediate_size`），远小于 dense MLP 的 18432——所以激活 8 个专家 + 1 个共享专家的总计算量约等于 `8 × 2048 + 2048 ≈ 18432`，恰好是一个 dense MLP 的计算量。**MoE 的设计目标不是减少计算总量，而是在相同计算量下获得 256 倍于 dense 的参数容量**。

`FusedMoE` 使用 group GEMM 在一次 kernel 中完成多个 expert 的计算——将不同 token 按照路由到的 expert 分组，同一 expert 的 tokens 拼接为 batch，然后做 batched GEMM。

> **Prefill 的差异**：`(N, 7168)` 的 hidden_states 中，不同 token 可能路由到不同 experts。`FusedMoE` 的 `fused_moe` kernel 按 expert 分组 reorder tokens，group GEMM 并行计算。如果启用了 EP（Expert Parallelism），tokens 还会通过 all-to-all 发送到持有对应 experts 的 GPU 上——这是 DeepSeek-V3 在多节点部署中最昂贵的通信操作。

### 5.6 Sequence Parallel MoE

TP=8 时，可通过 `use_sequence_parallel_moe` 启用 SP MoE（`deepseek_v2.py:1310-1317`）：

```python
if self.use_sequence_parallel_moe:
    tp_world_size = get_tensor_model_parallel_world_size()
    sp_pad = (-hidden_states.shape[0]) % tp_world_size
    hidden_states = torch.nn.functional.pad(hidden_states, (0, 0, 0, sp_pad))
    hidden_states = tensor_model_parallel_reduce_scatter(hidden_states, 0)
```

reduce-scatter 将 hidden_states 按序列维度均匀切分到各 TP rank，各 rank 独立完成 MoE 计算。相比 EP，SP MoE 避免了 all-to-all 通信——代价是每个 rank 必须持有全部专家的权重副本（而非 EP 中每个 rank 只持有一部分专家）。对于单节点 8×GPU 且模型权重可全部装入的场景，SP MoE 是比 EP 更高效的替代方案。

---

## 六、第五阶段：LM Head — 从 7168 维到 129280 个 logits

### 6.1 Final RMSNorm

经过全部 61 层后，hidden_states 传入 final norm：

```python
# deepseek_v2.py:1384-1387
if get_pp_group().is_last_rank:
    self.norm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
```

最后一次 normalization 将 hidden_states 的数值范围稳定在 FP16 的安全区间。61 层中每层的 residual add 让残差流的绝对值持续增长——final norm 确保进入 lm_head 的值不会溢出。

### 6.2 ParallelLMHead 与 LogitsProcessor

```python
# deepseek_v2.py:1882-1887
def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
    logits = self.logits_processor(self.lm_head, hidden_states)
    return logits
```

`lm_head` 是 `(vocab_size, 7168)` 的线性层，输出 `(1, 129280)` 的 logits。TP>1 时，`ParallelLMHead` 按词表维度切分——每个 rank 输出 `(1, 129280/tp)` 的部分 logits，需要 all-gather 后才得到完整分布。

`LogitsProcessor` 负责将 logits 转换为概率并采样：temperature scaling → top-k filtering → top-p filtering → 应用 penalties → softmax → 采样。采样出的 `next_token_id` 成为下一轮 decode 的输入——循环开始。

> **为什么不跳过 §5 和 §6？** 这两个阶段在 decode 中的耗时占比虽小（各 ~1-2%），但 FMHA（Fused Multi-Head Attention）+ MoE + LM Head 三者的通信-计算交错模式决定了 TP/PP/EP 策略的选择。省略它们会丢失"为什么 DeepSeek-V3 的推理部署如此复杂"的完整图景。

---

## 七、第六阶段：MTP — 用一个 forward 预测多个 token

MTP 像是阅读时的"预判"。你读到"法国的首都是"——还没看到下一个词，脑子里已经自动浮现了"巴黎"。MTP 在主模型算完当前 token 之后、采样出新 token 之前，用几个中间层的"理解"快速预测接下来几个 token 可能是什么。这些预测不是最终输出——它们是草稿，下一轮主 forward 会并行验证。如果猜对了，一轮 forward 就推进了多个 token；猜错了也无妨，只丢弃草稿，恢复主路径。这就是 self-speculation：**模型既是 draft 的作者，也是 verifier**。

MTP（Multi-Token Prediction）的原理已有专文（[MTP 多 Token 预测](mtp-multi-token-prediction.md)），这里简化为它在 forward pipeline 中的位置。

### 7.1 MTP 在 call stack 中的位置

MTP forward 发生在**主模型 forward 之后、采样之前**。在 vLLM 中，MTP 通过 EAGLE3 框架集成——`DeepseekV2ForCausalLM` 继承了 `SupportsEagle3`（`deepseek_v2.py:1783`），调度器在每个主 forward 结束后检查是否需要运行 MTP 头。

### 7.2 aux_hidden_state_layers

MTP 头不读最终 hidden_states，而是从指定中间层获取 hidden states——每个 MTP 头有不同粒度的语义信息：

```python
# deepseek_v2.py:1863-1865
def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
    num_layers = len(self.model.layers)
    return (2, num_layers // 2, num_layers - 3)  # → (2, 30, 58)
```

- **Layer 2**：浅层特征——token 的语法和局部上下文信息
- **Layer 30**：中间层特征——token 的中等粒度语义
- **Layer 58**：深层特征——接近输出的高维语义，最接近最终预测

MTP 头（DeepSeek-V3 的 `num_nextn_predict_layers=1` 表示 1 个 MTP 模块）同时读取这三个中间层的 hidden states，融合不同深度的语义信息来预测后续 token——这比只读最终 hidden_states 更能捕捉序列的多层次特征。

### 7.3 MTP 在 decode 中的作用

MTP 不是每步都运行。在 speculative decoding 流程中：

```text
Step T:   主 forward → sampled token t_{T+1}
          → MTP forward → 一批 draft token 候选 d1...dk

Step T+1: 输入 [..., t_T, t_{T+1}, d1...dk]（draft tokens 拼接进输入）
          主 forward → 并行验证 d1...dk，同时产生 sampled token t_{T+2}
          → MTP forward → 新一批 draft tokens
```

每步主 forward 处理 1 个 token（无需 speculate 时），当 MTP 的 draft tokens 被接受时，一个 forward 可以验证 2 个 token——这就是 self-speculation 的加速来源。DeepSeek-V3 配置 `num_nextn_predict_layers=1`（1 个 MTP 头），在低熵场景下约能实现 1.5× 的吞吐提升。

---

## 八、完整时间线：一个 token 的延迟拆解

将八个阶段串联为一条完整的时间线。以下数据基于 H100 单卡 (TP=1) 的近似 profile（实际受 batch size、序列长度、EP 通信等因素影响）：

```text
一个 decode token 的完整旅程（~12ms，TP=1, H100）:

  Embedding:      ~0.1ms   ▌
  Layer 0 (dense): ~0.2ms   ▌▌
  Layer 1 (dense): ~0.2ms   ▌▌
  Layer 2 (dense): ~0.2ms   ▌▌
  Layer 3 (MoE):   ~0.18ms  ▌▌
  ... Layers 4-60: ~0.18ms × 57 ≈ 10.3ms   ▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌▌
  Final Norm:      ~0.02ms  ▏
  LM Head:         ~0.15ms  ▌
  Sampling:        ~0.1ms   ▌
  ─────────────────────────
  总计 ~11.6ms

延迟热点：
  - 61 层 forward 占 ~97%（其中 ~93% 在 MoE 层）
  - LM Head + Sampling 占 ~2%
  - Embedding 占 < 1%
```

每层的内部延迟分配（以 MoE 层为例）：

```text
  input_layernorm:   ~5μs    ▏
  MLA Attention:    ~40μs    ▌▌▌
    fused_qkv_a_proj: ~15μs
    q/kv_b_proj:      ~10μs
    attention:        ~10μs
    o_proj:            ~5μs
  post_attn_norm:    ~5μs    ▏
  MoE FFN:          ~130μs   ▌▌▌▌▌▌▌▌▌▌
    gate + routing:   ~10μs
    expert GEMM:     ~110μs  (8 experts × ~14μs)
    shared expert:     ~5μs
    combine:           ~5μs
  ─────────────────────────
  总计 ~180μs / layer
```

MoE FFN 占每层延迟的 ~72%——这是 DeepSeek-V3 decode 的最大性能瓶颈。任何 MoE 加速（FP8 GEMM、专家剪枝、更激进的 top-k 缩减）都比 attention 加速有更高的杠杆效应。

### 两个框架的关键差异

| 实现点        | vLLM                                         | SGLang                                                         |
| ------------- | -------------------------------------------- | -------------------------------------------------------------- |
| QKV 低秩投影  | `fused_qkv_a_proj`：一次 GEMM 合并 Q+KV 投影 | 分开投影：`q_a_proj` + `kv_a_proj_with_mqa`                    |
| MoE 路由      | `FusedMoE` 内部完成 gate + expert 计算       | `DeepseekV2MoE.forward()` 单独 gate，含 `biased_gate` 辅助平衡 |
| SP MoE        | `use_sequence_parallel_moe` 可选             | 默认不启用 SP MoE，优先使用 EP                                 |
| FP16 溢出修复 | 缩放 hidden_states，第 0 层特殊处理 residual | 类似，具体实现略有差异                                         |
| MTP 框架      | EAGLE3（`SupportsEagle3`）                   | 自研 MTP 框架                                                  |

这些差异不影响数学正确性——同一份权重在两个框架上产出相同的 token——但决定了相同模型在相同硬件上的吞吐和延迟。

---

## 九、总结

回顾一个 token 的 61 层旅程：

- **Embedding**：token ID → 7168 维向量，TP 下按词表切分
- **MLA Attention（每层）**：7168 → `fused_qkv_a_proj` → 低秩空间 1536(Q)+512(KV) → `q/kv_b_proj` 展开 → 标准 attention 计算 → KV cache 存入压缩后的 576 维（~1.15 KB），而非标准 MHA 的 ~80 KB
- **前 3 层 dense → 后 58 层 MoE**：浅层提供一致的早期特征，深层用 256 选 8 的稀疏激活在固定计算量下获得 256 倍参数容量
- **FP16 溢出修复**：attention 输出预除以 `routed_scaling_factor=2.5`，第 0 层特殊处理 residual
- **Shared Expert**：每个 token 必定激活，保证"通识"不被路由牺牲
- **Final Norm + LM Head**：最后一个 normalization + 7168 → 129280 投影 → 采样
- **MTP**：在主 forward 后、采样前运行，通过 speculative decoding 让一个 forward 处理 2 个 token

这套 pipeline 的复杂性来自一个基本矛盾：**671B 参数太大，一次 forward 装不进 GPU；1M 上下文太长，KV cache 存不下。** MLA 解决了 KV cache 问题（压缩到 ~1.8%），MoE 解决了参数规模问题（激活 37B/671B），FP8 进一步压缩权重的显存和带宽，MTP 在 decode 阶段提供额外的吞吐。四组技术各自独立设计，但在前向传播中以严格的顺序交织——这就是理解 DeepSeek-V3 推理需要"走读"而非"分读"的原因。

---

## 延伸阅读

- [vLLM `deepseek_v2.py` 源码](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/deepseek_v2.py)
- [SGLang `deepseek_v2.py` 源码](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/deepseek_v2.py)
- [DeepSeek-V4 端到端推理 Pipeline 走读](deepseek_v4_inference_pipeline.md) — V4 架构的根本性重构：mHC 多流残差、MQA+CSA/HCA、Hash-MoE
- [DeepSeek-V3 配置](https://huggingface.co/deepseek-ai/DeepSeek-V3-Base/blob/main/configuration_deepseek.py)
- [MLA 的 TP 切分：为什么 8 张 GPU 存了同一份 KV cache](mla_tp_kv_redundancy.md)
- [投机解码方法全景：六种草拟策略的工程选择](speculative_decoding_landscape.md)
- [MTP 多 Token 预测：训练、推理与 Self-Speculation](./mtp-multi-token-prediction.md)（`model_optimization/` 目录）
- [专家并行（EP）深度解析](../../parallelism/expert_parallelism_deep_dive.md)
