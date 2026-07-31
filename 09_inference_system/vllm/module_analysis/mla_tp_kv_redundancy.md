# MLA 的 TP 切分：为什么 8 张 GPU 存了同一份 KV cache

> DeepSeek V3 的 MLA 将 KV cache 压缩到标准 MHA 的 ~1.8%——从每 token 每层 32KB 压到 576 bytes。但当 TP=8 时，每个 GPU 的 KV cache 分配和单 GPU 部署完全一样——TP 没有为 KV cache 节省任何显存。这不是 bug——这是 MLA 的「跨 head 共享表示」撞上 TP 的「按 head 切分」时，二者在数学上根本不可调和的结果。

## 一、两个独立的好设计，合在一起就不对了

Tensor Parallelism（TP）的 KV cache 切分逻辑，自 vLLM 诞生那天就很清晰：K 和 V 各自按 head 维度切成 $N$ 份，每个 rank 拿一份，互不相干，零冗余。以 Llama-70B（GQA，8 个 KV head）TP=8 为例：

```text
rank 0: K 的 head 0, V 的 head 0
rank 1: K 的 head 1, V 的 head 1
...
rank 7: K 的 head 7, V 的 head 7
```

每个 head 的 K 和 V 只存在于一个 rank 上。这套逻辑在标准 MHA / GQA 上运行了三年，从未出过问题。

MLA（Multi-head Latent Attention）也是一个好设计。它不缓存显式的 K、V 矩阵，而是缓存一个低秩潜变量 + 解耦的 RoPE 分量。以 DeepSeek V3 为例：

|                          | 标准 MHA（128 个 KV head）               | MLA                   |
| ------------------------ | ---------------------------------------- | --------------------- |
| 每 token 每层存储（FP8） | $128 \times (128 + 128) = 32{,}768$ 元素 | $512 + 64 = 576$ 元素 |
| 每 token 每层字节        | ~32 KB                                   | ~576 bytes            |
| 等效压缩比               | —                                        | **~1.8%**             |

> 标准 MHA 基线按 `head_dim=128` 简化计算。若包含 RoPE 维度（DeepSeek V3 的 `qk_rope_head_dim=64`），K 的 head_dim 应为 192，基线为 $128 \times (192 + 128) = 40{,}960$，压缩比约 1.41%。两种口径不影响本文核心论点——MLA 的 576 维全部与 head 无关，TP 无法切分这一事实与基线取法无关。

`kv_lora_rank = 512`，`qk_rope_head_dim = 64`。关键是 `k_rope` 采用 MQA 风格——这 64 维是**所有 128 个 head 共用**的，不是每个 head 一份。整个 MLA 缓存只有 $512 + 64 = 576$ 维，全部与 attention head 无关。

问题是这两个好设计从未被放在一起审视过。MLA 设计出来是为了节省显存，TP 设计出来也是为了节省显存——一个压缩表示，一个切分表示。但当 MLA 的「所有 576 维都不归属任何单个 head」撞上 TP 的「按 head 切分」时，**没有 head 可以切。** 答案藏在 `kv_a_proj_with_mqa` 的实现里。

---

## 二、ReplicatedLinear：不是「不想切」，是「没法切」

### 2.1 标准 MHA 怎么切

标准 MHA 中，K 和 V 的投影权重分别是 column-parallel 线性层：

$$
W_K \in \mathbb{R}^{\mathrm{hidden size} \times (N_{\mathrm{kv heads}} \times D_{\text{head}})}
$$

TP=8 时使用列切分：每个 rank 拿总列数的 $1/8$，即 $\frac{N_{\mathrm{kv heads}}}{8}$ 个 head 对应列。每个 rank 算出的 K、V 只覆盖自己的 head，不相重叠。**切分后的所有权重列都是唯一的，没有一列出现在两个 rank 上。**

### 2.2 MLA 的投影：没有 head，也就不需要切

MLA 不再分别投影 K 和 V，而是通过 `kv_a_proj_with_mqa` 一步完成压缩投影。vLLM v0.20.0 的 DeepSeek V2/V3 模型代码揭示了真实实现：

```python
# vllm/model_executor/models/deepseek_v2.py
self.kv_a_proj_with_mqa = ReplicatedLinear(
    hidden_size,
    kv_lora_rank + qk_rope_head_dim,  # = 512 + 64 = 576
    bias=False,
    params_dtype=torch.float32,
)
```

三个事实同时成立：

1. **它是 `ReplicatedLinear`，不是 `ColumnParallelLinear`。** 权重矩阵形状为 `[hidden_size, 576]`——不存在「部分 replicated、部分 sharded」的列切分，全部 576 列在 8 个 rank 上持有完整副本。

2. **`k_rope` 是 MQA 风格共享的。** `qk_rope_head_dim = 64`，在公式里以 `+ qk_rope_head_dim`（而非 `+ num_heads × qk_rope_head_dim`）的形式出现。所有 128 个 attention head 共用这 64 维 RoPE 分量——它就像 `kv_c` 一样，是与 head 无关的共享表示。

3. **输出是 576 维，不是 $512 + 128 \times 64 = 8704$。** 不存在「per-head k_rope 8192 维」这一截。576 维中的每一维，对 128 个 head 平等地提供信息。

这三个事实指向同一个结论：**MLA 的整个 KV cache 表示——全部 576 维——从一开始就是跨 head 共享的。** TP 的列切分前提是「存在可以按 head 拆分的列」，而 MLA 的输出中没有这样的列。`ReplicatedLinear` 是唯一正确的选择。

### 2.3 如果强行用 ColumnParallelLinear 呢？

可以想象一个替代设计：将 576 维也切成 8 份（每 rank 72 维），解码时 all-gather 回完整的 576 维再做 attention。代价是每次 attention 多一次通信——每个 token 每层需要 all-gather 576 bytes（FP8）。以 batch=64 为例，单层单次 all-gather 的数据量 ~36 KB，走 NVLink 延迟大约几微秒。但 decode 阶段是逐 token 生成的——每个 decode step 都需要这 36 KB 的 all-gather，累积起来远超在显存中多放 576 bytes per token per layer 的空间开销。

vLLM 的选择不是设计偏好，是成本计算。而成本计算的结果是一个反直觉的事实：**TP=8 时，MLA 的 KV cache 在每个 rank 上都是完整的 576 维副本。** 8 个 TP rank 没有为 KV cache 节省任何显存。

---

## 三、576 维，全部是复制品

### 3.1 每个 rank 到底存了什么

vLLM 的 `MLAAttentionSpec` 确认了每个 rank 的存储规格：

```python
# vllm/model_executor/layers/attention/mla_attention.py
self.head_size = kv_lora_rank + qk_rope_head_dim  # = 576
self.num_kv_heads = 1

# get_kv_cache_spec() 返回:
MLAAttentionSpec(
    block_size=vllm_config.cache_config.block_size,
    num_kv_heads=1,           # K 和 V 融合为单一潜变量
    head_size=576,            # 即 512 (kv_c) + 64 (k_rope, MQA 共享)
)
```

LMCache v0.5.1 的格式检测确认了 3-D 存储：

```python
# lmcache/v1/gpu_connector/kv_format/detectors/vllm.py
if list_depth == 1 and tensor_ndim == 3:  # MLA
    return lmc_ops.EngineKVFormat.NL_X_NB_BS_HS, kv_caches

# lmcache/v1/kv_layer_groups.py
kv_size = 1 if mla else 2    # MLA: 1 个压缩张量
nh = 1 if mla else ...       # MLA: head 维度已压缩
```

每个 TP rank 的 GPU 显存上，每层的 KV cache 是：

```text
[NB, BS, 576]
  ↑
  全部 576 维在 8 个 rank 上完全相同
  (元素 0-511 = kv_c, 元素 512-575 = k_rope)
```

### 3.2 量化冗余

**单 rank 视角**：

$$
\frac{\text{有效独占}}{\text{总存储}} = \frac{0}{576} = 0\%
$$

GPU 0 上的 576 维，和 GPU 1 上的 576 维，**逐 bit 相同**——它们是同一批 hidden states 经过 `ReplicatedLinear` 的同一份权重算出来的。

**跨 rank 视角**：

8 个 rank 共分配：$8 \times 576 = 4{,}608$ 元素。去重后实际信息量：$576$ 元素。

$$
\text{冗余率} = \frac{4{,}608 - 576}{4{,}608} \approx 87.5\%
$$

对比标准 MHA（GQA, 8 个 KV head, TP=8）：冗余率 **0%**。TP 为 KV cache 节省的显存：

$$
\text{标准 MHA TP=8：节省 } 87.5\%，\quad \text{MLA TP=8：节省 } 0\%
$$

**MLA 抹去了 TP 对 KV cache 的显存收益。** TP 仍然为模型权重、激活值节省显存——但 KV cache 这部分，被 MLA 的跨 head 共享设计完全归零了。

### 3.3 这不是实现缺陷

这里没有 bug。`ReplicatedLinear` 是正确的选择，576 维复制是最优解。任何不复制这 576 维的方案都需要通信，而通信延迟远超存储成本。

这是 **MLA 的「跨 head 共享表示」与 TP 的「按 head 切分」之间的结构性摩擦**。它不取决于 TP 的尺寸、不取决于 vLLM 的版本、不取决于你用的是 FlashMLA 还是 Triton MLA backend——它取决于 MLA 缓存的所有维度都不按 head 分组这个**数学事实**。只要注意力需要完整的 `kv_c + k_rope` 来做计算，每个 TP rank 就需要一笔不差的 576 维。

---

## 四、LMCache 的 is_kv_writer：一个印证这个事实的工程选择

LMCache v0.5.1 在 `ParallelStrategy` 中，对 MLA 的 KV cache 保存逻辑做了特殊处理：

```python
# lmcache/integration/vllm/vllm_multi_process_adapter.py

@property
def is_kv_writer(self) -> bool:
    if not self.use_mla:
        return True                    # 非 MLA：每个 rank 都是 writer
    # MLA：只有每组的第一个 rank 是 writer
    return self.vllm_worker_id % (self.tp_size // self.n_servers) == 0
```

非 MLA 模型（如 Llama）：TP=8 时 8 个 rank 都是 writer——每个 rank 存的是不同 head 的 K 和 V，互补，缺一不可。

MLA 模型（如 DeepSeek V3）：TP=8 时只有 rank 0 是 writer——8 个 rank 的 KV cache **完全一样**，读 1 份和读 8 份没有任何区别。

```python
# lmcache/integration/vllm/lmcache_mp_connector.py
def wait_for_save(self):
    # In MLA scenario, only the first rank of the pipeline group
    # needs to save the KV cache.
    if not self.worker_adapter.is_kv_writer:
        return    # rank 1-7 直接跳过
```

8 个 TP rank，7 个在 `wait_for_save` 里直接 return。如果有工程师在代码 review 时看到这段，第一反应可能是「这会不会丢数据？」——不会。数据在 rank 0 上也有一份，一模一样。

这个实现揭示了一个比工程优化更根本的事实：**MLA 的 KV cache 中，「份数」这个概念本身没有物理意义。** 它的真实信息熵只有一份。出现在 8 个 rank 上，完全是 `ReplicatedLinear` 的一个副作用——就像你用 8 台相机拍同一块白板，物理上有 8 张照片，信息上只有一块白板。

---

## 五、这会影响你吗？

### 5.1 如果你在做显存规划

不要用压缩公式直接推算 per-GPU KV cache 分配量。$512 + 64 = 576$ 元素/token/层（576 bytes in FP8）既是信息量，也是**per-rank 分配量**——二者相等，因为 MLA 在 TP 下没有节省任何 KV cache 显存。

标准 MHA TP=8 的公式：$\frac{N_{\mathrm{kv heads}} \times D_{\text{head}} \times 2}{\mathrm{tp size}}$——有除以 tp_size。

MLA TP=8 的公式：$R_{\mathrm{kv lora}} + D_{\mathrm{qk rope}}$——**没有除以 tp_size。**

$$
\text{MLA per-rank KV (bytes per token per layer)} = kv lora rank + qk rope head dim = 576
$$

如果你部署 DeepSeek V3 在 8×GPU 上，每个 GPU 的 KV cache 预算和单 GPU 部署完全一样。TP 帮你分担了权重和激活值，但对 KV cache 无能为力。

### 5.2 如果你在做 KV cache offloading

确保你的 connector 没有把 8 份完全相同的 KV cache 都搬到远端存储。LMCache v0.5.1 的 `is_kv_writer` + rank-0-only 保存策略是一个经过验证的参考实现。这个策略的有效性依赖于一个前提——MLA 的 KV cache 在 TP 下是完完全全的复制，不是部分重叠。

### 5.3 如果你在理解 MLA 的 MLAAttentionSpec

| 字段           | 值  | 含义                                                                |
| -------------- | --- | ------------------------------------------------------------------- |
| `num_kv_heads` | 1   | 不是「1 个 head」——是「K 和 V 融合为单一潜变量，不按 head 区分」    |
| `head_size`    | 576 | `kv_lora_rank (512) + qk_rope_head_dim (64)`，不是 per-rank 的 1536 |
| `kv_size`      | 1   | LMCache 侧对应值：1 个压缩张量（vs 标准 MHA 的 2：K+V 分离）        |

不要用标准 MHA 的 `num_kv_heads × head_size × 2（K+V）/ tp_size` 公式去推算 MLA 的 KV cache 大小。MLA 不是一个更小的 MHA——它是一个与 「head」概念解耦的数据结构。

### 5.4 如果你跑的是标准 MHA 模型

不用担心。Llama、Qwen、GLM（非 MLA 变体）仍然遵循标准 TP 切分规则——按 head 切分 K 和 V，跨 rank 零冗余。本文讨论的现象仅出现在 vLLM 的 `is_deepseek_mla()` 判定返回 True 的模型上——目前覆盖 `deepseek_v2/v3/v32/v4`、`glm_moe_dsa`、`kimi_k2`、`longcat_flash`、`pangu_ultra_moe`、`bailing_hybrid` 等 MLA 家族模型。

---

## 六、DeepSeek V4：全部四组都落入同一个陷阱

V3 的 61 层全部是 MLA，每层 576 维，全部复制。分析 V3 时是一个简单的答案：「是的，全部复制。」

DeepSeek V4 引入了更复杂的注意力体系——四组不同压缩比的 KV cache：MLA 主缓存（c4a/c128a，4×/128×）、Indexer（c4a 层专用，条目 128 维）、SWA 滑动窗口（1×）、Compressor 状态（1×）。直觉上，四组的 TP 行为应该像第三章描述的标准 MHA 那样——分别判断、各不相同。但 vLLM v0.20.0 的代码给出了一个意外的答案。

### 6.1 四组全部是 MLA 变体

vLLM v0.20.0 中，V4 的四组 KV cache 使用的是 MLA 家族 spec，不是标准 MHA spec：

| KV cache 组           | Spec 类型                              | kv_size     | 存储形状              |
| --------------------- | -------------------------------------- | ----------- | --------------------- |
| MLA 主缓存（c4a）     | `MLAAttentionSpec(num_kv_heads=1)`     | 1（rank-3） | `(NB, BS, head_size)` |
| Indexer（c4a 层专用） | `MLAAttentionSpec(num_kv_heads=1)`     | 1（rank-3） | `(NB, BS, head_size)` |
| SWA 滑动窗口          | `SlidingWindowMLASpec(num_kv_heads=1)` | 1（rank-3） | `(NB, BS, head_size)` |
| Compressor 状态       | `SlidingWindowMLASpec(num_kv_heads=1)` | 1（rank-3） | `(NB, BS, head_size)` |

`SlidingWindowMLASpec` 在 vLLM v0.20.0 的 `vllm/v1/kv_cache_interface.py` 中定义。它与标准 `SlidingWindowSpec` 的关键区别在于 `real_page_size_bytes`——移除了标准 spec 中的 `2 ×` 因子（K+V 分离的乘数）。`CompressorBackend.get_kv_cache_shape()` 返回 `(num_blocks, block_size, head_size)`——明确是 rank-3，单向量。

LMCache v0.5.1 的 `kv_layer_groups.py` 用假设性例子描述了 `engine_kv_format` 字段的作用——"a rank-5 K/V group alongside a rank-3 key-only indexer cache"——但这描述的是该字段设计意图中的**一般场景**（同一 engine group 内混合不同格式），并非特指 V4。V4 的四组在 vLLM v0.20.0 中全部使用 rank-3 MLA 变体格式。

### 6.2 冗余表：V3 vs V4

|              | V3（全 MLA）         | V4 MLA 主缓存          | V4 Indexer                | V4 SWA              | V4 Compressor          |
| ------------ | -------------------- | ---------------------- | ------------------------- | ------------------- | ---------------------- |
| 压缩方式     | 低秩（kv_lora_rank） | slot 压缩（c4a/c128a） | 条目 128 维（c4a 层专用） | 无（SWA attention） | 无（compressor state） |
| 存储格式     | rank-3               | rank-3                 | rank-3                    | rank-3              | rank-3                 |
| kv_size      | 1                    | 1                      | 1                         | 1                   | 1                      |
| num_kv_heads | 1                    | 1                      | 1                         | 1                   | 1                      |
| TP 冗余？    | **是**（100% 复制）  | **是**（100% 复制）    | **是**（100% 复制）       | **是**（100% 复制） | **是**（100% 复制）    |

V3 的答案是「全部复制，冗余 87.5%」。V4 的答案没有变——**全部四组都复制。**

V3 和 V4 的压缩机制不同（低秩 vs slot），但它们在 TP 面前的表现完全一致：「不按 head 区分」的表示 ×「按 head 切分」的 TP = 无 head 可切。所不同的只是复制的内容——V3 是 576 维的潜变量 + RoPE，V4 各组是各自维度的压缩表示。

### 6.3 HMA：管理上的复杂度，不是 TP 行为上的差异

vLLM 的 Hybrid KV Cache Manager（HMA）通过 `group_layers_by_identity()` 按 `(kv_size, num_heads, head_size, block_size, engine_group_idx, dtype, engine_kv_format)` 的 7 元组将层聚合为 KV cache 组，管理四组不同的 `tokens_per_block`、`slots_per_block` 和 `compress_ratio`（4× 到 128×）。

HMA 解决的是「如何在同一个显存池中管理不同 block 大小的页」——这是一个**显存分配**问题。但在**TP 冗余**问题上，四组都给出了相同的答案：`kv_size=1` → `is_mla=True` → `ReplicatedLinear` → 全部复制。V4 的冗余不是 V3 的「扩展」——它是同一个规则在四组上的四次应用。

LMCache v0.5.1 的 `is_kv_writer` 对 MLA 的特殊处理对 V4 同样有效——四组都需要 rank-0-only 策略。**V4 没有让问题变得更复杂，只是让同一个问题出现了四次。**

---

## 七、一个更大的问题

这篇文章讨论的「存储冗余」在单节点 8-GPU 部署中的实际影响，随上下文长度增长而急剧变化：

**常规对话（context < 8K tokens）**：KV cache ~0.26 GB per rank（FP8），冗余 ~1.8 GB 跨 8 rank。每个 rank 的 KV cache 只占 H100 80GB 的 0.3%。无关紧要。

**长上下文推理（context = 100K tokens）**：KV cache ~3.3 GB per rank，冗余 ~23 GB 跨 8 rank。3.3 GB 的 KV cache 占单卡 4.1%——仍然 fit。但如果你在这 8 张卡上同时跑多个长上下文请求，冗余开始变成显存的实质消耗。

**百万 token 上下文（V4 的目标场景）**：KV cache ~33 GB per rank——占 H100 80GB 的 41%。冗余 ~229 GB 跨 8 rank。单卡的 33 GB 已经逼近了留给出 activation 和 weight 的空间；而跨 8 rank 合计 ~229 GB 的冗余存储，全部是同一份数据的副本。

但数字不是重点。重点是：随着 MLA 家族模型从 V2 的「一种 MLA」扩展到 V4 的「全部变体都是 MLA」，TP 在 KV cache 上的失效从「一个模型的特殊现象」变成了「MLA 生态的系统性特征」。

根本的命题没有变：**只要注意力缓存中存在不按 head 分组的共享表示，TP 就无法为这部分显存提供任何节省。** V3 的 576 维是第一个实例，V4 的四组变体是第二个、第三、第四、第五个。它不是一个可以被「修好」的问题——它是 MLA 的「共享」哲学和 TP 的「切分」哲学之间，一道数学上不可跨越的裂痕。

---

_基于 LMCache v0.5.1 (`979719d7`) 和 vLLM v0.20.0 源码交叉验证。数据来自 HuggingFace `deepseek-ai/DeepSeek-V3` config.json。_
