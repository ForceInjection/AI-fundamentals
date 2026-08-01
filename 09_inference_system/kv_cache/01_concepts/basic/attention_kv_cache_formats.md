# 不同注意力类型的 KV Cache 到底长什么样——MHA、GQA、MQA、MLA、CSA/HCA 的存储形态

KV Cache 存的是每一层每个 token 的 Key 和 Value。但 "Key" 和 "Value" 到底有几个？这取决于注意力类型——MHA 下 64 个 Q head 就有 64 组 K/V，GQA 下 64 个 Q head 可能只共享 8 组 K/V，MLA 下 K/V 干脆被压缩成了 latent vector。KV Cache 的物理大小直接由注意力架构决定。

本文以 LLaMA-2 70B (GQA)、DeepSeek-V3 (MLA)、DeepSeek-V4 (CSA/HCA) 为主要实例，给出每种注意力类型下 KV Cache 的精确形状和显存占用，以及 vLLM 当前的支持状态。MHA 和 MQA 以假设配置展示公式，实际模型参见 §六。

---

## 一、MHA（多头注意力）：每个 Q head 配一组 K/V

标准 Transformer 的配置：Q head 数 = K head 数 = V head 数。

```text
LLaMA-2 70B 如果使用 MHA（实际使用 GQA，此处仅做假设）：
  num_q_heads = 64
  num_kv_heads = 64    ← 与 Q head 数相同
  head_dim = 128

单 token 单层的 KV Cache：
  K: (64, 128) = 8192 个 float16 = 16 KB
  V: (64, 128) = 8192 个 float16 = 16 KB
  合计：32 KB / token / layer

全模型（80 layers, seq_len=4096）：
  KV Cache = 2 × 80 × 64 × 128 × 4096 × 2 bytes ≈ 10 GB
```

> 注：此处按 seq_len=4096 估算，仅为展示公式。长上下文场景（如 1M）下 MHA 的 KV Cache 会膨胀到 TB 级，详见 §六横向对比。

MHA 的 KV Cache 最大，因为每个 Q head 都需要自己独立的 K 和 V。没有任何共享。

![MHA KV Cache 存储形态](img/format-mha.svg)

---

## 二、GQA（分组查询注意力）：多个 Q head 共享一组 K/V

这是当前最主流的方案。Q head 被分成若干组，每组内的所有 Q head 共享一组 K 和 V。

```text
LLaMA-2 70B（实际配置）：
  num_q_heads = 64
  num_kv_heads = 8      ← 8 组 K/V，每组服务 8 个 Q head
  head_dim = 128

Q head 到 KV head 的映射：
  Q head 0-7   → KV head 0
  Q head 8-15  → KV head 1
  ...
  Q head 56-63 → KV head 7

单 token 单层的 KV Cache：
  K: (8, 128) = 1024 个 float16 = 2 KB
  V: (8, 128) = 1024 个 float16 = 2 KB
  合计：4 KB / token / layer    ← 仅为 MHA 的 1/8

全模型（80 layers, seq_len=4096）：
  KV Cache = 2 × 80 × 8 × 128 × 4096 × 2 bytes ≈ 1.25 GB
```

GQA 是 `num_kv_heads` 和 `num_q_heads` 的比例决定了 KV Cache 的缩减倍数。LLaMA-2 70B 的比例是 8:1，所以 KV Cache 是 MHA 的 1/8。Qwen-2 同样使用 GQA，比例因模型大小而异。

**在 vLLM 中**：GQA 是默认支持最广泛的注意力类型——LLaMA-2、LLaMA-3、Qwen-2、Mistral 等主流模型全部使用 GQA。启动时 vLLM 自动从模型 config 读取 `num_key_value_heads`，无需用户干预。

![GQA KV Cache 存储形态](img/format-gqa.svg)

---

## 三、MQA（多查询注意力）：所有 Q head 共享唯一一组 K/V

GQA 的极端版本：只保留 1 组 K 和 V，所有 Q head 全部共享。

```text
假设 MQA 配置：
  num_q_heads = 32
  num_kv_heads = 1      ← 只有 1 组 KV
  head_dim = 128

单 token 单层的 KV Cache：
  K: (1, 128) = 128 个 float16 = 256 bytes
  V: (1, 128) = 128 个 float16 = 256 bytes
  合计：512 bytes / token / layer
```

MQA 的 KV Cache 极小，但代价是注意力质量下降——只有一组 K/V 意味着所有 Q head 从同一个视角观察输入，表达能力受限。PaLM 和 Falcon 使用了 MQA，但 GQA 出现后 MQA 基本被取代。

![MQA KV Cache 存储形态](img/format-mqa.svg)

---

## 四、MLA（多头潜在注意力）：K/V 被压缩后再存储

DeepSeek V2/V3 提出的 MLA 彻底改变了 KV Cache 的物理形态。传统注意力下 K 和 V 是直接存储的——形状是 `(num_kv_heads, head_dim)`。MLA 的核心思路是：**不存完整的 K 和 V，只存一个压缩后的 latent vector，注意力和生成时再实时解压。**

````text
DeepSeek-V3（MLA）：
  d_model = 7168
  num_q_heads = 128
  qk_nope_head_dim = 128   ← Q/K 的非位置编码部分
  qk_rope_head_dim = 64    ← Q/K 的 RoPE 位置编码部分
  v_head_dim = 128
  kv_lora_rank = 512       ← KV 压缩后的 latent 维度

传统 MHA 单 token 单层的 KV Cache（假设无压缩）：
  K: 128 heads × 128 dim = 16384 个元素 = 32 KB
  V: 128 heads × 128 dim = 16384 个元素 = 32 KB
  合计：64 KB / token / layer

MLA 实际存储（两部分）：
  KV latent (K/V 共享压缩向量): (512,) = 512 fp16   = 1 KB
  Decoupled K (RoPE 位置编码): (64,) = 64 fp16 = 128 B  ← 共享于所有 head，RoPE 只需一份
  合计：~1.13 KB / token / layer   ← 约为传统 MHA 的 1/57

> DeepSeek 发现位置信息只需一个共享的 RoPE key 即可编码——内容部分 (k^C) 通过低秩分解承载 per-head 语义，位置部分 (k^R) 是所有 head 共用的一维信号。因此 decoupled RoPE K 不乘 head 数，仅为 `qk_rope_head_dim = 64` 维。

MLA 的压缩效果来自两个机制：(1) K 和 V 共享一个下投影矩阵 `W^{DKV}`，将 128 head × 128 dim 的高维空间压缩到 `kv_lora_rank = 512` 维；(2) 位置编码（RoPE）因旋转操作无法直接压缩，从 latent 中解耦后单独存储——但因为所有 head 共享同一个 RoPE key，仅需 64 维而非 128×64 维。实际 attention 计算时，latent 通过 `W^{UK}` 和 `W^{UV}` 实时还原 per-head K 和 V，RoPE key 广播到所有 head。

**在 vLLM 中**：MLA 通过独立的 attention backend 支持。vLLM 的 `DeepseekV2Attention` 后端自动处理 KV latent 和 decoupled K 的存储和实时解压，用户只需正常启动服务。关键差异：MLA 下 KV Cache 的物理格式不再是单一的 `(heads, head_dim)`，而是两种形态共存——`(kv_lora_rank,)` 的共享 latent 和 `(qk_rope_head_dim,)` 的共享 RoPE K。vLLM 的 block table 通过 `KVCacheGroupSpec` 为每种形态创建独立的 block 池（参见 `kv_cache_groups.py`）。

![MLA KV Cache 压缩管线](img/format-mla.svg)

---

## 五、CSA / HCA（压缩稀疏注意力 / 混合压缩注意力）：DeepSeek V4 的 KV 多级压缩

DeepSeek V4 在 MLA 的基础上引入了一个更激进的思路：**不仅压缩每个 token 的 K/V 维度，还压缩 token 的数量。** 这就是 CSA（Compressed Sparse Attention）和 HCA（Hybrid Compressed Attention）。

### 5.1 CSA：把连续多个 token 的 KV 合并为一个

不再给每个 token 都存一份 KV。而是跨 token 做加权合并：c4a 将 8 个连续 token 合并为 1 个压缩 token（步长 4），c128a 将 128 个 token 合并为 1 个（步长 128）。

```text
原始 token 序列（1M context）:
  t0 t1 t2 t3 t4 t5 t6 t7 t8 t9 t10 t11 ...

c4a 压缩后（每 8 个 token → 1 个压缩 token，步长 4，有重叠）:
  [t0-t7]  [t4-t11]  [t8-t15]  [t12-t19] ...
  每个压缩 token 存一份 KV → KV 数量 ≈ 原始的 1/2（步长 4 导致 2× 冗余）

c128a 压缩后（每 128 个 token → 1 个压缩 token，无重叠）:
  [t0-t127]  [t128-t255]  [t256-t383] ...
  每个压缩 token 存一份 KV → KV 数量 ≈ 原始的 1/128
````

压缩 token 的值是原始 token K/V 的加权和——不是简单平均，而是可学习的投影权重。注意力计算时，Q 直接与压缩后的 K 做点积，跳过了逐 token 展开的步骤。

### 5.2 HCA：混合使用多种压缩策略

V4 的不同层使用不同的压缩策略——部分层使用温和的 c4a 保留更多细节，大部分层使用激进的 c128a 最大化压缩。所有层都附带 128-token sliding window 保留局部信息。这就是 HCA（Hybrid Compressed Attention）：

```text
V4 共 61 层，每层均带 128-token sliding window：
  30 层 c4a：   8:1 token 压缩（有效 4:1，步长 4 重叠）
  31 层 c128a： 128:1 token 压缩（无重叠）
```

### 5.3 DeepSeek V4 的 KV Cache 形态

由于不同层使用不同压缩策略，V4 的 KV Cache 不再是一个统一的 `(layers, heads, seq, dim)` 张量，而是多种形态的混合：

| 层类型      | 存储内容 (per compressed entry) | 等效 per original token | 说明                      |
| ----------- | ------------------------------- | :---------------------: | ------------------------- |
| MLA 压缩    | 512 latent + 64 RoPE = 576 dim  |         ~1.1 KB         | 完整 MLA 存储             |
| c4a 压缩    | 64 B shared-KV + 8 B indexer    |          ~18 B          | 8:1 压缩, 步长4 → 有效4:1 |
| c128a 压缩  | 64 B shared-KV                  |         ~0.5 B          | 128:1 压缩, 无重叠        |
| Sliding win | 未压缩 KV (仅 128 token)        |     128 token 窗口      | 局部窗口保留              |

> 全模型合计（bf16）：**~9.62 GiB**（vs MLA-only fp16 的 ~67 GB 或 FP8 量化后 ~34 GiB[^1]，缩减 3.5×~7×）。实际部署中 indexer cache 使用 FP4，attention cache 使用 FP8，KV Cache 可进一步减半至 ~5 GB。
>
> [^1]: MLA @ 1M context：fp16 未量化 ≈ 1.13 KB × 61 × 1M ≈ **~67 GB**。实际 V3.2 部署使用 FP8 量化后约 576 bytes/token/layer，对应 1M context 约 **~34 GiB**（数据来源：[vLLM 博客](https://vllm.ai/blog/2026-04-24-deepseek-v4)）。

### 5.4 vLLM 对 V4 的支持

vLLM 通过 `KVCacheGroupSpec` 为每种压缩类型创建独立的 KV block 池——sliding window 的 block 和 c128a 的 block 物理大小完全不同，需要不同的内存分配器。`--block-size 256`（而非默认的 16）是 V4 推荐配置，因为大 block 对压缩 token 的批量 I/O 更友好。

关键启动参数：

```bash
--block-size 256 \                          # 压缩 token 场景下大 block 更高效
--kv-cache-dtype fp8 \                      # attention cache 使用 FP8
--attention_config.use_fp4_indexer_cache=True  # indexer cache 使用 FP4
```

![CSA/HCA token 维压缩](img/format-csa-hca.svg)

---

## 六、五种注意力类型的横向对比

> 以上各节的 SVG 可视化图（§一~§五末尾）展示了每种注意力类型下 KV Cache 的物理存储形态和计算过程。下图汇总精确数值。

| 注意力类型  | 压缩维度                         | 单 token 等效 KV | vs MHA 缩减 | 代表模型          |
| ----------- | -------------------------------- | :--------------: | :---------: | ----------------- |
| MHA         | 无                               |      32 KB       |     1×      | 原始 Transformer  |
| GQA         | KV head 共享                     |       4 KB       |     8×      | LLaMA-2/3, Qwen-2 |
| MQA         | 极值 KV head 共享                |      0.5 KB      |     64×     | PaLM, Falcon      |
| MLA         | head 维压缩 (latent) + RoPE 共享 |     ~1.1 KB      |    ~29×     | DeepSeek V2/V3    |
| **CSA/HCA** | **head + token 双维压缩**        | **~0.17 KB**[^2] |  **~190×**  | **DeepSeek V4**   |

> 单 token 等效 KV 基于 80 层 Dense 模型、FP16/BF16、`head_dim=128` 的假设。MLA 的 ~1.1 KB 包含了 KV latent (1 KB) + 共享 RoPE K (128 B)。MLA 的 decoupled RoPE K 是所有 head 共享的单个向量（`qk_rope_head_dim = 64`），非 per-head 存储——这是 MLA 压缩率远超 GQA 的关键。CSA/HCA 的 ~0.17 KB 来自 vLLM 官方博客的 1M context 实测（9.62 GiB ÷ 61 layers ÷ 1M tokens ≈ 169 bytes/token/layer，bf16），是 30 层 c4a + 31 层 c128a 的加权平均值。实际模型结构不同（层数、MoE、混合压缩层），具体数值以模型 card 为准。
>
> 1M context 下 KV Cache 的量级：以 GQA（8 KV heads）为例，1M × 4 KB × 80 ≈ **305 GB**。MLA 下 1M × 1.13 KB × 61 ≈ **~67 GB（fp16）**，实际 DeepSeek-V3.2 部署使用 FP8 量化后约 576 bytes/token/layer ≈ **~34 GiB @ 1M**（数据来源：[vLLM 博客](https://vllm.ai/blog/2026-04-24-deepseek-v4)）。V4 的 CSA/HCA 通过 token 维压缩 + K=V 共享将 1M 场景压至 **~9.62 GiB（bf16）**，FP8/FP4 混合精度可进一步减半至 ~5 GB。

[^2]: CSA/HCA 的 ~0.17 KB 计算：vLLM 博客实测 V4 @ 1M context bf16 = 9.62 GiB，除以 61 层和 1M tokens 得到 169 bytes/token/layer。注意这是 30 层 c4a（有效 4:1 压缩）+ 31 层 c128a（128:1 压缩）的加权平均；其中 c128a 层等效仅 ~0.5 bytes/token（~0.0005 KB），c4a 层等效 ~18 bytes/token（~0.018 KB）。详见 [vLLM 博客](https://vllm.ai/blog/2026-04-24-deepseek-v4) 和 `kv_cache_calc.py`。

---

## 七、vLLM 多注意力架构共存

一个推理引擎需要同时支持从 GQA 到 CSA/HCA 的五种注意力类型。vLLM v1 引擎的 `KVCacheGroupSpec` 抽象了这种差异——为每种注意力类型创建独立的 block 池和内存布局，上层 scheduler 统一管理 block ID 分配：

```text
模型加载时：
  vLLM 读取 model config → 识别每层的 attention 类型 → 创建对应的 KV Cache 布局

GQA (LLaMA-2 70B):
  KVCacheGroupSpec: num_kv_heads=8, head_dim=128, block_size=16
  → KV block: 2 × 8 × 128 × 16 × 2 bytes = 64 KB

MLA (DeepSeek-V3):
  layer_names: 61 层 MLA 全部归入同一个 KVCacheGroupSpec
  kv_cache_spec: MLAAttentionSpec(kv_lora_rank=512, head_size=576, block_size=16)
  → 单个 spec 内部管理两种物理存储：KV latent + decoupled RoPE K
  → page_size_bytes = block_size × 656 bytes (FP8 部署, flashmla 自定义布局)
  → 通用 fp16 回退: 2 × block_size × num_kv_heads × head_size × dtype_size

CSA/HCA (DeepSeek V4):
  layer_names: 30 层 c4a → ChunkedLocalAttentionSpec(attention_chunk_size=...)
              31 层 c128a → ChunkedLocalAttentionSpec(attention_chunk_size=...)
              所有 61 层 → SlidingWindowSpec(sliding_window=128)
  → c4a / c128a / sliding window 三种 spec，各自独立 block 池和内存分配器
  → block_size=256，大 block 对压缩 token 批量 I/O 更友好
```

同一个 `KVCacheGroupSpec` 接口，不同的物理存储布局——这是 vLLM 能同时支持 MHA、GQA、MLA、Sliding Window、CSA/HCA 的关键抽象。V4 的实现细节详见 [vLLM 中的 DeepSeek V4](../../../vllm/module_analysis/deepseek_v4_attention_support.md) 和 [MLA 到 CSA/HCA 进化](../../../vllm/module_analysis/deepseek_attention_evolution_mla_to_csa_hca.md)。

---

## 八、相关资源

- [A Visual Guide to Attention Variants in Modern LLMs](https://magazine.sebastianraschka.com/p/visual-attention-variants) — Sebastian Raschka, MHA→GQA→MLA 架构图和 KV Cache 对比（推荐与本文互补：有图 + 有精确计算）
- [KV Cache 原理简介](kv_cache_basics.md) — KV Cache 的工作机制和显存公式
- [KV Cache 为什么叫 KV Cache？——Q 去哪了](why_only_kv.md) — 为什么 Q 不参与缓存
- [为什么 GPU 生成每个 token 时利用率不到 5%？——Prefill 与 Decode 深度拆解](../../../prefill_decode/prefill_decode_qkv_calculation.md) — 两阶段计算过程详解
- [大模型 KV Cache 压缩技术详解](../compression/kv_cache_compression.md) — GQA/MQA 之外的压缩手段
- [vLLM kv_cache_groups.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/utils.py) — `KVCacheGroupSpec` 的源码实现
- [kv_cache_calc.py](kv_cache_calc.py) — 本文所有 KV Cache 数值的可复现计算脚本
