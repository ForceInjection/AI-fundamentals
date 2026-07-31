# KV Cache 量化深度解析：Per-Tensor、Per-Token-Head 与 NVFP4

KV Cache 每 token 每层都在吃显存——GQA 下是 4 KB，MLA 下约 1.1 KB，CSA/HCA 下不到 0.2 KB。架构压缩（GQA/MQA/MLA/CSA）做到了极致后，**量化** 是第二条压缩路径：不改模型结构，只降低每个数值的位宽——FP16（2 bytes）→ FP8（1 byte），显存直接减半。

但 KV Cache 量化比权重量化更敏感。权重量化误差在数百层 forward 的末端才体现，而 KV Cache 的量化误差直接作用在 Attention 的 SoftMax 输入上——指数放大后，微小的数值偏差可能将注意力完全导向错误的 token。**量化的粒度**（per-tensor、per-token-head）决定了误差的控制精度。

本文以 vLLM 源码为基准，拆解三种 KV Cache 量化粒度的原理、精度差异与工程实现。

> **前置阅读**：[KV Cache 压缩技术详解](kv_cache_compression.md) — 四维冗余模型中的 "特征维冗余" 是量化的理论基础；[不同注意力类型的 KV Cache 到底长什么样](../basic/attention_kv_cache_formats.md) — 不同架构下 KV 的形状决定了量化的适用方式。

---

## 一、KV Cache 量化为什么比权重量化更难？

### 1.1 误差传播路径不同

权重量化（如 GPTQ、AWQ）作用于模型参数——$W \cdot x$ 中 $W$ 的精度降低。权重误差分布在数百层中，逐层累积，但由于每一层的计算对权重噪声有一定容忍度，整体精度退化相对平缓。

KV Cache 量化作用于注意力计算的输入——$Q \cdot K^T$ 中 $K$ 的精度降低。误差传导路径是：

$$\text{量化 } K \rightarrow Q \cdot K_{\text{quantized}}^T \rightarrow \text{SoftMax} \rightarrow \text{权重偏差} \rightarrow V \text{ 聚合偏差} \rightarrow \text{输出偏差}$$

SoftMax 的指数放大是核心问题：假设量化误差导致 $K$ 中某些维度的值偏差 $\epsilon$，在 $Q \cdot K^T$ 的点积中，$\epsilon$ 被 $Q$ 的模长放大，然后通过 $\exp(\cdot)$ 将差异指数化。原来应该 attention 到 token A 的 query，因为量化误差可能将注意力转移到 token B。

### 1.2 量化粒度决定误差控制能力

量化的本质是将连续的浮点值映射到离散的整数等级上。这个映射需要两个参数：

- **scale**（缩放因子）：决定了"每个量化等级代表多大的浮点范围"
- **zero point**（零点）：偏移量（对称量化中为 0）

scale 的精度决定了量化误差的大小。**每个数值共享同一个 scale**（per-tensor）vs **每组数值有独立的 scale**（per-channel/per-token-head），是一种用额外存储换精度控制的权衡。

对于 KV Cache 来说，$K$ 的形状是 `(num_tokens, num_kv_heads, head_dim)`——这个三维结构提供了两种粒度选择：

| 量化粒度           | Scale 覆盖范围                    |         Scale 数量          |           额外存储            |
| ------------------ | --------------------------------- | :-------------------------: | :---------------------------: |
| **Per-Tensor**     | 整个张量共享一个 scale            |              1              |           几乎为零            |
| **Per-Token-Head** | 每个 token × 每个 head 独立 scale | `num_tokens × num_kv_heads` | 每 token-head 额外 ~2-4 bytes |

---

## 二、三种量化模式的原理与实现

vLLM 通过 `KVQuantMode` 枚举定义了五种量化模式（外加 `NONE`）[^1]：

```python
# vllm/v1/kv_cache_interface.py
class KVQuantMode(IntEnum):
    NONE = 0
    FP8_PER_TENSOR = 1       # per-tensor scales (当前 fp8 默认路径)
    INT8_PER_TOKEN_HEAD = 2  # per-token-head 动态 int8 量化
    FP8_PER_TOKEN_HEAD = 3   # per-token-head 动态 fp8 量化
    INT4_PER_TOKEN_HEAD = 4  # per-token-head 动态 int4 量化
    NVFP4 = 5                # 打包 fp4 数据 + fp8 block scales
```

### 2.1 FP8 Per-Tensor：最简路径

Per-Tensor 是整个 KV Cache 张量共享一个 scale：`K_fp8 = K_fp16 / scale`。写入 cache 时存 FP8 值和 scale；读取时 `K_dequant = K_fp8 * scale`。

**优点**：实现极简，额外存储仅一个 float32 scale（4 bytes），不需要修改 attention kernel。

**缺点**：一个 scale 覆盖所有 token 和所有 head。长序列下，不同位置的 token 的 $K$ 值分布差异很大——开头 token（Attention Sink）的 $K$ 值可能远大于中间 token。一个全局 scale 无法精确表示所有 token，尾部 token 的量化误差最严重。

**vLLM 中的配置**：`--kv-cache-dtype fp8` 或 `--kv-cache-dtype fp8_e4m3`[^2]。这是当前生产环境中最常用的量化配置。

### 2.2 Per-Token-Head（FP8/INT8）：精度换来的压缩

Per-Token-Head 为每个 token 的每个 head 独立计算 scale：`scale[i][j]` 覆盖第 `i` 个 token 的第 `j` 个 head 的 $K$ 向量（`head_dim` 维）。

```text
Per-Token-Head 量化的 scale 计算：
  for token i in [0, num_tokens):
    for head j in [0, num_kv_heads):
      k_vec = K[i, j, :]               # (head_dim,) 向量
      scale[i][j] = max(|k_vec|) / FP8_MAX
      K_quant[i, j, :] = round(k_vec / scale[i][j])
```

**相比于 Per-Tensor 的优势**：每个 token-head 组合有自己的 scale，量化误差被限制在该组合内部。Attention Sink token（高 $K$ 值）和普通 token（低 $K$ 值）不再共享同一个 scale——各自按自己的数值范围做最优映射。

**额外的存储开销**（注意 K 和 V 需要各自独立的 scale，因为两者数值分布不同，下表已将 K 和 V 的 scale 合计）：

| 模型               | num_tokens | num_kv_heads | scale 总数 (K+V) | 额外存储 (FP16) |
| ------------------ | :--------: | :----------: | :--------------: | :-------------: |
| LLaMA-2 70B @ 32K  |   32768    |      8       |     524,288      |      ~1 MB      |
| LLaMA-2 70B @ 128K |   131072   |      8       |    2,097,152     |      ~4 MB      |

相对于 KV Cache 本身（32K context × 80 layers × 4 KB ≈ 10 GB），~1 MB 的 scale 开销仍然可忽略不计。

**vLLM 中的配置**：`--kv-cache-dtype fp8_per_token_head` 或 `--kv-cache-dtype int8_per_token_head`。FP8 vs INT8 的选择主要取决于硬件支持——FP8 在 H100/H200 上有 native tensor core 加速，INT8 在 A100 上更成熟。

### 2.3 NVFP4：4-bit 的极致压缩

NVFP4 是 NVIDIA Blackwell（B200）引入的专用浮点格式——每个数值仅 4 bits，配合 FP8 block scale 实现 ~4× 压缩比（相对 FP16）。不同于传统的 INT4 量化，NVFP4 保留了浮点表示的动态范围优势，配合硬件原生支持的 Blackwell tensor core 做高效解压。

vLLM 通过 `NVFP4` 枚举值支持此模式[^1]，配置为 `--kv-cache-dtype nvfp4`。但需要注意：

- **硬件依赖**：仅 Blackwell GPU（B200）支持，H100/H200 不可用
- **精度敏感**：4-bit 下量化误差显著高于 8-bit，对长序列中关键 token（Attention Sink、Heavy Hitter）的保护需要额外策略
- **block scale**：NVFP4 使用 FP8 block scale——沿 `head_dim` 维度每 16 个元素为一个 block，每个 block 一个 scale。这与 Per-Token-Head（沿 token×head 维度分组，覆盖整个 `head_dim`）的粒度选择的"轴"不同——NVFP4 在特征维度上更细粒度，能更好地控制 outlier 维度的误差；Per-Token-Head 在 token/head 维度上更细粒度，能隔离不同 token 的数值分布差异。两者是正交的精度控制策略

---

## 三、量化对 Prefix Caching 的影响

### 3.1 Hash 依赖精度

Prefix Caching 通过 `hash(parent_hash, block_tokens)` 做 block 匹配。量化的关键问题是：**量化前后的 $K$ 值不同，但 hash 基于 token ID 计算，不依赖 $K$ 值**。

这意味着量化不影响 Prefix Caching 的命中逻辑——相同 token 序列的 block hash 在 FP16 和 FP8 下完全一致。但**命中的 block 的 $K$ 值精度降低**——如果 producer 用 FP8 存储 KV，consumer 命中后拿到的就是 FP8 精度的 $K$，而非 FP16。

### 3.2 FP8 与 FP16 的跨精度缓存

如果 producer 请求用 FP16 写入 KV Cache，consumer 请求用 FP8 模式（`--kv-cache-dtype fp8`）尝试复用——此时 producer 的 FP16 block 的 hash 与 consumer 的 token 序列匹配，但 consumer 期望从 block 中读取 FP8 数据，这会导致精度不匹配。

vLLM 当前对此的处理方式是：**在同一 vLLM 实例内，所有请求共享同一个 `kv_cache_dtype` 配置**——不存在跨精度缓存的问题。在 PD 分离场景下，通常建议 Prefill 节点和 Decode 节点使用相同量化精度，以避免在线转换开销。如果必须混合精度，可以在 Prefill 端以高精度计算，传输时量化为 Decode 端的低精度格式再发送——当前 vLLM V1 的 KV Connector API 不内置精度转换，connector 实现者需要在传输层自行处理。

---

## 四、量化精度损失的来源与量级

### 4.1 量化误差的三个来源

**Scale 不够精确**。这是 Per-Tensor 量化的主要误差源。全张量共享一个 scale 意味着 scale 必须容纳张量中最大的绝对值——如果有一个 outlier token 的 $K$ 值非常大，scale 被它撑大，其他所有 token 的量化精度都被拖累。这就是 per-token-head 量化要解决的核心问题。

**舍入误差**。FP32/FP16 → INT8/FP8 的映射不是连续的——`round(x / scale)` 引入了 ±0.5 量化等级的舍入误差。这个误差在大多数情况下是可接受的（FP8 的动态范围相对均匀），但在 INT8 下，如果 $K$ 值分布高度不均匀（大部分值很小，少数值很大），小值的相对舍入误差会很大。

**累积效应**。量化误差不是一层的问题——每一层的 Attention 输出都会受到该层 $K$ 值量化的影响。80 层模型下，量化误差可能逐层累积，但 LayerNorm 的归一化操作在一定程度上抑制了这种累积。

### 4.2 精度损失的参考量级

| 量化方式            | 显存节省（相对 FP16） | 典型精度退化 (PPL) | 依赖硬件                     |
| ------------------- | :-------------------: | :----------------: | ---------------------------- |
| FP8 Per-Tensor      |          2×           |   < 0.1 PPL 退化   | H100/H200 (FP8 tensor core)  |
| FP8 Per-Token-Head  |          2×           |     几乎可忽略     | H100/H200                    |
| INT8 Per-Token-Head |          2×           |  0.1-0.3 PPL 退化  | A100/H100 (INT8 tensor core) |
| NVFP4               |          4×           |  0.3-1.0 PPL 退化  | B200 (Blackwell)             |

> 注：精度退化数据为文献中常见范围的综合，实际值取决于模型、序列长度和任务类型。长序列（>32K）下量化误差通常大于短序列，因为 outlier token 的概率随序列长度增加。

---

## 五、取舍

### 5.1 量化粒度的选择

如果你追求最小的实现复杂度和最广泛的硬件兼容性，**FP8 Per-Tensor 是当前 vLLM 的默认推荐**——额外存储仅 4 bytes，attention kernel 无需修改，H100/H200 上开箱即用。它的主要风险在于 outlier token：当长序列中某个 token 的 $K$ 值远大于其他 token 时，全局 scale 被这一个 outlier 撑大，其余 token 的量化精度被拖累。

如果你需要最高的精度保持，**FP8 Per-Token-Head 是当前最优解**。它为每个 token 的每个 head 独立计算 scale，outlier token 不再影响其他 token 的量化质量。额外的 scale 存储开销（32K context 下 K 和 V 各一套 scale，合计约 1 MB）相对于 KV Cache 本身可忽略不计，精度上的收益远超存储成本。

如果你追求极致压缩，且运行在 Blackwell（B200）上，**NVFP4 提供了 4× 的压缩比**——但 4-bit 的精度损失不可忽视，可以与淘汰策略（如丢弃低重要性 token）配合使用——淘汰策略移除不重要但可能成为量化 outlier 的 token，从而间接改善剩余关键 token 的量化精度。如果你没有 B200，目前最务实的方案是在 FP8 Per-Token-Head 的基础上，结合 Attention Sinks 和 Heavy Hitter 淘汰策略，通过减少 token 数量来间接弥补量化压缩率的天花板。

### 5.2 一句话总结

**KV Cache 量化的核心矛盾不是"能不能压"，而是"scale 该有多细"——Per-Tensor 一个 scale 管全张量，实现简单但 outlier 敏感；Per-Token-Head 把 scale 下放到每个 token 每个 head，精度高但引入微小的 scale 存储开销；NVFP4 是硬件的未来，但当前仅在 B200 上可用。** FP8 Per-Token-Head 在精度与复杂度之间取得了最佳平衡，是当前生产环境中最值得推荐的配置——除非你在 Blackwell 上，那就直接 NVFP4。

---

## 相关阅读

- [KV Cache 压缩技术详解](kv_cache_compression.md) — 四维冗余模型 + 压缩全景
- [不同注意力类型的 KV Cache 到底长什么样](../basic/attention_kv_cache_formats.md) — 量化适用性因 attention 类型而异（MLA 的 latent 量化与 GQA 的 K/V 量化的区别）
- [Attention Sinks 与 KV Cache 淘汰策略](../eviction/attention_sinks_and_eviction.md) — 量化 + 淘汰 = 压缩的两条腿
- [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750) — INT4 K + FP16 V 的非对称量化方案

[^1]: vLLM 量化模式枚举 [`vllm/v1/kv_cache_interface.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/kv_cache_interface.py) — `KVQuantMode` 定义五种量化模式：`NONE`/`FP8_PER_TENSOR`/`INT8_PER_TOKEN_HEAD`/`FP8_PER_TOKEN_HEAD`/`INT4_PER_TOKEN_HEAD`/`NVFP4`；`get_kv_quant_mode()` 根据 `kv_cache_dtype` 字符串映射到对应模式。

[^2]: vLLM Cache 配置 [`vllm/config/cache.py`](https://github.com/vllm-project/vllm/blob/main/vllm/config/cache.py) — `cache_dtype: CacheDType = "auto"`，可选值包括 `auto`/`fp8`/`fp8_e4m3`/`fp8_e5m2`/`fp8_per_token_head`/`int8_per_token_head`/`nvfp4`。`--kv-cache-dtype` 启动参数映射到此配置项。
