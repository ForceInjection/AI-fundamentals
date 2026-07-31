# PagedAttention 这个让 vLLM 成名的技术，为什么走到了尽头？

> PagedAttention 的 CUDA 实现在 v0.23.1rc0 中最后一次完整出现，在 v0.25.0 中被彻底删除（仅 ROCm 路径保留 `paged_attention_rocm` 函数名）。本文对比两版源码，结合 PR [#47361](https://github.com/vllm-project/vllm/pull/47361) 和 [v0.25.0 Release Notes](https://github.com/vllm-project/vllm/releases/tag/v0.25.0)，分析其退役的技术原因。

## 1. 它曾经是最好的答案

2023 年，vLLM 诞生时，所有的 LLM 推理引擎都在做同一件事：把 token 的 Key 和 Value 连续地铺在显存上，提前为 batch 中最长的序列分配一整块连续空间，然后任由 80% 的空间被 `<pad>` 浪费。PagedAttention 用一张 `[num_seqs, max_blocks]` 的 int32 映射表打破了这个困境——让 KV cache 像操作系统的虚拟内存页一样，离散分布在显存里，按需映射、按需换入换出。

```cpp
// attention_kernels.cuh:201, 249-250 — PagedAttention 的核心抽象
const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;
const int64_t physical_block_number = block_table[block_idx];
```

这套机制让 vLLM 在吞吐量上远超同期引擎，PagedAttention 成为 vLLM 的标志性设计。

2026 年 7 月，v0.25.0 发布。在 Release Notes 的 Highlights 中，**「PagedAttention has been removed」被列为第一条**。同时列出的还有新注意力后端矩阵——`FLASH_ATTN_MLA_SPARSE`、`FLASHINFER_MLA_SPARSE`、FlashAttention 3——以及「V1/MRv2 backends are the standard path」。这是一个版本内同时完成「旧事物退场」和「新体系确立」的时刻。

好的抽象有保质期。PagedAttention 的退役，不是因为它有什么缺陷，而是因为它面对的世界已经完全变了。

## 2. 问题出在「注意力」本身变了

PagedAttention 要计算的注意力长这样：

$$
QK^\top V, \quad K, V \in \mathbb{R}^{\text{seq\_len} \times \text{head\_size}}
$$

K 是 `[num_blocks, num_heads, head_size/x, block_size, x]` 的五维张量，V 是类似的四维张量。这是标准 Multi-Head Attention 的存储形态——K 和 V 各占一块显存，互不干扰。

```cpp
// attention_kernels.cuh:89-92
const cache_t* __restrict__ k_cache,  // [num_blocks, num_kv_heads,
                                       //   head_size/x, block_size, x]
const cache_t* __restrict__ v_cache,  // [num_blocks, num_kv_heads,
                                       //   head_size, block_size]
```

但 2025 年以后，DeepSeek V2 引入 MLA——Multi-head Latent Attention。MLA 把 K 和 V 从独立的矩阵压缩为一个低秩表示 `[c_kv, k_rope]`。在标准 MHA 中，KV 的维度是 $2 \times \text{num\_layers} \times \text{num\_heads} \times \text{head\_size}$；在 MLA 中，这个数字缩小到了 $\text{kv\_lora\_rank} + \text{qk\_rope\_head\_dim}$——约为原来的 $27\%$。

PagedAttention 的 kernel 期望拿到两个独立的张量，一个叫 `k_cache`，一个叫 `v_cache`。MLA 只给一个压缩表示。PagedAttention 没有从 `[c_kv, k_rope]` 解压回 K 和 V 的代码路径。**这不是一个可以「修一修」就解决的问题——需要从存储格式、计算流程、到显存布局都重新设计。**

当 DeepSeek V4-Flash 把这件事推向极致——4 组不同压缩比的 KV cache 共存（MLA 主缓存 $4\times$、Indexer $128\times$、SWA $1\times$、Compressor $1\times$）——PagedAttention 不仅无法计算，对「注意力的输入是什么」这个基础假设都已经不成立了。

v0.25.0 的发布说明从侧面印证了这一点。DeepSeek V4 相关条目占了整整一屏：

> - `FLASH_ATTN_MLA_SPARSE` Hopper sparse-MLA backend
> - `token_to_req_indices` cache for DSv4 ($5{-}6\times$ kernel speedup)
> - better DSv4 MXFP8 kernel
> - Mooncake connector GDN + MLA (DeepSeek-V4-Flash) support
> - fused_indexer_q_rope_quant Triton kernel ($1.9{-}3.3\%$ E2E throughput)

一个模型有专属的 indexer kernel、专属的 sparse MLA 后端、专属的 MXFP8 kernel、甚至专属的 connector 支持——这说明 DeepSeek 系列已经不是「ML 中的一个模型」，而是 vLLM 注意力体系围绕其重构的基准。

## 3. 遍历两次同一个 block table

即使只看标准 MHA——不考虑 MLA——PagedAttention 的 kernel 结构也有一个被硬件演进放大的瓶颈：**它对 KV cache 做了两遍完整的遍历**。先遍历一遍 KV blocks 算 $QK$ 和 softmax，再一模一样地遍历第二遍算 softmax 加权的 V。两次遍历之间，整个 block 的 128 个线程在 barrier 处等待第一遍的 K 遍历和 softmax reduction 完全结束，才能开始第二遍的 V 遍历。K 和 V 之间没有任何 overlap。

这件事在 2023 年是可以接受的。当时 GPU 是 A100（HBM2e，2.0 TB/s），batch 不过个位数，KV 长度在几千 tokens 级别。两遍遍历的开销被 SXM 带宽掩盖。

但 DeepSeek V4-Flash 的 batch 可以达到数十甚至上百——每个请求的 KV 长度在 prefix caching 开启后可能上万 tokens。当 KV 总量远超 L2 cache 时，两遍遍历意味着对 HBM 的访问量翻倍——而 H100 的 FP8 matmul 是 A100 的 $6\times$，HBM 带宽只增长了 $1.7\times$。

FlashMLA 和 FlashAttention 系列（FA1→FA2→FA3）通过 IO-aware tiling 把 QKV 融合为单次遍历——中间 attention matrix 永不离开 SRAM。这是整个 FlashAttention 家族共有的设计，而非 FA3 独有的特性。在 v0.25.0 中，FlashAttention 3 被明确标注为「built against the torch stable API」，意味着它不再是实验性后端，而是稳定接口之上的标准实现。这比 PagedAttention 快的最直接原因，就是少了一遍 HBM 遍历。

## 4. 当模型是 FP8 的，为什么还要 dequant？

DeepSeek V4-Flash 的权重和 KV cache 都是 FP8（e4m3）。它生来就是 FP8。

PagedAttention 对 FP8 的支持局限在「存储」：K 和 V 以 FP8 存在显存中，但加载时 dequant 到 FP16/BF16：

```cpp
// attention_kernels.cuh:275-281 — 只有 dequant，没有 FP8 计算
Quant_vec k_vec_quant = *reinterpret_cast<const Quant_vec*>(k_ptr + ...);
k_vecs[j] = fp8::scaled_convert<K_vec, Quant_vec, KV_DTYPE>(
    k_vec_quant, *k_scale);  // FP8 → FP16/BF16
```

$QK$ 是 FP16/BF16 矩阵乘，softmax 是 FP32，加权 V 也是 FP16/BF16。从显存加载是 FP8（带宽省了一半），但一到 shared memory 和 register 就膨胀回 FP16/BF16。

v0.25.0 的 `FLASH_ATTN_MLA_SPARSE` 和 MXFP8 kernel 不一样：decode kernel 以 FP8 执行整个 $QKV$ 运算，只在最后输出时做一次转换。在 H100 上，shared memory 是 228 KB/SM 的稀缺资源。dequant 膨胀意味着更多 shared memory 占用，更低的 occupancy，更低的吞吐。

## 5. 通用性即浪费

PagedAttention 的编译策略是模板参数笛卡尔积：

```cpp
// paged_attention_v1.cu:94-128
switch (head_size) {
    case 32:  LAUNCH_PAGED_ATTENTION_V1(32);  break;
    case 64:  LAUNCH_PAGED_ATTENTION_V1(64);  break;
    case 128: LAUNCH_PAGED_ATTENTION_V1(128); break;
    // ...
}
```

$9 \times 3 \times 2 \times 2 \times 2 \times 2 \approx 400$ 个 kernel 实例。每个实例一份完整的 ~500 行模板展开。目的是「支持所有可能的模型配置」。

代价是每一个 kernel 实例都必须在最低公分母硬件上运行——这意味着放弃了 H100 上三个关键能力：**TMA**（Tensor Memory Accelerator，硬件异步从 HBM 搬运数据到 shared memory，不占用 CUDA core 的计算周期）、**wgmma**（warp group 级别的异步 FP8 矩阵乘累加指令，一次发射代替多次 FMA）、**persistent kernel**（thread block 在 SM 上驻留不退出，跨多次迭代复用 register 和 shared memory，消除 kernel launch 开销）。PagedAttention 用的是一套从 A100 时代继承下来的编程模型——`WARP_SIZE=32`、`__syncthreads()` 同步、shared memory 手动管理。对 A100 来说，这套模型是当时最好的选择；对 H100 来说，它放弃了硬件近一半的有效吞吐。

v0.25.0 的新后端体系正好相反。FlashMLA 按 SM 代独立实现——sm90 decode、sm90 sparse fp8 decode、sm100 prefill、sm100 decode——每一个 SM 代都有自己完整的 `.cu` 文件。但更关键的数字是：v0.25.0 的 `vllm/v1/attention/backends/` 目录下有 **21 个通用 attention backend**，`mla/` 子目录另有 **18 个 MLA 专用后端**——`flashattn_mla.py`、`flashattn_mla_sparse.py`、`flashinfer_mla.py`、`flashinfer_mla_sparse.py`、`flashmla.py`、`flashmla_sparse.py`、`triton_mla.py`、`cutlass_mla.py`、`rocm_aiter_mla.py`……这不是冗余。这是承认注意力已经不再是一个可以用单一实现覆盖全部场景的问题。

## 6. 退役，不是重构

v0.25.0 的 Release Notes 把 PagedAttention 删除放在 Highlights，同时把 V1/MRv2 标记为 standard path。这不是「v1→v2」的平滑升级——这是两个迥异的注意力体系之间的切换。

|          | PagedAttention 时代                 | v0.25.0 的注意力体系               |
| -------- | ----------------------------------- | ---------------------------------- |
| 输入格式 | K 和 V 分离，五维张量               | 压缩表示，MLA 的 `[c_kv, k_rope]`  |
| 计算流程 | 两遍遍历（K 遍 + V 遍）             | 单遍融合                           |
| 精度模型 | 存储 FP8 → 计算 FP16/BF16           | 原生 FP8 计算路径                  |
| 后端数量 | 1 个（CUDA kernel，内置在 `_C` 中） | 21 个通用 backend + 18 个 MLA 变体 |
| 硬件定位 | sm_80+ 通用                         | sm_90+ 和 sm_100+ 独立优化         |
| 编译策略 | 模板参数笛卡尔积（~400 实例）       | 按 SM 代 + 精度组合（~20 实例）    |

V1/MRv2 成为 standard path 是一个关键信号。它意味着「每个模型有一个专属注意力后端」的模式已经从实验性走向生产就绪——不是由 vLLM 团队在为一个通用后端修修补补，而是由 FlashMLA、FlashInfer、AITER、Triton 各自提供最优实现，vLLM 做编排。

在这个体系里，保留一个不再被任何新模型路径使用的旧 kernel 没有意义。9 个文件，400 个 kernel 实例，每次变更都要确保它们不被破坏——而它们已经不再服务于任何生产模型。

对用户来说，唯一的变化是标准 MHA 模型（如 Qwen、Llama）的注意力计算自动路由到了 FlashAttention v3 或 FlashInfer 后端；DeepSeek 系列自动路由到了 FlashMLA 或 Triton MLA 后端。不需要任何配置变更——vLLM 根据模型架构和 GPU 型号在后端矩阵中自动选择。A100 用户跑标准 MHA 模型不受影响（FA2 支持 sm_80）；H100 用户自动获得 FA3/FA4 后端。唯一保留 PagedAttention 这个名字的代码在 `vllm/_custom_ops.py:120`——一个 ROCm 专用的 `paged_attention_rocm()` 函数，它在 AMD GPU 上仍然需要手动分页逻辑。除此之外，PagedAttention 的 CUDA 实现已经没有任何一行代码存在于 v0.25.0 的源码树中。

**It is time.**
