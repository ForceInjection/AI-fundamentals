# 推理量化技术基础：FP8、INT8 与 FP4 的格式、粒度与算法

> 2026-07-31 | FP8 规范 (NVIDIA/ARM/Intel, 2022)、AWQ/GPTQ/SmoothQuant 论文分析

量化是 LLM 推理中最"便宜"的优化——不改模型结构、不增加层数、不需重新训练（PTQ 路径），只降低每个参数的位宽，就能将显存占用和内存带宽需求减半甚至更多。但其代价是精度损失——同样的数值用更少的比特表示，必然引入误差。量化技术的全部复杂性，都集中在**如何用最少的精度损失换最多的位宽节省**。

本文覆盖推理量化的四个基础问题：**FP8 格式本身怎么设计（E4M3 vs E5M2）**、**量化的粒度如何影响精度（per-tensor/per-token/per-group）**、**权重量化和 KV Cache 量化为何难度不同**、以及**三种主流权重量化算法各自怎么解决精度问题（SmoothQuant/AWQ/GPTQ）**。

> 已有专文覆盖 KV Cache 量化的工程实现（[KV Cache 量化深度解析](../kv_cache/01_concepts/compression/kv_cache_quantization.md)），本文聚焦于**通用量化原理和权重量化算法**，仅在与权重量化做对比时涉及 KV Cache。

---

## 一、FP8 格式：E4M3 与 E5M2 的分工

### 1.1 IEEE 754 不够用

标准的 FP16（1 符号 + 5 指数 + 10 尾数）和 BF16（1+8+7）已经是"压缩过的浮点数"。FP8 将其再次减半——只有 8 位，必然在指数和尾数之间做出比 FP16 更激进的选择。

### 1.2 两种 FP8：E4M3 和 E5M2

2022 年，NVIDIA、ARM、Intel 联合提出了两种 FP8 格式，按使用场景分工：

| 格式     | 符号位 | 指数位 | 尾数位 | 最大正规值 | 最小正规值 | 核心场景                 |
| -------- | ------ | ------ | ------ | ---------- | ---------- | ------------------------ |
| **E4M3** | 1      | 4      | 3      | 448        | 2⁻⁶        | **forward 的权重和激活** |
| **E5M2** | 1      | 5      | 2      | 57344      | 2⁻¹⁴       | **backward 的梯度**      |

为什么不是一种格式覆盖全部？

- **Forward 需要精度**：权重和激活的张量分布通常集中在 0 附近，4 位指数范围（最大 448）已经足够覆盖。3 位尾数提供足够的精度来区分相近的数值。
- **Backward 需要范围**：梯度在反向传播中可能极端缩放——某些层梯度极小（10⁻⁶），某些极大（10²）。5 位指数（最大 57344）确保梯度不溢出或下溢。精度在 backward 中不重要——只需足够大的指数范围来捕获梯度量级。

实际部署中，推理只涉及 forward（无训练 backward），所以**E4M3 是推理场景的主力格式**。vLLM 的 `--kv-cache-dtype fp8` 和 `--kv-cache-dtype fp8_e4m3` 均使用 E4M3。

### 1.3 FP8 的动态范围代价

E4M3 的动态范围（最大 448）远小于 FP16（最大 65504）和 BF16（最大 3.39×10³⁸）。这意味着无法直接用 FP8 表示 outlier 值——那些在 FP16 下合法的"极端大的 activation 值"在 E4M3 下会直接溢出到无穷。这正是 per-tensor 量化在 E4M3 下精度不足的根本原因：一个 scale 无法同时保护 outlier 和正常值。

---

## 二、量化的粒度：用存储换精度

量化的本质是计算 `x_quant = round(x / scale) × scale`。scale 是核心——它决定了每个量化等级覆盖多大的浮点范围。**多少数据共享一个 scale**，就是量化的粒度。

### 2.1 四种粒度

| 粒度            | scale 覆盖范围            | scale 数量（以 4096×32×128 的 K 为例） | 额外存储               | 精度                         |
| --------------- | ------------------------- | -------------------------------------- | ---------------------- | ---------------------------- |
| **per-tensor**  | 整个张量                  | 1                                      | ~忽略不计（一个 FP32） | 最低，outlier 污染全局       |
| **per-channel** | 每个 head/输出通道        | 32                                     | ~128 bytes             | 中等，跨 token 差异未处理    |
| **per-token**   | 每个 token 的整行         | 4096                                   | ~16 KB                 | 较好，token 间分布差异被吸收 |
| **per-group**   | 每组 g 个元素（如 g=128） | 4096×32×128/128 ≈ 131K                 | ~0.5 MB                | 最好，但存储开销显著         |

每一级粒度用更大的 scale 存储换取更低的量化误差。对于 1M 上下文的极限场景，per-tensor 用 4 bytes 换 50% 压缩率但精度损失严重，per-group 精度最优但 0.5 MB 的额外 scale 存储可能抵消压缩收益。实际部署中，**per-tensor（FP8 E4M3）是当前生产环境的主流折中**——精度可接受、实现简单、额外存储可忽略。

### 2.2 动态量化 vs 静态量化

- **静态量化**：scale 在推理前离线计算（calibration），推理时直接用固定 scale。要求 calibration 数据覆盖推理分布——分布外输入可能导致精度崩溃。
- **动态量化**：scale 随输入实时计算。理论上更鲁棒，但每步 forward 都需额外扫描 tensor 找 max——增加的延迟在小 batch 场景下可能抵消量化的收益。

权重量化通常用静态（权重在部署后不变），激活/KV Cache 量化可用静态（E4M3 per-tensor，通过 calibration 确定 scale）或动态。

---

## 三、权重量化 vs 激活量化 vs KV Cache 量化

量化在推理流程中作用于不同的对象，它们的量化难度不同：

| 量化对象     | 典型精度 | 难度 | 原因                                                            |
| ------------ | -------- | ---- | --------------------------------------------------------------- |
| **权重**     | INT4/FP8 | 中   | 静态、分布稳定、可离线 calibration；但 outlier 通道需要特殊处理 |
| **激活**     | FP8      | 高   | 动态变化、分布随输入漂移；outlier 值常见（LLaMA 类模型）        |
| **KV Cache** | FP8/INT8 | 中高 | 类似激活但长期累积；量化误差经过 SoftMax 指数放大               |

这三种量化的难度差异源于一个根本原因：**SoftMax 的指数放大效应**。权重量化误差在多层 forward 平摊，KV Cache 量化误差直接在 attention 的指数核内放大。已有文章（[KV Cache 量化深度解析](../kv_cache/01_concepts/compression/kv_cache_quantization.md)）详细拆解了 KV Cache 量化的 per-tensor/per-token-head 模式，这里不再重复。

---

## 四、三种权重量化算法：SmoothQuant、AWQ、GPTQ

权重量化的核心问题是：**LLM 的权重中有个别通道（channel）值异常大——outlier 通道。** 这些 outlier 在 INT8/INT4 的有限范围内被截断（clipping），引入的误差远大于其他通道。三种算法从不同角度解决这个问题。

### 4.1 SmoothQuant：将量化难度从权重转移到激活

SmoothQuant（Xiao et al., 2023）的核心洞察：weight 的 outlier 通道和 activation 的 outlier 通道是**对偶**的——weight 中 outlier 大的通道，activation 中该通道的值通常较小。

算法的做法：在 forward 前，对每个通道 j 计算一个平滑因子 `s_j = max(|W_j|)ᵃ / max(|X_j|)¹⁻ᵃ`（α 是迁移强度，通常 0.5），然后将 `W_j` 缩小 `s_j` 倍，`X_j` 放大 `s_j` 倍——保证数学等价（`W_j·X_j` 不变），同时将量化难度从 outlier 多的 weight 通道转移到分布更均匀的 activation 通道。

```text
变换前: Y = W · X           ← W 有 outlier 通道，精度损失大
变换后: Y = (W / s) · (X · s)  ← 数学完全等价，但 W/s 分布均匀，X·s 范围稍大
```

效果：per-tensor INT8 量化权重和激活，精度接近 FP16。这是当前 vLLM、TensorRT-LLM 等框架中 **W8A8 量化的理论基础**。

### 4.2 AWQ：用 per-channel scaling 保护显著通道

AWQ（Activation-aware Weight Quantization, Lin et al., 2023）从另一个角度切入：不是所有通道的量化误差对最终输出影响相同——**激活值越大的通道，该通道权重的量化误差对点积结果的影响越大。**

AWQ 的做法：根据激活值的量级为每个权重通道计算一个 per-channel scaling 因子 `s = s_X^α`（其中 `s_X` 是激活的平均幅值，α ∈ [0,1] 通过最小化层输出误差搜索得到）。放大显著通道的权重使其远离量化截断区，等效于保护这些通道免受量化噪声。缩放因子的逆操作被融合进前置层，推理时无额外开销。注意：与常见的误解不同，AWQ **不保留 1% 的 FP16 通道**——那只是论文中的消融实验验证思路，实际方案是纯 per-channel scaling，保持统一精度以兼容硬件。

```text
关键洞察: Y_j = Σ W_{jk}·X_k → dY_j/dW_{jk} = X_k
  X_k 大的通道 → 该通道的量化误差对输出影响大 → 用更大的 scale 保护
```

效果：INT4 权重量化 + FP16 activation，精度退化极小。这是 **W4A16 量化的主流方案**，显存节省 4×（权重从 FP16 的 2 bytes → INT4 的 0.5 bytes），而计算时反量化回 FP16。

### 4.3 GPTQ：逐层最优的权重量化

GPTQ（Frantar et al., 2023）用更数学化的方法解决 outlier 问题。核心思想源于 1990 年代的 Optimal Brain Quantization（OBQ）算法：**对每一层，逐列选择量化误差最小的列来量化，量化一列后立即补偿剩余列的权重以消除累积误差。**

GPTQ 的简化：OBQ 的复杂度是 O(d⁴)（d 为权重矩阵大小），无法直接用于 LLM。GPTQ 的改进是将权重矩阵分块（block-wise），每次量化 128 列，块内同步补偿——复杂度降到 O(d²)，可以处理 70B+ 模型。

不同于 SmoothQuant（转换权重分布→量化）和 AWQ（选择性保护重要通道），GPTQ **不需要分析激活值**——它通过逐列量化 + 误差补偿，在数学上保证量化后的权重点积与原权重点积的最小二乘误差。但它需要校准数据集来准确评估量化误差。

三者的工程定位：

| 算法            | 量化精度   | 关键需求              | 典型场景                              |
| --------------- | ---------- | --------------------- | ------------------------------------- |
| **SmoothQuant** | W8A8       | 校准集（小，~512 条） | 对延迟敏感、低精度损失、即插即用      |
| **AWQ**         | W4A16      | 校准集（小）          | 显存极度受限、需要最大压缩（4× 权重） |
| **GPTQ**        | W4A16/INT4 | 校准集（中）          | 追求最优精度-压缩比、可接受校准成本   |

---

## 五、总结

推理量化的四个关键设计选择：

- **格式**：E4M3（forward，精度优先）vs E5M2（backward，范围优先）。推理用 E4M3。
- **粒度**：per-tensor（最小开销，容差最低）→ per-channel → per-token → per-group（最大开销，精度最高）。实际生产主流是 per-tensor FP8。
- **对象**：权重量化（静态、最成熟，SmoothQuant/AWQ/GPTQ 各有解法）、KV Cache 量化（动态、受 SoftMax 放大影响，详见已有文章）、激活量化（最难，outlier 问题最严重）。
- **算法**：SmoothQuant（迁移量化难度，W8A8，最轻量）、AWQ（per-channel scaling 保护显著通道，W4A16，压缩率最大）、GPTQ（逐列补偿误差，W4A16，精度最优但需校准）。

量化的终极权衡是**精度 vs 带宽**。FP16→FP8 减半带宽、精度损失通常在 1% 以内（SmoothQuant 等方法的加持下），是当前"应默认开启"的基础设施级优化。FP16→INT4 再减半、精度损失需要 AWQ/GPTQ 的算法补偿，是"显存极端受限时的选择"。

---

## 延伸阅读

- [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433) (Micikevicius et al., 2022) — E4M3/E5M2 格式规范
- [SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs](https://arxiv.org/abs/2211.10438) (Xiao et al., 2023)
- [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978) (Lin et al., 2023)
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) (Frantar et al., 2023)
- [KV Cache 量化深度解析](../kv_cache/01_concepts/compression/kv_cache_quantization.md) — per-tensor/per-token-head 模式与 vLLM 实现
- [NVIDIA Model Optimizer 技术详解](nvidia_model_optimizer.md) — 量化工具链
