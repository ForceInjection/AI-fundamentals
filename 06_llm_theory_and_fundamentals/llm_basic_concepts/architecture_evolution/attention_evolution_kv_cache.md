# Attention 演进与 KV Cache 之变：每 token 从 4.5 MiB 到 890 B

> 2026-09-20 | 覆盖 2020–2026 年 17 个代表性模型，全部字段对照官方 `config.json`、技术报告原文或推理引擎源码核对（标注于各节），核对方法与置信度见文末索引
>
> **口径说明**：文中「每 token KV」均按训练精度 2 字节/元素（fp16/bf16）计算，公式随各节给出；2026 年模型的池结构数字为 config + 池代码公式推算，未跑引擎实测。GPT-3 精度论文未声明（按 2 字节假设），单独标注。
>
> 相关文章：[七张图讲透 KV Cache](kv_cache_seven_charts.md)——本文的图解传播版；[GPT-2 到 Kimi K3 的注意力机制演进](from_gpt2_to_kimi_k3_attention_evolution.md)——同一时间线上注意力**计算侧**的演进，本文专注**存储侧**的那笔账；[KV 压缩推到极限三部曲](../../../09_inference_system/kv_compression/README.md)——2026 年序列压缩路线的深挖。

---

## 一、引言：注意力的存取结构，决定推理的成本结构

自回归生成有一个绕不开的依赖：生成第 t 个 token 时，注意力要对前 t−1 个 token 的 Key 和 Value 做检索。这些 K、V 每一步都要用，于是只有两个选择——每步重算所有历史 token 的 K/V 投影，或者算一次就存下来。

存下来的那份东西就是 KV Cache。它是推理显存里唯一**随上下文长度线性增长**的部分，模型权重反而是固定的。这就是为什么：

- 上下文越长，首字越慢、每 token 越贵；
- 一张 80 GB 的卡，装 8B 模型的权重绰绰有余，却装不下四五个 128K 上下文请求的 KV（每个 16 GiB，见 §二）；
- 批量越大，KV Cache 越先成为瓶颈。

过去七年，主流模型在注意力结构上的每一次大改——MQA、GQA、MLA、滑窗、稀疏、线性、压缩——都在回答同一个问题：**这份必须存下来的账，能不能变小**。本文沿这条时间线走一遍，每一站记三个数：存了什么、省了几倍、代价是什么。

这条线走到 2026 年，分成了三条路线。先给结论：**序列压缩把 KV 压到近常数，稀疏选择只省计算不省存储，线性混合干脆把 KV 换成固定状态**。三者的账完全不同，选型前必须分清。

---

## 二、MHA：基线，以及那笔最贵的账

### 2.1 手算一遍

用一个迷你模型把账建立起来：2 层、每层 2 个注意力头、head_dim = 4、fp16（2 字节）。每个 token 在每层每头要存 K 和 V 两组向量：

$$\text{每 token KV} = 2 \times n_{\text{layer}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times \text{dtype\_bytes} = 2 \times 2 \times 2 \times 4 \times 2 = \text{64 字节}$$

10 个 token 的上下文就是 640 字节。迷你模型无所谓，但这个公式后面会反复用到。

### 2.2 真实模型：多头的代价按头数线性复制

MHA（多头注意力）下每个头都有独立的 K、V，账按头数线性放大：

| 模型               | 层  | 头数 | head_dim | 每 token KV                 | 出处                                                  |
| ------------------ | --- | ---- | -------- | --------------------------- | ----------------------------------------------------- |
| GPT-3 175B（2020） | 96  | 96   | 128      | **2×96×96×128×2 = 4.5 MiB** | 论文 Table 2.1（arXiv:2005.14165，d_head 列显式给出） |
| Llama-2 7B（2023） | 32  | 32   | 128      | **512 KiB**                 | NousResearch/Llama-2-7b-hf config                     |

两个细节值得说明。其一，GPT-3 论文 §2.1 写明其注意力是「alternating dense and locally banded sparse attention patterns」——隔层带局部稀疏，但头结构上仍是 96 头全量 KV；讲 KV Cache 时它是 MHA，讲注意力模式时它不是纯 dense。其二，GPT-3 论文全文没有声明训练精度（全文 grep 无 fp16/bf16 字样，只提 V100），4.5 MiB 这个数按 2 字节/元素假设；若按 fp32 则是 9 MiB。GPT-3 权重从未公开，无 config 可查。

4.5 MiB/token 是什么概念：GPT-3 的上下文是 2048，一个请求的 KV 就要 2048 × 4.5 MiB ≈ **9 GiB**。模型还没到最大的规模，账已经付不起了。Llama-2 7B 的 512 KiB 看着小，乘上 128K 上下文同样是 64 GiB，恰好等于一张 80 GB 卡扣掉权重后的全部余量。

MHA 的问题就此定型：**每个头的独立性，是用每头一份完整 KV 买来的**。下面七年的演进，全是围绕「这份独立性值多少钱」做文章。

---

## 三、MQA 与 GQA：第一轮省法——共享

### 3.1 机制的谱系

既然账按 KV 头数线性走，最直接的省法是让多个 Q 头共享同一份 K/V。按共享程度排：

| 机制                 | 每 token KV（元素数） | 检索能力 | 出处                                                                   |
| -------------------- | --------------------- | -------- | ---------------------------------------------------------------------- |
| MHA                  | 2·n_h·d_h·l           | 强       | DeepSeek-V2 论文 Table 1（arXiv:2405.04434，该表按元素数计，不分精度） |
| GQA（n_g 组）        | 2·n_g·d_h·l           | 中       | 同上                                                                   |
| MQA（全部共享 1 组） | 2·d_h·l               | 弱       | 同上                                                                   |

MQA（采用者已核对：Falcon-7B 的 config 显式 `multi_query: true`；PaLM 无公开权重，但其技术报告架构清单列明「Multi-Query Attention」——key/value 投影共享为 [1,h]，并称对质量与训练速度中性）把账砍到极限，质量掉得也明显；GQA 取折中：8～32 个 Q 头共享一组 KV，成为 2023 年之后几乎所有开源模型的标准配置。

### 3.2 数字：70B 的账反而比 7B 小

GQA 时代的三个锚点（config 直接核对）：

| 模型        | 结构                         | 每 token KV                | 说明                                           |
| ----------- | ---------------------------- | -------------------------- | ---------------------------------------------- |
| Llama-2 70B | 80 层，64 Q 头 / **8 KV 头** | 2×80×8×128×2 = **320 KiB** | 同规模 MHA 等价配置为 2.5 MiB，GQA 省 **8 倍** |
| Mistral 7B  | 32 层，32 Q / 8 KV           | **128 KiB**                | config 另带 `sliding_window: 4096`，见 §五     |
| Llama-3 8B  | 32 层，32 Q / 8 KV           | **128 KiB**                | 128K 上下文 ≈ 16 GiB                           |
| Qwen2.5 7B  | 28 层，28 Q / 4 KV           | **56 KiB**                 | KV 头砍到 4，更激进                            |

最值得记的一笔对比：**Llama-2 70B（GQA）每 token 320 KiB，反而低于 Llama-2 7B（MHA）的 512 KiB**。头共享让 10 倍大的模型背上了更小的 KV 账。这就是 GQA 在 2023 年横扫开源界的原因。

### 3.3 代价

共享是有损的：KV 头数从 64 砍到 8，不同 Q 头被迫在同一个检索子空间里工作，质量有可感知的下降（各家的消融结论幅度不一，此处不给统一数字）。Qwen2.5 把 KV 头砍到 4、Llama-3 70B 系停在 8，就是各家在「账」与「质量」之间各自画的线。

另一个工程注脚：Llama 系 config.json 不含 `head_dim` 字段，需以 `hidden_size ÷ num_attention_heads` 推导（4096/32 = 128），五个经典模型里只有 Mistral 论文显式写了 head_dim。

---

## 四、MLA：第二轮省法——不共享，压缩

GQA 的思路是「少存几份」；DeepSeek 在 V2（2024）提出的 MLA 换了一条路：**把 K 和 V 投影到一个低秩的隐空间，只存隐向量**。

### 4.1 存的是什么

MLA 把每 token 的 K/V 压成一个 `kv_lora_rank`（d_c）维的隐向量 c^KV，另加一段为兼容 RoPE 而解耦出来的位置键（`qk_rope_head_dim` = d_h^R 维）。推理时只需要缓存这两样（V2 论文 §2.1.1/§2.1.3 原文：「only needs to cache c_t^KV」「the decoupled key … should be cached with c_t^KV together」）：

$$\text{每 token KV} = (d_c + d_h^R) \times n_{\text{layer}} \times \text{dtype\_bytes}$$

V2 论文 Table 1 的 caption 给出了换算：按 d_c = 4·d_h、d_h^R = d_h/2 的配置，**MLA 的账相当于只有 2.25 组的 GQA，但检索能力「Stronger」**。账面到了 MQA 的水平，能力保住 MHA 的档位。

### 4.2 数字

| 模型                | 层数 | 配置              | 每 token KV                  | 出处                                             |
| ------------------- | ---- | ----------------- | ---------------------------- | ------------------------------------------------ |
| DeepSeek-V2（236B） | 60   | d_c=512，d_h^R=64 | (512+64)×60×2 = **67.5 KiB** | config + 论文                                    |
| DeepSeek-V3（671B） | 61   | 同上              | (512+64)×61×2 = **68.6 KiB** | config（另含 1 层 MTP，启用投机解码时 +1.1 KiB） |

两笔横向对比。其一，同配置 MHA 的反事实账：2×128 头×128 维×60 层×2 字节 ≈ 3.8 MiB/token，**MLA 只保留了 1.76%**。其二，V2 论文摘要口径：相比 DeepSeek 67B，KV Cache 减少 93.3%、生成吞吐提升 5.76 倍（注意基准是 DeepSeek 67B，不是 Llama）。

671B 的 V3 和 7B 的 Llama-3 摆在一起：68.6 KiB 对 128 KiB。**大 84 倍的模型，KV 反而省一半**。模型规模和 KV 账从此脱钩。

### 4.3 代价

省账的钱花在了工程上。MLA 的吸收变换与 RoPE 解耦让注意力内核无法直接复用 GQA 那套实现，推理引擎要写专门的路径；排障时还要小心一个 config 陷阱：MLA 模型的 `num_key_value_heads` 字段是 Llama 系 schema 的遗留物（V2/V3 里都写着 128），**拿它算 KV Cache 会得出完全错误的结论**——实际缓存的从来不是按头的 K/V，是 576 维的隐向量。

---

## 五、滑窗与局部-全局混合：第三轮省法——少存历史

共享和压缩都在「每层存多少」上做文章；滑动窗口（SWA）换了个维度：**窗口外的 KV 根本不存**。

| 模型                | 结构                                          | 全局层每 token KV             | 滑窗层固定开销                              | 出处                                             |
| ------------------- | --------------------------------------------- | ----------------------------- | ------------------------------------------- | ------------------------------------------------ |
| Mistral 7B          | 全层 SWA 4096                                 | 128 KiB（窗口内）             | 长文总占用封顶 ≈512 MiB                     | config `sliding_window: 4096`                    |
| Gemma-3 27B（2025） | 62 层 = **52 滑窗（1024）+ 10 全局**          | 10 层 × 80 KiB = 80 KiB/token | 滑窗层 52 × 1024 × 8 KiB ≈ **416 MiB** 固定 | unsloth 镜像 config + transformers 源码推导      |
| gpt-oss 20B（2025） | 24 层 = **12 滑窗（128）+ 12 全局**，1:1 交替 | 12 层 × 2 KiB = 24 KiB/token  | 滑窗层合计仅 **3 MiB**                      | openai/gpt-oss-20b config `layer_types` 逐层清单 |

三个说明。第一，Gemma-3 的模式是「每 6 层 1 个全局」（`(i+1) % 6 == 0`），62 层落在 5.2:1 而不是整数 5:1；gpt-oss 则是严格的隔层交替。第二，滑窗层的开销是**常数**：gpt-oss 的窗口只有 128（且按 HF 实现含当前 token，即只看 127 个前文），12 个滑窗层合计 3 MiB，长上下文下可以忽略；账全部压在少数全局层上，于是全局层的 KV 头数可以给得很省（8 头 × 64 维）。第三，代价是**窗口外彻底丢失**——这不是压缩是丢弃，长文档问答里「看过但想不起」的错误多数来自这里。

混合架构的账因此变成两栏：少数全局层贡献随长度线性增长的部分，多数滑窗层贡献一笔固定开销。这个「两栏记账」的结构，是理解 2026 年混合模型的直接前置。

---

## 六、稀疏选择：DSA——存储照旧，计算省

2025 年 9 月的 DeepSeek-V3.2 引入了 DSA（DeepSeek Sparse Attention，arXiv 侧随 V3.2 报告发布）：每个 query 用一个轻量 indexer 从全部历史里选出 top-k 个 token（k=2048）做注意力。GLM 从 GLM-5 起采用同构结构，一路沿用到 5.3。

**这条路线的账要单独算，它最容易被误读。** 以 GLM-5.3（2026，`model_type: glm_moe_dsa`，753B，78 层）为例，config 逐字段核对如下：

| 字段         | 值                                                                                            |
| ------------ | --------------------------------------------------------------------------------------------- |
| MLA          | kv_lora_rank 512 + rope 64（同 DeepSeek V3 系形状），64 头                                    |
| DSA indexer  | 32 头 × 128 维，每 query 选 top-2048                                                          |
| 索引跨层复用 | `index_topk_freq=4`：78 层中 21 层 full、57 层 shared（头 3 层全量，此后每 4 层重算一次索引） |
| 滑窗         | 无                                                                                            |

三个容易写错的点（都是核对中实际踩到的）：

1. **KV 仍在线性增长。** 主 KV 每层还是 MLA 的 576 元素，78 层 bf16 约 87.8 KiB/token，加 indexer K（132 B/层）约 **98 KiB/token**；1M 上下文 60–103 GiB，和 V3.2 同量级。DSA 省的是**注意力计算和 indexer 开销**，存储一分没省。
2. **「每 4 层复用索引」不是压缩。** 那 57 个 shared 层不建 Indexer 模块但保留 top-k 槽位（SGLang `model_config.py:414-429`），复用的是「选哪些 token」这个决策，每层该存的 KV 照存。
3. **GLM-5.3 相对 GLM-5.2 注意力零改动。** 两代 config 逐字段一致，HF model card 原话「same base model as GLM-5.2 — every gain comes from post-training」。DSA 是 GLM-5 引入的，不是 5.3 的新东西。

稀疏路线的价值在于计算近常数化：1M 上下文里每个 query 只和 2048 个 token 做注意力。但把它写进「KV 变小」的叙事就是错的：第九节的三路线表里，它单独占一行。

---

## 七、线性注意力回归：把 KV Cache 换成固定状态

第五条路最激进：不再按 token 存任何东西，把历史压缩成一个**固定大小的循环状态**，每来一个 token 就地更新。Gated DeltaNet（GDN，Qwen 系）与 KDA（Kimi）是这条路的两个代表。

### 7.1 状态的大小与「两栏记账」

以 Qwen3-Next-80B（2025-09，48 层 = 36 GDN + 12 GQA，config `full_attention_interval=4`）为例：

- **GDN 层：零 token 级 KV。** 每层状态为 32 头 × 128 × 128 = 52 万元素，bf16 约 1 MiB/层，36 层共 36 MiB（fp32 姿态 72 MiB）——**按请求数计费，与上下文长度无关**。依据是状态池的容量公式 `num_layers × slots × num_heads × head_v_dim × head_k_dim`（SGLang `mamba_checkpoint_pool.py:281`）里没有 seq_len 因子。
- **12 个 GQA 层照常记账**：KV 头砍到 2、head_dim 256，每 token 24 KiB。

Kimi K3（2026，93 层）是同一思路的规模化：69 个 KDA 层 + 24 个 Gated MLA 层（23 组「3 KDA : 1 MLA」宏循环 + 末层额外 1 个 MLA；层数依据 config 的 `full_attn_layers`/`kda_layers` 逐层清单）。KDA 状态 96 头 × 128 × 128 ≈ 3 MiB/层，69 层共 207 MiB（bf16）；24 个 MLA 层贡献 24 × 1.125 KiB ≈ 27 KiB/token——1M 上下文下线性部分约 27 GiB，对照全 MLA 的 GLM-5.3 约 98 GiB，差出 3.6 倍。

### 7.2 代价

固定状态是**有损压缩的极限形态**：整个历史被摘要成一个定长向量，精确检索靠不了它。所以 K3 每 12 层插一次 AttnRes（注意力残差）、保留近四分之一的 MLA 层做完整上下文的 softmax 检索——**「线性记忆 + 周期性精确检索」是这条路线的标准形态**，纯线性模型至今没有成为主流旗舰。

另一个只属于这条路线的新瓶颈：状态池按请求数计费，**并发上限先于 KV 池到来**。GLM-5.3-Flash 的官方文档明说了这一点（`GLM-5.3-Flash.mdx:113`），这在第七节之前是不存在的问题。

---

## 八、序列压缩：DeepSeek 的 CSA/HCA 与 V4.1

第六条路保留「存」但改变「存什么粒度」：不存逐 token 的 KV，存**按块压缩后的条目**。这是 DeepSeek-V4 系（2026）的路线。

### 8.1 DeepSeek-V4：一个层两种压缩率 + 一层滑窗

V4 的每一层是两条分支：global 分支按压缩率交错配置 CSA（压缩稀疏注意力，4 token 压成 1 条 entry）与 HCA（重度压缩，128:1），外加一路滑窗 128 的未压缩注意力（头两层只有 SWA）。V4-Pro 的 61 层 = 30 个 c4a + 31 个 c128a（config `compress_ratios`，经本地 02 篇对照核对）。

压缩后的账（SGLang `deepseek_v4_memory_pool.py:216-218` 的布局断言，本轮独立复核）：

```text
压缩 entry = 584 B（448 nope FP8 + 128 rope BF16 + 8 scale）
折合每原始 token：C4 层 ≈ 146 B/层；C128 层 ≈ 4.6 B/层
indexer K：132 B/token/层（FP8）
```

再加未压缩的 SWA 分支和 indexer，V4-Pro 在 1M 上下文约 **9.62 GiB/序列**（vLLM 博客口径，转引须注明）。存储结构因此异构化：SGLang 为它建了**七个池**（kv/swa/c4/c4_indexer/c128/c4_state/c4_indexer_state），一个 token 的状态不再是连续张量。

### 8.2 V4.1-Flash：890 B/token

V4.1-Flash（2026-09）把这条路线推到当前终点：40 层切两半（前 20 层含 2 层纯 SWA + 18 层 CSA2 m=2，后 20 层 CSA2 m=1），38 个 CSA2 层里只有 **8 层真干活**——4 层 Full 生产 main KV、4 层 Reindex 重打分、30 层直接复用。main KV 走 MXFP4。结果是 **global KV 890 B/token，为 V4-Flash 的 1/4**；持久化 KV 为 1/8。池结构随之变成八个（c1/c2 及各自 indexer/scale 池），V4 的状态池整类消失。

DeepSeek 自家的历代账（V4.1 报告 Figure 1(b) 口径，见本地 02 篇）：

```text
V1 389,120 B → V3.2 48,068 B → V4-Flash 3,514 B → V4.1 890 B
```

从 V1 到 V4.1，每 token 的账缩了 **437 倍**。代价同样是工程性的：跨层复用让「一个 token 的状态」碎成八个异构池，前缀缓存的命中判定、多级存储的备份粒度都要跟着重写（引擎侧的连锁反应见第十一节）。

---

## 九、2026 年的三条路线

把第八节的事实摆在一起，2026 年的旗舰注意力分成三条路线。**它们的账不可直接互相比较**——「每 token KV」对三条路线的含义都不同：

| 路线                                 | KV 随上下文                 | 精确检索                  | 代表（本文已核对）                         |
| ------------------------------------ | --------------------------- | ------------------------- | ------------------------------------------ |
| **序列压缩**（CSA/HCA/CSA2）         | 近常数，池异构化            | 全量（压缩 entry 上检索） | DeepSeek-V4、V4.1-Flash                    |
| **稀疏选择**（MLA + DSA）            | **线性**（计算近常数）      | top-k 内精确              | GLM-5.3（=V3.2 同构）                      |
| **线性混合**（GDN/KDA + 少数全局层） | 少数层线性 + 每请求固定状态 | 周期性精确（全局层）      | Kimi K3、GLM-5.3-Flash、Qwen3.8-Flash-Next |

1M 上下文的账各算各的：GLM-5.3 约 60–103 GiB（线性路线的常态）；V4-Pro 约 9.6 GiB；V4.1-Flash 的 global KV 仅 0.9 GiB 量级；Kimi K3 的线性部分约 27 GiB，但每个并发请求背 207–414 MiB 的固定状态。

两条路线之外的变量还在进场：GLM-5.3-Flash 把 indexer K 也按 4-token 池压缩（`index_kpool=4`），Qwen3.8-Flash-Next 的 QSA 选择每层独立压缩索引、刻意不做跨层复用。压缩的粒度和复用的边界，仍是各家的活跃分歧。

---

## 十、总表

17 个锚点模型的完整账目（bf16/fp16 口径，2 字节/元素；混合模型分别列线性部分与固定开销）：

| 模型                | 年代    | 机制                  | 每 token KV                        | 固定开销          |
| ------------------- | ------- | --------------------- | ---------------------------------- | ----------------- |
| GPT-3 175B          | 2020    | MHA（交替稀疏模式）   | ≈4.5 MiB（精度未验证，按 2B 假设） | —                 |
| Llama-2 7B          | 2023    | MHA                   | 512 KiB                            | —                 |
| Llama-2 70B         | 2023    | GQA ×8 组             | 320 KiB                            | —                 |
| Mistral 7B          | 2023    | GQA + SWA 4096        | 128 KiB                            | 窗口封顶 ≈512 MiB |
| Llama-3 8B          | 2024    | GQA                   | 128 KiB                            | —                 |
| Qwen2.5 7B          | 2024    | GQA（4 组）           | 56 KiB                             | —                 |
| DeepSeek-V2         | 2024    | MLA                   | 67.5 KiB（同配 MHA 的 1.76%）      | —                 |
| DeepSeek-V3         | 2024-12 | MLA                   | 68.6 KiB                           | —                 |
| Gemma-3 27B         | 2025    | 52 SWA + 10 全局      | 80 KiB（全局层）                   | 416 MiB           |
| gpt-oss 20B         | 2025    | 12 SWA + 12 全局      | 24 KiB（全局层）                   | 3 MiB             |
| Qwen3-Next 80B      | 2025-09 | 36 GDN + 12 GQA       | 24 KiB（GQA 层）                   | 36–72 MiB         |
| Kimi K3             | 2026    | 69 KDA + 24 Gated MLA | 27 KiB（MLA 层）                   | 207–414 MiB       |
| DeepSeek-V4-Pro     | 2026    | CSA/HCA/SWA 七池      | @1M ≈ 9.62 GiB/序列                | —                 |
| DeepSeek-V4.1-Flash | 2026-09 | CED/CSA2/FP4 八池     | 890 B                              | —                 |
| GLM-5.3             | 2026    | MLA + DSA（稀疏路线） | ≈98 KiB（fp8 ≈59 KiB）             | —                 |
| GLM-5.3-Flash       | 2026    | 34 KDA + 11 MLA+DSA   | ≈5.9 KiB                           | 68–136 MiB        |
| Qwen3.8-Flash-Next  | 2026-09 | 36 GDN + 12 QSA       | ≈12 KiB + 768 B indexer            | GDN 状态按请求计  |

未验证项如实声明：DeepSeek-V4 主仓 config 在 HF 为 gated（结论依赖 V4-Pro config 与池源码的既有核对）；GPT-3 训练精度论文未声明；Gemma-3 官方仓库 gated，config 取自 unsloth 镜像（未与原件比对）；GLM-5.3 活跃参数量 model card 未标注；2026 系的 per-token 数字为 config + 池代码公式推算，未跑引擎实测。

---

## 十一、存储侧的账，把推理引擎逼成了什么样

模型侧每省一段账，引擎侧就多一类问题。这条因果链在 2026 年的三条路线上各留了一处印记：

1. 池的异构化。序列压缩把「一个 token 一段连续 KV」打碎成七个池（V4）、八个池（V4.1），其中混着不是 KV 的 indexer 与压缩器状态。前缀缓存的命中判定、备份粒度、逐出策略都要按池重写（SGLang HiCache 的应对见[分层缓存一篇](../../../09_inference_system/kv_compression/03-multilevel-cache.md)）。
2. 命中判定的多边界化。滑窗与压缩并存后，Full 可复用的范围与 SWA 可复用的范围出现差值，前缀缓存第一次要维护两个边界——SGLang 的 SWA 分叉点缓存（v0.5.20，命中率 43.8%→60.8%，官方口径）正是为这个缺口而生（源码解析见[分叉点缓存一篇](../../../09_inference_system/sglang/sglang-swa-branching-point-cache.md)）。
3. 并发上限的迁移。线性混合路线的状态按请求计费，GLM-5.3-Flash 官方文档明确「KDA 状态池会先于 KV 池限制并发」——容量规划的公式从「上下文 × 每 token KV」变成两条公式取小。

引擎的支持门槛也成了选型变量：GLM-5.3-Flash 与 Qwen3.8-Flash-Next 的支持分别在 SGLang v0.5.20 才落地（前者还要求专用镜像）。

---

## 十二、config 索引与参考

### 模型 config（HuggingFace raw，2026-09-20 抓取核对）

| 模型                    | config 地址                                                               |
| ----------------------- | ------------------------------------------------------------------------- |
| Llama-2 7B/70B          | `NousResearch/Llama-2-7b-hf`、`NousResearch/Llama-2-70b-hf`（官方 gated） |
| Mistral 7B              | `mistralai/Mistral-7B-v0.1`                                               |
| Llama-3 8B              | `unsloth/llama-3-8b`（官方 gated）                                        |
| Qwen2.5 7B              | `Qwen/Qwen2.5-7B`                                                         |
| DeepSeek-V2 / V3        | `deepseek-ai/DeepSeek-V2`、`deepseek-ai/DeepSeek-V3`                      |
| Gemma-3 27B             | `unsloth/gemma-3-27b-it`（官方 gated）                                    |
| gpt-oss 20B             | `openai/gpt-oss-20b`                                                      |
| Qwen3-Next 80B          | `Qwen/Qwen3-Next-80B-A3B-Instruct`                                        |
| Kimi K3                 | `moonshotai/Kimi-K3`                                                      |
| GLM-5.3 / GLM-5.3-Flash | `zai-org/GLM-5.3`、`zai-org/GLM-5.3-Flash`                                |
| Qwen3.8-Flash-Next      | `Qwen/Qwen3.8-Flash-Next`                                                 |

### 论文与技术报告

- GPT-3：arXiv:2005.14165（Table 2.1；§2.1 稀疏交替模式）
- PaLM：arXiv:2204.02311（架构清单「Multi-Query Attention」；无公开权重，经论文原文核对）
- Falcon：`tiiuae/falcon-7b` config `multi_query: true`（40B config 无此字段，HF FalconConfig 默认值为 true，即同样 MQA）
- DeepSeek-V2：arXiv:2405.04434（§2.1.1/2.1.3 MLA 缓存式；Table 1 及 caption）
- DeepSeek-V3：arXiv:2412.19437（§2.1.1）
- DeepSeek-V4 / V4.1-Flash：arXiv:2606.19348 及 V4.1 报告（config 双向核对见[本地 02 篇](../../../09_inference_system/kv_compression/02-deepseek-v41-flash.md)）
- Kimi K3：arXiv:2607.24653
- GLM-5 系：arXiv:2602.15763；SGLang cookbook `GLM-5.3.mdx`、`GLM-5.3-Flash.mdx`、`Qwen3.8-Flash-Next.mdx`

### 引擎源码（SGLang，`c475ac5eaf`）

- MLA 缓存维度：`python/sglang/srt/models/deepseek_v2.py`（`kv_a_proj_with_mqa` 输出 576 维）
- DSA 索引复用：`python/sglang/srt/configs/model_config.py:414-429`、`:187-216`
- 线性状态池公式（无 seq_len 因子）：`python/sglang/srt/mem_cache/mamba_checkpoint_pool.py:281`
- Qwen3-Next 层型推导：`python/sglang/srt/configs/qwen3_next.py:259-269`
- V4 池布局断言：`python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py:216-218`、`:39-43`、`:730`
- QSA 每 token 字节：`python/sglang/srt/mem_cache/qsa_kv_pool.py:39-48`

### 本地关联文章

- [GPT-2 到 Kimi K3 的注意力机制演进](from_gpt2_to_kimi_k3_attention_evolution.md)——计算侧同时间线
- [KV 压缩推到极限三部曲](../../../09_inference_system/kv_compression/README.md)——序列压缩路线的深挖（V4/V4.1 七池八池、多级缓存）
- [SGLang 分叉点缓存源码解析](../../../09_inference_system/sglang/sglang-swa-branching-point-cache.md)——滑窗与压缩并存后，引擎侧的命中判定改造
