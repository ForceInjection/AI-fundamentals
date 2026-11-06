# KV 压缩：论文专题与架构重构

本目录收两个互补的子系列。**01–04 是 FAST'26/NeurIPS/AAAI 四篇论文的精读单元**：压缩不只是把存储账做小，压缩产物本身会成为后续决策的输入（该读谁、该算谁、丢了怎么办）。**05–07 是一条架构侧时间线**：DeepSeek-V4 与 Kimi K3 从架构层面重写 attention 之后，KV Cache 在 1M 上下文下从 250GB 量级降到个位数 GB，V4.1-Flash 再压到 1/4——压缩改掉的不只是体积，还有 KV 的形态本身。

> **分工说明**：引擎侧的技术手段（量化、稀疏化、淘汰）在 [`01_concepts/06_compression/`](../01_concepts/06_compression/01_kv_cache_compression.md)；本目录讲压缩的**决策价值**（论文四篇）与**模型架构侧把 KV 变成什么样**（时间线三篇）。三边结论不重叠。

## 1. 论文专题：压缩表示的二次决策（01–04）

- **[01 · 当压缩表示被继续使用：三篇论文给出的三种二次决策](01_kv-compression-overview.md)** — 三篇正式录用论文的横切总纲，按条件选路的对号表与「数字怎么读」口径清单。
- **[02 · Self-Indexing KVCache](02_self-indexing.md)**（AAAI '26）— 压缩码兼任稀疏检索索引。
- **[03 · SALS](03_sals.md)**（NeurIPS 2025）— pre-RoPE 潜空间选 token。
- **[04 · EchoKV](04_echokv.md)**（NeurIPS 2026）— 丢弃维度 + 轻量重建。
- 精读报告（Agent 逐条页码核对）：[Self-Indexing](references/self-indexing-report.md) · [SALS](references/sals-report.md) · [EchoKV](references/echokv-report.md)；论文原文 PDF 在 `references/papers/`。

## 2. 架构侧：KV 被推到极限之后（05–07）

2026 年，杠杆从「每层压多少」转向「几层共用一份」；一个 token 的状态不再是一段连续张量，而是一组异构池，里面还混着不是 KV 的压缩器状态。三篇按发生的先后顺序排列。

- **[05 · 当百万 Token KV Cache 从 250GB 降到 5GB](05-post-kv-cache-era.md)** — V4 与 K3 各自的稀疏化路线、1M 上下文下每 token 的 FLOPs 与 KV 体积账，以及旧优化技术在新架构下的位置重排（对照 vLLM/SGLang 源码交叉验证，逐条标注 ✓）。**从这里读起。**
- **[06 · 把 KV Cache 压缩推到极限](06-deepseek-v41-flash.md)** — DeepSeek-V4.1-Flash 技术报告精读：全局 KV 再压到 1/4、持久化压到 1/8，CED 架构、CSA2 跨层复用、单入口 mHC、Engram、FP4 main KV。它推翻了 05 的三处判断（报告章节与官方 `config.json` 对照）。
- **[07 · 七池与八池](07-multilevel-cache.md)** — 压缩之后的系统后果。以 SGLang HiCache 为对象，对照 V4 的七个池与 V4.1 的八个池（数量只差一个，组成几乎全换），拆三个结构性冲突与上游对策，以及 V4.1 落在这套机制之外的两处例外（SGLang 源码逐条核对，引用均带 `文件:行号`）。
- Engram（条件记忆 / N-gram Embedding）是 V4.1-Flash 的配套机制但主题独立，专文见 [`../../engram/`](../../engram/README.md)。

## 阅读顺序

论文四篇（01 → 04）与时间线三篇（05 → 07）各自成列、可独立读。交叉点在：论文们压的是「压缩表示怎么用」，时间线压的是「KV 本身的形态」——读完后对「压缩」这个词的三层含义（引擎手段、决策输入、架构重构）会有完整的分层。

## 相关链接

- [KV Cache 技术体系](../README.md)——引擎侧的全景导航（PagedAttention、Prefix Caching、淘汰、offloading、多级存储）
- [线性注意力与推理系统](../../linear_attention/README.md)——另一半答案：干脆不要 KV Cache 的那条路线，以及它自己的系统难题
- [HBF 是 HBM 的替代吗](../../hbf-vs-hbm.md)——算法把每步读取压下去之后，介质本身成了下一个变量
