# KV 压缩推到极限：从架构改造到分层缓存

2026 年，DeepSeek-V4 与 Kimi K3 从架构层面重写了 attention，KV Cache 在 1M 上下文下从 250GB 量级降到个位数 GB，「KV Cache 是首要矛盾」这句旧判断随之作废。但压缩没有让问题消失，只是把它换了位置：单层的绝对量小到极点之后，杠杆从「每层压多少」转向「几层共用一份」；一个 token 的状态不再是一段连续张量，而是一组异构池，里面还混着不是 KV 的压缩器状态。

本目录是这个转折的三篇深挖，按发生的先后顺序排列。

> **和 [`kv_cache/01_concepts/compression/`](../kv_cache/01_concepts/compression/kv_cache_compression.md) 的分工**：那边讲的是**引擎侧的技术手段**（量化、稀疏化、淘汰），这边讲的是**模型架构侧把 KV 变成什么样**，以及这个形态变化对缓存系统提出了什么新要求。两边的结论不重叠。

## 内容导航

- **[01 · 当百万 Token KV Cache 从 250GB 降到 5GB](01-post-kv-cache-era.md)** — V4 与 K3 各自的稀疏化路线、1M 上下文下每 token 的 FLOPs 与 KV 体积账，以及旧优化技术在新架构下的位置重排（对照 vLLM/SGLang 源码交叉验证，逐条标注 ✓）。**从这里读起。**
- **[02 · 把 KV Cache 压缩推到极限](02-deepseek-v41-flash.md)** — DeepSeek-V4.1-Flash 技术报告精读：全局 KV 再压到 1/4、持久化压到 1/8，CED 架构、CSA2 跨层复用、单入口 mHC、Engram、FP4 main KV。它推翻了 01 的三处判断（报告 §章节与官方 `config.json` 对照）。
- **[03 · 七池与八池](03-multilevel-cache.md)** — 压缩之后的系统后果。以 SGLang HiCache 为对象，对照 V4 的七个池与 V4.1 的八个池（数量只差一个，组成几乎全换），拆三个结构性冲突与上游的对策，以及 V4.1 落在这套机制之外的两处例外（SGLang 源码逐条核对，引用均带 `文件:行号`）。

## 阅读顺序

01 → 02 → 03 是一条时间线，也是逐步收窄的视角：先是「架构变了，旧账要重算」，再是「这一代压到了什么程度」，最后是「缓存系统要跟着改什么」。三篇可独立读，但 02 与 03 的多处结论以 01 提出的问题为锚点。

## 相关链接

- [KV Cache 技术体系](../kv_cache/README.md)——引擎侧的全景导航（PagedAttention、Prefix Caching、淘汰、offloading、多级存储）
- [线性注意力与推理系统](../linear_attention/README.md)——另一半答案：干脆不要 KV Cache 的那条路线，以及它自己的系统难题
- [HBF 是 HBM 的替代吗](../hbf-vs-hbm.md)——算法把每步读取压下去之后，介质本身成了下一个变量
