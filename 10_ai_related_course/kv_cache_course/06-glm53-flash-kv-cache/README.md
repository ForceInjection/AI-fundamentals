# 第 6 讲：KV Cache 在 GLM-5.3-Flash 里怎么变成三本账

> 120 分钟 | 课程第 6 讲 | 前置：第 1、2、5 讲

GLM-5.3-Flash 有两个塔：45 层文本塔，24 层 vision encoder。本讲说的 45 层、34 层 KDA、11 层 full attention，都只指文本塔；vision encoder 不属于这三本账，也不要把 45 当成模型总层数。

文本塔的 45 层都带注意力，但类型不同。34 层是 KDA（linear attention），11 层是 full attention。这里说的 full 是层类别，config 里写作 `deepseek_sparse_attention`；它的注意力算法是 DSA，KV 的存储布局是 MLA latent。DSA 和 MLA 是同一批 11 层的两个侧面，不是两种 full 层。

缓存因此分成三本账：11 层 MLA latent KV 按 token 计，11 层 DSA indexer 也按 token 计，34 层 KDA state 按 state slot 计。三本账的单位不同，不能合成一个“每 token 多少字节”。

> 文章口径：单 rank、attn_tp=1、scaled FP8、64-token page 预留、无 MTP。下文写「文章口径」都指这组条件；换条件要按同一公式重算。scaled FP8 只约束 11 层 latent KV 和 indexer，KDA state 的 dtype 单独看 conv/temporal。单位：MB = 10^6 B，不是 MiB。147.6 MB ≈ 140.8 MiB，738 MB ≈ 703.8 MiB（按精确字节是 703.9 MiB，差在舍入）。

默认 HBM 里只有一个 `HybridLinearKVPool`；`full_kv_pool` 和 `mamba_pool` 是它的两个子池。所谓双预算，是调度器对这两个子池的容量约束，不是两个池对象。

KDA 的记账方式和 full 层不同。每层维护一个固定大小的循环状态 `S_t = f(S_{t-1}, x_t)`，由 short conv 的卷积窗口状态和 temporal state 两部分组成。形状只由 heads、head_dim 和 conv kernel 决定，与 token 数无关：序列从 10 个 token 增到 10,000 个 token，state 的形状和字节数不变，变的是 11 层 full 的 token KV 和 indexer。所以一个请求在每个 KDA 层只需要一份 state，34 层的 state 合起来占一个 state slot 的 147.6 MB（文章口径），不需要每 token 存一份 KV。

三个词先定义：

- token slot：token 级 KV 和 indexer 在 HBM 里的位置。
- state slot：KDA state 在 HBM 里的位置；一个 slot 包含 34 层的 conv state 和 temporal state。
- compress tail：请求级尾缓冲。`index_kpool_compress=True` 时，每个请求用一段环形缓冲保存 compress 边界附近的 pooled index K 和 score，按 rank 整块分配，不进 HiCache。

HiCache 的覆盖范围也要限定：本模型 CUDA hybrid 路径的 host pool 只有 KV 和 MAMBA 两个 entry，indexer 和 compress tail 不进 HiCache（retraction 的临时 CPU copy 另算）。bare DSA 路径会给 indexer 建 `PoolName.INDEXER` sidecar；NPU 的 `page_first_kv_split` 只改 Mamba host layout（强制 `page_first_direct`），不要套到本模型的 CUDA 路径。

## 学习目标

- 说清文本塔 45 层的构成：34 层 KDA、11 层 full；full 的 DSA 是注意力算法，MLA 是 KV 布局。
- 用 `S_t = f(S_{t-1}, x_t)` 解释 KDA 的 state 为什么与 token 数无关。
- 分别列出三本账的单位：B/token、MB/state slot、MB/request。
- 复算 7,260 B/token 和 147.6 MB/state slot（文章口径），并说清 147.6 是 34 层合计。
- 画出 `HybridLinearKVPool → full_kv_pool / mamba_pool → kv_buffer / IndexKeyCache` 的对象层级。
- 从启动日志的 `max_total_num_tokens` 与 `max_mamba_cache_size` 判断并发先被谁限制。

## 本讲产出

一张三本账容量表、一张三级对象图，以及一句“先爆哪本账”的判断。表里的数字要带口径和单位。

## 目录

- [课程总览](../README.md)
- [教学大纲](syllabus.md)
- [PPT 提纲](ppt-outline.md)
- [课堂实验](hands-on-exercise.md)
- [课后作业](homework.md)
- [实验环境](lab-environment.md)

## 前后衔接

第 5 讲讲了通用前缀复用与分层；本讲看混合注意力如何把缓存拆成不同对象。第 7 讲换到 DeepSeek-V4.1-Flash，看另一条压缩路线。
