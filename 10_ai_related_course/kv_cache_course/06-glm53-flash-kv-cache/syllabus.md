# 第 6 讲：KV Cache 在 GLM-5.3-Flash 里怎么变成三本账 — 教学大纲

## 基本信息

| 项目 | 内容 |
| --- | --- |
| 课时 | 120 分钟 |
| 先修 | 第 1 讲 KV 公式；第 2 讲 block/page；第 5 讲 RadixTree / HiCache |
| 代码基线 | 本地 SGLang，HEAD `f4de6abee6` |
| 模型 config | [zai-org/GLM-5.3-Flash config.json](https://huggingface.co/zai-org/GLM-5.3-Flash/raw/main/config.json) |
| 本讲定位 | 把混合注意力的缓存拆成 token KV、indexer、KDA state 三本账 |

> 文章口径：单 rank、attn_tp=1、scaled FP8、64-token page 预留、无 MTP。下文所有容量数字除非另有说明，都按这组条件；标了「bf16」「raw FP8」的除外。scaled FP8 只约束 11 层 latent KV 和 indexer，KDA state 的 dtype 单独看 conv/temporal。
>
> 单位：MB = 10^6 B，MiB = 2^20 B。147.6 MB ≈ 140.8 MiB；738 MB ≈ 703.8 MiB（精确字节 703.9 MiB）。启动日志里的 `GB` 是 `1024^3` B（`memory_pool.py:107`），不要和这里的 MB 直接相加。
>
> 两个词：state slot 指 KDA state 在 HBM 里的一个座位，一个 slot 含 34 层 state；compress tail 指请求级尾缓冲，`index_kpool_compress=True` 时每个请求用一段环形缓冲保存 compress 边界附近的 pooled index K 和 score，按 rank 整块分配，不进 HiCache。

## 教学目标

- **知识目标**：理解文本塔 34 层 KDA 与 11 层 full 的缓存形态差异；说清 DSA 是 full 层的注意力算法、MLA 是它的 KV 布局；掌握三本账的计量单位。
- **能力目标**：能从 config 和启动日志复算 KV、indexer、KDA 容量；能画三级对象图；能判断先爆的是 token 池还是 KDA 池。
- **素养目标**：报 GLM-5.3-Flash 容量时同时说明模型变体、精度、缩放因子和统计范围。

## 时间安排

| 时间 | 环节 | 内容 |
| --- | --- | --- |
| 30 min | 概念 | 两个塔：45 层文本 + 24 层 vision；文本塔 34 KDA + 11 full；DSA 与 MLA 的关系；KDA 状态递推；三本账的单位 |
| 40 min | 源码/演示 | `HybridLinearKVPool`、`full_kv_pool`、`mamba_pool`、`IndexKeyCache`；5/4/3 slots；启动日志字段；RadixTree / HiCache 覆盖范围 |
| 40 min | 实验 | 无 GPU：公式复算、源码三级图、示例日志；有 GPU：启动 SGLang、改 mamba 参数、对照日志 |
| 10 min | 答疑 | 528 与 512、147.6 是否乘 34、5 slots 来源、TP 均分、indexer/tail 例外 |

## 重点与难点

| 类型 | 内容 | 处理方式 |
| --- | --- | --- |
| 重点 | 三本账的边界 | 每本账单独列单位、精度、统计范围；7,260 不含 KDA state 和 tail |
| 重点 | 一个 wrapper，两个子池的双预算 | 用 `max_total_num_tokens` 和 `max_mamba_cache_size // slots` 两条上限判断，不把两个子池说成两个池对象 |
| 难点 | KDA state 为什么不是 per-token | 从 `S_t = f(S_{t-1}, x_t)` 和固定 shape 解释；10 token 到 10K token，state 大小不变 |
| 难点 | 7,260 与 147.6 的口径 | 先拆字节，再标 64-token page 预留、34 层合计、5 slots 预留；147.6 已是 34 层，不再乘 34 |
| 难点 | 单位 | MB=10^6 B；日志 GB=2^30 B；必要时换算 MiB |

## 本讲产出

下表由学员填写；第一行是示例，示范单位和填法。

| 账本 | 对象 | 单位 | 精度 | 每单位容量 | 默认配置下的总量 |
| --- | --- | --- | --- | --- | --- |
| token KV（示例） | 11 层 MLA latent | B/token | scaled FP8（文章口径） | 5,808 | `max_total_num_tokens × 5,808` |
| indexer | 11 层 DSA indexer | 待填 | 待填 | 待填 | 待填 |
| KDA state | 34 层 conv + temporal | 待填 | 待填 | 待填 | 待填 |

表后写一句：默认配置下先限制并发的是哪个池，依据是哪两行日志或哪两个参数。

## 源码锚点

`HybridLinearKVPool` 和 `MambaPool` 都在 `python/sglang/srt/mem_cache/memory_pool.py`，不是两个独立文件；`IndexKeyCache` 在 `index_key_cache.py:14`。

| 文件 | 关键锚点 | 用途 |
| --- | --- | --- |
| `python/sglang/srt/mem_cache/memory_pool.py` | `:401` `MambaPool`、`:3862` `HybridLinearKVPool`、`:4893` `DSATokenToKVPool`、`:4996-5048` compress tail、`:878-900` Mamba 日志 | 两个池类在同一文件；tail 在 `DSATokenToKVPool` 内分配 |
| `python/sglang/srt/mem_cache/index_key_cache.py` | `:14-38` `IndexKeyCache`、64-token page 行宽 | indexer buffer 的管理者，不是池 |
| `python/sglang/srt/configs/mamba_utils.py` | `:60-107` dtype 默认、`:116-125` `mamba_cache_per_req`、`:324-369` state shape | conv 默认 bf16、temporal 默认 fp32；字节数已乘 34 层 |
| `python/sglang/srt/mem_cache/kv_cache_configurator.py` | `:174-184` ratio 常量、`:2232-2261` `_calculate_mamba_ratio`、`:2300-2335` mamba cap、`:2585-2627` FP8 layout | 5/4/3 slots 的来源；scaled FP8 528 与 raw 512 |
| `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py` | `:1118-1222` `build_hybrid_mamba_stack`、`:1165-1174` `page_first_kv_split` 回退、`:1785-1838` `_MambaStrategy`、`:2015-2047` bare DSA 的 `PoolName.INDEXER` sidecar | host pool 只有 KV + MAMBA 两个 entry；sidecar 与 NPU layout 是两类例外 |
| `python/sglang/srt/mem_cache/unified_cache/components/base.py` | `:47-55` `ComponentData` 的 `value` / `host_value` / lock | 树节点存索引，不存数据本体 |
| `python/sglang/srt/mem_cache/unified_cache/components/mamba.py` | `:156-186` `mamba_branching_seqlen`、`:219-245` state 挂 checkpoint、`:305-315` split 时 `host_value` 留在 child、`:737-790` loadback | KDA 的 backup/loadback 与 KV 同契约、不同 transfer 形状 |
| `python/sglang/srt/mem_cache/pool_host/mamba.py` | `:74-123` `MambaPoolHost`、`page_size=1` | host 侧 KDA state 不按 token page 组织 |

## 参考材料

- [三本账：图解 GLM-5.3-Flash 的 KV Cache 布局](../../../09_inference_system/sglang/glm53-flash-kv-layout.md)
- [zai-org/GLM-5.3-Flash config.json](https://huggingface.co/zai-org/GLM-5.3-Flash/raw/main/config.json)
- [SGLang GLM-5.3-Flash cookbook](https://github.com/sgl-project/sglang/blob/0bcd822377da7b5718e674eaf9c870d349424dd1/docs/cookbook/autoregressive/GLM/GLM-5.3-Flash.mdx)
- 本地源码：`~/Project/ai-infra/sglang`，HEAD `f4de6abee6`
