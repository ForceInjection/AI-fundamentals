# 第 3 讲教学大纲：KV cache 管理

## 基本信息

| 项目     | 内容                                                             |
| -------- | ---------------------------------------------------------------- |
| 课时     | 120 分钟                                                         |
| 基线     | nano-vllm `b9dd55a`；vLLM `1acec5c009`(v0.34.0rc0-1153)          |
| 先修     | 第 1 讲（EngineCore 初始化链）、第 2 讲（allocate_slots 调用点） |
| 授课形式 | 演进对照 + 源码走读 + 上手实验                                   |

## 教学目标

- **知识目标**：说清四层结构（门面/协调/类型/池）各自职责与分层动机；讲出 PagedAttention 退役证据链三步；知道前缀缓存三个升级（group 键、哈希前移、partial block）。
- **能力目标**：在 HEAD 上用 grep 独立验证退役证据；读 `init engine` 日志辨认 KV 分组数与 GPU blocks；解释一次前缀命中的完整路径（哈希前移 → 查表 → touch）。
- **素养目标**：识别「概念名已死、思想仍在」类演进(PagedAttention)；文档能力边界滞后（「仅 2 组」「只缓存整块」）的核对习惯。

## 时间安排

| 部分                | 时长   | 内容                                                                 |
| ------------------- | ------ | -------------------------------------------------------------------- |
| nano 回顾           | 10 min | 单一 BlockManager 三个「单一」：一个管理器、一种类型、一个哈希键维度 |
| 演进                | 15 min | 一层变四层；两个分层动力（混合注意力 / 唯一显存池）                  |
| 现状走读：五环节    | 45 min | 见下方函数级阅读清单；课堂重点讲初始化链、allocate_slots、BlockPool  |
| PagedAttention 生死 | 15 min | 退役证据链三步 + 思想下沉 + 简历措辞建议                             |
| 实验                | 25 min | 同前缀命中观察 + 分组日志                                            |
| 答疑                | 10 min |                                                                      |

## 函数级阅读清单（课前预读）

| 文件                                       | 函数 / 类                  | 行号              | 看什么                                                |
| ------------------------------------------ | -------------------------- | ----------------- | ----------------------------------------------------- |
| `vllm/v1/engine/core.py`                   | `_initialize_kv_caches`    | 288-419           | 初始化总装：层配置 → profile → 分组 → block_size 定夺 |
| `vllm/v1/core/kv_cache_planning.py`        | `get_kv_cache_configs`     | 1456-1600         | `num_blocks` 换算(1566-1569)；跨 rank 取 min;auto-fit |
| `vllm/v1/kv_cache_interface.py`            | `KVCacheGroup`             | 681-742           | 组 = 共享 block table 的一组层；同组同 page size      |
| `vllm/v1/kv_cache_interface.py`            | `KVCacheTensor`            | 628-655           | 组间 overlay 同一 allocation（639-640 注释）          |
| `vllm/v1/kv_cache_interface.py`            | `KVCacheConfig`            | 829-850           | 引擎级产物：num_blocks / groups / hash_block_size     |
| `vllm/v1/core/kv_cache_manager.py`         | `KVCacheManager.__init__`  | 131-216           | 门面装配；watermark（`:200`）                         |
| `vllm/v1/core/kv_cache_coordinator.py`     | `get_kv_cache_coordinator` | 850-904           | 工厂三分支：NoPrefixCache / Unitary / Hybrid          |
| `vllm/v1/core/kv_cache_manager.py`         | `allocate_slots`           | 391-639           | 三阶段；None 判定（557-559 准入门 / 595-599 常规）    |
| `vllm/v1/core/block_pool.py`               | `BlockPool.__init__`       | 107-173           | 唯一池；free_block_queue(153)；哈希 map(156)          |
| `vllm/v1/core/block_pool.py`               | `get_new_blocks`           | 681-750           | 分配即逐出候选（712-713 `_maybe_evict_cached_block`） |
| `vllm/v1/core/block_pool.py`               | `touch` / `free_blocks`    | 774-792 / 798-839 | 命中引用；LIFO（无哈希）/LRU（有哈希）双语义          |
| `vllm/v1/core/kv_cache_utils.py`           | `hash_block_tokens`        | 697-724           | 链式哈希（父哈希+本块 token+extra_keys）；默认 sha256 |
| `vllm/v1/core/kv_cache_utils.py`           | `BlockHashWithGroupId`     | 50                | 哈希+group 联合键（对照 nano 的 hash→block）          |
| `vllm/v1/request.py`                       | `Request.__init__`         | 243-248           | 哈希前移：构造即算全前缀哈希                          |
| `vllm/v1/attention/ops/paged_attn.py`      | 全文                       | 1-53              | 退役证据：只剩布局辅助；唯一使用方 ROCm               |
| `vllm/v1/attention/backends/flash_attn.py` | `forward`                  | 1387 / 1583       | 分页读取交给 FA 原生 block_table 参数                 |

预读提示：四层类名以本清单为准；`KVCacheGroupId`、`KVCacheGroupSpec` 在 HEAD 已不存在，不要去找。

## 重点与难点

- **重点**：四层各自职责与分层动机；退役证据链三步；`(hash, group)` 联合键。
- **难点**:Hybrid 不动点命中算法（多组互约束）;LIFO/LRU 双语义（无哈希块 LIFO 复用利好局部性，有哈希块 FIFO 即 LRU）；两处 None 判定的区别（running 触发抢占、waiting 只 break）。
- **易错点**：去找 `KVCacheGroupId`；把「PagedAttention 退役」理解成「分页管理没了」；以为哈希在 schedule 时才算。

## 参考材料

- 官方：`docs/design/paged_attention.md`（文档自述为历史文档，现状描述准确）、`prefix_caching.md`（主体准确，Note 1 与组件图滞后）、`hybrid_kv_cache_manager.md`（文档自述为早期阶段；「仅 2 组」已失效）
- 深读（演进素材，旧锚点）：仓库 `09_inference_system/vllm/module_analysis/pagedattention_retirement.md`、`hybrid_kv_cache_manager_deep_dive.md`
- 纵深：KV Cache 八讲第 4-5 讲（对象图与复用分层）
- 论文：PagedAttention SOSP 2023(arXiv:2309.06180)——机制思想仍然有效
