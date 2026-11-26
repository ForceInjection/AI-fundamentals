# LMCache v0.5.8 · CacheBlend 的 MP 模块组实现

> 核对基线：LMCache v0.5.8（tag `651617ef`）；写作日期 2026-11-25
> 本文所有路径均可在 `git show v0.5.8:<path>` 核对。

## 一句话总述

CacheBlend 在 v0.5.8 换了实现位置和机制：v0.5.1 是推理进程内的层间注入（`LMCBlender.process_qkv` 在 forward 层内比对重算），v0.5.8 把它挪进独立 MPCacheServer 进程，变成 `multiprocess/modules/blend/` 的一组模块，外加 coordinator 的集群级指纹索引；复用判定从「层内选择性重算」改为「chunk fingerprint 匹配 + 定位 + reorder + retrieve 时重 RoPE」。

## 旧实现（v0.5.1，整代已删）

| v0.5.1 组件         | 位置                                  | 说明                                                                                                                                           |
| ------------------- | ------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `LMCBlender`        | `lmcache/v1/compute/blend/blender.py` | 层内注入：`process_qkv` 在 `blend_check_layers` 指定的层比对重算前后 K 的差异，按 `blend_recompute_ratios` 选 topk token 重算                  |
| `LMCBlenderBuilder` | `lmcache/v1/compute/blend/utils.py`   | 构建入口                                                                                                                                       |
| config 键           | `lmcache/v1/config.py`                | `enable_blending` / `blend_check_layers` / `blend_recompute_ratios` / `blend_min_tokens`（默认 256）/ `blend_thresholds` / `blend_special_str` |
| 环境变量            | —                                     | `LMCACHE_ENABLE_BLENDING`、`LMCACHE_BLEND_CHECK_LAYERS` 等；blending 要求 `LMCACHE_USE_LAYERWISE=True`                                         |
| 隐式行为            | config 校验段                         | `enable_blending=True` 时自动把 `save_unfull_chunk` 置 True                                                                                    |
| 示例                | `examples/blend_kv_v1/`               | 端到端 blending 演示                                                                                                                           |

删除发生在 92eb56cd（#6135，随 InProcess 核心模块一起移出主仓）。以上符号在 v0.5.8 树上全部不存在。

## 新实现：server 侧模块组

入口是 `lmcache/v1/multiprocess/modules/blend/` 下的 `BlendModule`：仅在 server 以 `--engine-type blend` 启动时由 `create()` 构建，声明依赖 `LMCacheDrivenTransferModule`（包装其 STORE 路径注册指纹，并读取其 `cache_contexts`）。目录共 11 个文件：

| 文件                      | 职责                                                                                                                                                           |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `module.py`               | 组装五个 mixin；`cb_protocol_handshake`（协议版本 = 1）、`cb_reorder_plan`（BLOCKING）两个 RPC                                                                 |
| `matcher.py`              | `BlendTokenRangeMatcher`：token 级探测（任意偏移）+ 全 hash 碰撞排除，直寻表存 chunk poly-hash（2^20 槽，Fibonacci 散列）                                      |
| `lookup.py`               | `LookupMixin`：`cb_unified_lookup` 四腿 FSM（prefix / local-match / coordinator / sparse）；腿未走完返回 None 由调用方重发；对无法确认的 chunk 记 stale strike |
| `retrieve.py`             | `RetrieveMixin`：`cb_retrieve_pre_computed`，先规划后执行 H2D 拷贝 + 重 RoPE + 逐 token scatter                                                                |
| `store.py`                | `StoreMixin`：STORE 钩子异步注册 chunk fingerprint（`CB_FINGERPRINTS_KIND` 派发，`FpJob` 队列 + drainers）                                                     |
| `reorder.py`              | reorder 规划器（纯 Python + numpy）：`LMIN=16` / `GLUE=3` 窗口哈希，把新 prompt 排成「以某个已缓存 prompt 的精确副本开头」，KV 侧即变成精确前缀命中            |
| `rope.py`                 | `_CBRopeState` 与重 RoPE 几何（纯函数，不含 server 上下文）；fp8 e4m3/e5m2 的 dtype dispatch                                                                   |
| `registration.py`         | `RegistrationMixin`：`cb_register_rope` / `cb_unregister_rope`，vLLM 侧 RoPE 状态经 IPC 附着到 server                                                          |
| `fingerprint_eviction.py` | `FingerprintEvictionTracker`：按存储 key 事件主动摘除失效 chunk 的指纹（无 L2 时 L1 逐出即失效；单 L2 时须 L1、L2 都确认；多 L2 时保守忽略）                   |
| `read_set.py`             | `_BlendReadGroups` / `_classify_cb_read_groups`：prefix 腿读 attention + recurrent state，blend 腿读 attention + connector 私有 aux 组                         |
| `__init__.py`             | 公开面 `BlendModule` + 兼容旧 import 路径的 re-export                                                                                                          |

## coordinator 侧组件

| 组件              | 位置                                                                  | 职责                                                                                                                                       |
| ----------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `blend_index`     | `lmcache/v1/mp_coordinator/blend_index.py`                            | 内容指纹索引，key directory token 绑定的派生视图（数据来自统一 cache-event 流，无独立 publish 路径）；`--enable-blend-lookup` 开启，默认关 |
| `blend_client`    | `lmcache/v1/mp_coordinator/blend_client.py` `BlendCoordinatorClient`  | server 侧同步 httpx 客户端，submit-once / poll-on-recall 模式，`PENDING` 哨兵；未配 coordinator URL 时退化为纯本地匹配                     |
| blend_lookup 端点 | `lmcache/v1/mp_coordinator/http_apis/directory_api.py` `blend_lookup` | `POST /directory/blend-lookup`，集群级 fragment 查询 HTTP 面                                                                               |

设计要点：索引按 fleet 的 chunk-size 协商结果绑定窗口（`PUT /config/chunk-size`）；存过但未开 lookup 的 chunk 不回溯索引；缺 `token_offset` 的条目只填内容不入索引（位置不明会导致重 RoPE 出错而非 miss）。旧的独立 blend 目录（`blend_directory.py`、`POST /blend/fingerprints`）已被这套机制替换。

## 配置键

server 侧（`lmcache/v1/multiprocess/config.py` `MPServerConfig`）：

| 键（CLI）                                        | 默认      | 说明                                                                                           |
| ------------------------------------------------ | --------- | ---------------------------------------------------------------------------------------------- |
| `engine_type`（`--engine-type {default,blend}`） | `default` | `blend` 组合 blend 模块，做非前缀 KV 复用                                                      |
| `blend_timeout`                                  | `1.0`     | 一次集群 blend lookup 允许的秒数：既是 HTTP 请求超时，也是 blend client 的 per-lookup 匹配预算 |
| `blend_match_concurrency`                        | `8`       | blend client 同时在途的集群匹配往返上限（须为正）                                              |
| `enable_blend_reorder`                           | `False`   | 记录存入 prompt 的 token id 并应答 `cb_reorder_plan`                                           |
| `enable_segmented_prefix`                        | `False`   | 中段 prefix 的 L2 取回失败时保留断口前缀，断口后 chunk 留在 L1 走 sparse 腿                    |
| `enable_dedup_content_fingerprint`               | `False`   | 已索引内容不重复注册指纹（KV 照存）                                                            |
| `enable_dedup_content_kv`                        | `False`   | 指纹命中覆盖的内容不重复存 KV                                                                  |

coordinator 侧（`lmcache/v1/mp_coordinator/config.py` `MPCoordinatorConfig`）：

| 键                    | 默认    | 说明                                                                         |
| --------------------- | ------- | ---------------------------------------------------------------------------- |
| `enable_blend_lookup` | `False` | 开启内容索引与 `POST /directory/blend-lookup`；关闭时 fleet 不 hash 任何内容 |
| `blend_probe_stride`  | `1`     | 匹配探测点间隔，1 为全召回                                                   |

vLLM 侧：`lmcache.mp.save_unfull_chunk` 默认 False。v0.5.1 的对应键默认也是 False，但 `enable_blending=True` 会被配置校验自动翻成 True；v0.5.8 没有这条自动翻转，需要存未满 chunk 就显式传。in-tree 的 `lmcache_mp_connector.py` 不含任何 blend 配置键，CB 感知的 vLLM 插件在主仓之外（见 `docs/design/v1/mp_observability/blend_observability.md` 对 `lmcache_cacheblend` 插件的引用）。

## 与 v0.5.1 的机制对照

| 维度     | v0.5.1（in-process）                                           | v0.5.8（MP）                                                                            |
| -------- | -------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| 运行位置 | vLLM 推理进程内（engine + gpu_connector）                      | 独立 MPCacheServer 进程；vLLM 侧只剩外部 CB 插件                                        |
| 复用判定 | forward 层内 `check_layers` 比对 K 差异，按 ratio 选 topk 重算 | server 端 chunk fingerprint 匹配：本地直寻表 + coordinator 集群索引                     |
| 注入点   | `process_qkv` 逐层改写 q/k/v/residual                          | RPC 组合：`cb_unified_lookup` 定位，`cb_retrieve_pre_computed` 搬运 + 重 RoPE + scatter |
| 集群面   | 无（单进程）                                                   | coordinator `blend_index` 全集群匹配，cache-event 流驱动                                |
| 顺序保证 | `blend_special_str` 分隔符约定拼接顺序                         | `enable_blend_reorder` 可把 prompt 重排成以缓存 prompt 精确副本开头                     |
| 失效处理 | 无（KV 随 engine 生命周期）                                    | fingerprint eviction 主动摘除 + lookup stale strike 兜底                                |
| 配置面   | `enable_blending` 等 6 键 + `LMCACHE_*` env                    | `engine_type=blend` + 6 个 blend 相关键（见上文配置表）+ coordinator 2 键               |

## 论文背景

CacheBlend 的选择性重算原理（EuroSys '25，arXiv:2405.16444）见旧篇 [../01_lmcache/14_cache_blend.md](../01_lmcache/14_cache_blend.md) §1，常见疑问在该篇附录 B（FAQ）。注意该篇 §2/§3 描述的层内注入实现钉在 v0.5.1，与本文的 MP 实现已换代，配置与代码路径不可混用。

## 相关篇

- 总架构与请求生命周期：[01_mp_architecture.md](01_mp_architecture.md)
- vLLM 集成与 `lmcache.mp.*` extra-config：[02_lmcache_mp_connector.md](02_lmcache_mp_connector.md)
- MPCacheServer 端点与配置：[03_mp_server.md](03_mp_server.md)
- CB 指标与分布式追踪：[07_observability.md](07_observability.md)
