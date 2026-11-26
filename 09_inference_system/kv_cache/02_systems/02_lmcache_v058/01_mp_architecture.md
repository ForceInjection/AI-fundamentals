# LMCache v0.5.8 · MP 总架构

> 核对基线：LMCache v0.5.8（tag `651617ef`）；写作日期 2026-11-25
> 本文所有路径均可在 `git show v0.5.8:<path>` 核对。

## 四个角色

| 角色                    | 位置                                                                                                                                                                                                                            | 职责                                                                                                                                                                           |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **MPCacheServer**       | `lmcache/v1/multiprocess/server.py`                                                                                                                                                                                             | 独立 cache server 进程：L1/L2 存储执行、blend、HTTP 前端（status/healthcheck/cache-clear/checksum）+ Prometheus                                                                |
| **Connector / Adapter** | `lmcache/integration/vllm/lmcache_mp_connector.py`（`LMCacheMPConnector`，`KVConnectorBase_V1 + SupportsHMA`）+ `vllm_multi_process_adapter.py`（scheduler 侧 `LMCacheMPSchedulerAdapter`、worker 侧 `LMCacheMPWorkerAdapter`） | vLLM 进程内接 KVConnector 接口，跨进程转发到 server                                                                                                                            |
| **Client**              | `lmcache/v1/client/`（heartbeat `HeartbeatThread`、传输 `transport/` ZMQ/gRPC 实现、`transfer_context` 共享内存）                                                                                                               | connector ↔ server 之间的传输与保活                                                                                                                                           |
| **Coordinator**         | `lmcache/v1/mp_coordinator/`（独立控制平面）                                                                                                                                                                                    | HTTP API（instances/cache/events/metrics/health）、cache-event 流（HTTP/Kafka，`ingest/`）、持久化（`persistence/checkpoint.py`）、key_directory、blend 索引、prefetch_manager |

## 请求生命周期（以 prefix 命中为例）

1. vLLM scheduler 准入请求 → connector `get_num_new_matched_tokens` → MP 传输查询 server（或 coordinator 目录服务 `views/key_directory.py`）；
2. 命中 → worker 侧 `start_load_kv` 走 client 传输从 server 拉取（server 侧 L1 命中直返；L2 命中经 l2_adapters 回源）；
3. 未命中部分在 decode 过程中由 server `POST /cache/prefetches`（`multiprocess/http_apis/cache_api.py`）驱动的 prefetch_manager 预取；
4. KV 写满块 → store 路径写入 L1，超容量按驱逐策略（`eviction/`：trigger_watermark + eviction_ratio）落 L2；
5. 全程事件经 `mp_coordinator/ingest/`（HTTP 或 Kafka）上报，供控制面观测。

## 与 v0.5.1 in-process 架构的本质差异

| 维度     | v0.5.1（in-process）                    | v0.5.8（MP）                                                             |
| -------- | --------------------------------------- | ------------------------------------------------------------------------ |
| 进程模型 | connector/engine 与 vLLM 同进程         | server 独立进程；vLLM 侧只剩 connector/adapter                           |
| 存储执行 | StorageManager + 后端线程池             | server 内 L1 Manager + L2 adapters（独立演进）                           |
| 控制面   | engine 内 EventManager/HealthMonitor    | 独立 coordinator（HTTP + 事件流）                                        |
| 可观测   | 进程内 LMCStatsMonitor/PrometheusLogger | OTel + coordinator pull（见 [07_observability.md](07_observability.md)） |
| 失败域   | 与 vLLM 同生命周期                      | server 崩溃不影响 vLLM 推理（KV 降级重算）                               |

## 相关篇

- vLLM 集成细节：[02_lmcache_mp_connector.md](02_lmcache_mp_connector.md)
- server 与控制面：[03_mp_server.md](03_mp_server.md)
- 存储两级：[04_storage_l1_l2.md](04_storage_l1_l2.md)（二期）
