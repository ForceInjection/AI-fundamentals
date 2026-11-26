# LMCache v0.5.8 · 可观测体系

> 核对基线：LMCache v0.5.8（tag `651617ef`）；写作日期 2026-11-25
> 本文所有路径均可在 `git show v0.5.8:<path>` 核对。

## 旧指标全部失效

v0.5.1 的观测栈在推理进程内，两件套都在 `lmcache/observability.py`：

| v0.5.1 组件        | 位置                         | 说明                                                             |
| ------------------ | ---------------------------- | ---------------------------------------------------------------- |
| `LMCStatsMonitor`  | `lmcache/observability.py`   | 统计聚合单例，engine 内采集请求/令牌/命中率                      |
| `PrometheusLogger` | 同文件                       | 指标注册，经 vLLM `/metrics` 或 LMCache internal API server 暴露 |
| 健康检查           | `lmcache/v1/health_monitor/` | 检查结果喂给 gauge `lmcache:lmcache_is_healthy`                  |

`PrometheusLogger` 注册的 `lmcache:*` 指标共 72 个名字（部分随特性开关启用）。仪表盘常用的代表：

- 请求/令牌：`lmcache:num_retrieve_requests`、`lmcache:num_store_requests`、`lmcache:num_lookup_requests`、`lmcache:num_requested_tokens`、`lmcache:num_hit_tokens`、`lmcache:num_stored_tokens`
- 命中率：`lmcache:retrieve_hit_rate`、`lmcache:lookup_hit_rate`、`lmcache:request_cache_hit_rate`
- 容量：`lmcache:local_cache_usage`、`lmcache:remote_cache_usage`、`lmcache:local_storage_usage`
- 远端/P2P 时延：`lmcache:remote_time_to_get`、`lmcache:remote_time_to_put`、`lmcache:p2p_time_to_transfer`、`lmcache:p2p_transfer_speed`
- 健康：`lmcache:lmcache_is_healthy`

v0.5.8 树上 `lmcache/observability.py` 文件已删，`LMCStatsMonitor` / `PrometheusLogger` 无符号残留，`lmcache:*` 指标名一个不剩。沿用旧面板会全部断线。

## 新体系之一：server 进程内 mp_observability（OTel）

代码在 `lmcache/v1/mp_observability/`，事件驱动 + OpenTelemetry：

| 层     | 位置                                     | 职责                                                                                                        |
| ------ | ---------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| 事件源 | `event.py` `Event`/`EventType`           | L1Manager / StorageManager / MPCacheServer 发布事件（如 `mp.lookup_prefetch.end`、`cb.lookup.end`）         |
| 总线   | `event_bus.py`                           | 异步队列 + drain 线程，tail-drop（默认 10000）                                                              |
| 订阅者 | `subscribers/{metrics,logging,tracing}/` | metrics 按 l1/l2/lookup/mp_transfer/phase_timing/cb_server 等域拆分；logging 落 debug；tracing 出 OTel span |
| 初始化 | `otel_init.py` `init_otel_metrics`       | 配置 MeterProvider/TracerProvider 与资源属性（`service.instance.id` = `--instance-id`）                     |

指标命名：server 指标一律 `lmcache_mp.*` 前缀，CacheBlend 专属族用 `lmcache_blend.*`，gRPC 运行时指标走上游 `grpc.*`（依赖 `grpcio-observability`，仅 Linux）。转 Prometheus 时 `.` 变 `_`、counter 加 `_total`、unit 附着在名字里（`lmcache_mp.l1_read` 单位 chunks → `lmcache_mp_l1_read_chunks_total`）。

导出两条路：生产用 OTLP push（`--otlp-endpoint`，经 collector 进 Prometheus/Grafana）；dev/debug 用内建 Prometheus pull（`--prometheus-port`，默认 9090）。开关集中在 `ObservabilityConfig`（`config.py`）与 CLI 旗标：`--disable-observability`、`--disable-metrics`、`--disable-logging`、`--enable-tracing`（需 `--otlp-endpoint`）等。

追踪另有两条独立线：OTel span（`--enable-tracing`）与本地 trace 文件（`lmcache server --trace-level {storage,events} --trace-output ...`），离线经 `lmcache trace`（info / replay / replay-events）和 `lmcache bench trace-analyze` 消费。设计文档在 `docs/design/v1/mp_observability/`（METRICS.md 列全量指标，EVENTS.md 列事件契约，blend_observability.md 为跨进程追踪的 proposal）。

server 的 HTTP 前端也带指标端点：`lmcache/v1/multiprocess/http_apis/common/metrics_api.py` 的 `GET /metrics`；同文件 `POST /metrics/reset` 是兼容 no-op，OTel 累计型 instrument 不能原地清零，官方口径是对两次 scrape 做差。

## 新体系之二：coordinator 控制面

coordinator 不用 EventBus，直接用 plain OTel observable instruments（`lmcache/v1/mp_coordinator/observability.py`，`_PREFIX = "lmcache_coordinator."`），scrape 时回调读各组件的 `stats()` 快照，不持锁扫索引：

| 指标族                                  | 例                                                                                                                          | 回答什么                       |
| --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- | ------------------------------ |
| `lmcache_coordinator.key_directory.*`   | `keys_tracked`、`placements`、`lookups`、`lookup_chunks_found`                                                              | 目录规模与查询命中             |
| `lmcache_coordinator.blend_index.*`     | `chunks_indexed`、`unique_chunk_contents`、`namespaces`                                                                     | 集群 blend 索引规模            |
| `lmcache_coordinator.ingest.*`          | `event_batches_received`、`server_events_dropped`、`server_seconds_since_last_event`、`server_event_delivery_delay_seconds` | 事件流丢失与新鲜度（观测重点） |
| `lmcache_coordinator.background_loop.*` | `last_success_timestamp_seconds`、`passes`                                                                                  | 后台循环是否还活着             |

暴露：`lmcache/v1/mp_coordinator/http_apis/metrics_api.py` 的 `GET /metrics`，pull 模式挂在 coordinator HTTP 端口（默认 9300）；配了 OTLP push 或 metrics 关闭时该路由返回 404。资源属性 `service.name=lmcache-mp-coordinator`、`service.instance.id` 为 `主机名:端口`。

### 事件流：mp_coordinator/ingest/

这条流既是控制面状态源，也是被观测对象：

- 两种入口：HTTP push `POST /events`（`http_event_source.py`，非持久）或 Kafka pull（`kafka_event_source.py`，持久，需 `lmcache[kafka]` extra，启用后 `POST /events` 返回 404）；
- `event_gate.py` `EventGate` 做准入：incarnation fencing（重启实例的 L1 状态作废）、seq 去重、gap 检测；
- `event_broadcaster.py` `CacheEventBroadcaster` 分发给 consumers（key_directory 为主消费者）；
- 丢失情况不静默：batch 带 `dropped_events` 字段，进上面的 `ingest.*` 指标。

## vLLM 侧挂接

vLLM 进程内不再跑 LMCache 统计单例，挂接点在 `lmcache/integration/vllm/vllm_multi_process_adapter.py`：

- KV events：`enable_kv_events=True` 时对每个 backing server 的传输客户端建一个 `MPKVEventSubscriber`（类定义在 `lmcache/v1/client/kv_event_subscriber.py`）订阅该 server 的 KV-event 流，scheduler 经 `get_kv_events()` 排空缓冲；
- 指标：用 `prometheus_client` 注册进 vLLM 的 multiprocess registry，共 5 个 `vllm:lmcache_mp_kv_events*` 名字（`_buffered` 为 livesum gauge，`_generated/_drained/_batches/_resyncs` 为 counter），标签 `model_name`/`worker_id`；`_resyncs` 带 `reason`（固定三种：`server_restart`、`events_lost`、`stream_closed`），表示调度器因事件流无法精确跟随而撤销已公告的 LMCache placement。

## 仪表盘迁移对照

| 旧名（v0.5.1）                                         | 新数据来源（v0.5.8）                                                                                                                                                                                              |
| ------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `lmcache:num_lookup_requests`                          | `lmcache_mp_lookups_requests_total`                                                                                                                                                                               |
| `lmcache:num_requested_tokens`                         | `lmcache_mp_lookup_requested_tokens_total`                                                                                                                                                                        |
| `lmcache:num_hit_tokens`                               | `lmcache_mp_lookup_hit_tokens_total`（可按 `lmcache_mp_lookup_hit_l1/l2_tokens_total` 拆层）                                                                                                                      |
| `lmcache:lookup_hit_rate`                              | `rate(lmcache_mp_lookup_hit_tokens_total) / rate(lmcache_mp_lookup_requested_tokens_total)`；请求级用 `lookup_request_hit / lookup_prompt`                                                                        |
| `lmcache:local_cache_usage`                            | `lmcache_mp_l1_memory_usage_bytes`（gauge，另有 `l1_staging_bytes`）                                                                                                                                              |
| `lmcache:remote_cache_usage`                           | `lmcache_mp_l2_usage_bytes{l2_name}`（每 adapter 一个观测点）                                                                                                                                                     |
| `lmcache:remote_time_to_get` / `remote_time_to_put`    | 无同名直译：吞吐看 `lmcache_mp_l2_store_throughput_GBs` / `lmcache_mp_l2_load_throughput_GBs` 直方图，在途积压看 `lmcache_mp_num_inflight_l2_stores/loads`                                                        |
| `lmcache:p2p_time_to_transfer`                         | MP transfer 计数（`lmcache_mp_num_submitted/finished_stores/retrieves_total`）+ transfer phase 直方图 `lmcache_mp_transfer_staging_throughput_GB_per_second`                                                      |
| `lmcache:lmcache_is_healthy`                           | 无对应 gauge：探活改走 HTTP（server `GET /healthcheck`、coordinator `/status`），故障计数看 `lmcache_mp_l1_allocation_failure_chunks_total` / `l1_read_failure_chunks_total` / `l2_prefetch_failure_chunks_total` |
| `lmcache:num_retrieve_requests` / `num_store_requests` | server 端 RPC 计数：`lmcache_mp_num_submitted_retrieves_total` / `lmcache_mp_num_submitted_stores_total`（按 device 维度）                                                                                        |

迁移时注意三点：统计位置从 vLLM 进程换到独立 server 进程；维度从 Prometheus labelnames 换成 OTel attributes（`model_name`、`cache_salt`，后者高基数，可在 scrape 端 relabel 丢弃）；命中率类指标从预聚合 gauge 改为计数器做 rate，PromQL 写法要跟着改。

## 相关篇

- 总架构：[01_mp_architecture.md](01_mp_architecture.md)
- MPCacheServer 端点：[03_mp_server.md](03_mp_server.md)
- CacheBlend 模块组（`lmcache_blend.*` 指标与 CB 事件的产生方）：[06_blend_mp.md](06_blend_mp.md)
- vLLM 集成：[02_lmcache_mp_connector.md](02_lmcache_mp_connector.md)
