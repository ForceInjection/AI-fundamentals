# LMCache v0.5.8 · MPCacheServer 与控制平面

> 核对基线：LMCache v0.5.8（tag `651617ef`）；写作日期 2026-11-25

## 入口

- CLI 收敛后只剩 `lmcache` 一个入口（`lmcache/cli/main.py`），server 是子命令：`lmcache server`；
- server 进程本体：`lmcache/v1/multiprocess/server.py` `MPCacheServer`（组合器：L1 Manager + L2 adapters + blend 模块 + HTTP 前端）；
- 旧 `lmcache-server <host> <port>`（legacy TCP）与 `lmcache_controller` 入口已删除。

## 端点

| 面                                | 实现                                                                                   |
| --------------------------------- | -------------------------------------------------------------------------------------- |
| 数据面                            | ZMQ 或 gRPC（`--transport {zmq,grpc}`）                                                |
| HTTP 前端                         | status / healthcheck / cache-clear / checksum                                          |
| 控制平面（coordinator，独立部署） | HTTP API：instances / cache / events / metrics / health（`mp_coordinator/http_apis/`） |

## HTTP API（server 侧）

- `POST /cache/prefetches`（`http_apis/cache_api.py`）——驱动 prefetch_manager 预取；
- `POST /cache/clear`、`DELETE /cache/objects`——清理入口；
- 事件上报：`mp_coordinator/ingest/`（HTTP 或 Kafka）。

## L1 / L2 与驱逐

- L1：DRAM Manager（`distributed/l1_manager/dram.py`）与 GDS L1（`GdsL1Config`，cuFile/hipFile/uGDS/Phoenix）；
- L2：`l2_adapters/` 注册表 20 类（fs/fs_native/raw_block/dax/hf3fs/nixl×2/mooncake_store/s3/resp/valkey/bigtable/aerospike/p2p/mock/fault_inject/plugin 等）；
- 驱逐：`--eviction-policy {LRU,ARC,IsolatedLRU,noop}` + `trigger_watermark` / `eviction_ratio`（`distributed/config.py`）——v0.5.1 只有 LRU；「无驱逐会 OOM」的旧叙述被显式驱逐配置取代。

## 控制平面持久化

- `mp_coordinator/persistence/checkpoint.py`：coordinator 重启恢复；
- key 目录：`views/key_directory.py`；blend 索引：`blend_index/blend_lookup/blend_client`。
