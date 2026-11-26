# LMCache v0.5.8 · P2P 重设计：gRPC 控制面

> 核对基线：LMCache v0.5.8（tag `651617ef`）；写作日期 2026-11-25
> 本文所有路径均可在 `git show v0.5.8:<path>` 核对。

## 旧 → 新一句话总述

v0.5.1 的 P2P 是存储后端（`v1/storage_backend/p2p_backend.py` `P2PBackend`）：控制信令走 ZMQ REQ/REP，取数指令压在 `BatchedLookupAndGetMsg` 一类 msgspec 消息里，数据走 `v1/transfer_channel/` 的 `PySocketChannel`/`NixlChannel`。v0.5.8 把这一整代删掉，换成 **server 侧 L2 adapter + 独立 gRPC 控制面**：控制 RPC 收敛到 `lmcache/v1/distributed/p2p.py` 一个文件，数据面复用 transfer channel（nixl/mooncake_te），peer 发现交给 coordinator。

| v0.5.1                                                                                                                        | v0.5.8                                                                                  | 备注                                                                               |
| ----------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `v1/storage_backend/p2p_backend.py` `P2PBackend`（ZMQ 控制信令）                                                              | 已删                                                                                    | 整个 `v1/storage_backend/` 目录不存                                                |
| `v1/transfer_channel/py_socket_channel.py` `PySocketChannel`、`mock_memory_channel.py` `MockMemoryChannel`、`nixl_channel.py` | 已删                                                                                    | 测试桩改由 L2 adapter 注册表的 `mock` 类型（`l2_adapters/mock_l2_adapter.py`）承担 |
| `v1/multiprocess/protocols/p2p.py`                                                                                            | 已删                                                                                    | ZMQ 协议随控制面一起退场                                                           |
| （无对应物）                                                                                                                  | `v1/distributed/p2p.py` `P2PClient`                                                     | 新增，gRPC 控制面客户端                                                            |
| `v1/distributed/l2_adapters/p2p_l2_adapter.py`、`v1/multiprocess/modules/p2p_controller.py`                                   | 同路径保留，内容按 gRPC 重写                                                            | 两文件在 v0.5.1 已存在                                                             |
| `v1/distributed/transfer_channel/`（api/abstract/factory/impl/nixl_impl）                                                     | api 拆到 `v1/common/transfer_channel/`；nixl_impl 延续，新增 `impl/mooncake_te_impl.py` |                                                                                    |

## 控制面：gRPC-only P2PService

`lmcache/v1/distributed/p2p.py` 首行 docstring：「gRPC-only control RPCs between cache servers, separate from worker RPCs」。

- 协议定义：`lmcache/v1/transport/grpc/protos/p2p_service.proto`，service `P2PService`，五个 RPC：
  - `P2PLookupAndLock`（keys + group_layout_descs → task_id）
  - `P2PQueryLookupResults`（task_id → `TransferChannelAddressList` 或空）
  - `P2PUnlockObjects`（keys）
  - `P2PReserveWrite`（keys + layout_desc → 每 key 一个地址；offset 无效表示未预留）
  - `P2PFinishWrite`（keys + succeeded → committed）
- 客户端 `P2PClient`：方法名与 RPC 一一对应，另加 `close()`；pb 消息经 `compile_request_codec_for_types` 生成编解码器，调用返回 `MessagingFuture`；通道带 `GrpcClientAuthenticationInterceptor`（`v1/client/transport/grpc_impl/auth.py`）。
- 服务端独立监听：`v1/multiprocess/server.py` 在 `p2p_config.enabled` 时用 `build_grpc_request_server(modules_for_service(components.modules, "p2p"), replace(mp_config, port=p2p_config.rpc_port))` 起一个专属 gRPC server 并 `attach_p2p_server`。它和 worker 的 ZMQ/gRPC 传输端口分开，防火墙只放行 worker `--port` 不够，`--p2p-rpc-port` 也要放行（官方 `docs/source/mp/p2p.rst` 明确提示）。
- 服务端方法（`v1/multiprocess/modules/p2p_controller.py` `P2PController`）与客户端五个 RPC 同名，统一装饰 `@request_handler(HandlerType.BLOCKING)`（`multiprocess/request_handler.py`）。

## Peer 发现：coordinator 成员表 + reconcile

`P2PController(DiscoverableModule)` 的 docstring：「peer discovery, adapter lifecycle, lookup and write serving」。

- `_fetch_instances()` 用 httpx 轮询 coordinator `GET {coordinator_url}/instances`，解析每个实例的 `instance_id` / `ip` / `p2p_advertised_url` / `mq_port`；心跳间隔兼作发现轮询间隔（p2p.rst）。
- `mq_port` 字段名沿用旧 client 传输的 mq 命名，实际语义是「该 peer 的 P2P 控制 gRPC 端口，P2P 关闭时为 0」（coordinator 侧 `v1/mp_coordinator/cache_events.py` `InstanceIdentity` 字段注释）。
- `_reconcile` / `_add_adapter` / `_remove_adapter`：每个活 peer 建一个 `P2PL2Adapter`，配置为 `peer_mq_server_url = grpc://{ip}:{mq_port}`（控制面）加 `peer_transfer_channel_server_url = p2p_advertised_url`（数据面）；peer 离开即摘除 adapter。无静态 peer 列表。

## 数据面：P2PL2Adapter + transfer channel

`v1/distributed/l2_adapters/p2p_l2_adapter.py` `P2PL2Adapter`（`PeerL2AdapterBase` 子类，注册名 `"p2p"`，见 `register_l2_adapter_type("p2p", P2PL2AdapterConfig)`），内部持有一个 `P2PClient`。模块 docstring 给出机制边界：控制 RPC 发给 peer 的独立 gRPC 服务；定位到的对象经 transfer channel 从 peer 的 L1 拉取；`write_enabled` 时 store 在 peer L1 预留 buffer、推送、提交；adapter 永不在 peer 上驱逐或删除。

读路径（lock-and-locate）：

1. `submit_lookup_and_lock_task` → peer 端 `p2p_lookup_and_lock` 对 L1 驻留 key 加读锁（sparse，允许空洞）；
2. `p2p_query_lookup_results` 返回 `TransferChannelAddress`（offset/size，定义在 `v1/common/transfer_channel/api.py`），即对象在 peer 注册 L1 区内的位置；
3. 本端 transfer channel 单侧 RDMA 读入自己的 L1（p2p.rst：the node that owns the data is not interrupted to serve it），随后 `p2p_unlock_objects`。

RPC 与 transfer channel 都不暴露完成 fd，所以 lookup/load/store 三个事件 fd 由 `PeriodicEventNotifier` 单例周期脉冲（间隔 `_PERIODIC_NOTIFIER_INTERVAL_MS = 5` 毫秒）；lookup RPC 本身超时 `_LOOKUP_RPC_TIMEOUT_S = 3.0` 秒，配置层超时由 `lookup_timeout` 兜底。

transfer channel 工厂：`v1/distributed/transfer_channel/factory.py` `register_transfer_channel_factory` / `create_transfer_channel_context`，实现自注册：

| 引擎           | 注册处                     | 说明                                                                                  |
| -------------- | -------------------------- | ------------------------------------------------------------------------------------- |
| `nixl`（默认） | `impl/nixl_impl.py`        | RDMA（IB/RoCE），随可选 extra `lmcache[nixl]` 安装                                    |
| `mooncake_te`  | `impl/mooncake_te_impl.py` | Mooncake Transfer Engine，默认 TCP；设 `MC_TE_PROTOCOL=rdma` + `MC_TE_DEVICE` 切 RDMA |

transfer channel 启动时把整个 L1 buffer 注册给 peer 远程读，所以要求单一连续 L1 区：与 GDS L1（`--gds-l1-path`）和 Device-DAX L1（`--l1-devdax-path`）互斥，冲突时 server 拒绝启动。

## 写路径：`--enable-p2p-write`

默认只读。开启后：

- 自动选用 `p2p` store 策略（`v1/distributed/storage_controllers/p2p_store_policy.py` `P2PStorePolicy`，`register_store_policy("p2p", ...)`）和 `skip_l1` prefetch 策略（`prefetch_policy.py` `BufferOnlyPrefetchPolicy`），除非显式设置 `--l2-store-policy` / `--l2-prefetch-policy`；
- 本地 DRAM L1 用量达到 `--p2p-write-l1-usage-threshold`（0.7）后，新对象改写进 peer 并丢弃本地副本；一个 store 任务至多落一个 peer，write-enabled 的 adapter 轮流接活；
- 顺序：`p2p_reserve_write`（peer L1 分配 buffer，返回地址）→ transfer channel 推送字节 → `p2p_finish_write`（按 key 上报成败，返回 peer 侧 committed 结果）；
- 语义是「peer 扩展本机 DRAM」（典型是 headless peer），一对节点只在一侧开启（flag help 与 p2p.rst 同口径）。

## 配置表

位置：`v1/multiprocess/config.py` `P2PConfig`（字段默认值、`__post_init__` 校验）、`add_p2p_args` / `parse_args_to_p2p_config`（CLI 翻译）、`MPServerConfig.p2p_config` 字段；`DEFAULT_P2P_CONFIG` 为全默认实例。

| CLI 旗标                         | P2PConfig 字段             | 默认     | 说明                                                                                         |
| -------------------------------- | -------------------------- | -------- | -------------------------------------------------------------------------------------------- |
| `--p2p-advertise-url`            | `advertise_url`            | `""`     | 数据面 host:port，须 peer 可达；非空即启用 P2P，同时还要求 `--coordinator-url`（启动时校验） |
| `--p2p-listen-url`               | `listen_url`               | `""`     | 数据面绑定地址，缺省取 advertise_url（绑 `0.0.0.0`、对外通告路由 IP 用）                     |
| `--p2p-rpc-port`                 | `rpc_port`                 | `0`      | 控制面 gRPC 端口，绑在 `--host`；0 为自动分配                                                |
| `--p2p-lookup-timeout`           | `lookup_timeout`           | `30.0`   | 秒，超过计 miss                                                                              |
| `--p2p-load-timeout`             | `load_timeout`             | `30.0`   | 秒，超过计 load 失败                                                                         |
| `--p2p-transfer-engine`          | `transfer_engine`          | `"nixl"` | transfer channel 实现                                                                        |
| `--enable-p2p-write`             | `enable_write`             | `False`  | 开启写 peer，见上节                                                                          |
| `--p2p-write-l1-usage-threshold` | `write_l1_usage_threshold` | `0.7`    | (0,1]，须低于 `--eviction-trigger-watermark`，`__post_init__` 校验                           |
| `--p2p-store-timeout`            | `store_timeout`            | `30.0`   | 秒，超过计 store 失败                                                                        |

`enabled` 是 `advertise_url` 非空的派生属性；`effective_listen_url` 在 `listen_url` 为空时回落到 `advertise_url`。

## 与 coordinator 目录服务的关系

两条独立的路径，不要混：

- **成员表（P2P 依赖）**：coordinator 只管成员——p2p.rst 原文「It only manages membership — it never sees KV data or participates in lookups」。P2PController 从 `/instances` 拿活 peer 列表，lookup 数据从不经过 coordinator。
- **key 目录（fleet 级提示）**：`v1/mp_coordinator/views/key_directory.py` 把 `ObjectKey` 映射到全 fleet 的 placements（instance/tier/backend/size），由 cache-event 流构建，最终一致，lookup 结果只是 hint、须到 owner 验证。这是 vLLM connector 跨实例查询的路径，P2P lookup 不走它。

观察：`GET /status` 里看 `p2p_state`（registered 为已入 fleet）与 `p2p_peer_count`；coordinator `GET /instances` 列 fleet（p2p.rst「Verifying P2P is working」一节）。

## 约束与边界（p2p.rst 官方口径）

- 必须配 coordinator：只给 `--p2p-advertise-url` 不给 `--coordinator-url` 拒绝启动；
- 默认只读、单跳（直接向持有 prefix 的 peer 读，不跨 peer 链式）；
- 需要 RDMA 网络才有生产级性能；建议参与 P2P 的 server 设 `--l1-align-bytes 65536`，让 RDMA 读更大更齐；
- GDS L1 / DevDAX L1 与 P2P 互斥。

## 相关篇

- server 入口与端口：[03_mp_server.md](03_mp_server.md)
- 旧 ZMQ 代 `P2PBackend` 档案（v0.5.1）：[../01_lmcache/07_p2p_backend.md](../01_lmcache/07_p2p_backend.md)
- 本目录总览与旧新映射表：[README.md](README.md)
