# LMCache v0.5.8 · vLLM 集成（MP Connector）

> 核对基线：LMCache v0.5.8（tag `651617ef`）；写作日期 2026-11-25

## 入口与类

| 组件                                                       | 位置                                                        |
| ---------------------------------------------------------- | ----------------------------------------------------------- |
| `LMCacheMPConnector`（`KVConnectorBase_V1 + SupportsHMA`） | `lmcache/integration/vllm/lmcache_mp_connector.py`          |
| scheduler 侧                                               | `vllm_multi_process_adapter.py` `LMCacheMPSchedulerAdapter` |
| worker 侧                                                  | `vllm_multi_process_adapter.py` `LMCacheMPWorkerAdapter`    |

与 v0.5.1 的 `LMCacheConnectorV1Impl`（`vllm_v1_adapter.py`，已删）对照：四接口名 `start_load_kv` / `save_kv_layer` / `wait_for_layer_load` / `get_num_new_matched_tokens` 保留，但**实现全部重写**为 MP 传输路径——`retrieve_layer`/`store_layer` 层级生成器流水线已删除，`wait_for_layer_load` 退化为兼容空壳。

## 配置：`lmcache.mp.*` extra-config

v0.5.8 的 LMCache 配置不再使用 `LMCacheEngineConfig`/`LMCACHE_*` 环境变量，而是经 vLLM 的 `--kv-transfer-config` extra-config 传入，键前缀 `lmcache.mp.`（`lmcache_mp_connector.py:614` 起逐键翻译）。要点：

- `save_unfull_chunk`：**默认 False**——与 v0.5.1「开启 blending 时自动置 True」的语义反转（旧键 `enable_blending` 在 v0.5.8 已移除，blending 走 server 侧 blend 模块）；
- 完整 server 侧配置见 [03_mp_server.md](03_mp_server.md) 的 MPServerConfig/distributed config。

## 请求追踪

- `LMCacheMPRequestTracker`：`lmcache_mp_metadata.py:69`；
- 每请求元数据 `LMCacheMPRequestMetadata`：同文件 `:340`；
- KV events：`multiprocess/modules/kv_events.py`（`enable_kv_events` 仍在）。

## key 计算

- chunk 大小 256 不变；hash 默认从 sha256 改 **blake3**：`v1/common/token_hasher.py` `TokenHasher`；
- key 线上编码：`distributed/key_codec.py`。

## 附:lmcache.mp.* 完整键表

逐键核对 `git show v0.5.8:lmcache/integration/vllm/lmcache_mp_connector.py` 的翻译段(docstring `:594-621` 逐键说明,直读代码 `:679` 起),并补齐 adapter 与辅助模块读取的键。全集共 **24 个** `lmcache.mp.*` 键,按翻译位置分组(`use_vmm_api` 在 connector 与 adapter 两侧都出现,计一次)。布尔键接受字符串写法 `1/true/yes/on`(大小写不敏感,`vllm_multi_process_adapter.py:179-183`)。

### connector 直读(lmcache_mp_connector.py)

| 键                                         | 默认值              | 语义                                                                                                                                                          | 翻译处                 |
| ------------------------------------------ | ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------- |
| `lmcache.mp.server_urls`                   | 无                  | 多 server URL 列表或逗号分隔串(如 `tcp://host1:6667,tcp://host2:6667`);设置后覆盖 host/port,server 数由此推导                                                 | `:728-737`             |
| `lmcache.mp.host`                          | `"tcp://localhost"` | 单 server 主机,可带 scheme;裸 host 自动补 `tcp://`(`_ensure_transport_scheme`,`:585-587`)                                                                    | `:740-742`             |
| `lmcache.mp.port`                          | `5555`              | 单 server 端口,最终 URL 为 `{host}:{port}`                                                                                                                    | `:743-746`             |
| `lmcache.mp.save_decode_cache`             | `False`             | decode 生成的 token 是否也按 chunk 存;非布尔直接 `ValueError`                                                                                                 | `:679-683`             |
| `lmcache.mp.save_unfull_chunk`             | `False`             | 存 prompt 未满块(边界在 vLLM prefix-cache 网格上)并允许 lookup 命中停在块中;与 speculative decoding、`CircularBufferSpec`、`lazy_offload` 互斥(构造期 `ValueError`) | `:684-713`、`:820-823` |
| `lmcache.mp.eager_prefetch`                | `False`             | 请求进入 vLLM waiting 队列即提交 lookup                                                                                                                       | `:720-724`             |
| `lmcache.mp.lazy_offload`                  | `False`             | store 先缓冲、由下刷策略驱动;要求 vLLM prefix caching(否则构造期 `ValueError`)                                                                                | `:817-830`             |
| `lmcache.mp.transfer_intermediate_tensors` | `False`             | 实验特性:向 server 的 qstore 模块请求 `TRANSFER_QUERY`,传输中间张量                                                                                          | `:982-989`             |
| `lmcache.mp.use_vmm_api`                   | `False`             | KV 经 CUDA VMM 分配时用 VMM IPC 注册;显式值权威,缺省按 vLLM `enable_extensible_kv_cache`/`enable_cumem_allocator` 自动判定后仅注入 worker 侧(`_worker_extra_config`,`:156-184`) | `:153`、adapter `:149` |

### adapter 侧 ExtraConfigDefault 枚举(vllm_multi_process_adapter.py:115-149,经 resolve_extra_config_value 加前缀解析)

| 键                                     | 默认值   | 语义                                                                                                                              | 翻译处               |
| -------------------------------------- | -------- | --------------------------------------------------------------------------------------------------------------------------------- | -------------------- |
| `lmcache.mp.mq_timeout`                | `300.0`  | 阻塞 MQ 请求超时:首查 chunk size、KV 注册/注销等同步操作                                                                          | adapter `:592-593`   |
| `lmcache.mp.heartbeat_interval`        | `10.0`   | 心跳间隔(秒),同时是单次探测超时;3×间隔超 server reap 下限 30s 时启动告警(`:1426-1437`)                                         | adapter `:593`       |
| `lmcache.mp.nonblocking_lookup_status` | `True`   | lookup 状态查询默认不阻塞 scheduler;`False` 则等待在途回复                                                                        | adapter `:588-595`   |
| `lmcache.mp.mp_transfer_mode`          | `"auto"` | `create_transfer_context` 路由:`auto`/`engine_driven`/`lmcache_driven`;优先级高于 `LMCACHE_MP_TRANSFER_MODE` 环境变量             | adapter `:1309-1312` |
| `lmcache.mp.isolated_ipc`              | `False`  | IPC 需跨隔离容器(无共享 host IPC namespace 或 /dev/shm);必须与 server `--isolated-ipc` 一致                                       | adapter `:1316`、connector `:1923` |

### autostart(mp_server_launcher.py:26-28)

| 键                                  | 默认值 | 语义                                                                                                | 翻译处     |
| ----------------------------------- | ------ | --------------------------------------------------------------------------------------------------- | ---------- |
| `lmcache.mp.autostart`              | `False` | connector 自动拉起本地 MP server;仅支持单 server,`server_urls` 多于一个时构造期报错(connector `:754-763`) | `:138-150` |
| `lmcache.mp.autostart.server_args`  | `""`   | 透传给自动拉起 server 的 CLI 参数;`--host/--port/--http-host/--transport` 在禁改清单                | `:302-307` |
| `lmcache.mp.autostart.wait_timeout` | `90.0` | 等待自动拉起 server 就绪的秒数(常量 `_DEFAULT_WAIT_TIMEOUT`,`:30`)                                 | `:304-308` |

### lazy offload 子键(lazy_offload_policy/)

| 键                                             | 默认值             | 语义                                                       | 翻译处                                  |
| ---------------------------------------------- | ------------------ | ---------------------------------------------------------- | --------------------------------------- |
| `lmcache.mp.lazy_offload_policy`               | `"EVICTION_AWARE"` | 下刷策略名,`FIFO` 或 `EVICTION_AWARE`                     | `lazy_offload_policy/__init__.py:27,58` |
| `lmcache.mp.lazy_offload_threshold`            | `100`              | FIFO 专用:积压多少个完成请求触发一次下刷                  | `fifo.py:47-52`                         |
| `lmcache.mp.lazy_offload_select_count`         | `10`               | FIFO 专用:单次下刷最多选多少请求                          | `fifo.py:53-58`                         |
| `lmcache.mp.lazy_offload_horizon_steps`        | `2.5`              | eviction_aware:视为「临近驱逐」的 block 消耗步数,须 > 0  | `eviction_aware.py:39,117-126`          |
| `lmcache.mp.lazy_offload_max_drain_per_step`   | `64`               | eviction_aware:单个 step 最多下刷的操作数,须 ≥ 1         | 同上                                    |
| `lmcache.mp.lazy_offload_max_deferral_seconds` | `0.0`              | eviction_aware:操作最长缓冲时长,0 表示完全由驱逐压力驱动 | 同上                                    |

### qstore ring

| 键                        | 默认值 | 语义                                                                              | 翻译处                       |
| ------------------------- | ------ | --------------------------------------------------------------------------------- | ---------------------------- |
| `lmcache.mp.q.ring_depth` | `2`    | q ring 深度系数:`ring_blocks = ceil(max_num_batched_tokens / block_size) × depth` | `sdk/qringbuffer.py:320-330` |

相邻前缀但不在本表的键:`lmcache.q.ring_blocks`(直接指定 ring 块数,跳过 ring_depth 推导,`qringbuffer.py:320-322`)、`lmcache.tag.revision`(缓存命名空间 tag,connector `:151`)、`lmcache.temporary_unfull_chunk`(单请求 `kv_transfer_params`,配合 `save_unfull_chunk` 只读一次存未满块)。
