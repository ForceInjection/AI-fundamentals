# LMCache v0.5.8 · 两级存储(L1/L2)

> 核对基线:LMCache v0.5.8(tag `651617ef`);写作日期 2026-11-25
> 本文所有路径均可在 `git show v0.5.8:<path>` 核对,定位用「文件路径 + 符号名」。

## 两级模型

server 进程内的存储由 `lmcache/v1/distributed/config.py` 的 `StorageManagerConfig` 描述,在 `distributed/storage_manager.py` 装配,由 `distributed/storage_controllers/` 的 store_controller / prefetch_controller / eviction_controller 执行:

| 组件                | 位置                                                                    | 说明                                                                                         |
| ------------------- | ----------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| L1                  | `distributed/l1_manager/`                                               | server 本地层级,至少配一个;介质可以是 pinned DRAM、NVMe slab(GDS)、Device-DAX 或 shadow 索引 |
| L2                  | `distributed/l2_adapters/`                                              | 经 adapter 外接的层级,0 到多个,按 `--l2-adapter` 给出顺序排列                                |
| 驱逐                | `distributed/eviction_policy/` + `distributed/eviction.py`              | L1/L2 各挂一份 `EvictionConfig`                                                              |
| store/prefetch 策略 | `distributed/storage_controllers/store_policy.py`、`prefetch_policy.py` | 决定每个键写到哪些 adapter、从哪个 adapter 读                                                |

排序规则:`--l1-manager` 列表里第一个占内存的 L1 接收引擎写入,其余 L1 是附加 tier;shadow L1 不占内存,须排在占内存的 L1 之后。

L2 读写走向由两个策略旗标决定,注册名来自 `store_policy.py` / `prefetch_policy.py`:

| 旗标                   | 注册名    | 行为                                                                                                                      |
| ---------------------- | --------- | ------------------------------------------------------------------------------------------------------------------------- |
| `--l2-store-policy`    | `default` | 写所有 adapter,保留 L1                                                                                                    |
|                        | `skip_l1` | 写完 adapter 后删 L1 副本(`BufferOnlyStorePolicy`)                                                                        |
|                        | `p2p`     | `p2p_store_policy.py` `P2PStorePolicy`;`--enable-p2p-write` 在两个策略旗标均未显式设置时自动选中(同时选 skip_l1 prefetch) |
| `--l2-prefetch-policy` | `default` | 取第一个命中 adapter 的结果并保留在 L1                                                                                    |
|                        | `skip_l1` | 读完后删 L1 副本                                                                                                          |
|                        | `retain`  | `default` 的废弃别名                                                                                                      |

## L1:l1_manager/ 包

四类 L1 各有 config 类与 manager 类,工厂在 `l1_manager/factory.py`(`create_l1_manager` / `create_l1_managers`),基类 `l1_manager/base.py` `BaseL1Manager`:

| type(JSON) | config 类(`distributed/config.py`) | manager 类                    | 关键字段                                                                                              |
| ---------- | ---------------------------------- | ----------------------------- | ----------------------------------------------------------------------------------------------------- |
| `DRAM`     | `DRAML1ManagerConfig`              | `dram.py` `DramL1Manager`     | `size_gb`、`use_lazy`(默认 True)、`init_size_gb`(默认 20)、`shm_name`、`use_hugepages`、`mlock`(ROCm) |
| `DEVDAX`   | `DevDaxL1ManagerConfig`            | `devdax.py` `DevDaxL1Manager` | `path`、`offset_gb`、可选 `shared`(下详)                                                              |
| `GDS`      | `GDSL1ManagerConfig`               | `gds.py` `GdsL1Manager`       | `path`、`backend`、`direct_io`(默认 True)、`align_bytes`(≥4096)                                       |
| `SHADOW`   | `ShadowL1ManagerConfig`            | `shadow.py` `ShadowL1Manager` | `tag`、`segment`、`read_ttl_seconds` / `write_ttl_seconds`;无 `size_gb`、无驱逐                       |

配置有两条路,互斥——`distributed/config.py` `parse_args_to_config` 同时见到两路直接报错「Use either --l1-manager or legacy L1 flags」:

- legacy 旗标:`--l1-size-gb` 与 `--eviction-policy` 必配,另有 `--l1-init-size-gb`、`--l1-use-lazy`、`--l1-use-hugepages`、`--l1-align-bytes`、`--l1-devdax-path`、`--l1-devdax-offset-gb`、`--l1-mlock`(`distributed/config.py`)。hugepages 要求预分配,与 lazy 互斥,同时给出时自动关 lazy。另有 `--shm-name`(定义在 `multiprocess/config.py` MP Server 组):engine-driven 传输的共享内存段名,要求 `--no-l1-use-lazy`,不属于 L1 配置本身。
- `--l1-manager '<JSON>'`(可重复):每个 JSON 带 `type` 加类型字段,支持一个 server 配多个 L1。

JSON 公共字段(`_parse_l1_common`):`tag`、`size_gb`、`eviction`(per-L1 驱逐)、`read_ttl_seconds`(默认 300)/ `write_ttl_seconds`(默认 600)。`L1ManagerConfig` 本体还有 `write_ttl_seconds` / `read_ttl_seconds` 两个锁 TTL 字段。

DEVDAX 的 `shared` 对象(`SharedDevDaxConfig`)把整个 pool 交给外部 memory orchestrator 管:`orchestrator`(host:port)、`region_id`、`client_id`、`rpc_timeout_seconds`(默认 5.0);此时该 L1 的驱逐策略必须是 `noop`(带 `shared` 而未写 eviction 时自动补 noop,写了非 noop 报错)。配套的 orchestrator 进程由 CLI 的 `lmcache memory` 子命令启动。

SHADOW 是「借来的 L1」:它只是某个 L2 adapter 替 peer 字节维护的索引(docstring 原话是 "the index a borrowing L2 adapter keeps of a peer's bytes"),不占内存、不驱逐,按 `segment` 指向 peer 的内存段。

### GDS L1

`GdsL1Config`(同样在 `distributed/config.py`)挂在 `L1ManagerConfig.gds_l1_config` 上。一旦设置,L1 介质从 pinned DRAM 换成 NVMe slab(GDS DMA 路径),两者互斥:

| 字段            | 默认   | 说明                                                                   |
| --------------- | ------ | ---------------------------------------------------------------------- |
| `file_location` | 必填   | cuFile / hipFile / phx 的 slab 目录,或 uGDS 的 `ugds_drv` 字符设备路径 |
| `size_in_bytes` | 必填   | slab 容量,来自 `--l1-size-gb`;uGDS 下占用裸设备头部一段                |
| `use_direct_io` | True   | cuFile / hipFile 的 O_DIRECT;uGDS 忽略                                 |
| `backend`       | `auto` | `auto` 按平台选:CUDA→cuFile,ROCm→hipFile                               |
| `align_bytes`   | 4096   | cuFile / hipFile 与 O_DIRECT 要求 4 KiB 对齐                           |

CLI 旗标:`--gds-l1-path`(设置即启用 GDS L1)、`--gds-l1-use-direct-io`(默认 True)、`--gds-l1-backend`。backend 实现收在 `lmcache/v1/gpu_connector/gds_backends/` 包里,`available_backends()` 用 pkgutil 枚举模块名,`create_backend(name)` 用到才 import:

| 模块         | 后端                                                              |
| ------------ | ----------------------------------------------------------------- |
| `cufile.py`  | NVIDIA cuFile(CUDA 默认)                                          |
| `hipfile.py` | ROCm hipFile                                                      |
| `mufile.py`  | 摩尔线程 MUSA SmartIO muFile(`libmufile.so`)                      |
| `phx.py`     | Phoenix phxfs;可跑在 NFS / NVMe-oF 挂载上,但走非 DMA 路径性能降级 |
| `ugds.py`    | 裸 NVMe 字符设备(有限容量 raw device)                             |

`auto` 按模块名序逐个 import,取 `is_default()` 为真的第一个。帮助文本附带给部署者的提醒:开 GDS L1 后不要配 byte-array 型的 L2 adapter。

## L2:l2_adapters/ 注册表

注册机制分布在两个文件:

- `l2_adapters/factory.py`:`register_l2_adapter_factory(name, factory)` 登记工厂,签名 `(config, l1_memory_desc) -> L2AdapterInterface`;`ensure_adapter_loaded` / `load_all_adapters` 负责懒 import;`get_all_registered_names()` 供 CLI 枚举;
- `l2_adapters/config.py`:`register_l2_adapter_type(name, config_cls)` 把注册名映射到 config 类(`_L2_ADAPTER_CONFIG_REGISTRY`),`get_registered_l2_adapter_types()` / `get_type_name_for_config()` 做反查;
- `l2_adapters/__init__.py`:pkgutil 扫描本目录下所有 `*_l2_adapter.py`,经 `add_pending_module` 挂起,请求对应类型名时才 import——第三方依赖缺失的模块被跳过而不拖垮其他 adapter;
- 创建入口 `create_l2_adapter` / `create_l2_adapter_from_registry`,接口基类 `l2_adapters/base.py` `L2AdapterInterface`。

加一个新 adapter 只需放一个 `*_l2_adapter.py` 模块进目录,不改任何注册文件。

### 注册名总表

v0.5.8 共 21 个注册名(逐一核对过每个模块的 `register_l2_adapter_type` 调用):

| 注册名               | config 类                          | 后端                                                                 |
| -------------------- | ---------------------------------- | -------------------------------------------------------------------- |
| `aerospike`          | `AerospikeL2AdapterConfig`         | 原生 C++ Aerospike connector                                         |
| `bigtable`           | `BigtableL2AdapterConfig`          | Cloud Bigtable(v0.5.1 无)                                            |
| `dax`                | `DaxL2AdapterConfig`               | Device-DAX,核心在 `dax/core.py` `DaxCore`                            |
| `fault_inject`       | `FaultInjectL2AdapterConfig`       | 故障注入,测试用                                                      |
| `fs`                 | `FSL2AdapterConfig`                | 文件系统,aiofiles 异步 I/O                                           |
| `fs_native`          | `FSNativeL2AdapterConfig`          | 原生 C++ 文件系统 connector                                          |
| `hf3fs`              | `Hf3fsL2AdapterConfig`             | 原生 C++ 3FS connector(v0.5.1 无)                                    |
| `hfbucket`           | `HFBucketL2AdapterConfig`          | Hugging Face Buckets                                                 |
| `mock`               | `MockL2AdapterConfig`              | 内存 mock,测试用                                                     |
| `mooncake_store`     | `MooncakeStoreL2AdapterConfig`     | Mooncake Store 原生 connector;`setup_config` 原样透传给 mooncake SDK |
| `native_plugin`      | `NativePluginL2AdapterConfig`      | 动态加载第三方 pybind C++ connector                                  |
| `nixl_native`        | `NixlNativeL2AdapterConfig`        | 原生 NIXL(v0.5.1 无)                                                 |
| `nixl_store`         | `NixlStoreL2AdapterConfig`         | NIXL storage backend,预分配文件槽位池                                |
| `nixl_store_dynamic` | `DynamicNixlStoreL2AdapterConfig`  | NIXL 动态文件,一对象一文件                                           |
| `p2p`                | `P2PL2AdapterConfig`               | 单个 peer cache server(控制 RPC 走对端 gRPC)                         |
| `plugin`             | `PluginL2AdapterConfig`            | 动态加载外部 Python adapter 类                                       |
| `raw_block`          | `RawBlockL2AdapterConfig`          | 裸块设备,核心在 `raw_block/core.py` `RawBlockCore`                   |
| `resp`               | `RESPL2AdapterConfig`              | RESP 协议(Redis/Valkey),原生 C++ Redis connector                     |
| `s3`                 | `S3L2AdapterConfig`                | AWS CRT Python 绑定                                                  |
| `sagemaker-hyperpod` | `SageMakerHyperPodL2AdapterConfig` | SageMaker HyperPod;注意注册名带连字符                                |
| `valkey`             | `ValkeyL2AdapterConfig`            | 官方 `valkey-glide` 同步客户端,支持 standalone 与 cluster(v0.5.1 无) |

### 通过 CLI 配 adapter

`--l2-adapter '<JSON>'` 可重复,JSON 必带 `"type"`,其余键交给该 adapter 的 `from_dict`。通用键在 `l2_adapters/config.py` `L2AdapterConfigBase`:

| 键                | 说明                                                                                           |
| ----------------- | ---------------------------------------------------------------------------------------------- |
| `shared`          | bool,默认 False                                                                                |
| `eviction`        | per-adapter 驱逐:`{"eviction_policy": "...", "trigger_watermark": 0.8, "eviction_ratio": 0.2}` |
| `persist_enabled` | 默认 True                                                                                      |
| `serde`           | 如 `{"type": "fp8", "fp8_dtype": "float8_e4m3fn"}`                                             |

`affinity_tag` 已删除:JSON 里出现该键直接报错,L1 affinity 改由 memory tags 解析。

### 两个代表配置

`fs` 的 `FSL2AdapterConfig`(fs_l2_adapter.py):

| 字段               | 说明                                                         |
| ------------------ | ------------------------------------------------------------ |
| `base_path`        | KV 文件目录,必填                                             |
| `relative_tmp_dir` | 写入期间临时文件的相对子目录                                 |
| `read_ahead_size`  | 先读这么多字节触发内核 readahead                             |
| `use_odirect`      | O_DIRECT 绕过 page cache,默认 False;要求缓冲按文件系统块对齐 |

文件名本身编码了可逆的 `ObjectKey`,文件体只放裸 tensor 字节,无元数据头。

`nixl_store` 的 `NixlStoreL2AdapterConfig`(nixl_store_l2_adapter.py):

| 字段             | 说明                                                                                                                                         |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `backend`        | 七选一:`GDS` / `GDS_MT` / `POSIX` / `HF3FS` / `OBJ` / `AZURE_BLOB` / `IBM_SCALE`(`_VALID_NIXL_BACKENDS`)                                     |
| `backend_params` | 文件类 backend(`_FILE_BACKENDS` = GDS/GDS_MT/POSIX/HF3FS/IBM_SCALE)必带 `file_path` 与 `use_direct_io`;可选 `file_size`(缺省取 L1 page size) |
| `pool_size`      | 预分配的存储描述符数量,必须 > 0                                                                                                              |

`nixl_store_dynamic` 的 `DynamicNixlStoreL2AdapterConfig`(nixl_store_dynamic_l2_adapter.py):`backend` 取值域是 `FILE_DYNAMIC_BACKENDS`(定义在 `nixl_store_agents/file_dynamic_nixl_store_agent.py`)= `GDS` / `GDS_MT` / `POSIX` / `HF3FS` / `IBM_SCALE`;`backend_params` 必含 `file_path` 与 `use_direct_io`。类 docstring 只列了前四个,以常量为准。`persist_enabled` 控制关停时是否保留数据文件;miss 时总会查一次二级存储。

裸设备两侧的槽位元数据都落在 `DiskCacheMetadata`(`lmcache/utils.py`):`l2_adapters/raw_block/core.py` 的 `RawBlockCore`(配套 `RawBlockCoreConfig`、`io_engine.py`、`key_codec.py`)负责槽位分配与 checkpoint/recovery;`l2_adapters/dax/core.py` 的 `DaxCore` 同样用它记录槽位状态。

## 驱逐

策略集中在 `distributed/eviction_policy/`,工厂 `factory.py` `CreateEvictionPolicy`:

| 策略          | 类                                            | 说明                                                                                                                                                 |
| ------------- | --------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `LRU`         | `lru.py` `LRUEvictionPolicy`                  | 经典 LRU                                                                                                                                             |
| `ARC`         | `arc.py` `ARCEvictionPolicy`                  | 自适应平衡「最近用」与「频繁用」;v0.5.8 新增(v0.5.1 该目录没有 arc.py)                                                                               |
| `IsolatedLRU` | `isolated_lru.py` `IsolatedLRUEvictionPolicy` | 每 cache_salt 一条 LRU;要求按 cache_salt 配好 quota(`distributed/quota_manager.py` `QuotaManager`,字节预算经 HTTP API 设置,CLI 对应 `lmcache quota`) |
| `noop`        | `noop.py` `NoOpEvictionPolicy`                | 不驱逐;shared DEVDAX L1 强制此项                                                                                                                     |

`distributed/eviction.py` 提供接线:`L1EvictionPolicy`(实现 `L1ManagerListener`)与 `L2EvictionPolicy`(实现 `L2AdapterListener`)分别监听两侧的容量事件。

参数即 `EvictionConfig`(distributed/config.py),server 旗标:

| 旗标                           | 默认 | 说明                                                             |
| ------------------------------ | ---- | ---------------------------------------------------------------- |
| `--eviction-policy`            | 无   | `LRU` / `ARC` / `IsolatedLRU` / `noop` 四选一;legacy L1 路径必配 |
| `--eviction-trigger-watermark` | 0.8  | 触发驱逐的容量水位(0.0–1.0)                                      |
| `--eviction-ratio`             | 0.2  | 触发时一次驱逐的比例(0.0–1.0)                                    |

L2 adapter JSON 里的 `eviction` 键用同一组字段、同一组策略名。

## 旧 → 新对照(v0.5.1 → v0.5.8)

v0.5.1 同时存在两套存储:`lmcache/v1/storage_backend/`(in-process 引擎的后端树)与 `lmcache/v1/distributed/`(MP server 的雏形)。v0.5.8 删掉了前者,只留并扩充后者:

| v0.5.1                                                                                                                   | v0.5.8                                                                    | 说明                                                                    |
| ------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| `lmcache/v1/storage_backend/` 整目录                                                                                     | 已删,无对应路径                                                           | in-process 后端树整体移除                                               |
| `storage_backend/local_disk_backend.py` `LocalDiskBackend`                                                               | L2 adapter `fs` / `fs_native` / `raw_block`                               | O_DIRECT 概念延续为 `use_odirect`,readahead 延续为 `read_ahead_size`    |
| `storage_backend/local_cpu_backend.py` `LocalCPUBackend`                                                                 | L1 `DRAM`(`l1_manager/dram.py` `DramL1Manager`)                           | CPU 内存层级升格为 L1 Manager                                           |
| `storage_backend/gds_backend.py` `GdsBackend`                                                                            | L1 `GDS`(`l1_manager/gds.py`)+ `gpu_connector/gds_backends/` 包           | 后端拆为 cufile / hipfile / mufile / phx / ugds 五模块                  |
| `storage_backend/nixl_storage_backend.py` `NixlStorageBackend`(`NixlStaticStorageBackend` / `NixlDynamicStorageBackend`) | L2 adapter `nixl_store` / `nixl_store_dynamic` / `nixl_native`            | 静态槽位与动态文件各自成类;原生 NIXL 单列                               |
| `storage_backend/raw_block/` `RawBlockCore`                                                                              | `l2_adapters/raw_block/`(`core.py` + `io_engine.py` + `key_codec.py`)     | 归入 adapter 注册表,I/O 引擎独立成模块                                  |
| `distributed/l1_manager.py` 单文件 `L1Manager`                                                                           | `distributed/l1_manager/` 包(DRAM / DEVDAX / GDS / SHADOW + `factory.py`) | 支持一个 server 配多个 L1 与 shadow 索引                                |
| `distributed/eviction_policy/` 仅 LRU / IsolatedLRU / noop                                                               | 新增 `arc.py`                                                             | `--eviction-policy` 增加 ARC 选项                                       |
| `distributed/l2_adapters/` 16 个注册名                                                                                   | 21 个                                                                     | 新增 `bigtable`、`hf3fs`、`nixl_native`、`sagemaker-hyperpod`、`valkey` |

## 相关篇

- server 进程与配置面:[03_mp_server.md](03_mp_server.md)
- 总架构与请求生命周期:[01_mp_architecture.md](01_mp_architecture.md)
- P2P(`p2p` adapter、`--enable-p2p-write` 与 gRPC 控制面):[05_p2p_grpc.md](05_p2p_grpc.md)(二期)
- vLLM 侧接入:[02_lmcache_mp_connector.md](02_lmcache_mp_connector.md)
