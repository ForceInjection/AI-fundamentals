# LMCache v0.5.8 · MPCacheServer 内部深读

> 核对基线：LMCache **v0.5.8**（tag `651617ef`）；深读日期 2026-11-25
> 本文全部行号来自 `git show v0.5.8:<path>` 逐函数实读；文中行号均为 @v0.5.8，不再逐条标注。
> 定位：[01_mp_architecture.md](01_mp_architecture.md)（地图篇）的纵深篇——四角色总览与请求生命周期概览不重复；[03_mp_server.md](03_mp_server.md) 是端点/配置的速览面，本文下沉到函数与行号。

## 一、组合器本体：MPCacheServer

`MPCacheServer` 不是引擎，是一个纯组合器（`multiprocess/server.py:85`）：持有一个共享 `MPCacheServerContext`、一个按构建序排列的模块列表、一个可选的 P2P 监听器。全部成员如下：

| 成员                | 行号       | 事实                                                                                                                                                              |
| ------------------- | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `__init__`          | `:98-106`  | 只做三个赋值：`_context` / `_modules` / `_p2p_server`                                                                                                             |
| `context`           | `:109-111` | 返回共享上下文                                                                                                                                                    |
| `mq_port`           | `:114-116` | P2P 监听器的 `bound_port`；未启用返回 0。注册到 coordinator 时作为 `mq_port` 上报                                                                                 |
| `attach_p2p_server` | `:118-120` | 把已建好的 peer 监听器交给 `close()` 托管                                                                                                                         |
| `report_status`     | `:122-140` | 聚合状态：先取 `storage_manager.report_status()`，拼入 `chunk_size`/`hash_algorithm`/`active_sessions`，再逐模块 `status.update(module.report_status())` 覆盖合并 |
| `close`             | `:142-149` | 关闭顺序：先 `p2p_server.close(wait=True)`，再**倒序**关模块，最后 `context.close()`                                                                              |
| `storage_manager`   | `:153-156` | HTTP `/quota/*` 用的透传属性                                                                                                                                      |
| `cache_contexts`    | `:158-167` | `/cache/checksums` 用：从 `LMCacheDrivenTransferModule.context_entries_snapshot()` 解包                                                                           |
| `clear`             | `:169-175` | `/cache/clear` 用：找 `ManagementModule` 委托，找不到直接抛 `RuntimeError`                                                                                        |

两个细节值得记：

- `report_status` 的合并语义是「后建模块覆盖先建模块的同名 key」，与 gRPC handler 的「后注册覆盖先注册」同构（见二、三节）。
- `close` 的倒序依赖 `_build_server_components` 的正向构建序：`management` 模块 `deferred=True` 建在最后，所以它的 reaper 线程最先停（`modules/management.py:52-55` 注释明说了这个安排）。

## 二、模块装配：发现 → 定序 → 构建

v0.5.8 的装配不是「按 engine_type 走三条分支」，而是**扫描-发现 + 声明依赖 + `create()` 返回 None 自行退出**。引擎级代码只有两处看 `engine_type`：`server.py:437`（`is_blend` → `full_sw_kv`）和 BlendModule 的 create 门（`modules/blend/module.py:87`）。

### 2.1 发现与替换

| 步骤         | 函数                          | 行号                       | 事实                                                                                                                           |
| ------------ | ----------------------------- | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| 外置模块加载 | `load_server_module_classes`  | `ext_server_module.py:287` | 按 `--server-module` 配置导入类，同一模块类重复配置报错                                                                        |
| 内置模块发现 | `discover_modules`            | `engine_module.py:298-312` | `discover_subclasses` 扫 `lmcache.v1.multiprocess.modules` 包（含 blend/ 子包），import 失败直接抛 `ImportError`（`:315-317`） |
| 同名抢占检查 | `_build_server_components` 内 | `server.py:228-243`        | 外置类若用内置 `module_name` 但不是其子类，`ValueError`；是子类则**替换**内置                                                  |
| 合并序列     | 同上                          | `server.py:244-252`        | 外置在前、未被替换的内置在后                                                                                                   |

`plugin_modules/qstore.py` 的 `QStoreModule`（`:187-197`）不在 `modules` 包里，不参与自动发现——它是给 `--server-module` 显式加载的示范外置模块。

### 2.2 依赖定序

`order_modules`（`engine_module.py:237-285`）：把 `module_dependencies` 声明（`DiscoverableModule:207-210`）折成图，用 `graphlib.TopologicalSorter.static_order()` 求确定性拓扑序；依赖按 `module_name` 解析（外置子类顶替内置同名类，`:245-246`）；环依赖报 `ValueError("Module dependency cycle")`（`:284-285`）。排序键 `_module_sort_key`（`:288-295`）是 `(module_name, __module__, __qualname__)`。

构建分两相（`server.py:246-256`）：先建非 `deferred`，再建 `deferred`——`management` 是唯一 deferred 模块，它要等所有 liveness target 收集完（`ModuleBuildContext.liveness_targets`，`engine_module.py:129`）。

### 2.3 构建与注册

- `_build_module`（`server.py:185-192`）：调 `module_cls.create(build_ctx)`，返回非 None 才 `build_ctx.register(...)`。**`create` 返回 None 就是「本配置不装」**。
- `register`（`engine_module.py:175-185`）：重复注册报错；注册后用 `_is_liveness_target`（`:230-234`）探测四方法契约（`touch_instance` / `reap_stale_instances` / `tracked_instance_count` / `drop_instance_state`，`:20-25`），命中就收进 `liveness_targets`。
- `require`（`:151-173`）：按类找已建依赖，找不到按 `module_name` 兜底（外置子类场景），还没有就抛 `ValueError` 并列出已建模块。
- `deferred` 全部建完后，若外置 server module 注册了 handler，追加一个隐藏模块 `ExtServerModuleRouter`（`server.py:276-278`；类在 `ext_server_module.py:150`），对外只暴露两个 RPC：`server_module_call`（`ext_server_module.py:188-203`）和 `server_module_affinity_call`（`:205+`）。
- 最后打一条 `Composed %d engine modules: ...` 日志（`server.py:293`），列出每个模块名与依赖——核对线上装配，看这行日志最快。

### 2.4 内置模块清单（8 个 + 1 个条件模块）

| 模块                                | module_name                                                   | deferred | create 装配门                                                                       | 关键 RPC（HandlerType）                                                                                                                                           |
| ----------------------------------- | ------------------------------------------------------------- | -------- | ----------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `LookupModule`                      | `lookup`（`modules/lookup.py:157`）                           | 否       | 无条件（`:160-168`；KV 负载策略由 `--kv-load-policy` 选，`kv_load_policy.py:110`）  | `lookup`/`query_prefetch_status`/`wait_prefetch_status`/`free_lookup_locks`/`end_session` 全 BLOCKING（`:204,419,451,485,610`）                                   |
| `LMCacheDrivenTransferModule`       | `lmcache_driven_transfer`（`lmcache_driven_transfer.py:237`） | 否       | `supported_transfer_mode != "engine_driven"`（`:241-253`）                          | `register_kv_cache`/`unregister_kv_cache` SERIALIZED（`:530,631`）；`store`/`retrieve` BLOCKING+affinity（`:655,1052`）                                           |
| `EngineDrivenTransferModule`        | `engine_driven_transfer`（`engine_driven_transfer.py:88`）    | 否       | `supported_transfer_mode != "lmcache_driven"`（`:91-96`）                           | `register_kv_cache_engine_driven_context`/`prepare_store`/`commit_store`/`prepare_retrieve`/`commit_retrieve`                                                     |
| `KVEventModule`                     | `kv_events`（`kv_events.py:145`）                             | 否       | 无条件；`kv_event_log_size=0` 或总线关闭时只建空壳（`:154-159,166-190`）            | `subscribe_kv_events` SERIALIZED（`:268`）                                                                                                                        |
| `BlendModule`                       | `blend`（`blend/module.py:79`）                               | 否       | `engine_type == "blend"` 才建（`:87`）；依赖 `LMCacheDrivenTransferModule`（`:82`） | 见九节                                                                                                                                                            |
| `P2PController`                     | `p2p_controller`（`p2p_controller.py:184`）                   | 否       | 无条件建；`rpc_service = "p2p"`（`:185`）——全仓唯一非 worker 服务                   | `p2p_lookup_and_lock` 等五个全 BLOCKING（`:310,377,420,434,480`）                                                                                                 |
| `ManagementModule`                  | `management`（`management.py:52`）                            | **是**   | 无条件（`:58-69`）                                                                  | `ping`/`clear`/`report_block_allocation` BLOCKING；`get_chunk_size`/`negotiate_chunk_size`/`get_experimental`/`noop` SERIALIZED（`:129,165,174,191,196,204,213`） |
| `ExtServerModuleRouter`（条件追加） | —                                                             | —        | 有外置 server module 时由 `_build_server_components` 追加                           | `server_module_call` BLOCKING；`server_module_affinity_call` BLOCKING+affinity                                                                                    |

注意两个 transfer 模块是「互斥/共存」关系而非三选一：`lmcache_driven`（默认）只装前者，`engine_driven` 只装后者，`auto` 两个都装。`kv_events` 声明依赖两者（`kv_events.py:148-151`），保证倒序关闭时事件订阅先于 transfer 状态释放。

## 三、两条传输端点与 HandlerType 分发

### 3.1 声明侧：装饰器与校验

- `HandlerType`（`request_handler.py:18-23`）：`SERIALIZED` / `BLOCKING` / `NON_BLOCKING` 三值；`NON_BLOCKING` 在 ZMQ 与 gRPC 两条路径都直接 `NotImplementedError`（`zmq_impl/mq.py:323-324,553`；`grpc_impl/aio_server.py:159-162`）——实际只有前两值在用。
- `request_handler(...)`（`:44-64`）：operation 缺省取方法名；`requires_client_affinity` 只许配 BLOCKING（`:51-52`）。
- `iter_request_handlers`（`:111-137`）：`inspect.getmembers` 扫装饰方法，同 operation 重复报错；`validate_handler`（`:79-108`）逐参数对照 `get_rpc_spec` 的 payload/response 类型，不匹配即 `TypeError`——**启动期把协议钉死**。

### 3.2 服务面拆分：modules_for_service

`modules_for_service(modules, service)`（`engine_module.py:219-227`）按模块声明的 `rpc_service` 过滤，缺省 `"worker"`（`DiscoverableModule:210`）。server 装配时切两刀：

| 接线                                                                                                      | 行号                | 目标                                                                                                                                                  |
| --------------------------------------------------------------------------------------------------------- | ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `modules_for_service(components.modules, "p2p")` → `build_grpc_request_server(..., rpc_client=P2PClient)` | `server.py:457-469` | P2P 独立 gRPC 监听器，端口 `replace(mp_config, port=mp_config.p2p_config.rpc_port)`（`:466`），认证与主服务同源（`api_key_files` 随 config 替换保留） |
| `modules_for_service(components.modules, "worker")` → `create_request_server(modules, mp_config)`         | `server.py:489-491` | 主数据面                                                                                                                                              |

`rpc_service="p2p"` 的只有 `P2PController`；也就是说 P2P 监听器上只有五个 p2p RPC，主监听器上永远不会出现它们。`mp_config.port` 是主服务端口；P2P 的 `rpc_port` 独立（默认值见 `multiprocess/config.py` `P2PConfig`）。

### 3.3 工厂与三种 gRPC 实现

`create_request_server`（`transport/server_factory.py:17-45`）：`transport == "grpc"` 走 `build_grpc_request_server`，否则 ZMQ。**默认 transport 是 `zmq`**（`multiprocess/config.py:43`）。

`build_grpc_request_server`（`transport/grpc_impl/server_builder.py:45-84`）：`resolve_grpc_server_mode`（`:28-42`）把 `auto` 解析为 native（装了 `pygrpc_native`）或 aio，否则按配置取 `sync`/`aio`/`native`。三实现共享 `GrpcMultiprocessServerBase`（`grpc_impl/server_base.py:29-115`）：

- 每个服务一个 affinity 池 `AffinityThreadPool(max_gpu_workers)`（`:54-57`）+ 一个 `grpc_server_workers` 线程执行器（`:60-63`）；
- `add_modules`（`:72-94`）：`bind_module_handlers`（`server_handlers.py:58-76`，**同 operation 后模块覆盖先模块**）→ `build_service_handlers`（`:118-162`，没 handler 的方法保留 `handler=None`，运行期回 `UNIMPLEMENTED`，`:146-153`）；
- 绑定端口失败（`bound_port == 0`）即抛错（`server_base.py:106-111`）；gRPC 显式关掉 `so_reuseport`，避免双 server 静默分摊客户端（`server_handlers.py:36-38`）。

三实现的分发差异：

| 实现                                       | 分发函数                                              | SERIALIZED                                        | BLOCKING                                                                            | BLOCKING+affinity                                            |
| ------------------------------------------ | ----------------------------------------------------- | ------------------------------------------------- | ----------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| sync（`grpc_impl/server.py:142`）          | `_GeneratedServicer._dispatch` `:97-139`              | `_serialized_handler_lock` 串行（`:111-113`）     | gRPC 线程直接执行，`_normal_slots` 有界信号量限并发 `max_cpu_workers`（`:125-129`） | `affinity_pool.submit(hash(client_key))`（`:114-124`）       |
| aio（`grpc_impl/aio_server.py:76`）        | `dispatch` `:128-180`                                 | 单线程池 `_serialized_pool`（`:109-112,155-158`） | `_normal_pool`（`max_cpu_workers`，`:105-108,171-174`）                             | `affinity_pool.submit` + `asyncio.wrap_future`（`:163-170`） |
| native（`grpc_impl/native_server.py:107`） | `_dispatch_loop` `:327-348` → `_submitter` `:307-313` | `_serialized_pool`                                | `_normal_pool`                                                                      | `_submit_affinity` `:315-325`                                |

流式方法（KV 事件订阅）在三种实现里都走「每流一个读线程 + 信用信号量」：aio 版 `forward`（`grpc_impl/aio_server.py:182-241`）用 `threading.Semaphore(_STREAM_READ_AHEAD)` 防止读超前于客户端消费；sync 版起 `grpc-event-stream` 线程（`grpc_impl/server.py:83-90`）。

affinity key 取法：`client_affinity_key`（`server_handlers.py:165-182`）优先取客户端 metadata 里的随机 ID，退化为 `hash(context.peer())`。

### 3.4 ZMQ 实现

`build_zmq_request_server`（`zmq_impl/server.py:101-143`）：`MessageQueueServer` 绑 `tcp://host:port` 的 ROUTER socket（`zmq_impl/mq.py:161-163`）；按 handler 元数据把操作分池——SERIALIZED 主循环内联、affinity 进 `max_gpu_workers` 池、其余进 `max_cpu_workers` 池（`zmq_impl/server.py:128-141`；池判定 `HandlerSpec.pool` `:46-53`）。主循环单线程（`mq.py:432-454`）：`zmq.Poller` 同时盯请求 socket 与输出 eventfd，单轮最多 64 条请求；BLOCKING handler 提交线程池，完成回调把响应压 `output_queue` 并 `notify()` eventfd（`:252-308`）。affinity key 在 ZMQ 侧取 `hash(zmq identity)`（`mq.py:268`）。

`AffinityThreadPool`（`affinity_pool.py:60-193`）：每 worker 一条队列，`_slot_for_key`（`:131-167`）首次见到 key 就按轮转绑槽，槽用满后绕回并告警（`:156-166`）——同槽的客户端任务被串行化。`DeferredResponse`（`:32-45`）支持 handler 先返回占位 Future、设备事件完成后再补结果（ROCm host-event-wait 用，`lmcache_driven_transfer.py:672-680`）。

## 四、请求全链路：LOOKUP 与 STORE/RETRIEVE

### 4.1 一条 LOOKUP（prefix 查询）

入口 `LookupModule.lookup`（`modules/lookup.py:204-417`，BLOCKING → normal 池）：

1. 事件：`MP_REQUEST_START` + `MP_LOOKUP_PREFETCH_START`（`:222-233`）；
2. 按 `(model_name, world_size)` 查 `LayoutDescRegistry`，无注册即空 job 退出 `no_gpu_context`（`:235-243`）；取该注册的 `chunk_size` 与 `TokenHasher`，对 `token_bytes[:end]` 算 chunk hash（`:245-260`）；
3. `KVLoadPolicy.should_load`（`:325-347`；默认实现 `kv_load_policy.py:86-108`）可以拒绝整单——拒绝时注册「零命中」空 job，不取锁不预取；
4. 引擎已覆盖前缀 `[0, start)`：不锁不预取，只 `storage_manager.touch_cached_keys` 保 recency（`:351-363`）；
5. 对 `[start, end)` 组装 `PrefetchTaskSpec`，`storage_manager.submit_prefetch_task`（`:392-395`）转给 `PrefetchController`，job 按 `request_id` 登记（`:396-417`）。

`PrefetchController`（`storage_controllers/prefetch_controller.py:534`）在后台线程跑 `select.poll` 事件环（`_prefetch_loop :947-1021`）：监听提交 eventfd、每 L2 adapter 的 lookup/load eventfd。请求走两相状态机（`_advance_request :1023-1045`）：`LOOKUP` 相向各 adapter 撒查询，齐了转 `PLAN_AND_LOAD`（预留在 `_reserve_l1_cells :293`，再从 L2 装入 L1）。**是否需要查 L2 由 `_needs_l2_lookup` 决定（`:408-427`）：L1 已锁住的 cell 按 sliding window 折叠后命中长度够了，就不碰 L2——这是「L1 命中/落 L2」的真正分岔点。**

客户端侧两条跟进 RPC（都 BLOCKING）：

- `query_prefetch_status`（`:419-449`）：轮询；结果非 None 即消费掉 job（exactly-once），未知 request_id 返回 0；
- `wait_prefetch_status`（`:451-483`）：`storage_manager.wait_prefetch_status` 阻塞等，再走同一条消费路径。

### 4.2 一条 RETRIEVE（L1 → GPU）

`LMCacheDrivenTransferModule.retrieve`（`lmcache_driven_transfer.py:1057-1352`，BLOCKING+affinity → 该客户端绑定的 GPU 线程）：

1. ROCm host-event-wait 场景，生产者事件没完成就挂 `DeferredResponse`（`:1098-1112`）；
2. `get_and_touch_context_entry(instance_id)`（`:356`）取 GPU 上下文并顺带续命；未注册直接回 `(b"", False)`，不产生完成事件（`:1115-1124`）——注释明确：回显生产者 handle 会让对端 import 自己的事件，HIP 上非法；
3. `resolve_obj_keys` 按 object group 解 key（`:1131-1140`）；block id 数量不足直接放弃整单（fail-closed，`:1188-1204`），防止 transfer kernel 读越界显存；
4. 滑窗组只读窗口内后缀，`aux` 组整组跳过（`:1228-1245`）；`prepare_read_completion` 一次性捕获全部 retained 锁（`:1257-1259`）；
5. 逐组 `unsafe_read`（`:1271-1273`），读到就 `transfer_object_group(..., H2D)`（`:1282-1291`）；unfull chunk 由 `_retrieve_unfull_chunk_objects` 补（`:1293-1306`）；
6. 收尾在 copy stream 上挂 `finish_read_by_owner` host 回调放锁（`:1310-1317`）+ 流上发布 `MP_RETRIEVE_END`（`:1324-1340`）。

### 4.3 一条 STORE（GPU → L1 → L2）

`store` → `store_with_chunk_mask`（`:660-1025`）：

1. 同样的上下文检查与 fail-closed 检查（`:738-813`）；全 null block 的 chunk 先剔掉（`:819-826`）；
2. import 生产者事件 → stream wait → 用完即还（`hold_imported_event` + `release_imported_event` 回调，`:832-841`）；
3. 逐 object group：`storage_manager.reserve_write`（`:901-903`）→ `transfer_object_group(D2H)`（`:919-928`，`batch_size` 恒 1）；unfull chunk 单独走 `_store_unfull_chunk_objects`（`:930-937`）；
4. **全有或全无**：任一异常则 `stored_count=0`，已捕获 owner 的保留对象在 copy stream 上 `abort_write`（`:940-959`）；成功才 `finish_write`；
5. 返回三 元组：完成事件 IPC handle、整单成败、逐 chunk 掩码（`:1004-1025`）。

`finish_write` 进 `StorageManager`（`distributed/storage_manager.py:352-390`）：对象对读者可见，发布 `SM_WRITE_FINISHED`。**L1→L2 落盘不经过驱逐**：`StoreController` 通过 `StoreListener.on_l1_keys_write_finished`（`storage_controllers/store_controller.py:127`）收新 key，入队（`_process_new_keys :643-657`）、按预算提交 L2（`_submit_queued :659-666`）；预算是 `trigger_watermark * (1 - eviction_ratio)` 的 L1 容量（`_over_budget :687-695`）——在飞 store 的读锁不能把驱逐要腾的空间先吃掉。

### 4.4 收尾三件套

- `free_lookup_locks`（`lookup.py:485-609`）：按 session 里记录的 lookup 结果放锁；预取还在跑时什么都不放，锁随 read-lock TTL 过期（`:508-512` 注释）；
- `end_session`（`:610-646`）：发 `MP_VLLM_END_SESSION`、删 session、统一 touch 本次请求全部 key；
- 放锁的实际执行在 `finish_lookup_read`（`engine_context.py:58-84`）：按每个 object group 预取时预留的 `num_kv_readers` 计数，分组调 `storage_manager.finish_read_prefetched`。

## 五、模块间通信：EventBus 与 coordinator 链路

### 5.1 进程内 EventBus

`EventBus`（`mp_observability/event_bus.py:101`）：`publish`（`:215`）入队、后台线程 `_run/_drain_all`（`:331-338`）分发；`publish_on_stream`（`:186`）把事件挂到 copy stream 上，保证事件顺序与设备写入一致（store/retrieve 的 START/END 都走这条路）。`has_subscribers`（`:175`）让 lookup 在无订阅者时跳过重元数据组装（`lookup.py:269-288`）。

### 5.2 server → coordinator 的四条链路

| 链路            | 函数                                                                              | 传输                                                                                                                                                                                                                      | 行号 |
| --------------- | --------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---- |
| 注册 + 心跳     | `keep_registered`（`mp_coordinator/registrar.py:111-199`）                        | HTTP：`POST /instances` 注册、`PUT /instances/{id}/heartbeat` 每 `heartbeat_interval`（默认 5s，`:37`）一次、取消时 `DELETE`；**404 视为 coordinator 失忆，弃 id 重注册**（`:181-187`）；瞬时失败保 id 重试（`:190-194`） |
| 注册成功回调    | `on_registered=engine.storage_manager.publish_capacity`                           | 容量声明只在内存，重注册必须补发（`http_server.py:222`；`registrar.py:146-150`）                                                                                                                                          |
| chunk size 绑定 | `bind_coordinator_chunk_size`                                                     | HTTP `PUT /config/chunk-size`，挂成 ctx 的 chunk_size listener（`server.py:476-484`；`registrar.py:41-61`）                                                                                                               |
| cache 事件流    | `maybe_create_cache_event_subscriber`（`mp_coordinator/cache_events.py:831-887`） | sink 二选一或并存：`HttpCacheEventSink` / `KafkaCacheEventSink`（`create_cache_event_sink :807-828`）+ events 级 trace 文件；按 `flush_interval` 攒批（`_record/_flush_if_due :767-804`）                                 |
| P2P 对端发现    | `P2PController._start_orchestration`（`p2p_controller.py:228-265`）               | httpx 同步客户端按 `heartbeat_interval` 轮询 coordinator `/instances`（`_poll_cycle :613` → `_fetch_instances :643` → `_reconcile :715`），每对端一个 L2 adapter                                                          |

这条链路只在 **HTTP 前端模式**（`lmcache server`）存在：`keep_registered` 是 FastAPI lifespan 里 `asyncio.create_task` 起的（`http_server.py:198-225`）；纯 `run_cache_server` 模式（`server.py` `__main__`）不注册 coordinator，只有 P2PController 自己的发现轮询。

### 5.3 worker → server 的存活链

client 侧 `HeartbeatThread`（`client/heartbeat.py:41-102`，默认 10s，`ThreadLevel.CRITICAL`）发 `ping` → server 侧 `ManagementModule.ping`（`management.py:129-144`）遍历所有 liveness target `touch_instance` 续命。reaper 由 management 模块持有：`worker_reap_timeout_seconds`（默认 120s）的 1/4 为扫描周期（`:90-98`），`_reap_cycle`（`:146-163`）对每个 target 调 `reap_stale_instances`，再把被收割的 id 广播 `drop_instance_state`（BlendModule 用它丢 per-instance CB rope 状态，`blend/module.py:331-335`）。从未 ping 过的 worker 按 `worker_registration_grace_seconds`（默认 3600s）宽限；配置校验要求 grace ≥ reap、reap 为 0 或 ≥ 30s（`multiprocess/config.py:207-225`）。

coordinator 侧对 server 的对称机制：`_evict_stale`（`mp_coordinator/app.py:360-373`）按 `last_heartbeat_time` 扫 `InstanceRegistry.remove_stale`（`views/instance_registry.py:240` 附近）——server 心跳断了就从成员表除名。

## 六、HTTP 前端：路由注册与 handler 落点

入口链：`lmcache server`（`cli/commands/server.py:104-113`）→ `run_http_server`（`http_server.py:279-361`）→ uvicorn 跑模块级 `app`。`app` 在 import 时就装好三件事（`:266-276`）：lifespan、`HTTPAPIRegistry`、错误处理器。

- `HTTPAPIRegistry.register_all_apis`（`http_api_registry.py:29-44`）：扫 `http_apis/` 下名字以 `_api` 结尾、暴露模块级 `router` 的文件，全部 include。
- `common_api.py`（`:30-44`）再合并 `http_apis/common/` 子包的路由。

路由 → 落点对照（server 侧，行号 @v0.5.8）：

| 文件                    | 路由                                                                                                                                                                                      | 落点                                                                                                                                                                                                                                                                               |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `info_api.py`           | `GET /`、`/healthcheck`、`/status`、`/lmc_version`、`/commit_id`、`/version`（`:26,36,58,74,79,84`）                                                                                      | `/status` 直接 `engine.report_status()`（`:59-71`），即一节那套聚合                                                                                                                                                                                                                |
| `cache_api.py`          | `GET/DELETE /cache/objects`（`:58,81`）、`POST /cache/prefetches` 202（`:107`）、`GET /cache/prefetches/{id}`（`:129`）、`POST /cache/clear`（`:161`）、`POST /cache/checksums`（`:305`） | objects/prefetches 走 `MPHTTPContext` 的 `ObjectService`/`PrefetchService`（`dependencies.py:40-56`）；`clear` 调 `engine.clear` → `ManagementModule.clear`（`management.py:196-202`，`_clear_lock` 内 memcheck→clear→memcheck）；checksums 走 `MPCacheServer.cache_contexts` 透传 |
| `quota_api.py`          | `PUT/GET/DELETE /quota/{cache_salt}`、`GET /quota`（`:67,119,138,158`）                                                                                                                   | `storage_manager.quota_manager`（`QuotaManager` 注册表，`storage_manager.py:201-207`）                                                                                                                                                                                             |
| `config_api.py`         | `GET /config`、`/config/adapters`（`:30,63`）                                                                                                                                             | 反射 server 配置                                                                                                                                                                                                                                                                   |
| `reconfigure_api.py`    | `GET /reconfigure/{backend}/l2/status`、`POST /reconfigure/{backend}/l2/{operation}`（`:262,276`）                                                                                        | `storage_manager.reconfigure_l2_adapter` 系                                                                                                                                                                                                                                        |
| `l1_reconfigure_api.py` | `/reconfigure/dax/l1/{status,add,remove}`（`:110,133,171`）                                                                                                                               | devdax arena 增删                                                                                                                                                                                                                                                                  |
| `common/*`              | `GET /metrics`、`POST /metrics/reset`、`GET /env`、`/loglevel`、`/threads`、`/periodic-threads*`、`POST /run_script`（默认关闭，`multiprocess/config.py:146-149`）                        | 诊断面                                                                                                                                                                                                                                                                             |

两个澄清：`directory_api`、`health_api`（/health）、`instances_api`、`events_api` 属于 **coordinator** 的 `mp_coordinator/http_apis/`，server 侧没有；server 的 `/metrics` 是 common 包的即时拉取端点，Prometheus 抓取走 `init_observability` 起的独立端口（HTTP 模式下 `start_prometheus_http_server=False`，`http_server.py:160`，抓取由 coordinator pull，见 [07_observability.md](07_observability.md)）。

`get_context`（`dependencies.py:59-78`）：engine 未就绪时统一 503，路由层不写 try/except；领域错误由 `register_error_handlers`（`error_handlers.py:35`）集中映射。

## 七、驱逐接线：watermark 检查在哪个函数

`StorageManager.__init__`（`distributed/storage_manager.py:131-284`）一口气装四个控制器，驱逐接线全在这里：

| 组件                                         | 装配点     | 说明                                                                                                                                                                                 |
| -------------------------------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `L1EvictionController`（每 L1 manager 一个） | `:160-166` | 持 `L1EvictionPolicy` 桥接监听器，`register_listener` 挂到 manager                                                                                                                   |
| `QuotaManager`                               | `:201-207` | 每 cache_salt 配额注册表，无 L2 也常驻（HTTP `/quota` 的稳定引用）                                                                                                                   |
| `L2EvictionController`（全 adapter 一个）    | `:216-234` | 每个 adapter 包一个 `L2AdapterEvictionState`（自己的 policy + listener，`eviction_controller.py:201-215`）；是否启用由 `_should_enable_l2_eviction`（`storage_manager.py:1627`）过滤 |
| `StoreController`（每 L1 manager 一个）      | `:240-257` | 写后异步落 L2（见 4.3）                                                                                                                                                              |
| `PrefetchController`                         | `:259-271` | 带在飞水位 `prefetch_inflight_watermark`                                                                                                                                             |

**L1 watermark 检查点**：`L1EvictionController.eviction_loop`（`storage_controllers/eviction_controller.py:160-190`）——后台线程每秒 `get_memory_usage`，`usage < trigger_watermark` 就发一条 `L1_EVICTION_LOOP_TICK(triggered=False)` 事件睡觉；越线才调 `get_eviction_actions(eviction_ratio, key_eligible_filter=self._l1_manager.is_key_evictable)`，逐个 `execute_eviction_action`（`:192-198`：目的地只认 `DISCARD` → `l1_manager.delete`）。

**L2 watermark 检查点**：`L2EvictionController._eviction_loop`（`:303-311`，单线程轮询全部 adapter）→ `_check_and_evict`（`:313-317`）分叉：

- 全局支 `_check_and_evict_global`（`:319-345`）：`adapter.get_usage().usage_fraction` 对 watermark；`usage_fraction < 0`（adapter 未声明容量）不触发；
- 隔离支 `_check_and_evict_by_cache_salt`（`:347-404`）：`support_isolation`（`IsolatedLRU`，`eviction.py:31-43`）时逐 salt 比 `watermark * QuotaManager 限额`，**未注册/零配额 salt 一律全量清（effective_ratio=1.0）**（`:382-384`），按目的地合批后一次 `adapter.delete`（`:400-404`）。

事件桥：`L1EvictionPolicy`（`distributed/eviction.py:175-215`）把 L1Manager 的 reserve/write-finished/deleted/accessed 翻成 `on_keys_*`；recency 只来自显式 touch，读锁释放不算访问（`:195-198`）。策略工厂 `CreateEvictionPolicy`（`eviction_policy/factory.py:14-35`）：`LRU`/`ARC`/`IsolatedLRU`/`noop`，未知名直接 `ValueError`。`EvictionDestination` 枚举（`internal_api.py:186-195`）有 `L2_CACHE`，但 v0.5.8 两个控制器都只实现 `DISCARD`——**降级写 L2 是 StoreController 的活，驱逐只负责扔**。

## 八、生命周期：启动顺序、退出、超时

### 8.1 启动序（`_start_cache_server`，`server.py:358-515`）

| 步  | 行号       | 内容                                                                                                                                                                  |
| --- | ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | `:367`     | `set_isolated_ipc`——先于一切 event IPC 解析，让本进程所有 resolver 看到该设置                                                                                         |
| 2   | `:373-379` | `instance_id` 投影到 OTel `service.instance.id`；`grpc_metrics_enabled` 按 transport 解析                                                                             |
| 3   | `:381-389` | `init_observability`（返回 event_bus）→ `init_gc_monitor` → trace recorder                                                                                            |
| 4   | `:391-431` | engine-driven 之外的 SHM 校验：`--shm-name` 与 lazy/devdax 冲突告警；`/dev/shm` 余量不足则**自降级**清空 `shm_name` 走 inline                                         |
| 5   | `:437-447` | `is_blend = engine_type == "blend"` → 建 `MPCacheServerContext`（`full_sw_kv=is_blend`）                                                                              |
| 6   | `:449-453` | 建模块（二节），`engine = MPCacheServer(ctx, modules)`；**先注册 `engine.close` 进 ExitStack 再继续**（`:450-452`）——后续任何一步失败，已建模块与上下文都会被回滚释放 |
| 7   | `:457-469` | P2P 启用时建独立 gRPC 监听器并 attach                                                                                                                                 |
| 8   | `:471-486` | usage telemetry 四连：MP 上下文、连续用量（挂 chunk_size listener）、L2 connector、L1                                                                                 |
| 9   | `:477-484` | coordinator 在场时把 `bind_coordinator_chunk_size` 挂成第二个 chunk_size listener                                                                                     |
| 10  | `:489-493` | `create_request_server`（worker 服务面），同样进 ExitStack（`:493`）                                                                                                  |
| 11  | `:495-508` | 日志 + `torch_dev.init()`（后端不支持则跳过告警）                                                                                                                     |
| 12  | `:509-512` | `server.start()` → `p2p_server.start()` → `startup_resources.pop_all()`（回滚保险解除）                                                                               |

`run_cache_server`（`:300-356`）：`return_engine=False` 时 `while True: sleep(1)` 驻留；`KeyboardInterrupt` → `event_bus.stop()` → `server.close()` → `engine.close()`；`finally` 里 `shutdown_gc_monitor` + `shutdown_grpc_otel_metrics`（`:352-354`；后者非 daemon，启动失败也要补关，`:336-339`）。裸 `__main__` 入口先装 `SIGTERM → default_int_handler`（`:534`），让 SIGTERM 走同一条 KeyboardInterrupt 退出路径。

### 8.2 退出序

`MPCacheServer.close`（`:142-149`）：P2P 监听器（wait=True 排空）→ 模块倒序（management 的 reaper 先停，`management.py:124-127`；kv_events 关订阅流 `kv_events.py:232-236`；BlendModule 摘 session 监听器、停 fingerprint worker、关 coordinator 客户端，`blend/module.py:321-329`）→ `context.close()`（session 清理线程 + storage manager 全家：控制器线程、L2 adapter、GDS 上下文，`engine_context.py:417-422`）。

HTTP 模式的 lifespan 收尾（`http_server.py:244-263`）：取消注册任务（触发 `keep_registered` finally 里的 DELETE 注销）→ 关 httpx client → 停插件 → `event_bus.stop()` → `request_server.close()` → `engine.close()` → 关 gRPC OTel metrics。

### 8.3 三类超时与身后事

| 超时                         | 默认                                                   | 到期行为                                                                                                                                                                                                  |
| ---------------------------- | ------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| worker 静默（ping 断）       | 120s，扫描周期 30s（`management.py:90-98`）            | transfer 模块 `reap_stale_instances`（`lmcache_driven_transfer.py:416-461`）经 `_release_entries`（`:463`）释放该 worker 的 GPU 上下文，BlendModule 丢 rope 状态；worker 重连后需重新 `register_kv_cache` |
| 注册后从未 ping              | 3600s grace                                            | 同上；配置层强校验 grace ≥ reap（`config.py:221-225`）                                                                                                                                                    |
| request session 空闲         | 600s（`session.py:227`），60s 扫一次（`:228,352-366`） | `SessionManager.cleanup_expired`（`:294-313`）删 session 并逐个跑 `destroy_listeners`（`:315-332`）——BlendModule 借这里兜底释放 retrieve 没消费的读锁（`blend/module.py:169,302-319`）                    |
| server 心跳对 coordinator 断 | coordinator 侧 `instance_timeout`                      | `_evict_stale` 除名（`app.py:360-373`）；server 恢复后心跳 404 → 重注册 + `publish_capacity` 补发                                                                                                         |

## 九、blend 装配差异：`engine_type="blend"` 实际多了什么

| 差异点         | 位置                                                                                                                                                                                                                                                                                                                                                                                                                   | 内容                                                                                                                                                                                                                                                   |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 上下文开关     | `server.py:437,445`                                                                                                                                                                                                                                                                                                                                                                                                    | `full_sw_kv=True`：滑窗组缓存整 chunk KV，blended chunk 可在任意位置复用（`engine_context.py:533-535`）                                                                                                                                                |
| 模块组         | `blend/module.py:66-74`                                                                                                                                                                                                                                                                                                                                                                                                | `BlendModule` = `FingerprintEvictionMixin + LookupMixin + RegistrationMixin + RetrieveMixin + StoreMixin + InstanceLivenessTarget` 六合一                                                                                                              |
| 构建门         | `blend/module.py:85-122`                                                                                                                                                                                                                                                                                                                                                                                               | 非 blend 直接返回 None；`engine_driven` 模式报错；coordinator 有 URL 但没开 `--coordinator-event-reporting` 时降级为纯本地匹配并告警（`:98-105`）；`BlendCoordinatorClient.maybe_create` 建舰队指纹目录客户端                                          |
| 依赖约束       | `blend/module.py:82`                                                                                                                                                                                                                                                                                                                                                                                                   | 必须排在 `LMCacheDrivenTransferModule` 之后（`require` 取它的 `cache_contexts`）                                                                                                                                                                       |
| **store 覆盖** | `blend/store.py:61-67`                                                                                                                                                                                                                                                                                                                                                                                                 | blend 也注册名为 `store` 的 handler；`bind_module_handlers` 后者覆盖前者（`server_handlers.py:62-76`）——blend 先委托 transfer 的 store 写 KV，再（worker 0）把 chunk hash 投进指纹队列异步注册（`blend/module.py:210-220` 的 `cb-fingerprint-worker`） |
| blend 专属 RPC | `cb_protocol_handshake`（`blend/module.py:248-250`，协议版本 1）、`cb_register_rope`/`cb_unregister_rope`（`blend/registration.py:54,220`）、`cb_unified_lookup`（`blend/lookup.py:486`，BLOCKING、submit-once/poll-on-recall）、`cb_retrieve_pre_computed`（`blend/retrieve.py:565`，BLOCKING+affinity）、`cb_reorder_plan`（`blend/module.py:265`，仅 `--enable-blend-reorder` 且无 coordinator 时有料，`:156-163`） |
| 生命周期钩子   | `blend/module.py:169,302-319,331-335`                                                                                                                                                                                                                                                                                                                                                                                  | session destroy 监听器放未消费读锁；reaper 收割 worker 时丢 CB 状态；`close` 摘监听器、停指纹线程、关 coordinator 客户端                                                                                                                               |

一句话：blend 不是另起一套存储，而是在 default 装配之上**多挂一个模块组**——换掉 `store` 操作的入口、追加五个 CB RPC、把 `full_sw_kv` 打开、再接一条到 coordinator blend 索引的 HTTP 客户端。

## 十、总评与深读发现

总评：v0.5.8 的 server 是一个「微内核 + 声明式模块」结构——组合器本体不到 90 行（`server.py:85-175`，含 docstring），真正的装配逻辑全部压在模块发现、拓扑定序和 create 门里；传输端点不承载业务，只按 `HandlerType` 把 handler 调度到三种线程池。判断的折扣清单：

- **模块数比直觉多**：主 worker 面最多 7 个内置模块 + 1 个隐藏 plugin router，`report_status` 的合并顺序（后建覆盖先建）意味着排查状态字段时要先知道构建序，而构建序由拓扑排序决定，不完全等于包名顺序。
- **「engine_type 三种装配」的预期不成立**：`engine_type` 只有 `default`/`blend` 两档生效（`config.py:84-86`），且只影响 BlendModule 的 create 门和 `full_sw_kv`；没有 `p2p` engine_type——P2P 是 `p2p_config.enabled` 控制的**第二个 gRPC 监听器**，靠 `rpc_service` 字段从同一份模块列表里切出 `P2PController`。
- **L1 驱逐 ≠ L1→L2 落盘**：驱逐控制器只 `DISCARD`；落盘由 StoreController 在 write-finished 后异步做，两者通过读锁预算（`trigger_watermark * (1 - eviction_ratio)`，`store_controller.py:694`）间接协调。把「L2 命中率低」归因到驱逐参数会找错文件。
- **`HandlerType.NON_BLOCKING` 是死代码**：两套传输都在分发处抛 `NotImplementedError`；声明面允许它存在，读代码时可以跳过。
- **coordinator 注册链路只在 HTTP 前端模式生效**：`run_cache_server` 直跑（`server.py __main__`）不注册 coordinator、不报 cache 事件；P2P 的对端发现轮询倒是模块自己带、不受此限制。
- **store 的覆盖机制是隐式的**：blend 接管 `store` 依赖「后建模块覆盖同名 operation」这一规则，代码里没有任何显式声明；同规则也被外置模块用来顶替内置 handler，排查「store 怎么多干了活」时要从 `bind_module_handlers` 的循环顺序看起。
- 行号均实测于 `git show v0.5.8:`；`modules/` 各文件的 `create` 门与 handler 装饰器逐个打开核对过，无推测成分。

## 源文件索引

| 文件                                                                                                                                                                                          | 关键类/函数                                                                                                                        |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `lmcache/v1/multiprocess/server.py`                                                                                                                                                           | `MPCacheServer`（组合器）、`_build_server_components`（装配）、`run_cache_server`/`_start_cache_server`（启动/退出）、`parse_args` |
| `lmcache/v1/multiprocess/engine_module.py`                                                                                                                                                    | `DiscoverableModule`/`EngineModule` 协议、`ModuleBuildContext`、`modules_for_service`、`order_modules`、`discover_modules`         |
| `lmcache/v1/multiprocess/engine_context.py`                                                                                                                                                   | `MPCacheServerContext`（chunk_size 协商/绑定监听）、`finish_lookup_read`、`LayoutDescRegistry`                                     |
| `lmcache/v1/multiprocess/request_handler.py`                                                                                                                                                  | `HandlerType`、`request_handler` 装饰器、`iter_request_handlers`/`validate_handler`                                                |
| `lmcache/v1/multiprocess/transport/server_factory.py`、`grpc_impl/server_builder.py`、`grpc_impl/{server,aio_server,native_server,server_base,server_handlers}.py`、`zmq_impl/{server,mq}.py` | 两条传输 + 三种 gRPC 实现 + handler 绑定/分发                                                                                      |
| `lmcache/v1/multiprocess/affinity_pool.py`                                                                                                                                                    | `AffinityThreadPool`、`DeferredResponse`                                                                                           |
| `lmcache/v1/multiprocess/modules/{lookup,management,lmcache_driven_transfer,engine_driven_transfer,kv_events,p2p_controller}.py`                                                              | 七个内置模块                                                                                                                       |
| `lmcache/v1/multiprocess/modules/blend/*.py`                                                                                                                                                  | `BlendModule` 及 store/lookup/retrieve/registration/rope/reorder/fingerprint 混入                                                  |
| `lmcache/v1/multiprocess/ext_server_module.py`                                                                                                                                                | 外置模块加载、`ExtServerModuleRouter`                                                                                              |
| `lmcache/v1/multiprocess/{http_server,http_api_registry}.py`、`http_apis/*.py`、`cache_control/*`、`dependencies.py`                                                                          | HTTP 前端与路由落点                                                                                                                |
| `lmcache/v1/multiprocess/{config,session,kv_load_policy,rocm_host_event_wait}.py`                                                                                                             | 配置面、会话管理、KV 负载策略                                                                                                      |
| `lmcache/v1/distributed/storage_manager.py`                                                                                                                                                   | L1/L2/控制器装配总装                                                                                                               |
| `lmcache/v1/distributed/storage_controllers/{eviction_controller,store_controller,prefetch_controller}.py`                                                                                    | 驱逐/落盘/预取三控制器                                                                                                             |
| `lmcache/v1/distributed/{eviction,eviction_policy/*}.py`                                                                                                                                      | 驱逐策略基类、桥接、四策略工厂                                                                                                     |
| `lmcache/v1/mp_coordinator/{registrar,cache_events,app}.py`、`views/instance_registry.py`                                                                                                     | 注册/心跳、事件 sink、成员过期                                                                                                     |
| `lmcache/v1/client/heartbeat.py`                                                                                                                                                              | worker 侧心跳线程                                                                                                                  |
| `lmcache/cli/commands/server.py`                                                                                                                                                              | `lmcache server` 入口                                                                                                              |

## 相关篇

- 地图篇（四角色与请求生命周期）：[01_mp_architecture.md](01_mp_architecture.md)
- vLLM 侧对接：[02_lmcache_mp_connector.md](02_lmcache_mp_connector.md)
- server 端点/配置速览：[03_mp_server.md](03_mp_server.md)
- L1/L2 存储与 adapter：[04_storage_l1_l2.md](04_storage_l1_l2.md)
- `lmcache` CLI 子命令：[09_cli.md](09_cli.md)
