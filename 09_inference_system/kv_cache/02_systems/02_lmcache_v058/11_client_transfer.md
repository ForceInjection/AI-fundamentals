# LMCache v0.5.8 · Client 传输机制深读(SHM / ZMQ / gRPC)

> 核对基线:LMCache v0.5.8(tag `651617ef`);写作日期 2026-11-26
> 文中行号均为 @v0.5.8,可在 /Users/wangtianqing/Project/ai-infra/LMCache 用 `git show v0.5.8:<path>` 逐行核对。

## 传输面总览

v0.5.8 的 vLLM 进程内只剩 connector/adapter,全部 KV 数据移动发生在 client 侧的三条通道上。三条通道分工固定,不互为备份:

| 通道                | 实现位置                                    | 承载内容                                        | 数据封帧                                         |
| ------------------- | ------------------------------------------- | ----------------------------------------------- | ------------------------------------------------ |
| 共享内存(SHM 槽位)  | `v1/client/transfer_context/shm.py`         | engine_driven 模式的 KV chunk 本体              | 无封帧:worker 直接读写 server L1 池的字节        |
| inline(MessagePack) | `v1/client/transfer_context/inline.py`      | SHM 不可用时的 KV chunk 本体                    | `tensor_wire.py` 的 msgspec MessagePack envelope |
| 请求通道(ZMQ/gRPC)  | `v1/client/transport/{zmq_impl,grpc_impl}/` | 全部 38 个 RPC 契约方法 + 控制消息 + 槽位描述符 | msgspec MessagePack 帧 / protobuf                |

辅助机制两个:heartbeat(`v1/client/heartbeat.py`)保活,`v1/client/kv_event_subscriber.py` 订阅 server 的 CPU 缓存事件流。

RPC 契约本体是 `v1/transport/request_client.py` 的 `RequestClient` Protocol,38 个 `@rpc_method` 方法,唯一流式方法是 `subscribe_kv_events`(`request_client.py:254-257`)。ZMQ 与 gRPC 两个客户端类都从这一份契约自动生成自己的方法(`zmq_impl/client.py:80-94`、`grpc_impl/client.py:292-301`),加操作不需要改传输层代码。

通道选路按 URL scheme(`transport/factory.py:13-14`):`tcp://`、`ipc://`、`inproc://` 走 ZMQ,`grpc://`、`grpc+unix://` 走 gRPC;裸 `host:port` 默认补成 `tcp://`(`factory.py:22`)。connector 侧 `_ensure_transport_scheme` 对裸地址做同样的补全(`lmcache_mp_connector.py:585-587`)。

## 共享内存传输:transfer_context

### 模式选择与上下文族

入口 `create_transfer_context`(`v1/client/worker_transfer.py:1669-1732`)按 KV 张量设备类型路由,mode 可由参数、`LMCACHE_MP_TRANSFER_MODE` 环境变量(`:65`)或 `lmcache.mp.mp_transfer_mode` extra-config 指定。`MPTransferMode` 三个值(`v1/platform/transfer_mode.py:14-26`):`auto` 按设备声明默认(CUDA/NPU 走 lmcache_driven,其余走 engine_driven);`lmcache_driven` 是 server 经设备句柄拉数据的 IPC/SHM 零拷贝路径,不在本篇展开;`engine_driven` 是 worker 自己做 gather/scatter 的路径,其下再分 SHM 与 inline 两种实现。

engine_driven 的抽象基类 `EngineDrivenContext`(`transfer_context/base.py:100-283`)定义两阶段接口:`prepare_store`/`commit_store`、`prepare_retrieve`/`commit_retrieve`,工厂 `create_engine_driven_context`(`base.py:286-335`)按注册响应里是否带 `shm_name` 与正 `pool_size` 决定实例化哪个实现;SHM 初始化抛任何异常都回退 inline(`base.py:323-329`)。元数据载体 `EngineDrivenContextMetadata`(`base.py:72-93`)带 layout 描述、block_size、`use_mla` 与多组(hybrid-KV)的 `group_layouts`/`group_windows`。

### shm 段由 server 创建,worker 只挂载

段的所有权在 server 侧:

- L1 DRAM 分配器 `MixedMemoryAllocator` 创建 POSIX shm 段(`v1/distributed/memory_manager/l1_memory_manager.py:78-91`),默认名 `lmcache_l1_pool_<pid>`(`v1/distributed/config.py:275`),创建前先 `_unlink_stale_shm` 清残留段(`l1_memory_manager.py:27-44`);
- server 启动时 `_compute_shm_pool_info` 把池元数据放进共享上下文(`v1/multiprocess/engine_context.py:663-682`):L1 管理器数不为 1、名字为空、lazy 内存模式或 devdax L1,任一命中就返回空池,SHM 传输即关闭;
- worker 侧 `register` 发 `register_kv_cache_engine_driven_context`(`worker_transfer.py:1069-1084`),响应带回 `shm_name`、`pool_size`、`null_block_id`(`worker_transfer.py:1091-1095`),再交给工厂建上下文(`:1103-1109`)。日志明确打印 `mode=SHM` 或 `mode=inline`(`:1110-1117`)。

### worker 侧挂载:`EngineDrivenContextShm.__init__`

`transfer_context/shm.py:90-127`,逐行看做四件事:

1. 校验参数后 `shared_memory.SharedMemory(name=shm_name.lstrip("/"), create=False)` 挂载既有段(`:111-113`),不创建;
2. `resource_tracker.unregister(f"/{name}", "shared_memory")`(`:117`),把段从本 worker 的 resource tracker 注销,否则 worker 进程退出时 Python 会替 server 把段 unlink 掉;
3. `_pin_shm_buffer`(`:342-371`)用 `cudaHostRegister` 把整段页锁,pinned 失败只告警,D2H 拷贝退化为同步(`:365-371`);注释写明 pin 是每进程的,server 侧可能已 pin 过,worker 再 pin 是为了快速 DMA 拷贝(`:119-121`);
4. 失败路径把 `_shm`/`_shm_buffer` 置空后原样抛出(`:124-127`)。

槽位描述符 `ShmSlotDescriptor`(`:30-80`)四个字段:`offset`/`length`/`shape`/`dtype`,在 MQ 上下文字典里序列化与解析(`to_dict`/`from_dict`)。worker 收到描述符后 `_make_tensor_view`(`:129-151`)用 `torch.frombuffer(self._shm_buffer, dtype=..., count=length//itemsize, offset=offset)` 建一维视图再 `view(shape)`,这就是零拷贝入口:不复制字节,只建视图。

### store 全流程(逐函数)

| 步骤   | 函数                              | 行号(shm.py) | 行为                                                                                                                                                                                                               |
| ------ | --------------------------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 预约   | `prepare_store`                   | `:165-186`   | 发 `PREPARE_STORE`,`future.wait(timeout)` 先行以便超时只抛一个 `LMCacheTimeoutError`;响应 `context["slots"]` 为空列表时返回 `([], [])`,含义是「全部 chunk 已缓存,跳过 gather 与 commit」;非列表返回 `None` 当 miss |
| 建视图 | `_build_slot_tensors`             | `:153-163`   | 描述符逐个转 tensor 视图                                                                                                                                                                                           |
| gather | `gather_paged_kv_to_cpu`(base.py) | `:400-597`   | 把 paged KV 的 D2H 拷贝直接写进 SHM 视图(见下)                                                                                                                                                                     |
| 提交   | `commit_store`                    | `:256-263`   | `commit_store(key, instance_id, b"")`:KV 字节已在池里,载荷为空,只是通知 server 落账;超时返回 False                                                                                                                 |

`gather_paged_kv_to_cpu`(`base.py:400-597`)值得细看的点:

- 后端按 KV 张量的设备选(`:464-469`),`_detect_block_transfer_accepts_tensor`(`:50-66`)用签名内省判断原生 kernel 收 tensor 还是收指针;收指针的路径经 `transfer_with_address_kernel`(`:579-590`),此时 gather 目标必须 pinned(`requires_pinned = not accepts_tensor`,`:513`);
- 滑动窗口组只取每个 chunk 的尾部 `blocks_per_window` 个块(`bpw`,`:483-484`、`:548-554`),全注意力时等于全量;
- 返回的 chunk 形状:split-K/V 是 `[2, num_layers, window_tokens, hidden]`,单平面(MLA 与 fused-K/V,`kv_size == 1`)是 `[num_layers, window_tokens, hidden]`(`:518-537`);
- 结尾**刻意不同步设备**(`:592-597`),注释要求调用方在消费前自己 synchronize;`submit_store` 里因此在 gather 后补了一次无条件 `synchronize_device`(`worker_transfer.py:1185-1191`,注释解释 inline 路径若只在有 out_buffers 时同步,会序列化一个还在写入的缓冲)。

调用侧编排见 `EngineDrivenTransferContext.submit_store`(`worker_transfer.py:1145-1204`):`transfer_guard` 内先同步、prepare、gather、再同步、commit;`ContextClosedError` 与 `REQUEST_FAILURES` 都折成 `future.set_result(False)`,store 静默丢失,不打断推理。

### retrieve 全流程

对称结构:`prepare_retrieve`(`shm.py:265-294`)发 `PREPARE_RETRIEVE`,响应 `success` 为假或超时算 miss;槽位数与 `expected_count` 不符抛 `ValueError`(`:289-293`);命中返回 SHM 视图。worker 对视图做 H2D scatter(`scatter_cpu_to_paged_kv`,`base.py:600-769`):非块对齐的 `skip_first_n_tokens` 向下取整到整块并记 error 日志(`:686-700`),滑动窗口组用 `recalculate_blocks_to_skip` 把 APC 的整 chunk 前缀换算成窗口块(`:701-708`);结尾同样不同步(`:766-769`)。最后 `commit_retrieve`(`shm.py:296-301`)通知 server 释放读锁。

多组(hybrid-KV)变体:`prepare_store_grouped`(`:188-223`)与 `prepare_retrieve_grouped`(`:225-254`)返回平行的 `(tensors, chunk_indices, group_ids)`,server 按 group 跳过全空 chunk 的槽位预留。inline 侧对应方法名为 `prepare_retrieve_multigroup`(`inline.py:113-143`),返回的是数据本体而非槽位视图,两个类刻意不同名。

### close:先排水再解绑

SHM 池在传输进行中被解绑是 use-after-unmap 原生崩溃,所以 `close`(`shm.py:303-326`)的次序是:置 `_release_started` → `_drain_transfers(timeout=30.0)`(基类 `:153-176`,置 `_closing` 拒新转让在飞者结束,超时返回 False)→ 成功则 `_release_shm`(`:332-340`,先 unpin 再 `close()` unmap);超时则起名为 `lmcache-shm-release-*` 的 daemon 线程 `_release_after_transfers`(`:328-330`)无限等完再释放。注释给出的取舍:一个永不结束的传输会让池一直映射到进程退出,这比崩溃好(`:308-314`)。

传输期间的进入券是 `transfer_guard`(`base.py:132-151`):条件变量上的 inflight 计数,closing 时抛 `ContextClosedError`(`:96-97`)。

### inline 回退实现

`EngineDrivenContextInline`(`inline.py:25-155`)在 SHM 不可用时兜底,数据走请求通道:

- `prepare_store`(`:45-54`)发 `PREPARE_STORE` 后**吞掉超时**返回 `None`(inline 模式 server 回空槽位,worker 自己 gather 到新 CPU 张量);
- `commit_store`(`:56-83`)是唯一传数据的 commit:平铺 chunk 列表走 `encode_tensors`,多组的 `(chunk_indices, tensors)` 元组列表走 `encode_grouped_tensors`,序列化字节作为 `COMMIT_STORE` 的 `data` 载荷发出;
- `prepare_retrieve`(`:85-111`)反序列化响应字节,按注册的 layout 逐张校验 shape/dtype/count;
- `commit_retrieve`(`:145-152`)只是 no-op 通知;`close`(`:154-155`)无资源,空实现。

### 线格式:`tensor_wire.py`

inline 路径的全部字节格式集中在 `v1/client/transfer_context/tensor_wire.py`,核心是 msgspec Struct(类型化 MessagePack,`forbid_unknown_fields=True`):

| 结构                    | 行号     | 字段                                                                |
| ----------------------- | -------- | ------------------------------------------------------------------- |
| `TensorFrame`           | `:27-33` | `dtype: str`、`shape`(≤16 维)、`data: memoryview`(≤2 GiB)           |
| `TensorEnvelope`        | `:38-42` | `version: Literal[1]` + `TensorFrames`(≤65536 个)                   |
| `TensorGroup`           | `:48-52` | `chunk_indices`(严格递增)+ 该组的 frames                            |
| `GroupedTensorEnvelope` | `:58-73` | `version: Literal[1]` + `groups`(≤64);字段名与平铺版不相交,互不解码 |

上限常量(`:21-24`):单 tensor 2 GiB、总载荷 16 GiB、tensor 数 65536、每 tensor 元数据 1024 字节;dtype 必须在 `DTYPES` 白名单内(`v1/platform/base/ipc_wire.py` 导入,`:104-105`)。

编码 `encode_tensors`(`:76-92`)→ `_encode_frames`(`:95-118`):非 CPU 张量直接拒绝(`:101-102`),字节化是 `tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()`(`:109-111`)。这一步是实打实的内存拷贝,inline 路径因此至少多一次完整 memcpy。

解码 `decode_tensors`(`:121-160`)→ `_decode_frames`(`:163-209`):逐帧校验 dtype 名、shape 元组、字节数与期望布局严格相等(`:182-196`),累计字节过 16 GiB 上限即拒(`:197-199`);然后 `torch.frombuffer(frame.data, dtype=torch.uint8)` 再 `view(dtype).reshape(shape)`(`:201-208`),返回的张量直接以不可变的输入 payload 为底,注释要求调用方只读(`:137-138`)。grouped 版 `encode_grouped_tensors`/`decode_grouped_tensors`(`:212-241`/`:265-316`)在同一预算下多校验 chunk_indices 严格递增、每轴计数一致(`_check_chunk_indices`,`:244-262`)。

### 零拷贝路径在哪

- SHM store:KV 字节唯一的移动是 GPU 到池槽位的 D2H 拷贝(gather 直接写视图),请求通道上只有槽位描述符和空的 commit 通知,无序列化、无 KV 字节过 socket;
- SHM retrieve:反向同理,scatter 从池视图直接 H2D;
- inline store/retrieve 各多两次拷贝(编码 tobytes 一次,解码 frombuffer 零拷贝但随后 scatter 要读)加上整份 payload 走 ZMQ/gRPC;
- `lmcache_driven` 模式是第三条路:server 经 IPC 设备句柄自己拉数据(见 `MPTransferMode` docstring,`transfer_mode.py:21-24`),本篇不展开。

## ZMQ 通道:zmq_impl

### 套接字与保活参数

`MessageQueueClient.__init__`(`zmq_impl/mq_client.py:196-229`)建 `zmq.DEALER`,三个套接字级设置:`CONNECT_TIMEOUT=5000ms`、`RECONNECT_IVL_MAX=1000ms`(`:54-55`,重连退避上限 1 秒),再挂一个只监听 `EVENT_DISCONNECTED` 的 monitor(`:204`)。libzmq 自动重连,没有应用层 keepalive 帧;应用层保活是下文的 heartbeat。

### 消息封装

出站帧格式 `[request_uid, operation, *payloads]`,全部 msgspec MessagePack 编码;若配了 api_key,前面再插一帧版本化认证信封(`_with_authentication`,`:321-325`;帧格式在 `v1/transport/zmq/wire.py:98-135`,magic `LMCACHE-ZMQ-AUTH\x00`)。operation 编码兼容两代:老操作用冻结的整数 ID 表 `LEGACY_OPERATION_IDS`(`wire.py:62-95`,注释声明 ID 11 与 29-31 已退役不得复用),新操作用字符串名(`encode_operation`,`wire.py:143-146`)。`request_uid` 是进程内 `itertools.count` 自增整数(`:221`)。

`process_outbound_task`(`:231-319`)在共享轮询循环里执行,要点:

- payload 数量与契约 `rpc_spec.payload_types` 不符时抛错,错误消息直接提示「可能是 client 与 server 版本错配」(`:268-283`);
- socket 无房间(NOBLOCK 发送遇 `zmq.Again`)的帧连同其后所有排队帧一起滞留 `_unsent` deque,下轮先发(`:246-250`、`_send :551-564`);注释说明这里绝不能阻塞发送,否则会卡死共享同一轮询循环的全部客户端(`:554-558`);
- 已判死错传(`_wrong_transport`)时 `_fail_all` 清掉一切(`:242-244`)。

入站 `process_inbound`(`:327-395`)按 request_uid 分派:认证失败帧转 `PermissionError`(`:345-353`);流式响应直接喂单槽 queue(`:355-367`);一元响应先查保留错误操作 `__lmcache_rpc_error__`(`wire.py:14-15`)转 `MPRequestError`(`:372-382`),再查 operation 名一致(`:383-390`),最后按契约的 `response_type` 解码(`:391-395`)。错误类型族在 `transport/errors.py`:`MPServerUnavailableError` 同时继承 `TimeoutError` 与 `ConnectionError`(`errors.py:5-22`,注释解释 ZMQ 侧同一失败表现为超时),`MPRequestError` 是 server 明确拒绝。

### 共享轮询循环:`ClientPollingLoop`

每进程一个单例(`polling_loop.py:57-118`),引用计数管理:首个客户端注册时懒启动,最后一个 `release_instance` 时退出。一个线程 `zmq.Poller` 同时轮询所有客户端的 DEALER socket、monitor socket 与一个事件通知 fd(`:78-84`),`notify()`(`:146-148`)唤醒出站处理;poll 超时 1000ms(`:181`)兜底。有滞留发送的 socket 动态加挂 `POLLOUT`(`_send_outbound`,`:222-232`)。monitor 收到 `EVENT_DISCONNECTED` 调 `handle_disconnect`(`:196-203`);单个客户端的坏帧只丢帧不断循环(`:207-217`,注释:循环一死会拖垮进程内所有其他客户端的在途请求)。

### 流式订阅

`submit_request`(`:397-451`)对 streaming 契约建 `MessagingStream`:单槽 `queue.Queue(maxsize=1)` + 控制帧 `[request_uid, b"", b"ack"/b"cancel"]`(`:418-436`)。读者每读到一条先 ack;cancel 时塞 None 唤醒读者再发控制帧。断线时 `cancel_streams`(`:453-460`)唤醒所有流读者。

### 断线与「对端在说 gRPC」探测

ZMTP 握手失败的对端 libzmq 不再重连,发往 gRPC server 的请求会一直干等到超时,断线事件是唯一能察觉的机会。`handle_disconnect`(`:465-477`)触发 `PeerCheck`(`:130-184`)状态机:断线时扣留在途流并起一个后台探测 `_peer_speaks_http2`(`:63-90`,裸 TCP 读 9 字节,认 HTTP/2 SETTINGS 帧头),探测期间再断线就要求再查一次(`_again`);结果是 gRPC 则永久标记 `_wrong_transport`,`_apply_probe_results`(`:488-507`)在循环线程里生成一条明确报错:「server 在说 gRPC,请用 `grpc://host:port`,或以 `--transport zmq` 启动 server」,并 `_fail_all`(`:525-544`)放掉全部在途与排队请求。

## gRPC 通道:grpc_impl

### `GrpcMultiprocessClient`

`grpc_impl/client.py:97-302`。`__init__`(`:100-142`)五步:

1. `parse_grpc_target`(`v1/transport/grpc/endpoint.py:16-43`)把 `grpc://`/`grpc+unix://` 归一成 grpcio target;
2. `grpc.insecure_channel` + `GRPC_CHANNEL_OPTIONS`(`v1/transport/grpc/utils.py:22-31`):消息长度不限、每尝试连接超时(`min_reconnect_backoff_ms`)= 5000ms、初始退避 100ms、最大退避 1000ms、禁 HTTP 代理;注释说明这些界与 ZMQ 客户端相同,grpcio 默认是 20s/120s;
3. 有 api_key 时 `grpc.intercept_channel` 挂认证拦截器(`:106-109`);
4. 遍历生成的服务绑定建 stub,把每个方法连同编解码器存进 `_rpc_methods`(`:125-141`);方法名由 CamelCase 转 snake,特殊还原 `p2_p` → `p2p`(`v1/transport/grpc/descriptors.py:30-34`);`method_registry`(`v1/transport/grpc/method_registry.py:52-121`)在构建时双向校验:每个生成方法必须有同名 RPC 契约,每个契约方法必须有生成方法,流式标记必须一致;
5. 每次调用附 `lmcache-client-id-bin` 元数据,值是 `uuid4().bytes`(`:142`;key 定义 `endpoint.py:9`,server 用它做亲和路由)。

调用点 `_call`(`:213-262`):一元 RPC 走 `.future()` + `add_done_callback` 转 `MessagingFuture`;server-streaming 走同步 multi callable 转 `MessagingStream`。全部调用 `wait_for_ready=True`(`:222-223`、`:239-243`),注释:没有它,连到端口上但还没起动的 server 看起来就像永远不在。错误经 `request_error`(`utils.py:35-48`)转换:`UNAVAILABLE`/`CANCELLED` → `MPServerUnavailableError`,其余 → `MPRequestError`,二者都带 gRPC 状态码名与 details。

重连行为完全交给 grpcio 通道(退避 100ms 起、1s 封顶),客户端不做应用层 keepalive。

### 认证:`GrpcClientAuthenticationInterceptor`

`grpc_impl/auth.py:27-70`。实现 `UnaryUnaryClientInterceptor` 与 `UnaryStreamClientInterceptor` 两个接口,即覆盖一元与 server-streaming 两类调用。`_authenticated_details`(`:36-52`)先剥掉调用元数据里已有的 `authorization` 项再追加一个,值为 `bearer_authorization_header(api_key)`,即 `Bearer <key>`(`v1/security/bearer.py:28-35`);其余 call details 字段原样透传到 frozen dataclass `_AuthenticatedClientCallDetails`(`:17-24`)。server 侧分发前校验同一 header。

api_key 的来源链在 `v1/transport/auth.py:31-63` `load_request_client_api_key`:显式参数(非空校验)优先,其次 key 文件的第一个键,最后环境变量源;两个环境源同时配置直接 `ValueError`,多个键保留为 server 端轮换键但客户端只发一个。ZMQ 侧对等物是认证帧(`wire.py:110-135`),带 `version` 字段可演进,失败时 server 回 `LMCACHE-ZMQ-AUTH-ERROR\x00` 帧(`:99`)。

### 反向探测:server 其实在说 ZMQ

镜像实现:`_peer_speaks_zmtp`(`client.py:48-75`)裸 TCP 读 10 字节认 ZMTP 签名(首字节 `0xFF`、第十字节 `0x7F`)。触发点是通道订阅的连通性回调 `_on_connectivity`(`:183-198`),仅在 `TRANSIENT_FAILURE` 时起后台探测线程,5 秒节流(`_PEER_PROBE_INTERVAL_S`,`:43`);确认对端是 ZMTP 后置 `_wrong_transport`(报错文案指向 `tcp://host:port` 或 `--transport grpc`,`:203-207`),并 cancel 全部在途调用(`_track`/`_pending`,`:264-272` 保证判定与 cancel 扫描之间没有调用漏网)。

### P2P 五 RPC 的客户端:`P2PClient`

worker 数据通道之外,cache server 之间还有一套独立的 gRPC 控制面。客户端 `v1/distributed/p2p.py:77-161` `P2PClient`:同样 `insecure_channel` + `GrpcClientAuthenticationInterceptor`(`:88-99`),挂在 `P2PServiceStub` 上;pb 消息编解码器在模块顶层由 `compile_request_codec_for_types` 按 Python 类型编译(`:36-56`);`_future`(`:59-74`)把 grpc.Future 桥成 `MessagingFuture`,错误走同一个 `request_error`。

五个 RPC 与客户端调用点(P2PClient 方法 → 调用处 `v1/distributed/l2_adapters/p2p_l2_adapter.py`):

| RPC(方法名 @p2p.py)        | 行号       | 语义                             | 调用点                                                                                             |
| -------------------------- | ---------- | -------------------------------- | -------------------------------------------------------------------------------------------------- |
| `p2p_lookup_and_lock`      | `:101-111` | 提交 keys + 组 layout,回 task_id | `submit_lookup_and_lock_task`(`p2p_l2_adapter.py:331`),RPC 自身超时 `_LOOKUP_RPC_TIMEOUT_S = 3.0s` |
| `p2p_query_lookup_results` | `:113-123` | 轮询任务,None=仍在查             | `query_lookup_and_lock_result`(`p2p_l2_adapter.py:363`)                                            |
| `p2p_unlock_objects`       | `:125-133` | 放 peer 读锁                     | `submit_unlock`(`p2p_l2_adapter.py:379`)                                                           |
| `p2p_reserve_write`        | `:135-145` | 每 key 预留写 buffer,回地址      | `submit_store_task`(`p2p_l2_adapter.py:556`)                                                       |
| `p2p_finish_write`         | `:147-157` | 按上报成败提交/丢弃,回驻留结果   | `_discard_late_reservation`(`:738`)、`_check_write`(`:819`、`:826`)                                |

`P2PClient` 的实例化在 `PeerL2AdapterBase.__init__`(`p2p_l2_adapter.py:245-248`),端点是 peer 的 `--p2p-rpc-port` 控制面,与 worker 数据通道端口分开。

## heartbeat:HeartbeatThread

`v1/client/heartbeat.py:41-172`,继承 `PeriodicThread`(`v1/periodic_thread.py:62`),`ThreadLevel.CRITICAL`。单次探测 `send_ping`(`:16-38`):`req_client.ping(instance_id).result(timeout=timeout)`,任何失败都折成 `False`,异常只留 debug 日志。间隔与超时是同一个数:`interval` 既是探测周期又是单次探测的超时(`:63`、`:127-131`)。

状态机在 `_execute`(`:125-172`),一次循环的判定序:

1. `healthy = send_ping(...)`;
2. unhealthy → healthy 的边沿先跑 `_recover_callback`(`:138-147`),异常按恢复失败算;**恢复先于健康发布**,让 adapter 趁机把 server 侧注册补回来(类 docstring `:43-47`);
3. 恢复成功再跑 `_healthy_callback`(默认 `_set_health_event` 置事件,`:115-117`);
4. healthy → unhealthy 的边沿 `_publish_unhealthy`(`:119-123`):先跑 adapter 的 unhealthy 回调,finally 里清健康事件。

adapter 侧接线(vLLM 集成):

- scheduler adapter 每 server 一个 heartbeat,懒启动于首个 lookup/订阅(`vllm_multi_process_adapter.py:713-737`),recover 回调是重新核对 chunk size(`:739-746`);多 server 健康取 AND(`is_healthy`,`:709-711`);
- worker adapter 一个 heartbeat,懒启动于首个 store/retrieve(`:1702-1722`),recover 回调重注册 KV caches(`:1716`、`:1729` 起);启动健康,首次 store/retrieve 不被门控(docstring `:1695-1701`);
- `heartbeat_interval` 大到 3×间隔超过 30s(server reap 下限)时启动即告警(`:1426-1437`,常量 `_SERVER_REAP_TIMEOUT_FLOOR_SECONDS = 30.0` 在 `:162`)。

server 侧对应:`ManagementModule.ping`(`v1/multiprocess/modules/management.py:129-144`)带 `instance_id` 时对全部 liveness target 调 `touch_instance` 刷新 last-seen,恒返回 True;`InstanceLivenessTarget` 协议(`v1/multiprocess/engine_module.py:50-80`)定义 liveness owner(touch/reap/count)与 state mirror(`drop_instance_state`)两种角色;reaper 周期跑 `_reap_cycle`(`management.py:146-163`)。超时配置在 `v1/multiprocess/config.py`:`worker_reap_timeout_seconds` 默认 120.0(`:158`),0 关闭、非 0 下限 30s(`:215-219`),注释要求保持不低于 3×心跳间隔;从未 ping 过的 worker 用单独的 `worker_registration_grace_seconds` 默认 3600.0(`:177-180`)。

## kv_event_subscriber:MPKVEventSubscriber

`v1/client/kv_event_subscriber.py:32-151`。借用 adapter 的 request client,自持一条事件流和一个接收线程;生命周期(heartbeat、client 关闭、事件转换)归 adapter。

订阅方式:`_receive` 循环(`:117-151`)调 `subscribe_kv_events(instance_id, model_name, 0, 1024)`(`:121-123`),cursor 固定从 0 重放,批量上限 `_BATCH_SIZE = 1024`(`:17`);server 侧校验 `max_events ∈ [1, 1024]`(`v1/multiprocess/modules/kv_events.py:279-280`),事件 incarnation 取 `time.time_ns()`(`kv_events.py:179`),重放日志容量默认 32768 条(`v1/multiprocess/config.py:182-183`)。

事件类型(`v1/common/custom_types.py:350-372`):批载体 `KVEventBatch{enabled, incarnation, next_cursor, lost, events}`,单条 `KVEventRecord{seq, kind, medium, model_name, block_hashes, parent_block_hash, token_ids}`;kind 即 store/removed 两类 CPU 缓存变更,medium 目前为 `CPU`。

接收循环的失败语义(`:117-151`):

- 正常断流(含 server 回关闭)按 `ConnectionError` 处理:发 `KVEventStreamClosed()`(`error=None`,语义是「瞬时断开,重连后从零重放」,`:21-29`),等 `_RECONNECT_DELAY = 1.0` 秒(`:18`)后重订阅;
- 其他异常发 `KVEventStreamClosed(error=...)` 且**终止循环**,是否放弃由 adapter 决策(`:139-143`);
- `batch.enabled=False` 的批读完即停(`:132-133`),对应 server 端事件功能被关闭;
- 发布路径 `_publish`(`:110-115`)把批排进 `_pending` 后在锁外回调 `on_update`,契约要求消费者在回调里 `drain()` 清空(`:81-92`),不允许第二个并发消费者。

adapter 接线:开启 KV events 时 scheduler adapter 为每个 server 建一个订阅者(`vllm_multi_process_adapter.py:688-693`),`on_update` 是 `partial(self._drain_kv_event_updates, url)`;订阅者的 `instance_id` 取 `uuid4().int` 截断到 int64(`:684-686`),心跳先于订阅启动(`:694-697`)。关闭次序是先 join 接收线程、停心跳、最后才关借来的 client(`:937-943`)。

## 传输选型对照

| 维度              | 共享内存(SHM 槽位)                                            | inline(MessagePack)                                | ZMQ(DEALER)                                                  | gRPC(stub)                                                       |
| ----------------- | ------------------------------------------------------------- | -------------------------------------------------- | ------------------------------------------------------------ | ---------------------------------------------------------------- |
| 使用场景          | server 开了具名 L1 池且非 lazy/devdax 时的 engine_driven 存取 | SHM 不可用时的兜底;CPU-only worker                 | 默认数据通道(`tcp://` 裸地址自动落 here)                     | `grpc://` 显式选用;P2P 控制面只有 gRPC                           |
| KV 字节过不过网络 | 不过,只过内存                                                 | 过,整份 payload 走请求通道                         | 不直接传 KV(SHM 模式只传槽位描述符)                          | 同 ZMQ                                                           |
| 拷贝次数(store)   | 1 次(D2H 进池槽位)                                            | 2 次(D2H 进临时张量 + tobytes 序列化)              | 不适用                                                       | 不适用                                                           |
| 控制消息开销      | prepare/commit 各一往返,载荷为空或仅描述符                    | 每块一往返且载荷全量                               | 每请求一往返,MessagePack 帧,微秒级封送                       | 每请求一往返,HTTP/2 + protobuf,连接复用                          |
| 保活/重连         | 无(随上下文生命周期)                                          | 无                                                 | libzmq 自动重连,退避上限 1s;断线触发对端协议探测             | grpcio 自动重连,退避 100ms-1s;TRANSIENT_FAILURE 触发对端协议探测 |
| 认证              | 无(同机共享内存)                                              | 随请求通道                                         | 版本化认证帧(每请求首帧)                                     | `authorization: Bearer` 元数据(拦截器注入)                       |
| 失败表现          | prepare 超时抛 `LMCacheTimeoutError`,store 静默丢             | 同左,commit 超时返回 False                         | `MPRequestError`/`MPServerUnavailableError`(兼 TimeoutError) | 同左,经 `request_error` 带 gRPC 状态码                           |
| 关键上限          | 池大小 = L1 `size_in_bytes`;close 排水 30s                    | 单 tensor 2 GiB、总量 16 GiB、65536 tensors、64 组 | 无显式上限(受 HWM)                                           | 消息长度不限(`max_*_message_length=-1`)                          |

## 边界与观察

集中在这一节说判断,带折扣:

- SHM 路径的「零拷贝」只对 KV 字节成立;槽位描述符、prepare/commit 往返、以及 retrieve 端 scatter 前的只读视图都还是常规路径。收益主要在省掉序列化与 socket 字节,代价是 worker 必须与 server 同机且共享 /dev/shm(`lmcache.mp.isolated_ipc` 的存在说明跨容器场景这条路径会受限,需以 `platform/ipc_policy` 实测为准)。
- 两条请求通道的功能面是对齐的(同一份 38 方法契约、同一套错误类型、同样的重连上界 1s、互指对方的错传报错),选哪个基本由 URL scheme 决定,没有运行时切换。
- `tensor_wire` 的上限表(2 GiB/16 GiB/65536/64 组)是硬校验,超限直接 `ValueError`;chunk 布局与 `expected_shape/dtype/count` 逐项严格相等,字节多一个都拒。
- 心跳间隔即单次探测超时,默认 10s;server 默认 120s reap 是「3 次不到就收割」的口径,两端配置要成对改。
- 本文覆盖 engine_driven 的 SHM/inline 与全部请求通道;`lmcache_driven` 模式(`worker_transfer.py` 的另一族 `LMCacheDrivenTransferContext`、fd channel、VMM IPC)与多组(hybrid-KV)传输的逐函数细节只点到接口层,留作后续篇。

## 相关篇

- 总架构与请求生命周期:[01_mp_architecture.md](01_mp_architecture.md)
- server 端点、L1/L2 与驱逐:[03_mp_server.md](03_mp_server.md)
- L1 池与 L2 adapter 两级存储:[04_storage_l1_l2.md](04_storage_l1_l2.md)
- 本目录总览与旧新映射表:[README.md](README.md)
