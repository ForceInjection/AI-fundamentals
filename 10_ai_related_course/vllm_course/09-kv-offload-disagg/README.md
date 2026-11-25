# 第 9 讲 · KV 分载与 PD 分离

> 120 分钟 | 基线：nano-vllm `b9dd55a`；vLLM `2f5d8a2b1`（v0.34.1rc0-84,2026-11-25 核对）

本讲回答一个核心问题：**显存装不下的 KV，分载到 CPU、磁盘、另一台机器，分别怎么做？以及——你在 nano fork 里亲手写的 swap，离生产版还有几代？**

知识链：`--kv-offloading-size` / `--kv-transfer-config` → KVConnector（SCHEDULER/WORKER 两角色）→ 两条并行实现：`OffloadingConnector`（决策在 connector 侧 `OffloadingManager`，搬运在 `OffloadingWorker`，后端 CPU/Tiering）+ `SimpleCPUOffloadConnector`（完全独立的平行实现）→ 跨实例传输走 NIXL/Mooncake/LMCache 连接器 → disagg prefill = `kv_producer` + `kv_consumer` 两实例配对。

## 一、nano 已学形态（回顾）

你 fork 里亲手写的 `kv_swap.py`（`nanovllm/engine/kv_swap.py`）与设计文档（`docs/design/kv-offload.md`）：

- **决策与调度耦合**：swap 决策长在主 `Scheduler.schedule()` 里——waiting/running/swapped 三队列，decode 步内显式 `swap_in`/swap_out；
- **搬运**：`swap_out`/`swap_in` 在 `block_manager.py:142-180`，GPU↔CPU pinned buffer 整请求搬运；
- **约束**：`can_swap_out` 要求全部块 `ref_count == 1`（`block_manager.py:135-140`）；CPU 块**不参与前缀缓存**（只是临时停放）；
- **实测**(RTX 3090/Qwen3-0.6B):SWAP ≈ RECOMPUTE（PCIe 搬运 ≈ 重算），收益随模型规模增长。

这套实现对应 vLLM 的 **V0 时代**（CacheEngine swap + SWAPPED 队列）。它离生产版隔了两代——本讲沿三代对照表走。

## 二、三代演进对照（本讲锚点）

| 代                                     | 决策者                                                                                                                                                                                                                    | 搬运者                                                                                                                                                              | 淘汰策略                                                                                                                                     |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| **V0 CacheEngine swap**（已删除）      | 主 Scheduler:SWAPPED 队列，显式 `swap_in`/out                                                                                                                                                                               | `CacheEngine.swap_in/out`（整请求搬运）                                                                                                                             | GPU/CPU 双池 LRU                                                                                                                             |
| **v0.11-v0.2x 三件套**（课程旧文口径） | `OffloadingManager`（调度器进程）                                                                                                                                                                                         | `OffloadingHandler`（worker 进程，异步拷贝）                                                                                                                        | Manager 内 LRU（后补 ARC）                                                                                                                   |
| **HEAD**                               | 分载决策移入 **connector 侧**：`OffloadingConnectorScheduler` 或 `SimpleCPUOffloadScheduler` 驱动 `OffloadingManager.lookup/prepare_store`；主 Scheduler 只留 recompute 抢占（`scheduler.py:1829-1876`）与询问（`:1159-1163`） | `OffloadingWorker.submit_store/submit_load`（`base.py:741-770`）：CPU 走 `CPUOffloadingWorker`（`gpu_worker.py:811`，每方向独立 CUDA stream）；二级层一律经 CPU 中转 | 主层策略可插拔：`CachePolicy` lru（`policies/lru.py:26`）/**arc**（`policies/arc.py:28`，活跃维护）；二级层背压（`tiering/backpressure.py`） |

第 2 讲的伏笔在此回收：「调度器只剩 recompute」的另一半是——**swap 的决策权从调度器搬进了 connector**。措辞上注意：`OffloadingHandler` 抽象被 `OffloadingWorker` 取代，但「Handler」一词降级为内部实现名（如 `SingleDirectionOffloadingHandler`，`cpu/gpu_worker.py:314`）。

## 三、现状走读

### 环节 1：配置入口

两条互不重叠的入口：`--kv-offloading-size`(GB)+ `--kv-offloading-backend native|lmcache`（`config/cache.py:406-408`），在 `vllm/config/vllm.py:1250-1280` 翻译成 `KVTransferConfig`——native → `OffloadingConnector`（env `VLLM_USE_SIMPLE_KV_OFFLOAD=1` 则 → `SimpleCPUOffloadConnector`），lmcache → `LMCacheMPConnector`。或直接给完整 JSON：`--kv-transfer-config '{"kv_connector":"...","kv_role":"..."}'`（`config/kv_transfer.py:88`，`kv_role` 取 `kv_producer`/kv_consumer/kv_both）。**易混淆**：`vllm/config/offload.py` 是**模型权重** offload(UVA/Prefetch)，与 KV 无关。

### 环节 2：两条并行实现

**路径 A：`OffloadingConnector` + `vllm/v1/kv_offload/` 分层后端**。装配链：`OffloadingConnector.__init__`（`offloading_connector.py:82-103`）→ `OffloadingSpecFactory.create_spec`（`factory.py:54`；**内置仅两个 spec**：CPUOffloadingSpec、TieringOffloadingSpec，`:62-69`）。抽象三类全在 `base.py`（861 行）：`OffloadingManager`（`:329`，调度器侧决策者，接口 lookup/prepare_load/prepare_store 等 14 个方法）、`OffloadingWorker`（`:741`，worker 侧单介质异步搬运，`submit_store`/submit_load/get_finished）、`OffloadingSpec`（`:787`，配置→Manager+Worker 的装配）。CPU 层：`CPUOffloadingManager`（`cpu/manager.py:55`，容量不足时返回 `keys_to_retry`）与 `CPUOffloadingWorker`（`cpu/gpu_worker.py:811`，每方向独立 CUDA stream 的 DMA）。**Tiering**（`tiering/`）：CPU 主层 + 二级层（fs/obj/p2p/kvcr 四种，`tiering/factory.py` 注册），GPU↔二级层一律经 CPU 中转；二级层带背压(EMABackpressureDetector)与异步前缀查询（`HIT_PENDING`/RETRY 状态）。**注意**：顶层 `fs/` 的 `FSOffloadingWorker` 是未接线的孤儿（仅测试引用），生产 FS 路径走 tiering 二级层。

**路径 B：`SimpleCPUOffloadConnector` + `vllm/v1/simple_kv_offload/`**。入口：`VLLM_USE_SIMPLE_KV_OFFLOAD=1` 或显式 connector 名。**它与路径 A 是完全独立的平行实现**（类层次/manager/worker 均独立，零代码共享），不是「简化子集」——多了 hugepage 宿主缓冲、显式 `num_cpu_blocks`、磁盘后端（`disk_backend.py:49`，后台线程 writev/readv）、`cuMemcpyBatchAsync` 批量 DMA（`cuda_mem_ops.py`）。端到端：调度循环问 `get_num_new_matched_tokens`（`simple_kv_offload/manager.py:357`）→ `build_connector_meta`（`:552`，eager 逐块攒 store/惰性按空闲块决策）→ worker `start_load_kv`（`worker.py:326`，换入）/`wait_for_save`（`:343`，换出）→ `get_finished` 上报（`:367`）。

**共同点**：两条路径都把 swap 决策放在 **connector 自带的 scheduler 侧 manager**，主 `Scheduler` 只在每请求前缀查询时问一句 `connector.get_num_new_matched_tokens`（`scheduler.py:1159-1163`）；抢占善后走 `handle_preemptions`（`worker.py:408` / simple `:411`）。

### 环节 3:KV Connector 全家福

`vllm/distributed/kv_transfer/kv_connector/` 下的注册连接器（基类 `KVConnectorBase_V1`，`v1/base.py:182`，worker 侧 4 抽象 + scheduler 侧 3 抽象）：LMCacheConnectorV1 / LMCacheMPConnector（LMCache 单进程/多进程）、NixlConnector（Pull/Push 三态，NIXL RDMA）、MooncakeConnector（点对点带租约心跳）+ MooncakeStoreConnector（分布式 KV 池）、MultiConnector（有序组合多个）、OffloadingConnector / SimpleCPUOffloadConnector（本讲前两节）、FlexKV、HF3FS、HiSparse、MoRIIO(ROCm)、UMBP、DecodeBench、Example×2（教学）。v0 连接器目录已不存在，一切在 `v1/`。

### 环节 4:disagg prefill

`docs/features/disagg_prefill.md`：prefill 与 decode 分到不同 vLLM 实例，分别调 TP/PP 以独立优化 TTFT 与 ITL——**明确声明不提升吞吐**（动机是尾延迟与资源配比）。实现：`kv_role` 声明 producer/consumer（`kv_transfer.py:99`）；两实例经 connector 传 KV（NixlConnector 示例：`{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_connector_extra_config":{"backends":["UCX","GDS"]}}`，doc:24-50）；请求级协作走 `kv_transfer_params` 的 `do_remote_prefill`。XpYd 概念**不在 disagg 文档**，出处是 `mooncake_store_connector_usage.md:93`（MultiConnector 组合 Mooncake 点对点 + 共享 KV 池）。一个精巧细节：NIXL KV 租约（`docs/design/nixl_kv_cache_lease.md`）——P 端 KV 被 D 端 RDMA 读取前要「持住」块，租约初始 30 秒、心跳续 2/3，防 stale-read。

## 四、V0/旧教程 vs HEAD

| 旧说法                                          | HEAD 事实                                                                               |
| ----------------------------------------------- | --------------------------------------------------------------------------------------- |
| 「swap 决策在调度器里（SWAPPED 队列）」         | 决策在 connector 侧 manager；主 Scheduler 只剩 recompute（`scheduler.py:1829-1876`）    |
| 「OffloadingSpec/Manager/Handler 三件套」       | Spec 仍在；Handler 抽象被 Worker 取代，「Handler」降级为内部实现名                      |
| 「`vllm/config/offload.py` 是 KV offload 配置」 | 它是**模型权重** offload;KV 入口在 `cache.py:406` + `kv_transfer.py:88`                 |
| 「`simple_kv_offload` 是 `kv_offload` 的简化版」    | 平行实现，零代码共享；各有 metrics（`vllm:simple_kv_offload_*` vs `vllm:kv_offload_*`） |
| 「XpYd 见 `disagg_prefill` 文档」                 | 出处在 `mooncake_store_connector_usage.md:93`                                           |
| 「CPU 块不参与前缀缓存」（你 fork 的行为）      | HEAD 的 offload 块可被 lookup 命中——命中来源打 `source=host`/`disk` 标签(注:P2P no-op store 与 partial-tail CPU hit 的计数虚高已在新版修正 #69132/#69048)                |
| 顶层 `kv_offload/fs/` 是生产 FS 路径            | 孤儿代码（仅测试引用）；生产走 tiering 的 `FileSystemTierManager`                       |

## 五、实验与产出

见 [hands-on-exercise.md](hands-on-exercise.md)：实验 1 CPU offload 生效观察（纯单卡）；实验 2 两条路径对照(simple vs offloading)；实验 3（可选，双机）NixlConnector PD 分离。课后作业见 [homework.md](homework.md)。

## 六、边界

| 内容                           | 本讲           | 去处                                                                                  |
| ------------------------------ | -------------- | ------------------------------------------------------------------------------------- |
| LMCache / NIXL / Mooncake 内部 | 只讲接口与角色 | 各自官方文档；仓库旧文 `hybrid_kv_cache_manager_deep_dive.md`（LMCache 交叉核对沿用） |
| NIXL 协议与租约实现            | 只讲租约语义   | `docs/design/nixl_kv_cache_lease.md`                                                  |
| tiering 二级层各 manager 实现  | 只讲接口与注册 | 源码自读                                                                              |

## 七、前后衔接

第 2 讲的抢占(recompute-only)、第 3 讲的「三件套已重构」、你 fork 的 swap、第 6 讲的 deepep——在本讲汇合。第 10 讲进入生产运维：offload 的 metrics 与 PD 部署都是运维面的常客。
