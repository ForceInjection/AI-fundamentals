# 第 9 讲教学大纲：KV 分载与 PD 分离

## 基本信息

| 项目     | 内容                                                                |
| -------- | ------------------------------------------------------------------- |
| 课时     | 120 分钟                                                            |
| 基线     | nano-vllm `b9dd55a`（含你 fork 的 kv_swap）;vLLM `1acec5c009`      |
| 先修     | 第 2 讲（recompute-only 抢占）、第 3 讲（kv_offload 目录存在性）    |
| 授课形式 | 三代演进对照 + 双路径走读 + 实验                                    |

## 教学目标

- **知识目标**：复述三代演进对照表（决策者/搬运者/淘汰策略）；说清两条并行实现的关系（零代码共享的平行实现）；讲出 disagg prefill 的角色配对(kv_producer/consumer)与「不提升吞吐」的动机声明。
- **能力目标**：用 `--kv-offloading-size` 起 CPU 分载服务并从日志/指标确认生效；区分两条路径与两套 metrics；为 PD 分离写 NixlConnector 配置。
- **素养目标**:「决策权从调度器移入 connector」的架构演化视角；fork 实现与生产实现的三代差距量化方法。

## 时间安排

| 部分            | 时长    | 内容                                                                   |
| --------------- | ------- | ---------------------------------------------------------------------- |
| nano 回顾       | 10 min  | 你 fork 的 kv_swap：决策耦合调度/CPU 块不参与前缀缓存/3090 实测         |
| 三代演进对照    | 15 min  | V0 CacheEngine → v0.11 三件套 → HEAD（决策者/搬运者/淘汰策略三行对照） |
| 现状走读：六环节 | 45 min  | 见下方函数级阅读清单；课堂重点讲环节 2（双路径）与环节 4（全家福）     |
| 实验            | 25 min  | CPU offload 生效观察 + 双路径对照（见 hands-on-exercise.md）            |
| 答疑            | 10 min  |                                                                        |

## 函数级阅读清单（课前预读）

| 文件                                                  | 函数 / 类                        | 行号            | 看什么                                                 |
| ----------------------------------------------------- | -------------------------------- | --------------- | ------------------------------------------------------ |
| `vllm/v1/kv_offload/base.py`                          | `OffloadingManager`(ABC)        | 329-546         | 决策者 14 方法：lookup/prepare_load/prepare_store…     |
| `vllm/v1/kv_offload/base.py`                          | `OffloadingWorker`(ABC)         | 741-784         | 搬运者：submit_store/submit_load/get_finished          |
| `vllm/v1/kv_offload/base.py`                          | `OffloadingSpec`(ABC)           | 787-861         | 配置→(Manager, Worker);offload_prompt_only 默认 True  |
| `vllm/v1/kv_offload/factory.py`                       | 注册表                           | 62-69           | 内置仅 CPUOffloadingSpec / TieringOffloadingSpec       |
| `vllm/v1/kv_offload/cpu/manager.py`                   | `CPUOffloadingManager`           | 55；容量不足：270 | 决策者；keys_to_retry                                  |
| `vllm/v1/kv_offload/cpu/gpu_worker.py`                | `CPUOffloadingWorker`            | 811;handler:314 | 每方向独立 CUDA stream DMA                             |
| `vllm/v1/kv_offload/cpu/policies/arc.py`              | `ARCCachePolicy`                 | 28              | ARC 淘汰（T1/T2+ghost；活跃维护）                       |
| `vllm/v1/kv_offload/tiering/manager.py`               | `TieringOffloadingManager`       | 120 / 277       | CPU 主层 + 二级层级联；GPU↔二级层经 CPU 中转           |
| `vllm/v1/kv_offload/tiering/backpressure.py`          | 背压策略族                       | 9-189           | 慢层写满时的丢弃/节流                                  |
| `vllm/v1/simple_kv_offload/manager.py`                | `SimpleCPUOffloadScheduler`      | 143;362 / 557   | 平行实现的决策者；eager/惰性 store                      |
| `vllm/v1/simple_kv_offload/worker.py`                 | `SimpleCPUOffloadWorker`         | 33;323/340/364  | start_load_kv/wait_for_save/get_finished              |
| `vllm/v1/simple_kv_offload/cuda_mem_ops.py`           | `cuMemcpyBatchAsync` 封装        | 199 / 239       | 批量 DMA                                               |
| `vllm/config/vllm.py`                                 | `_post_init_kv_transfer_config`  | 1250-1280       | --kv-offloading-size 翻译成 KVTransferConfig          |
| `vllm/config/kv_transfer.py`                          | `KVTransferConfig`               | 85-148          | kv_connector/kv_role/kv_both                          |
| `vllm/distributed/kv_transfer/kv_connector/v1/base.py`| `KVConnectorBase_V1`             | 182;374/557     | worker 4 抽象 + scheduler 3 抽象                       |
| `vllm/distributed/kv_transfer/kv_connector/factory.py`| 注册表                           | 183-286         | 全部连接器                                             |
| `vllm/v1/core/sched/scheduler.py`                     | 询问点与抢占                      | 1161-1163 / 1830-1877 | 主 Scheduler 只问与 recompute                   |

## 重点与难点

- **重点**：三代对照表；两条平行实现（零代码共享）;「决策权在 connector 侧」;disagg 的 kv_role 配对。
- **难点**:OffloadingManager 的异步状态机（MISS/HIT/HIT_PENDING/RETRY 四态）;tiering 的「GPU↔二级层一律经 CPU 中转」；两条路径两套 metrics。
- **易错点**：把 `vllm/config/offload.py` 当 KV 配置；把 `kv_offload/fs/` 当生产 FS 路径；把 simple 当 kv_offload 的子集；引用 disagg 文档找 XpYd。

## 参考材料

- 官方：`docs/features/kv_offloading_usage.md`（587 行，示例可用）、`disagg_prefill.md`、`mooncake_store_connector_usage.md`(XpYd)、`docs/design/nixl_kv_cache_lease.md`（租约）
- 演进素材：仓库 `hybrid_kv_cache_manager_deep_dive.md`（LMCache 交叉核对，旧锚点）、`native_kv_offloading.md`（v0.18 三件套口径）
- 对照：你 fork 的 `nanovllm/engine/kv_swap.py` + `docs/design/kv-offload.md`（RTX 3090 实测）
