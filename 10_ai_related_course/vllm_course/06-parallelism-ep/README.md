# 第 6 讲 · 并行：TP 到 EP

> 120 分钟 | 基线：nano-vllm `b9dd55a`；vLLM `2f5d8a2b1`（v0.34.1rc0-84,2026-11-25 核对）

本讲回答一个核心问题：**一台机器装不下 MoE 的专家，把专家切到多台机器上，怎么切才算切对了？**

知识链：`ParallelConfig` 六轴(TP/PP/PCP/DP/DCP/EP)→ 进程组初始化（九种 group，按 rank 布局）→ attention 层走 TP 切分 + allreduce 优先级链 → MoE 层走 EP（dispatch/combine 抽象 + all2all 后端族）→ Wide-EP 部署形态（`EP_SIZE` = TP×DP 自动计算）。

## 一、nano 已学形态（回顾）

先修课第 8 课你看过的 nano TP（`nanovllm/engine/model_runner.py`）：

- `__init__`（`:42-49`）：`world_size > 1` 且 `rank == 0` 时创建共享内存；
- `loop()`（`:62-67`）：rank > 0 阻塞等 Event，从共享内存反序列化出 `method_name, args`，执行同名方法；
- rank0 经 `write_shm` / `call`（`:77-90`）广播调用。

**TP 之外什么都没有**：没有 PP/DP，没有专家，通信 = 「把方法调用广播给兄弟进程」。生产版的对比就从这里展开：轴多到六个、通信从「广播方法调用」变成「按维度选集合通信原语」、专家从无到「跨机器的 dispatch/combine」。

## 二、并行轴全景

`ParallelConfig`（`vllm/config/parallel.py:248`）有六根轴：`pipeline_parallel_size`(PP)、`tensor_parallel_size`(TP)、`prefill_context_parallel_size`（PCP，**新轴**：prefill 序列切分）、`data_parallel_size`(DP)、`decode_context_parallel_size`(DCP)、以及 `enable_expert_parallel`(EP)。三个容易记错的公式：

- `world_size = TP × PP × PCP`（`parallel.py:1182-1186`）——**不含 DP 与 EP**；DP 用 `world_size_across_dp = world_size × DP`（`:823`）单独表达；
- **EP 组大小恒等于 `DP × PCP × TP`**（不是独立配置项；EP 开关只决定「专家层是否用 EP 替代 TP」）；
- `EP_SIZE = TP_SIZE × DP_SIZE`（部署文档 `expert_parallel_deployment.md:49-55`，自动计算，不用手填）。

DP 与 EP 的语义区别，进程组初始化处的注释（`parallel_state.py:2200-2207`）说得最准：DP 组内各 rank 必须同时 forward（否则死锁）——DP 是「attention 数据并行（独立 KV cache、独立 batch）」；EP 组与 DP 组共享同一批 rank 但维度不同——「专家在 DP×TP 展开后的专家并行组」。

## 三、现状走读

### 环节 1：配置与校验

`ParallelConfig` 校验（`_validate_parallel_config`，`:681`）里与本讲相关的三条：all2all backend 为 "auto" 时的解析（`:697-702`，GB200/B300 优先 `flashinfer_nvlink_one_sided`）；EPLB 需要 `enable_expert_parallel` 且 TP×PCP×DP>1（`:723-748`）；TP 必须被 DCP 整除（`:756-770`）。executor backend 缺省逻辑（`:1272-1321`）与第 1 讲一致：**默认 mp 尽量单机**，`world_size` 超过本机 GPU 数且未设 nnodes 直接报错——「多机默认 Ray」是文档旧说法，nnodes>1 时只允许 mp/uni/external_launcher（`:1329-1338`）。

### 环节 2：进程组初始化

`initialize_model_parallel`（`parallel_state.py:2131`）按 rank 布局切出九种组：ExternalDP × DP × PP × PCP × TP（`:2200-2215`）→ TP（`:2219-2234`）→ DCP（`:2315-2329`）→ PCP（`:2346-2348`）→ PP（`:2351-2366`）→ DP（`:2379-2397`）→ **EP（`:2400-2437`，仅 MoE 模型创建）**→ EPLB（`:2440-2460`，独立同 ranks 组，隔离重平衡通信防死锁）。EP 组的 rank 布局取 DP×PCP×TP 维——**PP 不参与 EP**。各组由 `GroupCoordinator` 封装（getter 在 `:1613-1734`）。

### 环节 3:TP 通信

attention 与稠密层走经典 TP 切分：`ColumnParallelLinear`（列切，`linear.py:447`，输出分片；`gather_output=True` 时 `all_gather`，`:639`）与 `RowParallelLinear`（行切，`:1722`；`reduce_results` 时 `all_reduce`，`:1888`）。`all_reduce` 的后端是一条**优先级链**（`cuda_communicator.py:389-455`）：symm-mem NCCL → quick allreduce → flashinfer PCIe IPC → flashinfer NVLink → aiter(ROCm)→ vLLM custom allreduce → 兜底 pynccl；且 custom allreduce 系**只对 TP 组启用**（`:59-65`，DP/EP 组不会误用）。TP>1 时还有 DBO（all-reduce-overlap,comm/compute 双流，`parallel_state.py:762-772`）。

### 环节 4:EP——专家怎么切

MoE 层的并行判定在 `FusedMoEParallelConfig`（`fused_moe/config.py:1067`）：`use_ep = (DP×PCP×TP > 1) and enable_expert_parallel`（`:1260-1264`）；**开 EP 后专家层的 `tp_size` 置 1**——每只专家整只放在一张卡上，无张量切分（`:1292-1296`）。专家放置：`determine_expert_map`（`expert_map_manager.py:22`，linear 连续分段或 `round_robin`；`deepep_ll`/nixl_ep 强制 `round_robin`，`:137-139`）；EPLB 动态重平衡在 `vllm/distributed/eplb/`。

token 的往返走 **dispatch/combine 抽象**：接口 `FusedMoEPrepareAndFinalize`（`modular_kernel.py:241`），后端实现族在 `fused_moe/prepare_finalize/`（`deepep_ht` / `deepep_ll` / `deepep_v2` / `flashinfer_nvlink_one_sided`/two_sided / moonep / mori / `nixl_ep` / passthrough / batched 等），通信实体是 EP 组上的 all2all manager（`device_communicators/all2all.py`：AgRs 拼装版 / DeepEP HT / DeepEP LL）。deepep 的触发条件（`fused_moe/config.py:1094-1103`）：`use_ep` **且** DP（或 PCP/SP）>1 **且** `all2all_backend=="deepep_*"`。

一个反直觉结论：**纯 TP-EP（无 DP）不产生 all2all**——`use_all2all_kernels` 要求 dp/pcp/sp>1（`:1088`）；单机 TP=8 开 EP 时 token 本来就在 TP 组冗余，输出靠 TP allreduce 汇总，不需要专家间通信。这直接决定了实验设计：all2all 的现象要在 DP>1 的部署里看。

### 环节 5:Wide-EP 部署形态

`docs/serving/expert_parallel_deployment.md` 的两个台阶：

- **单机 8 卡**(H200/H20)：`--tensor-parallel-size 1 --data-parallel-size 8 --enable-expert-parallel`（`:85-96`）——attention 层 DP 复制(TP=1)，专家层 `EP_SIZE`=8 跨 DP 展开；
- **2 机 16 卡**（Wide-EP 形态）：Node1 用 `--all2all-backend deepep_low_latency --data-parallel-size 16 --data-parallel-size-local 8 --api-server-count=8`，Node2 加 `--data-parallel-start-rank 8 --headless`（`:166-193`）；多节点 prefill 换 `deepep_high_throughput`（`:32-46` 后端选择表）。

官方博客「DeepSeek @ 2.2k tok/s/H200 with Wide-EP」(2025-12-17)描述的就是这一形态；仓库文档没有 2.2k 数字本身，引用性能数字时以博客为出处、注明厂商口径。EPLB（专家负载均衡）的配置在 `:210-332`(`window_size` / `step_interval` / `num_redundant_experts`);PD 分离形态（prefill 用 HT、decode 用 LL、NIXL KV transfer）在 `:351-447`——第 9 讲回收。

### 环节 6:DP attention 与 DCP

**DP attention 不是独立实现**：没有「DP attention 层」的代码——就是 DP replica 复制（attention 权重按 DP 复制或 DP 内 TP 切分），只有 MoE 层经 EP 跨 DP 打通（`data_parallel_deployment.md:5-16`）。DP rank 非完全独立：前向要对齐，同步原语是 `dispatch_cg_and_sync_dp`（`dp_utils.py:232`，第 1 讲见过）+ 每隔 `dp_sync_interval`（默认 16 步）的 gloo CPU allreduce（`parallel.py:1074-1112`）；空闲 rank 跑 `_dummy_run`。coordinator 进程（`coordinator.py:23`）做跨 DP 的 finish-sync，第 1 讲已铺。

DCP(decode context parallel)按 head 交错切 KV cache，**上限 = `tp_size` / kv-head 数，只省 KV 重复不加卡**；案例：DeepSeek-R1(MLA H=1)tp8 可 dcp8、Qwen3-235B(H=4)tp8 只能 dcp2（`context_parallel_deployment.md`）。PCP（prefill 侧）分 chunk 算 QKV，有「partial Q full KV」与 ring-attention 两种策略。

## 四、V0/旧教程 vs HEAD

| 旧说法                                         | HEAD 事实                                                                                    |
| ---------------------------------------------- | -------------------------------------------------------------------------------------------- |
| 「`FusedMoE` 类的 forward」                    | 类已不存在；现在是 `FusedMoEFactory`（`fused_moe/layer.py:95`）→ `MoERunner`                 |
| `world_size = TP×PP×DP×EP`                     | `world_size = TP×PP×PCP`，不含 DP/EP（`parallel.py:1182`）                                   |
| 「EP=TP 时也走 all2all」                       | 纯 TP-EP 无 all2all;DP>1 才有（`fused_moe/config.py:1088`）                                  |
| all2all 后端 naive / pplx                      | 已删除，告警回退 `allgather_reducescatter`（`parallel.py:136-142`）                            |
| 「多机默认 Ray」                               | 默认 mp 尽量单机；超本机卡数未设 nnodes 直接报错（`:1284-1296`）                             |
| deepep 集成在 `vllm/distributed/ep/`           | 目录不存在；实际在 `device_communicators/all2all.py` + `fused_moe/prepare_finalize/deepep_*` |
| 「DP attention 是独立的 attention 实现」       | 不存在；DP replica 复制 + MoE 层 EP 打通                                                     |
| `arch_overview.md` 进程表「Worker = DP×PP×TP」 | 漏了 PCP 轴（代码 `world_size` 含 PCP）                                                        |

## 五、实验与产出

见 [hands-on-exercise.md](hands-on-exercise.md)：实验 1 并行配置组合校验（纯 CPU，错误组合读报错）；实验 2 TP 启动日志走读（单卡/双卡）；实验 3（可选）DP 配置解析与双卡观察。课后作业见 [homework.md](homework.md)。

## 六、边界

| 内容                                   | 本讲               | 去处                                     |
| -------------------------------------- | ------------------ | ---------------------------------------- |
| NCCL / custom allreduce 的 kernel 实现 | 只讲后端优先级链   | NCCL 官方文档；`02_gpu_programming` 目录 |
| MoE 路由数学（top-k、负载）            | 只讲放置与通信     | 模型侧内容，本课不展开                   |
| PD 分离                                | 只在部署形态处点名 | 第 9 讲                                  |

## 七、前后衔接

第 1 讲的 Executor（「进程怎么来」）在本讲补全「进程怎么协作」；第 1 讲的 DP 对齐（`dispatch_cg_and_sync_dp`）在本讲放进多机语境；第 9 讲的 PD 分离是「并行切分」在时间维度的延伸，deepep HT/LL 双后端正是为 PD 设计的。
