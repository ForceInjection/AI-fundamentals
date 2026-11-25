# 结课考试卷 · 参考答案与评分标准

> 对应试卷：`final-exam.md` | 基线：vLLM `2f5d8a2b1`(v0.34.1rc0-84)· nano-vllm `b9dd55a`
> 使用方式：先闭卷完成试卷，再对照本卷自评。简答与论述按要点给分，答出参考要点之外的合理机制同样得分；计算题公式对、数字错的按公式给分。

---

## 一、单项选择题（每题 2 分，共 20 分）

| 题号 | 答案  | 依据与解析                                                                                                      |
| ---- | ----- | --------------------------------------------------------------------------------------------------------------- |
| 1    | **C** | tokenizer 只在 API 层的 OutputProcessor(`async_llm.py:183-189`);EngineCore 只收发 token ids（第 1 讲 §2.3）     |
| 2    | **B** | 无阶段设计注释见 `scheduler.py:702-711`；chunked prefill V1 默认开；抢占只剩 recompute（第 2 讲 §二）           |
| 3    | **B** | 退役证据链三步：`csrc/attention/` 无 kernel、`paged_attn.py` 仅 53 行、读取交后端原生 page-table（第 3 讲 §四） |
| 4    | **D** | FLASHMLA 为 MLA 专用，`is_mla()` 与 dense GQA 互斥(`backend.py:332-336`)（第 4 讲矩阵）                         |
| 5    | **B** | 默认 -O2;O3 与 O2 逐字段相同(`vllm.py:366-386`)（第 5 讲 §四）                                                  |
| 6    | **B** | 纯 TP-EP 无 all2all：`use_all2all_kernels` 要求 dp/pcp/sp>1(`fused_moe/config.py:1088`)（第 6 讲 §四）          |
| 7    | **C** | Medusa 已从 HEAD 移除；在职五法见第 7 讲速查表                                                                  |
| 8    | **B** | `fp8.py:105-108` 拒绝无 checkpoint 量化，指引改用 `fp8_per_tensor`（第 8 讲 §三）                               |
| 9    | **C** | 换出决策在 connector 侧 manager(OffloadingConnectorScheduler / SimpleCPUOffloadScheduler)（第 9 讲 §三）        |
| 10   | **C** | /ping 仅存在于 SageMaker 适配层(`sagemaker/api_router.py:117`)；通用探活是 /health（第 10 讲 §二）              |

---

## 二、多项选择题（每题 4 分，共 20 分。多选/错选 0 分；漏选得 2 分）

| 题号 | 答案    | 解析                                                                                                                                             |
| ---- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| 11   | **ABC** | 每 token KV = 2×L×`kv_heads`×`head_dim`×`dtype_bytes`；`max_num_seqs` 影响并发总量，不影响单请求占用（第 3 讲；KV 八讲第 1-2 讲）                       |
| 12   | **ABC** | capture 走 `capture_model` 闭包工厂，不走 `_dummy_run`（后者只服务 profiling/warmup）（第 5 讲 §三）                                             |
| 13   | **ACD** | medusa 已移除（方法值/类/文档全无）；在职：ngram/suffix/eagle/eagle3/draft_model/mtp 等（第 7 讲速查表）                                         |
| 14   | **ACD** | `gptq_marlin`/`awq_marlin` 是别名；Marlin 恒 16-bit 激活（官方 README 更正）；bnb/gguf 为 OOT 插件(`__init__.py:65-101`)（第 8 讲 §二）           |
| 15   | **ABC** | 两路径零代码共享；决策在 connector 侧 manager；主 Scheduler 抢占只剩 recompute;offload 块可被前缀查询命中（命中标签 source=host）（第 9 讲 §三） |

---

## 三、填空题（每空 1 分，共 10 分）

| 题号 | 答案                                                                       | 依据                                                                                                         |
| ---- | -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| 16   | **tokenizer**                                                              | 在 API 层 OutputProcessor(`async_llm.py:183-189`);EngineCore 只收发 token ids（第 1 讲 §2.3）                |
| 17   | **`execute_model`**                                                        | `self.model_executor.execute_model(scheduler_output, non_block=True)`(`core.py:714`)                         |
| 18   | **`EngineCoreProc` 的 `process_input_sockets` / `process_output_sockets`** | 两个 daemon 线程入口 `core.py:1431/1442`；DEALER connect 2200-2203、输出 PUSH 2311-2318（第 1 讲 §三）       |
| 19   | **240**                                                                    | 256B FLIT = 240 数据 + 16 头部与流控                                                                         |
| 20   | **约 5mm**                                                                 | 另两前提：工艺代兼容、共享电压域                                                                             |
| 21   | **TP**                                                                     | `world_size` = TP×PP×PCP，不含 DP/EP（第 6 讲 §二）                                                          |
| 22   | **1.8 TB/s**                                                               | NVLink 5 单 GPU 双向（NVIDIA 官方）                                                                          |
| 23   | **16**(BF16)                                                               | Marlin 恒 16-bit 激活，仅做权重反量化（官方 README 更正）                                                    |
| 24   | **accepted/drafts**                                                        | drafts 取日志 Drafted token 计数；陷阱在含 bonus（日志均值）与不含 bonus（per-request 直方图）两套口径的混用 |
| 25   | **`--shm-size`**（如 `--shm-size=10g`）                                    | offload 8GB 需 shm 镜像共享区；实测 8GB offload 时报「8190 MiB required」，给 10g 留余量                     |

---

## 四、计算题（每题 10 分，共 20 分）

**26. 参考答案**：

每 token KV 公式：`2(K/V) × L(层) × kv_heads × head_dim × dtype_bytes`

= 2 × 28 × 8 × 128 × 2(B) = **114,688 B ≈ 112 KB/token**

总 KV = 112 KB × 32K × 64 并发 ≈ **224 GiB**（1024 进位，不含 activation 与临时块）

判断：224 GiB 远超单卡 80GB（若按 gpu-memory-utilization 0.9 扣除权重，可用约 70GB），**不能承载**。

可行路径（答出两条即满分）：

1. TP=4:KV 按头切分到 4 卡，每卡约 56GB——仍紧张；TP=8 或加节点；
2. FP8 KV dtype：占用减半至 ≈112GB——单卡仍不足，需配合 offload；
3. CPU offload(`--kv-offloading-size`)：溢出部分换出，吞吐受 PCIe 带宽约束；
4. 缩短上下文或降低并发（业务侧权衡）。

评分：公式 4 分；总量与判断 3 分；路径 3 分（每条 1.5）。

**27. 参考答案**：

进程拓扑：按第 1 讲「进程总数 = A + DP + N」公式，API server(A)1 个 + EngineCore(DP)4 个（`EngineCore_DP0`~DP3，每 DP rank 一个）+ Worker(N)8 个（每 DP 组内 TP=2）= **13 个相关进程**；另有 DP coordinator。

EP 组大小 = DP×PCP×TP = 4×2 = 8；专家层 `tp_size` 置 1（整只专家放卡）。

MoE 层**会产生 all2all**：判定条件 `use_ep 且 dp>1`(`fused_moe/config.py:1088`)——DP=4>1 满足；未显式指定时兜底走 `allgather_reducescatter`，显式 `--all2all-backend deepep_*` 才走 DeepEP。

评分：进程数 4 分（13 个，列出三类）；EP 组 3 分；all2all 判定 3 分（条件 + 后端名任一）。

---

## 五、简答题（每题 10 分，共 30 分）

**28. 参考要点**：

- 决策位置：V0 时 swap 决策在主 Scheduler（SWAPPED 队列）；HEAD 上已移入 connector 侧 manager(OffloadingConnectorScheduler / SimpleCPUOffloadScheduler)，主 Scheduler 抢占只剩 recompute(`scheduler.py:1829-1876`)；
- recompute 保留的原因：实现简单；小模型下 swap 的 PCIe 搬运成本 ≈ 重算成本（nano fork 在 RTX 3090 上的实测结论）；大模型/长上下文下 swap 开始有收益；
- swap 能力去向：offloading 子系统（`kv_offload` 分层后端 / `simple_kv_offload`）+ KV Connector 生态(LMCache/Nixl/Mooncake)——从「调度器内的紧急逃生」升级为「常态化的分层存储」（第 9 讲）。

**29. 参考要点**：

- 决策位置：两条路径的换出决策都不在主 Scheduler（它只剩 recompute，`scheduler.py:1829-1876`），而在 connector 侧的 scheduler 兼 manager——`OffloadingConnectorScheduler`(`offloading/scheduler.py:225`)与 `SimpleCPUOffloadScheduler`(`simple_kv_offload/manager.py:143`)；
- 结构关系：两者零代码共享、是平行实现而非「简化子集」——`kv_offload/` 分层后端（Spec 工厂仅 CPU/Tiering 两个内置 + CachePolicy 可插拔 lru/arc），`simple_kv_offload/` 自带 manager/worker 并多出 hugepage、显式块数与磁盘后端；
- 观测口径：两套 metrics 并存(`vllm:kv_offload_*` vs `vllm:simple_kv_offload_*`)，命中来源统一收敛到 `prompt_tokens_cached_by_source` 的 source=host/disk 标签；
- 评分：三个差异点各 3 分（决策位置/结构分层/观测口径），表述清晰 1 分。
- 依据：第 9 讲 §三；源码 `offloading/scheduler.py`、`simple_kv_offload/manager.py`。

**30. 参考要点**：

- 层次关系：编译（融合 pass + Inductor 子图）与 capture（`FULL_AND_PIECEWISE` = decode 整图消 launch + 其余分段图）是两层，由 `-O` 档位组合；-O2（默认）两层全开；
- --enforce-eager 两层全关：配置收口处无条件置 `CompilationMode.NONE + CUDAGraphMode.NONE`（`config/vllm.py:1878-1880`，日志原文「disabling torch.compile, CUDAGraphs, and JIT kernel warmup」），并非「只关 capture」；
- 排障用 --enforce-eager 的原因：一次关掉图与编译两层，把「图/编译引入的问题」与「模型本身的问题」分开。若 eager 下问题消失，定位在图或编译层，再二分 `-cc.cudagraph_mode=NONE` / `-cc.mode=0`；
- 生产默认 -O2:decode 占大头时稳态收益显著（launch 消除 + 融合），启动开销（capture 5~20s + 编译）是一次性成本。

---

## 附加题（10 分）

评分维度（每项 2 分，共 10 分）：

1. 并行组合与实例拓扑合理（引用第 6 讲的进程公式与 EP 判定）；
2. 调度与投机解码决策有机制依据（引用第 2/7 讲）；
3. 量化与 KV dtype 决策引用第 8 讲的 kernel 路由与精度权衡；
4. offload/PD 分离的取舍引用第 9 讲的两条路线；
5. 观测方案引用第 10 讲的端点与指标族，厂商口径与独立口径分开。
