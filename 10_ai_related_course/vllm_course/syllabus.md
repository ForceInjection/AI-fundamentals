# 课程总纲

> vLLM 十讲：从 nano-vllm 到生产推理引擎

## 基本信息

| 项目     | 内容                                                              |
| -------- | ----------------------------------------------------------------- |
| 课程目标 | 从 nano-vllm 的最小实现迁移到生产 vLLM：架构、源码、配置、调优     |
| 代码基线 | nano-vllm `b9dd55a`；vLLM `2f5d8a2b1`(v0.34.1rc0-84,2026-11-25) |
| 演进参照 | V0(2023-06)→ V1(2025-01)→ Model Runner V2(2026-03)                |
| 先修     | nano-vllm 实战课程 8 课，或等价源码阅读经验                        |
| 撰写状态 | 十讲全部成稿（README 正文/阅读清单/ppt-outline/实验作业齐备）       |

## 十讲安排

### 第 1 讲 · 引擎总架构：从单进程到三进程

- **nano 回顾**：单进程 `LLMEngine.step()`，API 与执行同进程，tokenizer 也在同进程（`add_request` 里 tokenize）。
- **演进**：V0 单进程 → V1 三进程拆分（`AsyncLLM` API 层 / `EngineCore` 核心层 / `Worker` 执行层，两套 IPC：前端↔EngineCore 走 ZMQ ROUTER/DEALER + PULL/PUSH,Executor↔Worker 走共享内存 MessageQueue）→ Model Runner V2（2025-11 落地 PR #25266,2026-11-07 旧版删除 PR #63373，现为唯一实现）。
- **现状走读**：`vllm/entrypoints/launchers/api_server/entry.py`（入口；`entrypoints/openai/api_server.py` 已是废弃 shim）、`vllm/v1/engine/async_llm.py`、`core.py`、`vllm/v1/executor/`、`vllm/v1/worker/gpu/model_runner.py`；`docs/design/arch_overview.md`、`model_runner_v2.md`。
- **实验**：起一个最小服务，用日志与 ps 观察三进程（纯 CPU 可完成进程观察）；`--enforce-eager` 开关对照。
- **外部材料**：Gordić「Inside vLLM: Anatomy of a High-Throughput LLM Inference System」（架构主线教材，V1 口径）。

### 第 2 讲 · 调度器：从 FIFO 到策略化

- **nano 回顾**：`scheduler.py` 三队列(waiting/running/swapped)、preempt 两层决策；**prefill 与 decode 互斥**（本轮有 prefill 就 return）、chunked prefill 仅限 batch 首条 seq（`scheduler.py:L52-53`）。
- **演进**：V0（两阶段调度、Policy 类、`preemption_mode` 可选 recompute/swap）→ V1（单一 `running` 列表、无阶段划分、抢占只剩 recompute、async scheduling 默认开启）。
- **现状走读**：`vllm/v1/core/sched/scheduler.py`（`schedule()` 单函数约 1114 行，`scheduler.py:700-1815`；「无阶段划分」注释 `:702-711`）、`request_queue.py`(SchedulingPolicy:FCFS/PRIORITY)、`async_scheduler.py`（只覆写 `_update_after_schedule`；overlap 实现在 EngineCore `step_with_batch_queue`，`core.py:746 起`）、`docs/usage/v1_guide.md`。
- **实验**：长短 prompt 混合负载，对照 chunked prefill 与 `--long-prefill-token-threshold` 的 TTFT/吞吐；观察 async scheduling（默认开启）的日志特征。
- **外部材料**：Sarathi-Serve 论文(arXiv:2403.02310,OSDI'24)。

### 第 3 讲 · KV cache 管理：PagedAttention 的生与死

- **nano 回顾**：`block_manager.py` 单一管理器（hash→block、`ref_count`、free/used 池）；fork 的 CPU 池与 swap（`can_swap_out` 的 `ref_count==1` 约束）。
- **演进**：单一 BlockManager → 四层（KVCacheManager 门面 → Coordinator 三选一：NoPrefixCache/Unitary/Hybrid → SingleTypeKVCacheManager×注意力类型 → BlockPool 唯一池）；哈希键加 group 后缀（`BlockHashWithGroupId`）；哈希前移到 `add_request`；partial block 缓存出现；混合注意力(GSA/NSA/SLA)触发多组协调。
- **退役证据链**：自研 PagedAttention CUDA kernel 已删除（`csrc/attention/` 无 `attention_kernels`.cu）；`paged_attn.py` 瘦身成 53 行布局辅助；分页思想下沉为 `block_table`/BlockPool，读取交给 FA/Triton/FlashInfer 的原生 page-table API。
- **现状走读**：`vllm/v1/core/kv_cache_manager.py:131`、`kv_cache_coordinator.py:828`（工厂三分支）、`block_pool.py:107`（唯一池+LRU 链）、`kv_cache_interface.py:682`（KVCacheGroup；**KVCacheGroupId 已不存在**）、`kv_cache_planning.py:1476`（容量换算）。
- **实验**：同前缀请求的命中观察(`prefix_cache_probe`)；混合注意力模型(Gemma-2/gpt-oss)的分组日志；`--kv-cache-dtype` 对照。
- **边界**：KVCacheManager 对象图细节 → [KV Cache 八讲](../kv_cache_course/README.md)第 4 讲；容量账本 → 同课第 1-2/8 讲。
- **演进素材**：`09_inference_system/vllm/module_analysis/pagedattention_retirement.md`（双版本对比）、`attention_mha_mla_nsa.md`、`hybrid_kv_cache_manager_deep_dive.md`（HMA 口径）。

### 第 4 讲 · 注意力后端：从 if-else 到后端矩阵

- **nano 回顾**：`layers/attention.py` 按 backend 字符串分支。
- **演进**：V0 硬编码 → V1 后端注册表 + platform 自动选择；FlashAttention / FlashInfer / Triton / FlashMLA / FlexAttention 矩阵；按模型架构(MHA/GQA/MLA/NSA)与硬件(CUDA/HIP/CPU)双重过滤。
- **现状走读**：`vllm/v1/attention/backends/`、backend 注册与选择逻辑、`docs/design/attention_backends.md`。
- **实验**：`--attention-backend` 强制指定不同后端（注意：旧教程的 `VLLM_ATTENTION_BACKEND` 环境变量已被移除），对比吞吐与告警日志。
- **外部材料**：仓库 `flashattention_deep_dive.md`（FA1→FA3 原理，演进素材）。

### 第 5 讲 · CUDA Graph 与编译

- **nano 回顾**：`model_runner.py` 的 `capture_cudagraph()`（`:239-274`），整段 decode 前向「录像」，eager 条件 = prefill / `enforce_eager` / batch>512（`run_model` `:212-229`）。
- **演进**：V0/V1 全图或 PIECEWISE → MRV2 默认 **-O2 = `FULL_AND_PIECEWISE`**（decode 整图、其余分段图，`vllm.py:359`）；capture 不走 dummy run（`capture_model` → 闭包工厂）；capture 尺寸两族网格（prefill token 族 / decode 请求族）；draft 多步融合进一张 FULL 图。
- **现状走读**：`vllm/v1/worker/gpu/model_runner.py:1172-1252`（`capture_model`）、`cudagraph_utils.py:154/641`（CudaGraphManager/Model 版）、`dp_utils.py:232-369`（`dispatch_cg_and_sync_dp` 每步落桶）、`config/compilation.py`（CUDAGraphMode 五值 `:165-222`、`splitting_ops` `:962-987`、四档字典 `vllm.py:303-386`）。
- **实验**：三态对照（`enforce_eager` / 默认 -O2 / -O0）TPOT;capture 日志解析（图数量/耗时/显存）；调试二分（`-cc.mode`/`-cc.cudagraph_mode`/TORCH_TRACE）。
- **外部材料**：官方博客「Introduction to torch.compile and How It Works with vLLM」(2025-08-20)；`docs/design/debug_vllm_compile.md`（旗标二分表）。

### 第 6 讲 · 并行：TP 到 EP

- **nano 回顾**：TP = spawn 多进程 + 共享内存广播方法调用（`model_runner.py:42-90`）；无 PP/DP/EP。
- **演进**：并行轴扩展到 TP/PP/**PCP/DCP**/DP/EP 六轴；`world_size = TP×PP×PCP`（不含 DP/EP）；EP 组 = DP×PCP×TP（仅 MoE 模型创建，PP 不参与）；Ray 是可选 backend（代码默认 mp 尽量单机）。
- **现状走读**：`vllm/config/parallel.py:248`（六轴配置与校验）、`parallel_state.py:2131`（进程组初始化）、`linear.py` TP 切分与 allreduce 优先级链（`cuda_communicator.py:389-455`）、`fused_moe/`（EP dispatch/combine：`prepare_finalize/` 后端族 + `all2all.py` 管理器；deepep 集成）、`docs/serving/` 四篇部署文档。
- **实验**：并行配置组合校验（错误组合看报错）；TP 启动日志走读；DP/EP 配置解析（双卡可选）。
- **外部材料**：官方博客「DeepSeek @ 2.2k tok/s/H200 with Wide-EP」（2025-12-17；对应 `docs/serving/expert_parallel_deployment.md` 的 2 机 DP=16 deepep 命令）；仓库 `deepseek_blackwell_wide_ep.md`（演进素材）。

### 第 7 讲 · 投机解码

- **nano 回顾**：无（自回归逐步生成；先修课明确标注「完全不涉及」）。
- **演进**：V1 proposer 体系 → MRV2 时代 **Speculator 体系**（改名 + 迁移到 `vllm/v1/worker/gpu/spec_decode/`）；**Medusa 已从 HEAD 移除**（方法值/类/文档全无，仓库旧文作演进素材）；验证融合进 Sampler 路径（非独立 verify 步）。
- **现状走读**：`vllm/v1/worker/gpu/spec_decode/`（BaseSpeculator 类层级 `speculator.py:74`；ngram Triton 实现 `ngram/speculator.py:151`）、`rejection_sampler.py:152`、`config/speculative.py:113-123`（method 合法值）、`docs/features/speculative_decoding/`。
- **实验**：ngram 开关对照（零依赖）；acceptance length 日志与 per-position 读取；动态 SD（可选）。
- **外部材料**：GPU MODE Lecture 22「Hacker's Guide to Speculative Decoding in vLLM」；仓库 `speculative_decoding_landscape.md`（选型框架，含已移除的 Medusa，演进素材）。

### 第 8 讲 · 量化

- **nano 回顾**：裸 BF16，无量化路径、无 kernel 路由、无 KV 量化。
- **演进**：量化方法文件合并（gptq/awq 系收敛 `auto_gptq.py`/`auto_awq.py`）；kernel 选择独立成注册表（`vllm/model_executor/kernels/linear/`，w4a16 顺序 Marlin/Machete/Conch…）；全新**在线量化子系统**（`quantization/online/` + `--quantization fp8_per_tensor` 等 shorthand）；bitsandbytes/GGUF 出树为 OOT 插件。
- **现状走读**：`quantization/__init__.py:15-58`（方法全集）、`config/model.py:1468-1625`（overrides 优先级表）、`base_config.py:311-436`(`resolve_quant_method`)、`fp8.py`（完整走查）、`kernels/linear/__init__.py:565-589/1006-1080`（kernel 路由）、`kv_cache.py`(KV scale)。
- **实验**：LLM Compressor 压 FP8（或加载现成量化 checkpoint）；启动日志的 kernel 选择观察；`--kv-cache-dtype` 对照。
- **外部材料**：DeepLearning.AI 短课量化模块；仓库 `02_gpu_programming/06_quant_gemm/01_marlin-deep-dive.md`（CUDA 层仍准，Python 封装已变）；`docs/features/quantization/`（表格自动生成，零滞后）。

### 第 9 讲 · KV 分载与 PD 分离

- **nano 回顾**：你 fork 里的 `kv_swap.py`（swap 抢占三队列，CPU pinned 镜像，RTX 3090 实测）。
- **演进**：V0 CacheEngine swap → V0.11 OffloadingSpec/Manager/Handler → 当前分层后端（`vllm/v1/kv_offload/` 的 base/factory/cpu/fs/tiering + `simple_kv_offload/`，ARC 淘汰）；KV Connector 生态(mooncake / nixl / LMCache);disagg prefill。
- **现状走读**：`vllm/v1/kv_offload/`、`vllm/v1/simple_kv_offload/`、`vllm/distributed/kv_transfer/`、`docs/features/kv_offloading_usage.md`、`disagg_prefill.md`。
- **实验**：CPU offload 开关下的长上下文负载对照；LMCache connector 起双实例共享前缀。
- **演进素材**：`09_inference_system/vllm/module_analysis/native_kv_offloading.md`（v0.18 口径，演进素材）、`hybrid_kv_cache_manager_deep_dive.md`；nano 仓 `docs/design/kv-offload.md`。

### 第 10 讲 · 生产运维与生态

- **nano 回顾**：`bench.py` 单机吞吐脚本。
- **演进**：OpenAI 兼容 server 成熟化；metrics 体系；`vllm bench` 统一；production-stack(K8s)、AIBrix（控制面）、llm-d（分布式推理）生态分层；引擎 → 控制面的行业演进。
- **现状走读**：`vllm/entrypoints/openai/`、`docs/design/metrics.md`、`docs/benchmarking/`。
- **实验**：压测 + metrics 采集面板；production-stack 起一个 K8s 部署（可选）。
- **外部材料**：DeepLearning.AI 短课（部署与基准模块）、「Keeping vLLM Production Quality」博文；arXiv:2609.23130（引擎 → 控制面，收官阅读）。

## 考核作业（框架）

| 作业   | 落点     | 内容                                                     |
| ------ | -------- | -------------------------------------------------------- |
| 作业 1 | 第 1 讲  | 画出一次请求在当前 vLLM 的全生命周期图（进程、类、方法级） |
| 作业 2 | 第 7 讲  | 选一个负载，投机解码选型报告：五法选型并给出量化依据         |
| 作业 3 | 第 9 讲  | 给 nano-vllm 的 swap 与 vLLM 当前 offloading 写对照分析  |
| 作业 4 | 第 10 讲 | 生产部署方案一份：容量、并行、量化、观测四选型 + 决策备忘 |
