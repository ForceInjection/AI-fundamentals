# 第 1 讲 · 引擎总架构：从单进程到三进程

> 120 分钟 | 基线：nano-vllm `b9dd55a`；vLLM `1acec5c009`(v0.34.0rc0-1153,2026-11-23)

本讲回答一个核心问题：**一次请求从 OpenAI 兼容接口进来，经过哪些进程、哪些类、哪些方法，怎么变成 token 流出去?**

知识链：OpenAI 请求 → `AsyncLLM`（API 层进程）→ ZMQ(ROUTER/DEALER + PULL/PUSH)→ `EngineCore`（核心进程：调度 + KV 状态）→ Executor → 共享内存 MessageQueue → `Worker`（GPU 进程：Model Runner 执行）→ 采样 → 输出原路流回。

## 一、nano 已学形态（回顾）

先修课第 1 课你已经沿 `LLM.generate` 走完了 nano-vllm 的全部执行路径，因为它的所有环节在**同一个进程**里：

- `LLMEngine.generate`（`nanovllm/engine/llm_engine.py:68-98`）：把每个 prompt 经 `add_request`（`llm_engine.py:46-50`，**在这里 tokenize**）变成 `Sequence` 加入调度器，然后循环调 `step()` 直到完成。
- `LLMEngine.step`（`llm_engine.py:52-63`）：每步只在 prefill 和 decode 中选一种执行（两者互斥），共用「调度 → 执行 → 回写」三段式。
- API 与执行没有边界：`LLM.generate` 直接持有 engine，同进程同循环。

这个形态教学的优点是路径完整、一眼到底。生产上它撞到三堵墙：API 的异步 IO 与 GPU busy loop 抢同一个进程的 GIL；单进程无法扩展到多卡多机；服务崩溃时引擎状态一起陪葬。V1 的三进程拆分就是对着这三堵墙去的。

## 二、演进：三步拆分

**V0（2023-06 起）**：单进程为主，调度与模型执行纠缠在 `LLMEngine` 里，多卡靠 Ray worker，扩展与运维都别扭。

**V1（2025-01 发布）**：拆成三类进程（官方 `docs/design/arch_overview.md` 的「V1 Process Architecture」一节）:API server、`EngineCore`、GPU Worker（外加 DP>1 时的 coordinator）。进程总数公式 `A + DP + N`（A 个 API server、DP 个核心进程、N 个 worker）。

**Model Runner V2（2025-11 落地，PR #25266）**:Worker 内部的 ModelRunner 从「一个 7,285 行的大文件」（`vllm/v1/worker/gpu_model_runner.py`，V1）重构成 `vllm/v1/worker/gpu/model_runner.py` + `gpu/` 子包（module_states、block_table、cudagraph_utils、sample 等）。2026-11-07(PR #63373)旧版被整体删除。**今天 HEAD 上 MRV2 不是可选项，是唯一实现**，没有 V1/V2 切换开关。

三个问题各被一堵墙对应：API 与引擎分进程，异步 IO 不再干扰计算循环；EngineCore 可以起 DP 个，每个持有独立调度器；Worker 只负责执行，崩溃与重启边界清晰。

## 三、现状：一次请求的全链路走读

以下全部按基线 `1acec5c009` 核对。函数级阅读清单见 [syllabus.md](syllabus.md)，这里按环节讲清每一步发生了什么。

### 环节 1：服务启动与请求进入（API 层进程）

服务入口在 `vllm/entrypoints/launchers/api_server/entry.py`：先 `AsyncLLM.from_vllm_config(...)`（`entry.py:94`）实例化引擎客户端，再 `build_and_serve`（`entry.py:115-161`）创建 FastAPI 并 `init_app_state`（`vllm/entrypoints/launchers/api_server/app_state.py:24-181`，把 engine_client 挂到 `app.state`）。请求路由 `POST /v1/chat/completions`（`vllm/entrypoints/openai/chat_completion/api_router.py:44-59`）进入 `OpenAIServingChat.create_chat_completion`（`vllm/entrypoints/openai/chat_completion/serving.py:290`），最终 `:411` 调 `self.engine_client.generate(...)`。

> 注意：`vllm/entrypoints/openai/api_server.py` 在 HEAD 上已是废弃 shim（将在 v0.37.0 移除），网上大量旧教程还在引用它。判断依据：文件开头一行 `warnings.warn`。

### 环节 2:AsyncLLM 的装配（`vllm/v1/engine/async_llm.py`）

`AsyncLLM.__init__`（`:97-249`）装配三件事：输入处理器（InputProcessor，`:171`）、输出处理器（OutputProcessor，`:181-187`，**tokenizer 在这里**，每请求配一个 `RequestOutputCollector` 信箱，见 `output_processor.py:54-127`）、以及 `engine_core = EngineCoreClient.make_async_mp_client(...)`（`:194-201`，核心层客户端）。请求主入口 `generate`（`:740-868`）走四步：建流、经 InputProcessor 转成 `EngineCoreRequest`、注册 detokenizer、`_add_request`（`:581-602`）里 `await self.engine_core.add_request_async(request)`（`:598`，跨进程的落点）。常驻后台任务 `_run_output_handler`（`:870-966`）不断 `get_output_async()` 取回输出，按块送进各请求的 collector。

### 环节 3：进程间通信（前端 ↔ EngineCore）

ZMQ 拓扑是 **ROUTER/DEALER（输入）+ PULL/PUSH（输出）**,msgpack 帧：前端进程 bind（`vllm/v1/engine/core_client.py:688-701`，`MPClient`）；每个 EngineCore 进程用固定 identity connect（`vllm/v1/engine/core.py:2213-2220` 输入 DEALER、`:2322-2327` 输出 PUSH）。这个设计支持多 API server 对多 EngineCore 的多对多路由（`docs/design/arch_overview.md` 所说「任意 API server 可路由到任意 engine core」）。EngineCore 进程内部用两个 daemon 线程把 socket 收发与计算循环解耦（`core.py:1434-1449`）,busy loop 本体只消费内存队列。

### 环节 4:EngineCore（核心进程）

进程入口 `run_engine_core`（`vllm/v1/engine/core.py:1655-1797`，被 `CoreEngineProcManager` 在 `vllm/v1/engine/utils.py:304` 作为 Process target 拉起）。`EngineCore.__init__`（`:133-286`）的初始化顺序值得记：**executor → KV cache（profile 显存后分配）→ StructuredOutputManager → scheduler**（`:165`、`:288`、`:187`、`:190-206`）。注意它**没有 tokenizer**——只收发 token ids。

busy loop（`run_busy_loop`，`:1814-1824`）每轮做三件事：`_process_input_queue()`（无活时阻塞等输入）、`_maybe_publish_request_counts()`（上报负载）和 `_process_engine_step()`（`:1881-1904`，调 `step_fn`）。核心是 `step()`（`:714-745`）：`scheduler.schedule()` 产出本步调度结果 → `self.model_executor.execute_model(scheduler_output, non_block=True)`（`:727`，非阻塞拿 Future）→ `_wait_for_output` → `scheduler.update_from_output`（`:738`）→ 打包 `EngineCoreOutputs` 回传。调度器细节是第 2 讲的内容，本讲只要看到「schedule 与 execute 在核心进程内、经 Future 解耦」。

### 环节 5:Executor 选择（`vllm/v1/executor/abstract.py`）

executor 的选择逻辑在 `Executor.get_class`（`abstract.py:66-107`）：`distributed_executor_backend` 显式指定就直用；否则按配置解析——单卡默认 `"uni"`，多卡默认 `"mp"`，有 Ray 环境 `"ray"`（缺省解析在 `vllm/config/parallel.py:1279-1317`）。三种实现：`UniProcExecutor`（`uniproc_executor.py:51`，同进程直调，单卡即它）、`MultiprocExecutor`（`multiproc_executor.py:150`，逐卡拉起 WorkerProc 子进程）、`RayExecutorV2`（`ray_executor_v2.py:239`，继承自 MultiprocExecutor）。

### 环节 6:Executor ↔ Worker 的第二套 IPC

这套通道和前端 ZMQ **不是一回事**：`MultiprocExecutor._init_executor`（`multiproc_executor.py:157-316`）建 `MessageQueue`（`:203-209`，实现在 `vllm/distributed/device_communicators/shm_broadcast.py:498`，共享内存环形队列 + zmq 通知），把 `SchedulerOutput` 广播给每个 worker,worker 回传 `ModelRunnerOutput`。Worker 侧是纯 busy loop（`worker_busy_loop`，`multiproc_executor.py:1189-1197`）：`dequeue` RPC 请求 → 按 方法名反射或 cloudpickle 闭包调用 → 结果入 response 队列。选共享内存而不是 ZMQ，是因为 `SchedulerOutput` 大（含全部 block table）,shm 避免序列化拷贝。

### 环节 7:Worker 与 Model Runner（执行层）

调用链：`EngineCore.step` → `MultiprocExecutor.execute_model`（`:431`）→ `collective_rpc` → worker busy loop → `WorkerWrapperBase.execute_model`（`vllm/v1/worker/worker_base.py:419`）→ `Worker.execute_model`（`vllm/v1/worker/gpu_worker.py:1513-1560`，PP 非 first rank 先收上游中间张量）→ `GPUModelRunner.execute_model`（`vllm/v1/worker/gpu/model_runner.py:1865`）。

Model Runner（当前即 MRV2）在 `execute_model` 里做的：请求状态增量维护（`finish/free/add/update_requests`，`:1872-1879`）→ `block_tables.apply_staged_writes()`（`:1888`，StagedWrite 模式：CPU 暂存 diff、打包拷 GPU、单 kernel 应用）→ CUDA graph 与 DP 同步决策（`dispatch_cg_and_sync_dp`，调用在 `:1933`、定义在 `dp_utils.py:232`）→ 执行前向。MRV2 的四个特征设计，与代码的对应关系：

| 设计文档 `model_runner_v2.md` 说的               | HEAD 代码里的落点                                             |
| ------------------------------------------------ | ------------------------------------------------------------- |
| Persistent Batch（请求占固定行，状态与输入解耦） | `gpu/model_states/` 的 ModelState 体系                        |
| StagedWriteTensor                                | `gpu/block_table.py:34` `BlockTables` + `apply_staged_writes` |
| 显式 CudaGraphManager                            | `gpu/cudagraph_utils.py:154/641`                              |
| GPU-native 采样(Gumbel-max)                      | `gpu/sample/gumbel.py` 等 Triton sampler                      |

### 教学彩蛋：文档说的 vs 代码看到的

本讲准备时实际发现的文档滞后（2026-11-24 核对），留作「以代码为准」方法的示范：

1. `docs/design/arch_overview.md`「LLM Engine」一节仍写 `LLMEngine` 在 `vllm/engine/llm_engine.py`——实际该文件只是别名 shim（`LLMEngine = V1LLMEngine`），实体是 `vllm/v1/engine/async_llm.py` 的 `AsyncLLM`。
2. 网上教程普遍引用 `vllm/entrypoints/openai/api_server.py`——HEAD 上它已是废弃 shim，实体迁到 `vllm/entrypoints/launchers/api_server/entry.py`。
3. 网络文章普遍说 EngineCore 有 `poll()` 方法——HEAD 上没有，轮询已改为 input_queue 阻塞 + 独立 IO 线程。

读源码的方法论：入口从 `vllm serve` 的 main 追；别名的 shim 文件点开看第一行；「文档没提的文件」（`launchers/` 目录）往往才是实体。

## 四、实验与产出

见 [hands-on-exercise.md](hands-on-exercise.md)：实验 1 用脚本观察进程拓扑并对照 `A + DP + N` 公式（纯 CPU）；实验 2 起服务追一次请求的全链路日志并标注环节（需 GPU）；实验 3（可选）`--enforce-eager` 对照。课后作业见 [homework.md](homework.md)。

## 五、边界

| 内容                               | 本讲                                             | 去处                    |
| ---------------------------------- | ------------------------------------------------ | ----------------------- |
| 调度器内部（token budget、优先级） | 只看到 `scheduler.schedule()` 调用点             | 第 2 讲                 |
| KV cache 管理器对象图              | 只看到 `_initialize_kv_caches`                   | 第 3 讲 / KV Cache 八讲 |
| Model Runner 前向内部              | 只到 `execute_model` 入口                        | 第 5 讲                 |
| ZMQ API                            | ROUTER/DEALER/PULL/PUSH 拓扑与 bind/connect 方向 | 不深入 zmq 协议本身     |

## 六、前后衔接

第 2 讲从本讲的 `scheduler.schedule()` 调用点深入 EngineCore 内部的调度器；第 5 讲从 `GPUModelRunner.execute_model` 入口深入执行层；第 9 讲会回到「显存不足时」这个本讲刻意跳过的分支。
