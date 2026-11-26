# 第 1 讲：引擎总架构 — PPT 大纲

## 设计说明

- **总页数**：约 22 页
- **演讲时长**：120 分钟（nano 回顾 10 min + 演进 15 min + 现状走读 45 min + 实验 40 min + 答疑 10 min）
- **核心视觉**：三类进程拓扑图（本讲锚点，第 8 页）；一次请求旅程泳道图；两套 IPC 对照（裸 ZMQ vs 共享内存 MessageQueue）；MRV2 文档↔代码对应表
- **教学策略**：每讲一个「核心问题」贯穿；先复用先修课第 1 课的 nano 锚点图建立「已知」，再逐环节放大生产拓扑；走读页每页只落一个环节，保持「我在总图的哪里」的位置感
- **配色**：API 层蓝色系、EngineCore 紫色系、Worker 绿色系、nano 对照橙色系；IPC 通道用灰底等宽字
- **边界**：调度器内部（第 2 讲）、KV 管理器（第 3 讲）、Model Runner 前向细节（第 5 讲）在本讲只出现调用点，不展开

---

## 开场（第 1–2 页）

### 第 1 页：封面

- **标题**：引擎总架构：从单进程到三进程
- **副标题**：vLLM 十讲 · 第 1 讲
- **课程信息**：120 分钟 | 基线：nano-vllm `b9dd55a` + vLLM `2f5d8a2b1`(v0.34.1rc0-84)
- **一张图**：左「nano-vllm ≈ 1,400 行，一个进程」，右「vLLM ≈ 20 万+ 行，一类进程家族」，中间箭头「这门课讲这段路」

### 第 2 页：本讲核心问题与议程

- **核心问题**（大字居中）：一次请求从 API 进来，经过哪些进程、哪些类、哪些方法，怎么变成 token 流出去？
- **四段议程**：
  - nano 已学形态回顾（10 min，第 3–4 页）
  - 演进：三步拆分（15 min，第 5–7 页）
  - 现状走读：七个环节（45 min，第 8–20 页）
  - 实验：进程拓扑与全链路日志（40 min，第 21 页）
- **先修声明**：假设已学 nano-vllm 第 1 课；本讲所有行号按基线核对，课件附核对日期

## 第一部分：nano 已学形态（10 min，第 3–4 页）

### 第 3 页：nano 回顾：一条路径走到底

- **复用先修课第 1 课锚点图**（`LLMEngine.generate` → step 循环 → 输出）
- **三个锚点**（橙色系，标注行号）：
  - `LLMEngine.generate`（`nanovllm/engine/llm_engine.py:68-98`）：`add_request`（tokenize 在此）→ 循环 step → decode 文本
  - `LLMEngine.step`（`llm_engine.py:52-63`）：每步只在 prefill / decode 中选一种
  - 三段式：调度 → 执行 → 回写
- **口播**：「先修课你们沿这条路径走完了一个完整引擎；今天这节课，这条路径会被拆到三个进程里」

### 第 4 页：nano 的三堵墙

- **标题**：教学形态很好，生产撞三堵墙
- **墙 1**：API 异步 IO 与 GPU busy loop 在同一进程抢 GIL——流式输出会卡计算循环
- **墙 2**：单进程无法扩展多卡多机；Executor 抽象不存在
- **墙 3**：API 层崩溃 = 引擎状态陪葬；没有故障边界
- **一张图**：nano 单进程框图，三个红色裂纹标注三堵墙
- **口播**：V1 的三进程拆分，每堵墙对应一个进程边界

## 第二部分：演进（15 min，第 5–7 页）

### 第 5 页：演进时间线

- **时间轴**（横向，三个节点）：
  - V0（2023-06 起）：单进程 `LLMEngine`，调度与执行纠缠，多卡靠旧 Ray worker
  - V1（2025-01 发布）：三类进程拆分；官方 `arch_overview.md`「V1 Process Architecture」
  - MRV2（2025-11 落地 PR #25266）：Worker 内部 ModelRunner 重构；2026-11-07 旧 V1 删除(PR #63373)
- **关键澄清**（红字）：HEAD 上 MRV2 不是开关，是唯一实现——网上「V1/V2 切换」的说法已过期
- **口播**：注意时间——V1 拆进程解决三堵墙，MRV2 解决的是 Worker 内部的可维护性，两者别混

### 第 6 页：三类进程与公式

- **一张图**：官方 `arch_overview` 的进程拓扑重绘（蓝色 API / 紫色 EngineCore / 绿色 Worker / 灰色 coordinator）
- **进程公式**：`A + DP + N`（A 个 API server,DP 个 EngineCore,N 个 Worker;DP>1 时 +1 coordinator）
- **示例**：tp=4 单机 → A=1 + DP=1 + N=4 = 6 进程
- **各进程持有什么**（三栏表）：
  - API 层：tokenizer、detokenizer、每请求输出信箱
  - EngineCore：调度器、KV cache 状态、StructuredOutputManager——**没有 tokenizer**
  - Worker:Model Runner（权重、CUDA Graph、采样 kernel）

### 第 7 页：两套 IPC（总览）

- **对照表**（灰底等宽）：
  - 前端 ↔ EngineCore：裸 ZMQ,ROUTER/DEALER（输入）+ PULL/PUSH（输出），msgpack 帧——为多对多路由
  - Executor ↔ Worker：共享内存 `MessageQueue`（`shm_broadcast.py:498`）+ zmq 通知——SchedulerOutput 大，避免序列化拷贝
- **口播**：「两套通道长得不一样，不是历史包袱，是流量特征不同：前端消息小而多，worker 消息大而重」

## 第三部分：现状走读（45 min，第 8–20 页）

### 第 8 页：全链路总图（本讲锚点页）

- **一张图**：请求旅程泳道图（课程主页同款），七个环节编号 ①–⑦，后续每页高亮一个环节
- 泳道：API 进程 | ZMQ | EngineCore 进程 | shm | Worker 进程
- ①入口 ②AsyncLLM ③前端 IPC ④EngineCore ⑤Executor 选择 ⑥第二套 IPC ⑦Worker/Model Runner
- **口播**：接下来七页，每页在总图上点亮一个环节；位置感比细节重要

### 第 9 页：环节① 入口在哪

- **上稿**：
  - 服务启动：`vllm/entrypoints/launchers/api_server/entry.py:94` → `AsyncLLM.from_vllm_config(...)`
  - 请求路由：`POST /v1/chat/completions`（`chat_completion/api_router.py:44-59`）→ `OpenAIServingChat.create_chat_completion`（`chat_completion/serving.py:290`）
  - 落点：`serving.py:411` → `engine_client.generate(...)`
- **红色警示框**：「`vllm/entrypoints/openai/api_server.py` 已是废弃 shim（v0.37.0 移除）——网上旧教程的重灾区。判断方法：文件第一行 `warnings.warn`」
- **口播**：这也是本讲方法论第一次出现——从 `vllm serve` 的 main 追入口，不信路径记忆

### 第 10 页：环节② AsyncLLM 装配

- **上稿**（`vllm/v1/engine/async_llm.py`）：
  - `__init__`（`:97-249`）装配三件：InputProcessor（`:173`）、OutputProcessor（`:181-187`，**tokenizer 在这**）、`EngineCoreClient.make_async_mp_client`（`:196`）
  - `generate`（`:742`）四步：建流 → EngineCoreRequest → 注册 detokenizer → `_add_request`（`:583-600`）
  - 跨进程落点：`:600` `await self.engine_core.add_request_async(request)`
- **一张图**：AsyncLLM 内部框图，三个装配件与输出后台任务（`_run_output_handler`，`:872`）

### 第 11 页：输出回流

- **上稿**：
  - 常驻 asyncio 任务 `_run_output_handler`（`:872`）：循环 `get_output_async()`（`:896`）→ 按块送 OutputProcessor（`:919` 起）
  - 每请求一个 `RequestOutputCollector` 信箱（`output_processor.py:54-127`）：DELTA 模式生产快于消费时合并（`add(output, aggregate=...)`）
- **一张图**：collector 信箱队列图（生产者 `output_handler` / 消费者 generate 循环）
- **口播**：tokenizer 在这层——这就是上一页说「EngineCore 没有 tokenizer」的另一半

### 第 12 页：环节③ 第一套 IPC

- **一张图**：ZMQ 拓扑图。前端进程 **bind** ROUTER（输入）/ PULL（输出）；每个 EngineCore **connect** DEALER(identity=`engine_index`)/ PUSH（`core_client.py:688-701`，`core.py:2200-2203`、`:2311-2318`）
- **上稿**：
  - 多对多：A 个 API server × DP 个 EngineCore，任意 API 可路由到任意 core（`arch_overview` 原话）
  - EngineCore 进程内两个 daemon 线程做 socket↔队列搬运（`core.py:1420-1449`），busy loop 只碰内存队列
- **口播**：bind 在前端——顺序反了会怎样？客户端要连多个引擎，服务端地址固定

### 第 13 页：环节④ EngineCore 初始化

- **上稿**（`EngineCore.__init__`，`core.py:131-288`），初始化顺序（箭头图）：
  1. `model_executor = executor_class(vllm_config)`（`:166`）
  2. `_initialize_kv_caches`（`:289`，profile 显存后分配）
  3. StructuredOutputManager（`:188`）
  4. Scheduler（`:192-208`）
- **口播**：顺序即依赖——调度器要吃 KV cache 配置，所以排最后；这个顺序在第 3 讲还会见到

### 第 14 页：EngineCore.step 五步

- **一张图**：step 流程横条（`core.py:701-731`）
  1. `scheduler.schedule()`（`:711`）
  2. `model_executor.execute_model(scheduler_output, non_block=True)`（`:714`，拿 Future）
  3. `get_grammar_bitmask`（`:715`）
  4. `_wait_for_output`(`:717`，future.result())
  5. `scheduler.update_from_output`（`:725`）→ `EngineCoreOutputs`
- **口播**：②和④之间就是「CPU 等\_GPU」——第 2 讲的 async scheduling 会把这个空隙填上
- **易错澄清**：EngineCore 上没有自己的 `execute_model` 方法，也没有 `poll()`——旧教程说法已失效

### 第 15 页：busy loop

- **上稿**（`run_busy_loop`，`core.py:1801-1812`，代码摘录 5 行）：

  ```python
  while self._handle_shutdown():
      self._process_input_queue()
      self._maybe_publish_request_counts()
      # （每轮触发两次:输入处理后与收尾各一）
      self._process_engine_step()
  ```

  - 无活时 `_process_input_queue` 阻塞等待输入（watchdog 关闭窗口）
  - 进程入口：`run_engine_core`（`core.py:1642`），被 `CoreEngineProcManager`（`utils.py:305`）作为 Process target 拉起

### 第 16 页：环节⑤ Executor 选择

- **上稿**（`vllm/v1/executor/abstract.py:66-107` `Executor.get_class`）：
  - 显式 `distributed_executor_backend` 直用；"ray"→`RayExecutorV2`；`"mp"`→`MultiprocExecutor`；`"uni"`→`UniProcExecutor`
  - 缺省解析（`config/parallel.py:1279-1317`）：单卡 → `"uni"`，多卡 → `"mp"`
- **一张小表**：三种实现的进程关系（uni 同进程 / mp 子进程 / ray 继承 Multiproc）
- **口播**：单卡实验你的服务里其实没有独立 Worker 进程——这解释了实验 1 里进程数比直觉少

### 第 17 页：环节⑥ 第二套 IPC

- **一张图**：MultiprocExecutor 与 N 个 WorkerProc 之间的 shm MessageQueue（`multiproc_executor.py:203-209`）
- **上稿**：
  - 广播：`rpc_broadcast_mq` 发 `SchedulerOutput`；回传：`worker_response_mq` 收 `ModelRunnerOutput`（`:750-782`）
  - Worker 是纯 RPC 循环：`worker_busy_loop`（`:1189-1197`）dequeue → 反射方法名或 cloudpickle 闭包 → 执行 → 入队
- **对照框**：为什么这里不用裸 ZMQ？SchedulerOutput 含全部 block table，大；shm 省序列化

### 第 18 页：环节⑦ Worker 调用链

- **一张图**：垂直调用链（五级，全带 file:line）：
  - `EngineCore.step` → `MultiprocExecutor.execute_model`（`:431`）→ `collective_rpc` → `WorkerWrapperBase.execute_model`（`worker_base.py:419`）→ `Worker.execute_model`（`gpu_worker.py:1513-1560`）→ `GPUModelRunner.execute_model`（`gpu/model_runner.py:1865`）
- **上稿**：PP 非 first rank 先 `irecv_tensor_dict` 收上游中间张量（`:1513-1541`）；pooling 模型补 `.pool()`
- **口播**：到 `GPUModelRunner.execute_model` 为止是本讲边界；它内部是第 5 讲

### 第 19 页：MRV2 四特征

- **标题**：MRV2：不是开关，是唯一实现
- **文档↔代码对应表**（`docs/design/model_runner_v2.md` ↔ HEAD）：

  | 设计文档说的                     | HEAD 代码落点                                                             |
  | -------------------------------- | ------------------------------------------------------------------------- |
  | Persistent Batch（请求占固定行） | `gpu/model_states/` ModelState 体系                                       |
  | StagedWriteTensor                | `gpu/block_table.py:34` + `apply_staged_writes`（`model_runner.py:1891`） |
  | 显式 CudaGraphManager            | `gpu/cudagraph_utils.py:154/641`                                          |
  | GPU-native 采样                  | `gpu/sample/gumbel.py`(Triton,Gumbel-max)                                 |

- **背景行**：MRV2 重构动机——V1 的 `gpu_model_runner.py` 已长到 7,285 行（PR #63373 删除时的大小）
- **口播**：「执行前向内部」是第 5 讲；本讲只确认「跑到哪一层、谁负责」

### 第 20 页：文档说的 vs 代码看到的

- **三个滞后案例**（每个：文档/教程原话 → 代码事实 → 核对方法）：
  1. `arch_overview.md` 仍写 `LLMEngine` 在 `vllm/engine/llm_engine.py` → 实为别名 shim（`LLMEngine = V1LLMEngine`），实体是 `vllm/v1/engine/async_llm.py`
  2. 旧教程入口 `entrypoints/openai/api_server.py` → 废弃 shim，实体在 `launchers/`
  3. 「EngineCore 有 `poll()`」→ 已不存在，`input_queue` 阻塞 + IO 线程
- **方法论框**（红底）：从 `vllm serve` main 追入口；shim 文件点开看第一行；「文档没提的目录」往往是实体；一切行号带核对日期

## 第四部分：实验与收尾（第 21–22 页）

### 第 21 页：本讲实验

- **三行实验卡**（对应 hands-on-exercise.md）：
  - 实验 1（纯 CPU）：`process_topology.py` 归类进程，对照公式 `A + DP + N`
  - 实验 2(GPU):DEBUG 日志追一次请求，标注七环节
  - 实验 3（可选）：`--enforce-eager` 对照，数据留给第 5 讲
- **产出要求**：进程归类表；「日志行 ↔ 环节号」标注表（≥7 行）

### 第 22 页：结课与预告

- **本讲你应能**：画三类进程拓扑；沿 file:line 追全链路；辨认可疑的过时教程
- **作业**：作业 1（计考核）——画请求全生命周期图（进程/类/方法/数据形态四级，一页）
- **预告**：下一讲打开 `EngineCore.step` 第一步——`scheduler.schedule()` 的 1,114 行；带上一个问题：长 prompt 和短请求混着来，batch 怎么拼？
- **下节材料**：第 2 讲 syllabus 函数级阅读清单（15 条，课前预读）

---

## 图资产清单（自制）

| 图                        | 页  | 说明                                        |
| ------------------------- | --- | ------------------------------------------- |
| nano 单进程框图（三堵墙） | 4   | 橙色系；裂纹标注                            |
| 演进时间线                | 5   | 三节点横轴                                  |
| 三类进程拓扑（重绘官方）  | 6   | 蓝/紫/绿/灰四色；公式入图                   |
| 请求旅程泳道总图          | 8   | 七环节编号；后续页复用高亮                  |
| AsyncLLM 装配框图         | 10  | 三装配件 + 后台任务                         |
| collector 信箱队列        | 11  | 生产者/消费者                               |
| ZMQ 拓扑图                | 12  | ROUTER/DEALER/PULL/PUSH + bind/connect 方向 |
| step 五步横条             | 14  | 与第 1 讲 README 五步一致                   |
| shm MessageQueue 图       | 17  | 与 ZMQ 图并列对照                           |
| Worker 调用链             | 18  | 五级垂直链，全 file:line                    |
