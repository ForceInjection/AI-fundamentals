# 第 1 讲教学大纲：引擎总架构

## 基本信息

| 项目     | 内容                                                   |
| -------- | ------------------------------------------------------ |
| 课时     | 120 分钟                                               |
| 基线     | nano-vllm `b9dd55a`；vLLM `1acec5c009`(v0.34.0rc0-1153) |
| 先修     | nano-vllm 实战课程第 1 课                              |
| 授课形式 | 演进对照 + 源码走读 + 上手实验                         |

## 教学目标

- **知识目标**：说出三类进程(API server / EngineCore / Worker)各自持有与不持有什幺（如 tokenizer 只在前端）；说清两套 IPC（前端 ZMQ、Executor↔Worker 共享内存）各自为什么。
- **能力目标**：沿 `file:line` 追踪一次请求的全链路；用进程公式 `A + DP + N` 解释自己机器上的进程数；判断一个网络教程是否用了过时入口(api_server.py shim)。
- **素养目标**：建立「文档说的要拿代码验证」的核对习惯（本讲给出三个真实滞后案例）。

## 时间安排

| 部分                          | 时长   | 内容                                                      |
| ----------------------------- | ------ | --------------------------------------------------------- |
| nano 已学形态回顾             | 10 min | 单进程 `LLMEngine`；三堵墙（GIL/扩展/故障隔离）              |
| 演进：三步拆分                 | 15 min | V0 → V1 三进程 → MRV2 唯一化（2025-11 落地，2026-11 删旧版） |
| 现状走读：七环节               | 45 min | 见下方函数级阅读清单，课堂重点讲环节 2/4/7，其余课前预读    |
| 实验（见 hands-on-exercise.md） | 40 min | 进程拓扑观察 + 全链路日志追踪 + eager 对照（可选）          |
| 答疑                          | 10 min |                                                           |

## 函数级阅读清单（课前预读）

| 文件                                                 | 函数 / 类                                          | 行号                  | 看什么                                                                |
| ---------------------------------------------------- | -------------------------------------------------- | --------------------- | --------------------------------------------------------------------- |
| `vllm/entrypoints/launchers/api_server/entry.py`     | `build_async_engine_client_from_engine_args`       | 69-112                | AsyncLLM 在哪实例化（不在旧 api_server.py）                             |
| `vllm/entrypoints/openai/chat_completion/serving.py` | `OpenAIServingChat.create_chat_completion`         | 290（调用点 411）       | 请求落到 engine_client.generate 的位置                                |
| `vllm/v1/engine/async_llm.py`                        | `AsyncLLM.__init__`                                | 97-249                | 三件装配：InputProcessor / OutputProcessor / EngineCoreClient          |
| `vllm/v1/engine/async_llm.py`                        | `generate`                                         | 740-868               | 请求四步；`_add_request:581` 的跨进程落点                              |
| `vllm/v1/engine/async_llm.py`                        | `_run_output_handler`                              | 870-966               | 输出如何回流到每请求 collector                                        |
| `vllm/v1/engine/core_client.py`                      | `MPClient.__init__`                                | 630 附近              | 前端 bind ROUTER / PULL(688-701)                                      |
| `vllm/v1/engine/core.py`                             | `process_input_sockets` / `process_output_sockets` | 2222-2231 / 2325-2327 | 引擎侧 DEALER / PUSH connect;IO 线程解耦(1434-1449)                   |
| `vllm/v1/engine/core.py`                             | `EngineCore.__init__`                              | 133-286               | 初始化顺序：executor → KV → structured output → scheduler；无 tokenizer |
| `vllm/v1/engine/core.py`                             | `step`                                             | 714-745               | schedule → execute_model(non_block)→ update_from_output               |
| `vllm/v1/engine/core.py`                             | `run_busy_loop`                                    | 1814-1824             | busy loop 两件事；无活阻塞输入队列                                     |
| `vllm/v1/executor/abstract.py`                       | `Executor.get_class`                               | 66-107                | executor 后端选择逻辑                                                 |
| `vllm/v1/executor/multiproc_executor.py`             | `_init_executor` / `worker_busy_loop`              | 157-316 / 1189-1197   | 共享内存 MessageQueue;worker 纯 RPC 循环                              |
| `vllm/v1/worker/gpu_worker.py`                       | `Worker.execute_model`                             | 1513-1560             | PP 中转；转发给 Model Runner                                           |
| `vllm/v1/worker/gpu/model_runner.py`                 | `GPUModelRunner.execute_model`                     | 1865 起               | StagedWrite(1883)与 CUDA graph 决策(1905)                             |

预读方法：每个条目先只看「行号段 + 一句话」，课堂上对答案。注意 `vllm/entrypoints/openai/api_server.py` 与 `vllm/engine/llm_engine.py` 是 shim，不要从它们读起。

## 重点与难点

- **重点**：三类进程各自持有什么（尤其 tokenizer 只在前端）；`EngineCore.step` 的五步；两套 IPC 的分工。
- **难点**:ROUTER/DEALER 多对多路由（多 API server × 多 EngineCore）;Executor↔Worker 为什么用共享内存而不是 ZMQ（SchedulerOutput 大，避免序列化拷贝）。
- **易错点**：从 `api_server.py` 或 `LLMEngine` shim 开始读源码；以为 MRV2 是要打开的开关（HEAD 上它是唯一实现）。

## 参考材料

- 官方：`docs/design/arch_overview.md`（V1 Process Architecture 一节）、`docs/design/model_runner_v2.md`
- 深读：Gordić「Inside vLLM: Anatomy of a High-Throughput LLM Inference System」（V1 口径，2025-08；本讲的进程模型部分与其一致，执行层以其后的 MRV2 为准）
- 演进：官方博文「vLLM V1」(2025-01-27);git 考古 PR #25266（MRV2 落地）与 #63373（V1 删除）
- 对照：nano-vllm 先修课第 1 课
