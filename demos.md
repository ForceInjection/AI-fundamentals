# Demos：仓库内可运行的代码与 Notebook

本仓库是知识库，但散落着 12 处**可运行的代码**——它们是各篇文章观点的「可运行证据」。本页是唯一的总索引：每项标注依赖要求、对应文章与验证状态。使用前请先读[验证状态说明](#验证状态说明)。

> 维护约定：新增 demo 时在本页登记一行；验证过就在「验证状态」列更新日期。发现跑不通，欢迎提 issue 或 PR。

## 一、AI Agent 与 MCP

| Demo                                                                                                                                  | 主题                                                                              | 依赖                                                           | 对应文章                                                                                                                           | 验证状态                                                                                                                                                                                                                                                                                                                   |
| ------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- | -------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [08_agentic_system/mcp/mcp_demo](08_agentic_system/mcp/mcp_demo/README.md)                                                            | MCP 客户端/服务端三模式（stdio/SSE/Streamable HTTP）+ Parallel 免匿名网页搜索 CLI | Python 3.11+，mcp、httpx（独立 venv）                          | 目录 README 自含教程                                                                                                               | ✅ 2026-10-12（7/7 测试 + 真实端点调用）                                                                                                                                                                                                                                                                                   |
| [08_agentic_system/multi_agent/multi_agent_system](08_agentic_system/multi_agent/multi_agent_system/README.md)                        | 多 Agent 系统（分组对话、监控、工作流）                                           | Python + requirements.txt，支持 docker-compose                 | [Multi-Agent 课程 §2/§4/§5](10_ai_related_course/multi_agent_system/multi_agent_training/02-LangGraph框架/02-LangGraph深度应用.md) | ✅ 2026-10-12 修复后实测 24/24 通过。修复内容：requirements 两处依赖冲突（langchain-langgraph、pytest-pytest-asyncio，已升至最新兼容组合）；MessageQueue.put 自锁死锁（Condition 复用 \_lock 不可重入）；订阅回调从未被调用的死代码补全；测试重写对齐真实 API（初版 27 个测试按未落地的 API 设计稿编写，提交时即全部失败） |
| [08_agentic_system/agent_design/coref-dialogue-system](08_agentic_system/agent_design/docs/coreference-resolution-dialogue-system.md) | 指代消解对话系统（实体识别、状态管理、多模态指代）                                | Python + requirements.txt                                      | [指代消解与对话系统设计](08_agentic_system/agent_design/docs/coreference-resolution-dialogue-system.md)                            | ✅ 2026-10-12 实测（tests/run_tests.py 4/4 通过，含性能测试 93s）                                                                                                                                                                                                                                                          |
| [08_agentic_system/memory/langchain/code](08_agentic_system/memory/langchain/langchain_memory.md)                                     | LangChain 记忆机制（基础示例 + LangGraph 记忆）                                   | Python + requirements.txt + LLM API key（`config.example.py`） | [LangChain 记忆机制](08_agentic_system/memory/langchain/langchain_memory.md)                                                       | ✅ 2026-10-12 实测（24/24 单文件测试过；注：unittest discover 全量模式与测试 patch 写法不兼容会报 12 errors，请按单文件跑）                                                                                                                                                                                                |

## 二、推理与显存规划

| Demo                                                                         | 主题                                                         | 依赖                         | 对应文章                                                                                                                                                                    | 验证状态                      |
| ---------------------------------------------------------------------------- | ------------------------------------------------------------ | ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------- |
| [09_inference_system/memory_calc](09_inference_system/memory_calc/README.md) | 模型显存与 KV Cache 计算器（Qwen3/DeepSeek-V4，含 GQA 分析） | 纯 Python 标准库，无三方依赖 | [KV Cache 原理](09_inference_system/kv_cache/01_concepts/basic/kv_cache_basics.md) · [B300 部署与 KV Cache](09_inference_system/deployment/b300-deployment-and-kv-cache.md) | ✅ 2026-10-11（纯标准库直跑） |

## 三、RAG 与知识图谱

| Demo                                                                                                 | 主题                                                            | 依赖                                                | 对应文章                                                                  | 验证状态                                                                                                                                          |
| ---------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- | --------------------------------------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| [07_rag_and_tools/synergized_llms_kgs/demo](07_rag_and_tools/knowledge_graph/neo4j_handson_guide.md) | LLM × 知识图谱协同（ETL 管道、Agent、API，docker-compose 编排） | Python + requirements.txt + docker-compose（Neo4j） | [Neo4j 实战指南](07_rag_and_tools/knowledge_graph/neo4j_handson_guide.md) | ✅ 2026-10-12 实测（依赖可装 + api/agent import ✓ + compose 语法 ✓；未起 Neo4j 全栈实测。运行：在 synergized_llms_kgs/ 目录下以 PYTHONPATH=. 跑） |
| [07_rag_and_tools/rag_basics/notebooks](07_rag_and_tools/rag_basics/notebooks/rag_lesson2.ipynb)     | RAG 基础教程 Notebook                                           | Python + notebook 内声明的依赖                      | 目录内教程                                                                | ⚠️ 2026-05-15                                                                                                                                     |

## 四、训练与微调

| Demo                                                                                                                      | 主题                               | 依赖                           | 对应文章                                                                                              | 验证状态      |
| ------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- | ------------------------------ | ----------------------------------------------------------------------------------------------------- | ------------- |
| [05_model_training_and_fine_tuning/sft_example](05_model_training_and_fine_tuning/sft_example/一文入门垂域模型SFT微调.md) | 垂域模型 SFT 微调（Qwen2）Notebook | Python + GPU + notebook 内依赖 | [一文入门垂域模型 SFT 微调](05_model_training_and_fine_tuning/sft_example/一文入门垂域模型SFT微调.md) | ⚠️ 2026-05-15 |
| [98_llm_programming/langgraph](98_llm_programming/langgraph/README.md)                                                    | LangGraph 入门 Notebook            | Python + notebook 内依赖       | [LangGraph 介绍](98_llm_programming/langgraph/langgraph_intro.md)                                     | ⚠️ 2026-03-23 |

## 五、GPU 编程（CUDA）

| Demo                                                                                   | 主题                                                                                          | 依赖              | 对应文章                                                             | 验证状态                                            |
| -------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- | ----------------- | -------------------------------------------------------------------- | --------------------------------------------------- |
| [02_gpu_programming/02_cuda/code](02_gpu_programming/02_cuda/code/README.md)           | CUDA 基础系列（NUMA、Stream 并发、Graphs、Tensor Core GEMM、Bank Conflict 等 8+ 个 .cu 基准） | nvcc + NVIDIA GPU | 目录 README 自含编译运行说明                                         | ✅ 入库前已在真实 GPU 环境验证 |
| [02_gpu_programming/04_profiling/code](02_gpu_programming/04_profiling/code/README.md) | PCIe/HBM 带宽与传输效率基准                                                                   | nvcc + NVIDIA GPU | [NCCL 基准系列](03_ai_cluster_ops/03_nccl/04_nccl_benchmark.md) 引用 | ✅ 入库前已在真实 GPU 环境验证 |

## 六、昇腾教程配套脚本

[02_npu_programming](02_npu_programming/README.md) 的编号教程（01 环境 → 14 量化）各配 Python 脚本，另有 07_rag_on_npu 的 RAG on NPU 示例（4 py）。依赖 MindSpore/CANN 等昇腾软件栈。| ✅ 入库前已验证 · 教程随系列文章维护

## 验证状态说明

- ✅ = 已验证可运行：标注复跑日期的为本轮在维护者机器上实测，标注「入库前已验证」的为入库时的真实验证；
- ⚠️ = 入库前已验证、入库后未再复跑，最后提交日期仅供参考——遇到依赖漂移请以目录内 README 为准，或提 issue；
