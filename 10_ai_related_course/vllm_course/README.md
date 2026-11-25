# vLLM 十讲：从 nano-vllm 到生产推理引擎

> 10 讲 × 120 分钟 | 2026-11 起滚动更新

nano-vllm 用约 1,400 行 Python 实现了推理引擎的全部核心机制：调度、KV cache 分页、前缀缓存、张量并行、CUDA Graph。学完它，你知道每个模块「为什么存在、最小形态长什么样」。但这 1,400 行与生产环境的 vLLM（20 万+ 行）之间，隔着三年的时间、两次架构换代(V0 → V1 → Model Runner V2)和一整套工程化改造。

这门课讲的就是这段路：每个主题从你在 nano-vllm 里见过的最小形态出发，看它在真实 vLLM 里演进成了什么、为什么这样设计，最后落到当前版本(v0.34)的源码与上手操作。

**适合谁**：学过 [nano-vllm 实战课程](https://github.`com/ForceInjection/nano-vllm/blob/main/docs/llm-inference-visual/README.md`)（8 课），或能独立读懂 nano-vllm 的 `scheduler.py` 和 `block_manager.py` 的工程师。目标是用上、用好 vLLM 这类生产推理框架。

| 项目   | 内容                                                     |
| ------ | -------------------------------------------------------- |
| 讲次   | 10 讲，每讲 120 分钟                                      |
| 形式   | 演进对照 + 当前源码走读 + 上手实验 + 答疑                |
| 语言   | 中文                                                     |
| 前置   | nano-vllm 实战课程（8 课），或等价的 nano-vllm 源码阅读经验 |
| 不需要 | 不要求读过 vLLM 源码；CUDA 经验仅第 5、6 讲部分实验需要   |

## 一、为什么要学

nano-vllm 教会你机制，生产环境要求你面对真实约束：模型形态从 Qwen3 单文本变成 MLA/NSA/混合注意力；调度从 FIFO 变成 chunked prefill 加策略选择；KV 管理从单一 BlockManager 变成多后端矩阵，连 PagedAttention 本身都在 2026 年退役了。这些变化nano 里没有，生产里天天遇到。

学完这门课，你能做四件事：

1. 部署一个生产配置的 vLLM 服务，并解释每个关键配置项（`--max-num-seqs`、`--gpu-memory-utilization`、`--kv-cache-dtype` 等）背后的机制。
2. 在当前 vLLM 源码里定位任何核心路径：一个请求从 API server 进来之后经过哪些进程、哪些类、哪些方法，`file:line` 级别。
3. 遇到性能问题知道往哪个子系统查：吞吐低查调度与并行，显存爆查 KV 管理与量化，首 token 慢查 prefill 与编译。
4. 评估进阶特性要不要上：投机解码五法选型、量化方案、KV offload、PD 分离，各自适合什么场景、代价是什么。

## 二、课程是什么

课程主线是一个问法：「nano 里你已经见过它的最小形态，它在生产 vLLM 里变成了什么，为什么？」

每讲按四层展开：

1. **nano 已学形态**（10 分钟回顾）：你在先修课里写的那个类、那条路径。
2. **演进**：V0 → V1 → Model Runner V2，这个机制经历了几次重写，每次为什么。
3. **当前现状**：以本地 vLLM 仓 HEAD（`2f5d8a2b1`，v0.34.1rc0-84）为基线的源码走读。
4. **上手实操**：部署参数、实验脚本、观测指标。

### 十讲一览

| #   | 主题                                      | 你会学到什么                                                                                                                                          | 材料                                      |
| --- | ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| 1   | **引擎总架构：从单进程到三进程**           | nano 单进程 `LLMEngine` 的边界；V1 的 AsyncLLM / EngineCore / Worker 三进程拆分与 ZMQ 通信；Model Runner V2 的模块化重构                                | [导读](`01-engine-architecture/README.md`)  |
| 2   | **调度器：从 FIFO 到策略化**               | nano 的 `Scheduler.schedule()` 与 preempt;vLLM 的 Scheduler 与策略解耦、chunked prefill(Sarathi-Serve)、prefill/decode 混合批、async scheduling       | [导读](`02-scheduler/README.md`)            |
| 3   | **KV cache 管理：PagedAttention 的生与死** | nano `block_manager`（xxhash 前缀链）；V0 BlockManager 到 V1 KVCacheManager；混合注意力催生 HMA;PagedAttention 为何在 2026 退役                           | [导读](`03-kv-cache-evolution/README.md`)   |
| 4   | **注意力后端：从 if-else 到后端矩阵**      | nano `attention.py` 的分支判断；vLLM 的 backend 注册与选择机制(FlashAttention / FlashInfer / Triton / FlashMLA)；按模型架构与硬件自动选型               | [导读](`04-attention-backends/README.md`)   |
| 5   | **CUDA Graph 与编译**                     | nano 预 capture 固定 batch;piecewise CUDA Graph;torch.compile 四档优化与调试                                                                          | [导读](`05-cuda-graph-compile/README.md`)   |
| 6   | **并行：TP 到 EP**                         | nano spawn + 共享内存的 TP;Ray + NCCL 的生产形态；EP 与 Wide-Expert Parallel;DP / PD 部署形态                                                          | [导读](`06-parallelism-ep/README.md`)       |
| 7   | **投机解码**                              | nano 没有；ngram / suffix / EAGLE / draft model / MTP 五法在职的选型框架（Medusa 已移除）、验证循环与 accept 指标                                        | [导读](`07-speculative-decoding/README.md`) |
| 8   | **量化**                                  | nano 裸 FP16;LLM Compressor 工作流；FP8 / AWQ / GPTQ 在 vLLM 里的加载路径；Marlin 类 kernel 的角色                                                      | [导读](`08-quantization/README.md`)         |
| 9   | **KV 分载与 PD 分离**                     | nano `kv_swap`（你在 fork 里实现的 swap 抢占）；V1 原生 offloading 的分层后端(cpu / fs / tiering);KV Connector(mooncake / nixl / LMCache);disagg prefill | [导读](`09-kv-offload-disagg/README.md`)    |
| 10  | **生产运维与生态**                        | OpenAI 兼容 server 与 metrics；`vllm bench`；production-stack / AIBrix / llm-d；从单机引擎到集群控制面的生态演进                                         | [导读](`10-production-ops/README.md`)       |

### 十讲各回答一个核心问题

| #   | 核心问题                                                       |
| --- | -------------------------------------------------------------- |
| 1   | 一次请求从 API 进来，经过哪些进程、哪些类，怎么变成 token 流出去？ |
| 2   | 长 prompt 和短请求混着来，每一步 batch 怎么拼才不互相拖累？       |
| 3   | KV 的分页管理用了八年，为什么 2026 年说退役就退役？               |
| 4   | 同一个模型，引擎怎么自动决定用哪个注意力 kernel?                 |
| 5   | decode 为什么离不开 CUDA Graph,piecewise 又解决了它的什么死穴？   |
| 6   | 一台机器装不下 MoE，专家怎么切到多台机器上才算切对了？            |
| 7   | 一次生成多个 token 还不出错，代价和收益怎么算？                   |
| 8   | FP8 和 FP16 的模型，vLLM 加载和执行路径差在哪？                   |
| 9   | 显存装不下的 KV，分载到 CPU / 磁盘 / 另一台机器，分别怎么做？     |
| 10  | 一个引擎变成一组服务，K8s、网关、控制面怎么接？                   |

### vLLM 内部，一个请求的旅程

```text
用户请求(OpenAI 兼容接口)
  → [第 1 讲] API 层 AsyncLLM:校验 → tokenize → add_request → ZMQ 发给 EngineCore
  → [第 2 讲] 调度器:waiting 队列取出 → chunked prefill 切块 → 与 decode 混合拼 batch
  → [第 3 讲] KV 管理:前缀缓存查询 → 分配 block → 构建本步 block table
  → [第 4 讲] attention 后端:按模型架构 × 硬件选 kernel → prefill/decode 分支
  → [第 5 讲] Model Runner:piecewise CUDA Graph / torch.compile 执行本步计算
  → [第 6 讲] 并行层:TP/EP 通信收集各卡结果(Ray + NCCL)
  → [第 7 讲] (可选)投机解码:草稿 N 个 token → 主模型一次验证
  → [第 8 讲] (可选)量化 kernel:反量化 GEMM(Marlin 类)参与执行
  → 采样 → 输出 token 经 ZMQ 回 API 层 → 流式返回用户
  → [第 9 讲] 显存不足时:offload 到 CPU / 磁盘,或跨实例传 KV(PD 分离)
  → [第 10 讲] 全程的 metrics、日志、压测与集群管控
```

### 动手，从第一行命令开始

每个主题不是听懂了就过去，每讲都有一个跑得出数字的实验：

- **第 1 讲**： `vllm serve` 起服务，`ps -ef | grep` 数出三进程，`curl` 一次请求抓全链路日志
- **第 2 讲**： 长短 prompt 混合压测，调 `--long-prefill-token-threshold` 看 TTFT 与吞吐互换
- **第 3 讲**： 同前缀冷热对照读 prefix cache 命中，混合注意力模型看分组日志
- **第 4 讲**： `--attention-backend FLASHINFER` 强切后端，对比吞吐与告警
- **第 5 讲**： `--enforce-eager` 与 `-O0` 三态对照，TPOT 差多少倍
- **第 6 讲**： 双卡 TP=2 起服务，小杯 MoE 开 EP 对照吞吐
- **第 7 讲**： Qwen3 开 MTP 投机，读 accept length 指标算实际加速比
- **第 8 讲**： LLM Compressor 压一个 FP8 模型，加载后对比精度与显存
- **第 9 讲**： CPU offload 长上下文，对比 with/without 的吞吐悬崖位置
- **第 10 讲**： `vllm bench` 规范压测，导出 metrics 配一面 Grafana

### 与其他课程的关系

| 内容                                  | 本课（vLLM 十讲）             | KV Cache 八讲                 |
| ------------------------------------- | ----------------------------- | ----------------------------- |
| KV cache 容量账本（每 token 多少）     | —                             | ✅ 全程推导与多模型计算        |
| KVCacheManager / BlockPool 对象图     | 演进叙事为主（第 3 讲）        | ✅ 对象图细节（第 4-5 讲）      |
| 引擎架构、调度、并行、编译            | ✅ 核心内容                   | —                              |
| 投机解码、量化、offload、PD 分离      | ✅ 核心内容                   | offload 的 KV 视角在第 5 讲    |
| 生产部署、压测、集群生态              | ✅ 核心内容（第 10 讲）        | 选型判断的容量输入（第 8 讲）   |
| 先修顺序                              | 先修 nano-vllm 8 课          | 可与本课穿插或后学            |

另有两类材料不参与边界切分：[nano-vllm 实战课程](https://github.`com/ForceInjection/nano-vllm/blob/main/docs/llm-inference-visual/README.md`)是本课先修（第 1 层「nano 回顾」贯穿全部十讲）；[AI-fundamentals 推理系统目录](../../09_inference_system/README.md)是深读材料库，若干主题的「演进脉络」一节会引用其源码精读文（旧文标注版本锚点，行号以本课基线重新核对为准）。

## 三、怎么学习

按 1 → 10 顺序学；第 1 讲是后面所有主题的地图。每讲 120 分钟：

| 阶段       | 时间   | 内容                                                      |
| ---------- | ------ | --------------------------------------------------------- |
| nano 回顾  | 10 min | 先修课对应内容快速回放，明确「已知什么」                   |
| 演进与现状 | 50 min | V1/V2 变化点 + 当前 HEAD 源码走读（函数级阅读清单课前预读） |
| 上手实验   | 50 min | 部署、配置、观测、改参数验证行为                          |
| 答疑       | 10 min |                                                           |

每讲闭环：读本章 README → 预读函数清单 → 上机跑实验 → 做课后作业 → 用自己的模型复现一遍。

每讲目录里的材料：

| 文件                   | 用途                                   |
| ---------------------- | -------------------------------------- |
| `README.md`            | 本章导读、学习目标、材料与前后衔接     |
| `syllabus.md`          | 120 分钟安排、函数级阅读清单、重点难点 |
| `ppt-outline.md`       | 页面提纲（按需生成 PPT）                 |
| `hands-on-exercise.md` | 课堂实验                               |
| `homework.md`          | 课后作业                               |
| `code/`                | 实验脚本                               |

每讲成稿的格式约定(参照 [ai-infra-course](https://github.com/ForceInjection/ai-infra-course) 模块格式)：开头给一个核心问题与知识链；正文含「与相邻课程的边界」对照表；实验脚本登记进 `code/README.md`；性能结论带口径。

撰写状态：**十讲全部成稿**（每讲 README 正文 + syllabus 函数级阅读清单 + ppt-outline 可生成粒度 + 实验/作业；第 1/2/3/5/10 讲含实验脚本）；全部 file:line 按基线 `2f5d8a2b1` 核对，换版本按「以代码为准」重核；基线换钉时第 5 讲三态实验已在新基线上运行时复现（2026-11-25，h34 + Qwen3-0.6B）。

## 四、课程边界

- 本课不做 KV 容量账本教学。算一个 token 多少 KV、给 Agent 长会话做容量规划，是 [KV Cache 八讲](../kv_cache_course/README.md)的事；第 3 讲向它单向导流（对方的前置是 ai-infra-course）。
- 不做论文串讲。论文（SOSP'23、Sarathi-Serve 等）用来补机制背景，课堂时间花在源码与实操。
- 以本地代码为准。nano-vllm 与 vLLM 各钉一个 commit(见[实验环境](lab-environment.md))；课上 `file:line` 按这些版本讲，换版本要重新核对。
- 性能数字一律带口径（模型变体、精度、硬件、统计范围）；厂商口径与独立测量分开标注。

## 附：资料与基线

### 共享文件

- [课程总纲](syllabus.md)
- [共用实验环境](lab-environment.md)
- [术语表](glossary.md)
- [可复用配图](visuals.md)

### 源码基线

| 材料      | 版本 / 路径                                                                  | 用途                        |
| --------- | ---------------------------------------------------------------------------- | --------------------------- |
| nano-vllm | 本地 `~/Project/study/nano-vllm`，HEAD `b9dd55a`                              | 每讲第 1 层的回顾对照       |
| vLLM      | 本地 `~/Project/ai-infra/vLLM`，HEAD `2f5d8a2b1`(v0.34.1rc0-84,2026-11-25) | 每讲第 3 层的现状走读与实验 |
| 官方文档  | docs.vllm.ai stable（2026-11-06 版），`docs/design/` 32 篇                      | 现状的权威校对              |

### 开始学习

- [第 1 讲：引擎总架构](`01-engine-architecture/README.md`)
- [课程总纲](syllabus.md)
