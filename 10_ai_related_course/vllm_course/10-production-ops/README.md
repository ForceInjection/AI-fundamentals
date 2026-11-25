# 第 10 讲 · 生产运维与生态

> 120 分钟 | 基线：nano-vllm `b9dd55a`；vLLM `2f5d8a2b1`（v0.34.1rc0-84,2026-11-25 核对）

本讲回答一个核心问题：**一个引擎变成一组服务，K8s、网关、控制面怎么接？以及——前九讲的所有机制，在生产上各自对应哪条命令、哪个端点、哪张面板？**

知识链：`vllm serve` 暴露运维端点(/health /metrics /version /load)→ Prometheus 双出口（Logging/Prometheus，`StatLoggerManager` 总管）→ `vllm bench` 注入负载取百分位 → DP 故障走显式恢复协议（`EngineCoreSentinel` + orchestrator）→ 上层三件套（production-stack 部署栈 / AIBrix 控制面 / llm-d 分布式框架）按需选层。

## 一、nano 已学形态（回顾）

nano 的「运维」就是 `bench.py` 一个单机吞吐脚本：没有 server（先修课 API 是离线 `LLM.generate`）、没有 metrics、没有健康检查、没有故障恢复。生产版的前九讲机制——调度、KV 管理、投机、量化、offload——每一个都需要对应的「可观测面」与「故障面」。本讲把它们收拢。

## 二、server 运维面

### 端点

HEAD 的路由拆在 `vllm/entrypoints/serve/instrumentator/`（`/health` 的 `health.py:22-33`——引擎死亡返回 **503**；`/metrics` 的 `metrics.py:56-81`，ASGI mount 绕 307 重定向；`/version` 与 `/load` 的 `basic.py:53/30-50`，/load 需 `--enable-server-load-tracking`）。**`/ping` 不是通用端点**——只在 SageMaker 适配层（`sagemaker/api_router.py:117-118`）。路由总装：`launchers/api_server/routers.py:12-78`。

### 中间件（洋葱顺序，`middleware/register.py:19-75`）

CORS → 认证（`--api-key` 优先于 `VLLM_API_KEY`）→ Request-Id → `elastic_ep` Scaling → 调试日志（带生产警告）→ 用户自定义 `--middleware a.b.Class`。**没有限流中间件**——限流靠上游网关；引擎侧唯一的准入拒绝是 `vllm:admission_rejections` 计数器（`loggers.py:861`）。

### 多 API server 与取消

`--api-server-count N`（一个引擎组配 N 个前端摊开销）；`--headless` 完全不起 API server（`cli/serve.py:69-77`，引擎裸跑等控制面接管）。请求断连：`listen_for_disconnect` 监听 ASGI disconnect（`serve/utils/api_utils.py:53-111`，注释明确不能用 `request.is_disconnected`），取消后引擎侧 `abort(request_id)`（`async_llm.py:818-826`）。

## 三、metrics 体系

双出口：`LoggingStatLogger`（打日志）与 `PrometheusStatLogger`，由 `StatLoggerManager` 总管（`loggers.py:1372`）；数据流：Scheduler 每步产 `SchedulerStats`（`stats.py:224-262`）→ EngineCoreOutputs → client 侧分发。指标族归类（全部 `vllm:` 前缀）：

| 族        | 代表指标（行号 `loggers.py`）                                                                                                          |
| --------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| 调度/引擎 | `num_requests_running/waiting`（`:541/551`）、`kv_cache_usage_perc`（`:633`）、`admission_rejections`（`:861`）                        |
| 吞吐      | `prompt_tokens`（`:762`）、`generation_tokens`（`:813`）、`request_success` 按 finish reason（`:823`）                                 |
| 缓存      | `prefix_cache_queries/hits`（`:675/686`）、**`cached_by_source`**（`:795-807`，device/host/disk/p2p 分层——第 3/9 讲的来源标签）        |
| 延迟      | `time_to_first_token_seconds`（`:944`）、`inter_token_latency_seconds`（`:954`）、TPOT（`:964`）、`request_num_preemptions`（`:1024`） |
| 外挂      | `spec_decoding`（第 7 讲）、`kv_connector`（第 9 讲）、`ec_connector`、PerfMetricsProm（出口在扩）                                           |

**ITL vs TPOT**（`metrics.md:54-55、74-80` 新增段落）：ITL 每个流式输出记一次，TPOT=(e2e−TTFT)/(n−1) 每请求一次；投机解码一步输出多 token 时两者分叉。**文档两处滞后**：`vllm:prompt_tokens_total`/`generation_tokens_total` 是旧名（代码无 `_total`）；`num_requests_swapped` 已删——以 `docs/usage/metrics.md` 为准。

## 四、vllm bench

七个子命令（`vllm/entrypoints/cli/benchmark/`）：serve / throughput / latency / forward / startup / sweep / mm-processor。`--dataset-name` 全集 13 种（`datasets.py:1614-1631`，默认 random；第 7 讲的 `spec_bench` 在其中）。输出字段（`serve.py:352-393`）：throughput + TTFT/TPOT/ITL/E2EL 均值与百分位。与 metrics 的关系互补：**bench 是注入负载的外部测量（黑盒），metrics 是引擎内部自报告**——两者的 TPOT 口径可分叉（见上）。

## 五、故障容忍：显式协议，不是自动自愈

`vllm/v1/fault_tolerance/` 的 `EngineCoreSentinel`（`engine_core_sentinel.py:55`）是一个**显式恢复协议**，不是自动退避：

- 状态机：HEALTHY → DIAGNOSING → UNHEALTHY → DEAD（`:67、:173-186`）；
- busy loop 异常 → `on_fault`（`:142-194`）：先 `_clear_contaminated_blocks`（`:231-255`，清除失败步骤可能写脏的 KV 块），再按策略 recompute 抢占或 finish；
- 恢复由**外部 orchestrator** 驱动：`POST /v1/fault_tolerance/apply`（retry 或 `scale_down`）+ `GET /status`（`api_router.py:41-103`）；retry 走幸存引擎 gloo 投票（`_recover_and_vote`，`:469-511`，MIN 归约一败俱败）；
- **Worker 崩溃没有自动恢复**：`start_worker_monitor`（`multiproc_executor.py:389-424`，第 1 讲提过的 monitor）盯到任一 worker 死亡即 `shutdown()` 整个 executor——留给上层（K8s 重拉）。API 层 `/health` 变 503（`health.py:32-33`）。

这个接口形状就是「引擎暴露原料、控制面做决策」——与下面三层生态衔接。

## 六、生态分层：引擎 → 部署栈 → 控制面

| 层       | 项目                                                                                                                    | 定位                                                                                                                                                                                                  |
| -------- | ----------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 引擎层   | vLLM（本课 1-9 讲）                                                                                                     | 「怎么执行」；暴露 metrics/KV 事件/FT 接口                                                                                                                                                            |
| 部署栈层 | [production-stack](https://github.com/vllm-project/production-stack)（vllm-project 官方）                               | Helm 一条装齐「多 engine + 智能路由器」；轻量参考实现；社区有迁往 llm-d 的迹象                                                                                                                        |
| 控制面层 | [AIBrix](https://github.com/vllm-project/aibrix)（字节）/ [llm-d](https://github.com/llm-d/llm-d)(Red Hat,CNCF Sandbox) | 集群级调度、弹性、路由、模型生命周期；AIBrix 管到 replica 级，llm-d 走 well-lit paths + **prefix-cache-aware routing**（读 KV 事件把请求路由到已持前缀的副本——第 3/9 讲的 KV 事件在控制面的消费方式） |

收官阅读：arXiv:2609.23130（2026-09；**单作者系统综述，非 vLLM 官方立场**）：推理正从「引擎内优化」变成「分布式控制问题」，稀缺资源从 FLOPs 转向「被管理的状态、放置、网络搬运、可靠性」——把本课第 2/3/6/9 讲的机制放回这张大图里看。

## 七、V0/旧教程 vs HEAD

| 旧说法                                   | HEAD 事实                                                                |
| ---------------------------------------- | ------------------------------------------------------------------------ |
| 「/ping 探活」                           | 通用端点是 /health;/ping 仅 SageMaker 适配                               |
| 「/metrics、/health 在 `api_server`.py」 | 迁到 `entrypoints/serve/instrumentator/`                                 |
| 「vLLM 自带限流」                        | 无限流中间件；限流靠网关                                                 |
| 「worker 崩溃自动重启」                  | worker 死 → 整个 executor shutdown，上层(K8s)重拉；DP 故障走显式 FT 协议 |
| 「vllm:`prompt_tokens_total`」           | 现名 `vllm:prompt_tokens`（metrics.md 滞后）                             |
| 「`num_requests_swapped`」               | V1 已删(running/waiting)                                                 |
| 「AIBrix 有 detector 组件」              | 公开文档无此组件(autoscaler/router/gateway)                              |

## 八、实验与收官

见 [hands-on-exercise.md](hands-on-exercise.md)：实验 1 运维面巡检（端点 + 指标，纯 CPU 可做）；实验 2 基线压测（`vllm bench serve`）；实验 3（可选）故障演练（杀 worker 观察）。课后作业见 [homework.md](homework.md)。

## 九、生产十条（收官清单）

十条真会用的命令/端点/配置（每条都能在前九讲找到机制解释）：

1. `curl localhost:8000/health` — 探活，503 = 引擎死(K8s liveness/readiness)
2. `curl localhost:8000/metrics | grep vllm:` — 全量指标；盯 running/waiting、`kv_cache_usage_perc`、TTFT 直方图
3. `curl localhost:8000/version` 与 `/load` — 版本与在途负载，网关分流依据
4. `vllm bench serve --dataset-name random --num-prompts 1000` — 上线前基线，取 TTFT/TPOT/E2EL 百分位
5. `vllm bench serve --dataset-name spec_bench` — 投机收益验证（第 7 讲闭环）
6. `VLLM_NO_USAGE_STATS=1` — 生产关匿名上报
7. `VLLM_LOGGING_LEVEL=DEBUG` / `VLLM_LOGGING_CONFIG_PATH` — 排障日志
8. `--api-server-count N` / `--headless` — 前端摊销 / 交控制面裸跑
9. `--api-key <token>` + `--middleware mypkg.MyMiddleware` — 最小认证 + 自定义限流审计
10. `GET /v1/fault_tolerance/status` + `POST /v1/fault_tolerance/apply` — DP 故障降级接口（orchestrator 标准动作）

## 十、课程收官与边界

前九讲的每一个机制——调度、KV、后端、图、并行、投机、量化、offload——在本讲都有一个「生产对应物」：指标、命令、面板或协议。课程边界：生产栈的 K8s 细节(HPA/PDB)属部署工程； llm-d/AIBrix 内部属各自项目文档；精度评估方法论见 `kv_cache/08_methodology/02_inference-benchmark-methodology.md`。

课程结束。回到第 1 讲的核心问题——「一次请求从 API 进来，经过哪些进程、哪些类」——现在你还能回答它在新问题下的所有变体：批怎么拼（第 2 讲）、KV 放哪（第 3 讲）、kernel 谁来跑（第 4 讲）、为什么快（第 5 讲）、怎么扩（第 6 讲）、怎么多 token（第 7 讲）、怎么省显存（第 8 讲）、装不下怎么办（第 9 讲）、生产上怎么盯（本讲）。
