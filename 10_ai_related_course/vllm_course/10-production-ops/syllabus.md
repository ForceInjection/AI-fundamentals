# 第 10 讲教学大纲：生产运维与生态

## 基本信息

| 项目     | 内容                                                                |
| -------- | ------------------------------------------------------------------- |
| 课时     | 120 分钟                                                            |
| 基线     | nano-vllm `b9dd55a`；vLLM `1acec5c009`(v0.34.0rc0-1153)            |
| 先修     | 第 1-9 讲（本讲是收官合流讲）                                       |
| 授课形式 | 运维面走读 + 指标族梳理 + 压测实验 + 生态分层                       |

## 教学目标

- **知识目标**：列出运维端点（/health /metrics /version /load，及 /ping 陷阱）；说清双出口 metrics 与指标族归类；区分 ITL 与 TPOT；复述 FT 显式协议（非自动自愈）与生态三层（引擎/部署栈/控制面）。
- **能力目标**：跑通「压测 → 读 metrics → 定位瓶颈」闭环；为生产写最小安全配置清单；辨认 /ping、限流、自动重启三类过期说法。
- **素养目标**：黑盒(bench)与白盒(metrics)测量的互补视角；引擎暴露原料、控制面做决策的分层思维。

## 时间安排

| 部分            | 时长    | 内容                                                                   |
| --------------- | ------- | ---------------------------------------------------------------------- |
| nano 回顾       | 10 min  | bench.py 单机脚本；运维面四缺（server/metrics/健康/恢复）               |
| server 运维面   | 15 min  | 端点；中间件洋葱；无限流；api-server-count/headless；取消链              |
| metrics 体系    | 15 min  | 双出口；指标族归类表；ITL vs TPOT；文档两处滞后                          |
| vllm bench      | 15 min  | 七子命令；13 种 dataset；输出字段；与 metrics 互补                       |
| 故障容忍与生态  | 20 min  | EngineCoreSentinel 显式协议；worker monitor；生态三层；arXiv 综述        |
| 实验            | 25 min  | 运维面巡检 + 基线压测（见 hands-on-exercise.md）                        |
| 答疑与收官      | 10 min  | 生产十条清单；课程总结                                                  |

## 函数级阅读清单（课前预读）

| 文件                                                  | 函数 / 类                        | 行号            | 看什么                                                 |
| ----------------------------------------------------- | -------------------------------- | --------------- | ------------------------------------------------------ |
| `vllm/entrypoints/serve/instrumentator/health.py`     | 健康端点                         | 22-33           | 引擎死 → 503                                          |
| `vllm/entrypoints/serve/instrumentator/metrics.py`    | /metrics 挂载                    | 56-81           | ASGI mount 绕 307；排除端点清单                        |
| `vllm/entrypoints/serve/middleware/register.py`       | 中间件洋葱                       | 19-75           | 认证/Request-Id/自定义；无 rate limit                  |
| `vllm/entrypoints/serve/utils/api_utils.py`           | `listen_for_disconnect`          | 53-111          | 断连监听（不能用 is_disconnected）                     |
| `vllm/v1/engine/async_llm.py`                         | 取消与 abort                     | 818-826         | CancelledError → abort；引擎死不再 abort               |
| `vllm/v1/metrics/loggers.py`                          | 指标族                           | 541/551/633/761/812/944/1024 | 调度/吞吐/缓存/延迟各族代表               |
| `vllm/v1/metrics/loggers.py`                          | `cached_by_source`               | 795-807         | device/host/disk/p2p 分层命中（第 3/9 讲回收）          |
| `vllm/v1/metrics/stats.py`                            | `SchedulerStats`                 | 224-262         | 统计出口的数据结构                                     |
| `docs/design/metrics.md`                              | 两分法/ITL vs TPOT               | 13-19 / 67-75   | server-level 与 request-level；两口径分叉              |
| `vllm/entrypoints/cli/main.py`                        | CMD_MODULES                      | 56-63           | bench 子命令注册                                       |
| `vllm/benchmarks/datasets/datasets.py`                | `--dataset-name` 全集            | 1614-1631       | 13 种；spec_bench 在 3011                              |
| `vllm/benchmarks/serve.py`                            | `BenchmarkMetrics`               | 352-393         | TTFT/TPOT/ITL/E2EL 百分位                              |
| `vllm/v1/fault_tolerance/engine_core_sentinel.py`     | `EngineCoreSentinel`             | 55;on_fault:142-194 | 状态机；清脏块；投票恢复                            |
| `vllm/entrypoints/serve/fault_tolerance/api_router.py`| FT 端点                          | 41-103          | apply(retry/scale_down)与 status                      |
| `vllm/v1/executor/multiproc_executor.py`              | `start_worker_monitor`           | 389-424         | worker 死 → 整 executor shutdown（第 1 讲行号已漂移）  |

## 重点与难点

- **重点**：运维端点四件；指标族归类；bench 与 metrics 互补；FT 显式协议。
- **难点**:ITL vs TPOT 口径分叉（投机解码下）;FT 两阶段恢复（orchestrator rendezvous + 投票）。
- **易错点**:/ping 当通用端点；以为有内置限流；以为 worker 崩溃自动重启；metrics.md 旧指标名（`_total`/`swapped`）。

## 参考材料

- 官方：`docs/design/metrics.md`（ITL/TPOT 段新增；两处指标名滞后）、`docs/usage/metrics.md`（权威清单）、`docs/benchmarking/cli.md`、`docs/usage/usage_stats.md`
- 生态：production-stack / AIBrix / llm-d 各自 GitHub 与 docs.vllm.ai 集成页；Red Hat「KV-cache aware routing」(2025-10)、「Demystifying llm-d and vLLM」
- 收官阅读：arXiv:2609.23130（单作者综述，非官方立场；引擎→控制面）
- 外部课：DeepLearning.AI「Fast & Efficient LLM Inference with vLLM」（9 节视频+3 lab,Compress/Serve/Benchmark 三阶段）
