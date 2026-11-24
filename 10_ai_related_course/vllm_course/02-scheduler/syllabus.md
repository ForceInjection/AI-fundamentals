# 第 2 讲教学大纲：调度器

## 基本信息

| 项目     | 内容                                                                |
| -------- | ------------------------------------------------------------------- |
| 课时     | 120 分钟                                                            |
| 基线     | nano-vllm `b9dd55a`；vLLM `1acec5c009`(v0.34.0rc0-1153)            |
| 先修     | 第 1 讲（EngineCore.step 的调用点）;nano 实战课第 3 课              |
| 授课形式 | 演进对照 + 源码走读 + 上手实验                                      |

## 教学目标

- **知识目标**：说清「无阶段划分」的调度模型（请求 = 计数器，无 prefill/decode 分支）；双预算的语义与扣减点；抢占只剩 recompute 且插队队首。
- **能力目标**：用混合负载实测 chunked prefill 的效果；按需调 `--long-prefill-token-threshold` / `--max-num-batched-tokens`；从日志辨认抢占事件与策略选择。
- **素养目标**：识别 V0 时代教程在新版本上的六类失效说法（本讲给出对照表）。

## 时间安排

| 部分              | 时长    | 内容                                                                   |
| ----------------- | ------- | ---------------------------------------------------------------------- |
| nano 回顾         | 10 min  | 互斥调度与「首 seq 才能分块」的 `break`（你亲手写过）                   |
| 演进              | 15 min  | V0 两阶段 + Policy 类 + preemption_mode → V1 无阶段 + recompute + async 默认开 |
| 现状走读：六环节   | 45 min  | 见下方函数级阅读清单；课堂重点讲环节 1-4,5/6 课前预读                  |
| 实验              | 40 min  | 混合负载 TTFT 对照（见 hands-on-exercise.md）                           |
| 答疑              | 10 min  |                                                                        |

## 函数级阅读清单（课前预读）

| 文件                                      | 函数 / 类                        | 行号          | 看什么                                                     |
| ----------------------------------------- | -------------------------------- | ------------- | ---------------------------------------------------------- |
| `vllm/v1/core/sched/scheduler.py`         | `schedule()`                     | 702-1816      | 四段结构；「无阶段划分」注释(704-713)；单函数 1114 行      |
| `vllm/v1/core/sched/scheduler.py`         | 预算初始化                       | 728-734       | token_budget / input_budget 双局部变量                     |
| `vllm/v1/core/sched/scheduler.py`         | 预算扣减                         | 991-992 / 1621-1622 | running 与 waiting 两阶段的扣减点                    |
| `vllm/v1/core/sched/scheduler.py`         | running 阶段切块                 | 824-842       | `min(剩余, token_budget, input_budget - draft_slots)`      |
| `vllm/v1/core/sched/scheduler.py`         | threshold 生效条件               | 761-776       | 「nobody to starve」：多请求才切块；adaptive 变体           |
| `vllm/v1/core/sched/scheduler.py`         | 抢占决策                         | 909-969       | victim 选择（FCFS 队尾 / PRIORITY）；连环抢占               |
| `vllm/v1/core/sched/scheduler.py`         | `_preempt_request`               | 1831-1877     | 只有 recompute；`num_computed_tokens = 0`；prepend 队首     |
| `vllm/v1/core/sched/scheduler.py`         | 被抢占者重新准入                 | 1596-1623     | 「按新请求全量重发」                                       |
| `vllm/v1/core/sched/scheduler.py`         | `Scheduler.__init__` 队列段      | 231-251       | 四条队列与 `policy` 字段                                   |
| `vllm/v1/core/sched/scheduler.py`         | `_enqueue_waiting_request`       | 2826-2841     | 入队分流：blocked / kv_holding / async_kv / waiting         |
| `vllm/v1/core/sched/request_queue.py`     | `SchedulingPolicy` / `create_request_queue` | 13-17 / 200-207 | FCFS=deque,PRIORITY=heap              |
| `vllm/v1/core/sched/async_scheduler.py`   | `AsyncScheduler._update_after_schedule` | 18-62   | 只覆写这一个方法：乐观输出占位                              |
| `vllm/v1/engine/core.py`                  | `step_with_batch_queue`          | 759-886       | overlap 的真身：先排下一批再收上一批                        |
| `vllm/config/vllm.py`                     | async_scheduling 自动解析        | 1755-1794     | None → True 的条件；不兼容时 raise                          |
| `vllm/v1/core/sched/output.py`            | `SchedulerOutput`                | 240-349       | 混合批的字段；`preempted_req_ids`(286)                     |

预读提示：`schedule()` 有 1,114 行，不要通读——按上表行号段跳读；课堂上对答案。

## 重点与难点

- **重点**：无阶段模型（计数器替代阶段）；双预算；抢占只剩 recompute + 插队首。
- **难点**:long_prefill_token_threshold 的真实语义（单请求上限 + 多请求才生效）;async scheduling 的乐观占位（为什么敢在结果未知时继续调度）。
- **易错点**：拿 V0 教程的 Policy 类 / preemption-mode / 两队列说法对号入座（见 README 对照表）；以为 `max_num_batched_tokens` 越大越好（它同时是输入预算与公平性旋钮）。

## 参考材料

- 论文：Sarathi-Serve(arXiv:2403.02310,OSDI'24)——chunked prefill 与 stall-free 调度的源头
- 官方：`docs/usage/v1_guide.md`（调度相关节）;config 注释（`vllm/config/scheduler.py:160-166` chunked prefill 默认开）
- 概念经典：Anyscale「Continuous Batching: LLM Inference Serving 的关键」（2023,V0 之前时代，概念仍成立）
- 对照：nano-vllm 先修课第 3 课（互斥调度与 swapped 队列）
