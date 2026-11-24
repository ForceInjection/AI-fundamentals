# 第 7 讲教学大纲：投机解码

## 基本信息

| 项目     | 内容                                                                |
| -------- | ------------------------------------------------------------------- |
| 课时     | 120 分钟                                                            |
| 基线     | nano-vllm `b9dd55a`；vLLM `1acec5c009`(v0.34.0rc0-1153)            |
| 先修     | 第 2 讲（draft_slots 与占位符）、第 5 讲(FULL_DECODE_ONLY)         |
| 授课形式 | 选型框架 + 验证循环走读 + 实测实验                                  |

## 教学目标

- **知识目标**：说出五法在职的选型维度（草稿来源/训练介入/典型场景）；讲清「一步 = target 前向 1 次（展开 1+K）+ 采样即验证 + draft K 步」的流水时序；acceptance length 的含 bonus 口径。
- **能力目标**：为给定负载选方法并写配置片段；实测 accept length（日志/Prometheus/per-request 三口径）；辨认 Medusa 已移除等过期说法。
- **素养目标**：理解「加速上限由 acceptance 决定，代价藏在草稿成本」的量化思维；两套口径（含/不含 bonus）的区分纪律。

## 时间安排

| 部分            | 时长    | 内容                                                                   |
| --------------- | ------- | ---------------------------------------------------------------------- |
| nano 回顾（反向） | 10 min  | 为什么没有；收益公式一句话；(1+a)/(1+c)                                |
| 演进            | 15 min  | Proposer → Speculator 改名迁移；Medusa 移除；方法全集（11+ 内置）      |
| 现状走读：五环节 | 45 min  | 见下方函数级阅读清单；课堂重点讲环节 3（验证循环）与环节 2（速查表）  |
| 实验            | 25 min  | ngram 开关对照 + 三口径读数                                            |
| 答疑            | 10 min  |                                                                        |

## 函数级阅读清单（课前预读）

| 文件                                              | 函数 / 类                        | 行号            | 看什么                                                 |
| ------------------------------------------------- | -------------------------------- | --------------- | ------------------------------------------------------ |
| `vllm/config/speculative.py`                      | `SpeculativeMethod`              | 113-123         | method 合法值全集（注意没有 medusa）                    |
| `vllm/config/speculative.py`                      | `__post_init__`                  | 1465-1551       | 方法推断：custom/ngram/无 model→draft_model/自动检测    |
| `vllm/v1/worker/gpu/spec_decode/speculator_config.py` | `_CONFIG_BY_METHOD`          | 355-371         | 方法→配置类注册表；checkpoint 检测(492)               |
| `vllm/v1/worker/gpu/spec_decode/speculator.py`    | `BaseSpeculator`                 | 74;propose:100-128 | 统一接口                                            |
| `vllm/v1/worker/gpu/spec_decode/ngram/speculator.py` | `NgramGPUSpeculator.propose`  | 151;213         | GPU Triton 全历史扫描                                  |
| `vllm/v1/worker/gpu/spec_decode/suffix/speculator.py` | `SuffixSpeculator.propose`   | 501;683         | 变长后缀 + 跨请求 corpus                               |
| `vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py` | 多步循环与图融合     | 24;227 / 276-321 | draft K 步；融合版 FULL 图                            |
| `vllm/v1/worker/gpu/spec_decode/mtp/speculator.py`| `MTPSpeculator`                  | 12-40           | draft=target；继承量化配置                              |
| `vllm/v1/worker/gpu/model_runner.py`              | `sample_tokens` propose 段       | 2426-2473       | draft 写回 req_states                                  |
| `vllm/v1/worker/gpu/model_runner.py`              | rejection 接线                   | 1788-1800 / 563 | 采样即验证的分叉点                                    |
| `vllm/v1/worker/gpu/spec_decode/rejection_sampler.py` | `RejectionSampler`           | 152；核心 kernel:1128 | greedy 退化 / probabilistic 概率比                |
| `vllm/v1/core/sched/scheduler.py`                 | draft_slots / lookahead 预留     | 730 / 384-391   | 第 2 讲种子的发芽处                                    |
| `vllm/v1/core/sched/scheduler.py`                 | `can_skip_drafts` / 动态 SD      | 349 / 1354-1403 | 高并发「仅 target」步；按 batch 调 K                   |
| `vllm/v1/spec_decode/metrics.py`                  | Logging / Prom                   | 64 / 195        | 三口径；均值含 bonus(115-116)                         |
| `vllm/benchmarks/serve.py`                        | `fetch_spec_decode_metrics`      | 1169-1181       | bench 自动抓 Prom 指标                                 |

## 重点与难点

- **重点**：验证循环的流水时序（易讲错的核心）；五法选型维度；acceptance 两套口径。
- **难点**:greedy 与 probabilistic 验证的差别（默认 greedy 时 draft_logits 为 None）;async 占位符与拒绝回滚；动态 SD 的查表时机。
- **易错点**：引用 `vllm/v1/spec_decode/` 找 Proposer 类（已迁移改名）；配置里写 `"method":"draft"`（合法名 `draft_model`）；把 Medusa 当在职方法；acceptance 数字混用两种口径。

## 参考材料

- 官方：`docs/features/speculative_decoding/`（README 选型表 + 各方法文件 + acceptance_metrics.md 对账表，与代码一致）
- 论文/讲座：GPU MODE Lecture 22（V1 时代机制；命名与路径按本讲校准）
- 演进素材：仓库 `speculative_decoding_landscape.md`（含已移除的 Medusa；选型决策框架仍有效）
- 深读：仓库 `model_optimization/illustrated-speculative-decoding.md`、`mtp-multi-token-prediction.md`
