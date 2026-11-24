# 第 6 讲教学大纲：并行：TP 到 EP

## 基本信息

| 项目     | 内容                                                                |
| -------- | ------------------------------------------------------------------- |
| 课时     | 120 分钟                                                            |
| 基线     | nano-vllm `b9dd55a`；vLLM `1acec5c009`(v0.34.0rc0-1153)            |
| 先修     | 第 1 讲（Executor 与进程拓扑）、第 2 讲（DP 对齐原语）               |
| 授课形式 | 轴全景 + 通信链走读 + 配置实验                                      |

## 教学目标

- **知识目标**：写出 `world_size = TP×PP×PCP`（不含 DP/EP）与 EP 组 = DP×PCP×TP；说出 EP 的三个反直觉点（专家整只放卡、纯 TP-EP 无 all2all、PP 不参与）；认识 all2all 后端族。
- **能力目标**：给定卡数与模型形态(MoE/dense)，选 TP/DP/EP 组合并写出部署命令；读懂 `init` 日志的九种 group 分配；识别 V0 时代的并行失效说法。
- **素养目标**：部署文档(docs/serving)与代码的互相核对习惯（文档示例 → config 字段 → 实现落点）。

## 时间安排

| 部分            | 时长    | 内容                                                                   |
| --------------- | ------- | ---------------------------------------------------------------------- |
| nano 回顾       | 10 min  | SHM 广播方法调用；TP 之外什么都没有                                    |
| 轴全景          | 15 min  | 六轴；三个公式；DP vs EP 语义（代码注释）                                |
| 现状走读：六环节 | 45 min  | 见下方函数级阅读清单；课堂重点讲环节 2（九组）、4（EP 通信）、5（部署）   |
| 实验            | 25 min  | 配置组合校验 + 日志走读（见 hands-on-exercise.md）                      |
| 答疑            | 10 min  |                                                                        |

## 函数级阅读清单（课前预读）

| 文件                                            | 函数 / 类                        | 行号            | 看什么                                                 |
| ----------------------------------------------- | -------------------------------- | --------------- | ------------------------------------------------------ |
| `vllm/config/parallel.py`                       | `ParallelConfig`                 | 248 起          | 六轴字段；EP/EPLB/all2all 相关字段(301-332)           |
| `vllm/config/parallel.py`                       | `world_size` 计算                | 1182-1186 / 823 | TP×PP×PCP;world_size_across_dp                        |
| `vllm/config/parallel.py`                       | `_validate_parallel_config`      | 681-770         | all2all auto 解析；EPLB 前置；DCP 整除校验              |
| `vllm/config/parallel.py`                       | backend 缺省                     | 1272-1321       | 默认 mp 尽量单机；报错条件                              |
| `vllm/distributed/parallel_state.py`           | `initialize_model_parallel`      | 2131-2460       | 九种 group 的 rank 布局；EP 组(2400-2437)             |
| `vllm/distributed/parallel_state.py`           | DP/EP 语义注释                   | 2200-2207       | DP=attention 数据并行；EP=专家展开组                    |
| `vllm/model_executor/layers/linear.py`         | `RowParallelLinear.forward`      | 1868-1898       | 行切 + all_reduce(1888)                               |
| `vllm/distributed/device_communicators/cuda_communicator.py` | `all_reduce`        | 389-455         | allreduce 后端优先级链；仅 TP 组启用 custom(59-65)     |
| `vllm/model_executor/layers/fused_moe/config.py`| `FusedMoEParallelConfig`        | 1067-1296       | use_ep 判定(1260);EP 时 tp=1(1292);all2all 条件(1088) |
| `vllm/model_executor/layers/fused_moe/expert_map_manager.py` | `determine_expert_map` | 22-152     | 专家放置；round_robin 强制场景                          |
| `vllm/model_executor/layers/fused_moe/modular_kernel.py` | `FusedMoEPrepareAndFinalize` | 241-454  | dispatch/combine 抽象                                  |
| `vllm/distributed/device_communicators/all2all.py` | 三种 manager                  | 57 / 257 / 316  | AgRs 拼装 / DeepEP HT / DeepEP LL                      |
| `docs/serving/expert_parallel_deployment.md`    | 部署示例                        | 85-96 / 166-193 | 单机 8 卡 EP;2 机 16 卡 deepep                         |
| `docs/serving/data_parallel_deployment.md`      | DP attention 动机               | 5-16            | DP replica + MoE EP 打通                               |
| `docs/serving/context_parallel_deployment.md`   | DCP 上限与案例                  | 52 附近         | dcp ≤ tp/kv_heads;R1/K2/Qwen3-235B 案例               |

## 重点与难点

- **重点**：三个公式（world_size / EP 组 / EP_SIZE）;EP 三个反直觉点；all2all 后端族与触发条件。
- **难点**:EP 组 rank 布局（DP×PCP×TP,PP 不参与）;「EP 时专家层 tp=1」的含义（整只专家）;DP 与 EP 共享同一批 rank 但维度不同。
- **易错点**：写 `world_size = TP×PP×DP×EP`；以为多机默认 Ray；以为纯 TP-EP 走 all2all；去找 `vllm/distributed/ep/` 目录（不存在）。

## 参考材料

- 官方：`docs/serving/parallelism_scaling.md`（选型指南）、`expert_parallel_deployment.md`、`data_parallel_deployment.md`、`context_parallel_deployment.md`
- 设计：`docs/design/fused_moe_modular_kernel.md`、`moe_kernel_features.md`、`dbo.md`
- 博客：「DeepSeek @ 2.2k tok/s/H200 with Wide-EP」（2025-12-17，厂商口径）
- 演进素材：仓库 `09_inference_system/vllm/module_analysis/` 的 `mla_tp_kv_redundancy.md`（MLA TP 冗余）、`deepseek_blackwell_wide_ep.md`
- 对照：nano-vllm 先修课第 8 课（TP 部分）
