# 可复用配图索引

原则：不重复造图。优先链接仓库已有图与外部权威图；课程自制图放各讲目录 `diagrams/`，登记到这里。

## 仓库内(09_inference_system)

| 图                                              | 来源文章                     | 可用于       |
| ----------------------------------------------- | ---------------------------- | ------------ |
| `09_inference_system/vllm/module_analysis/assets/kv-cache-comparison.svg` | KV 对比篇 | 第 3 讲      |
| `09_inference_system/vllm/module_analysis/assets/decode-path.svg`         | decode 路径 | 第 2、6 讲 |
| `09_inference_system/vllm/module_analysis/assets/c4a_animation.gif`       | —            | 第 3 讲      |
| `09_inference_system/vllm/module_analysis/assets/kimi_k3_attention_visual.html` | Kimi K3 篇 | 第 4 讲    |
| `09_inference_system/prefill_decode/` 交互可视化                            | `prefill_decode` 系列 | 第 2 讲 |
| `09_inference_system/assets/` 各 svg                                        | 按主题检索   | 按需         |

## nano-vllm 仓（先修课）

- `docs/llm-inference-visual/diagrams/`：8 课全部自制图，第 1 层「nano 回顾」直接复用。

## 外部权威图（引用时注明来源与版本口径）

| 图 / 来源                                                              | 可用于            |
| ---------------------------------------------------------------------- | ----------------- |
| Gordić「Anatomy of a High-Throughput LLM Inference System」约 17 张图（V1 口径，2025-08） | 第 1、2 讲架构总览 |
| `docs/design/arch_overview.md`、`model_runner_v2.md` 官方图             | 第 1 讲现状       |
| `docs/design/cuda_graphs.md`、`torch_compile.md` 图                     | 第 5 讲           |
| Wide-EP 博文(2025-12-17)部署图                                          | 第 6 讲           |
| KV Offloading Connector 博文(2026-01-08)架构图                          | 第 9 讲           |

## 待自制（随讲补充）

- 第 1 讲：三进程架构 + 一次请求全生命周期（nano 单进程对照）
- 第 2 讲：V0 → chunked prefill → async scheduling 调度时序对照
- 第 3 讲：PagedAttention 生死时间线（V0 2023 → 退役 2026）
- 第 9 讲：nano swap → OffloadingSpec 时代 → 当前分层后端的演进图
