# 第 4 讲教学大纲：注意力后端

## 基本信息

| 项目     | 内容                                                      |
| -------- | --------------------------------------------------------- |
| 课时     | 120 分钟                                                  |
| 基线     | nano-vllm `b9dd55a`；vLLM `2f5d8a2b1`(v0.34.1rc0-84)   |
| 先修     | 第 1 讲（Worker/Model Runner 位置）、第 3 讲(`block_table`) |
| 授课形式 | 决策树走读 + 支持矩阵 + 上手实验                          |

## 教学目标

- **知识目标**：说出五层决策树（显式 → 平台 → 优先级列表 → 统一校验 → 兜底）；知道 MLA 与标准后端互斥、GDN 走独立注册表；知道 `VLLM_ATTENTION_BACKEND` 已删除。
- **能力目标**：给定模型架构与算力代际，手推优先级列表并说出首选后端；用 `--attention-backend` 强切并读懂「拒绝理由列表」报错。
- **素养目标**：理解「控制流变数据」的重构模式（选择 = 列表 + 谓词）；文档自动生成（`attention_backends.md`）与手写文档的信任度差异。

## 时间安排

| 部分           | 时长   | 内容                                                                 |
| -------------- | ------ | -------------------------------------------------------------------- |
| nano 回顾      | 10 min | if-else 两个分支 + 固定 FA；三个生产约束逼出矩阵                     |
| 演进           | 15 min | if-else 短路链 → 优先级列表 + `validate_configuration`（控制流变数据） |
| 现状走读：五层 | 40 min | 见下方函数级阅读清单；课堂重点讲决策树与两个走查例                   |
| 支持矩阵       | 15 min | 后端×特性矩阵逐格给依据；两个结构性结论（MLA 互斥/GDN 独立树）       |
| 实验           | 30 min | 强切对照 + 决策树复算                                                |
| 答疑           | 10 min |                                                                      |

## 函数级阅读清单（课前预读）

| 文件                                                     | 函数 / 类                   | 行号                           | 看什么                                                        |
| -------------------------------------------------------- | --------------------------- | ------------------------------ | ------------------------------------------------------------- |
| `vllm/v1/attention/selector.py`                          | `get_attn_backend`          | 111                            | 选择入口；`AttentionSelectorConfig` 特征位(25-44)             |
| `vllm/v1/attention/backends/registry.py`                 | `AttentionBackendEnum`      | 34, 44-179                     | 注册表本体：成员名即后端名，值即 qualname                     |
| `vllm/v1/attention/backends/registry.py`                 | `register_backend`          | 361-433                        | 装饰器/显式注册；动态建 enum 成员                             |
| `vllm/platforms/cuda.py`                                 | `get_attn_backend_cls`      | 462                            | 平台入口；显式指定走校验(472-488)                             |
| `vllm/platforms/cuda.py`                                 | `_get_backend_priorities`   | 82-179                         | arch×算力优先级列表（MLA 分支 94-153/非 MLA 154-179）         |
| `vllm/platforms/cuda.py`                                 | 校验循环与兜底              | 495-527 / 500-517              | 取第一个通过者；全拒报错附理由列表                            |
| `vllm/v1/attention/backend.py`                           | `validate_configuration`    | 296-389                        | 十余项特性过滤；MLA 匹配(332-336)/稀疏(339-343)/滑窗(350-351) |
| `vllm/config/attention.py`                               | backend 解析                | 273-285                        | "auto"→None；非法名 → enum 报错                               |
| `vllm/v1/attention/backends/flash_attn.py`               | `FlashAttentionBackend`     | 396;supports:483-484 / 551-552 | cc≥8.0；滑窗支持                                              |
| `vllm/v1/attention/backends/flashinfer.py`               | `FlashInferBackend`         | 666;supports:761-762 / 835-842 | SM80-12.1;SM100 GQA 优先                                      |
| `vllm/v1/attention/backends/mla/flashattn_mla_sparse.py` | `FlashAttnMLASparseBackend` | 41;cc:86-87                    | 稀疏 MLA 的 SM90 专属                                         |
| `vllm/v1/worker/gpu/attn_utils.py`                       | `build_attn_metadata`       | 464;Common:540-563             | CommonAttentionMetadata 构建（执行期）                        |
| `vllm/model_executor/layers/attention/attention.py`      | `forward`                   | 537-655                        | 经 `unified_attention_with_output` 算子取 context→impl          |
| `vllm/v1/attention/backends/mla/prefill/selector.py`     | `get_mla_prefill_backend`   | 46                             | MLA 预填独立选择器                                            |

预读提示：本讲行号密集但模式单一——「列表 + 谓词」；先把第 1 讲 README 的五层记住，行号只是证据。

## 重点与难点

- **重点**：五层决策树；MLA/标准互斥（`is_mla()` 严格相等）；GDN 独立注册表；`VLLM_ATTENTION_BACKEND` 已删除。
- **难点**：优先级列表为什么按代际分化（SM100 FlashInfer 优先、SM90 `FLASH_ATTN` 优先）；稀疏 MLA 的 head 数分叉（bf16 ≤16 头 FA4-sparse 优先）；composite 后端的工厂生成（搜不到 class 是正常的）。
- **易错点**：写 `VLLM_ATTENTION_BACKEND`（已删）；说「FA3 后端」（是 `FLASH_ATTN` 的版本选项）；在 MLA 模型上强切标准后端（互斥必拒）。

## 参考材料

- 官方：`docs/design/attention_backends.md`（由 `validate_configuration` 自动生成，与 HEAD 强绑定，本课程发现的唯一零滞后文档）
- 深读（演进素材）：仓库 `09_inference_system/vllm/module_analysis/` 下的 `flashattention_deep_dive.md`（FA1→FA3 原理）、`attention_mha_mla_nsa.md`、`deepseek_attention_evolution_mla_to_csa_hca.md`
- 对照：nano-vllm 先修课第 7 课（if-else 分支）
