# 第 1 讲：KV Cache 为什么存在 — 教学大纲

## 基本信息

| 项目     | 内容                                                        |
| -------- | ----------------------------------------------------------- |
| 课时     | 120 分钟                                                    |
| 授课对象 | 有 Python / PyTorch 基础的工程师；不要求 Transformer 注意力细节 |
| 先修     | 矩阵乘法、PyTorch 基础；不要求 Transformer 注意力细节        |
| 本讲定位 | 建立 token 级 KV 成本模型，为分页、复用和容量规划打底       |

## 教学目标

- **知识目标**：理解自回归生成中 K/V 的复用条件，掌握每 token 与总 KV 的两行公式，以及注意力变体的影响。
- **能力目标**：给定模型 config、精度、序列长度和并发，算出 KV 显存，判断权重和 KV 谁先成为瓶颈。
- **素养目标**：报容量数字时同时说明模型、精度、序列长度和 batch。

本讲使用两行公式：

`per_token_bytes = 2 × b_kv × L × H_kv × D`

`total_bytes = per_token_bytes × T × batch`

| 符号    | 含义           | 说明                              |
| ------- | -------------- | --------------------------------- |
| `L`     | 层数           | 每层各有一份 K/V                  |
| `H_kv`  | KV head 数     | 不是 Q head 数；GQA/MQA 改这一项  |
| `D`     | head 维度      | 不是 hidden size / H_q            |
| `b_kv`  | 每元素字节     | FP16/BF16=2，FP8/INT8=1，INT4=0.5 |
| `T`     | 每请求序列长度 | 包含 prompt 和已生成 token        |
| `batch` | 并发请求数     | 统一写 `batch`，不用单字母 `B`    |

`T` 是本讲的“当前总序列长度”；讲 prefill 时如果只想说 prompt 长度，写 `T_prompt`，不要和这个 `T` 混用。

模型阵容：Qwen3-0.6B（`L=28`、`H_q=16`、`H_kv=8`、`D=128`）+ Qwen2.5-7B（`L=28`、`H_q=28`、`H_kv=4`、`D=128`）+ Qwen2.5-72B（`L=80`、`H_q=64`、`H_kv=8`、`D=128`）。Qwen3 没有 7B / 72B；Qwen3-0.6B 的 `head_dim` 是 128，不是 64。

## 时间安排

| 时间   | 环节       | 内容                                                                                                                                                                            |
| ------ | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 30 min | 概念       | 自回归循环；Q/K/V 的作用；因果掩码下历史 K/V 为什么不变；prefill 与 decode 的直觉差异；用 3-token 玩具例子手算一行注意力                                                         |
| 45 min | 公式与演示 | KV 两行公式与符号表；对照 Qwen3-0.6B 与 Qwen2.5 的 config 逐项代入；精度表；MHA / GQA / MQA / MLA；TTFT / TPOT / 吞吐；现场运行 `code/kv_estimator.py` 演示 batch、序列长度、精度的变化 |
| 35 min | 实验       | 5 min 安装与自检 + 30 min 填表：用 `code/kv_estimator.py` 算 Qwen3-0.6B / Qwen2.5-7B / Qwen2.5-72B 的每 token KV 与总量；72B 单独做 GPU 预算判断                                   |
| 10 min | 答疑       | 回答公式参数、GQA/MLA、TTFT/TPOT 口径问题；预告第 2 讲                                                                                                                          |

## 重点与难点

| 类型 | 内容                             | 处理方式                                                                                                                    |
| ---- | -------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| 重点 | KV 显存公式                      | 逐参数拆解，代入真实 config 手算；强调 `H_kv` 不是 Q head 数                                                                |
| 重点 | prefill 与 decode 的差异         | 对比矩阵形状和 arithmetic intensity；decode 说“典型小 batch、长上下文下带宽受限”，大 batch 可能 compute-bound               |
| 难点 | MLA 为什么不是简单的 `H_kv` 变小 | 先讲 latent 压缩 + 共享 RoPE key，公式 `b_kv × L × (kv_lora_rank + qk_rope_head_dim)`，不乘 2；第 6 讲落到 GLM 的 11 层 MLA |
| 难点 | 容量数字的口径                   | 每个数字标注模型变体、精度、序列长度、batch；权重与 KV 分开记账                                                             |
| 难点 | 72B 的预算判断                   | 单卡 40 GiB 只算每 token KV，说明权重先爆；并发要用 8×80 GiB 或明确 TP                                                      |

## 本讲产出

一页 KV 成本模型，至少包含：

- 模型与 config 来源；
- 精度与每元素字节；
- 每 token KV 字节；
- 给定序列长度和并发下的总 KV；
- 权重（按 FP16 参数估算）；
- 与 GPU 预算的对比，以及“KV 与权重谁更大 / 是否超预算”。

## 参考材料

- [第 1 讲实验脚本](code/kv_estimator.py)
- [AI-fundamentals KV Cache 技术体系](../../../09_inference_system/kv_cache/README.md)
- [KV Cache 基本概念](../../../09_inference_system/kv_cache/01_concepts/basic/)
- `~/Project/study/ai-infra-course/06-kvcache-optimization/code/calculate_qwen3_memory.py`
- `~/Project/study/ai-infra-course/06-kvcache-optimization/visuals/kv-cache-formula.html`

参考计算器的 Qwen3-0.6B 预设默认把 `head_dim` 当成 64，会算出 56 KiB；本讲以 `head_dim=128`、112 KiB 为准，对答案时先核对 `num_key_value_heads`、`head_dim` 和精度。
