# 第 1 讲：KV Cache 为什么存在

> 120 分钟 | 课程第 1 讲 | 前置：矩阵乘法、PyTorch 基础；不要求 Transformer 注意力细节

本讲从一次自回归生成出发，回答两个问题：decode 为什么必须缓存 K/V，一个 token 的 KV 账怎么算。课程后半段的分页、复用、容量规划，都建立在这本 token 账上。

## 学习目标

- 说清 Q、K、V 在注意力里各自做什么，历史 K/V 为什么可以复用。
- 区分 prefill 与 decode 的计算特征：decode 在典型小 batch、长上下文下受带宽限制，batch 增大后可能转向 compute-bound。
- 用两行公式估算每 token 与总 KV，并解释每个符号。
- 知道 MHA / GQA / MQA 改的是 `H_kv`，MLA 改的是缓存对象。
- 用 TTFT、TPOT、吞吐三个指标描述 KV Cache 的影响。

## KV 公式与符号

`per_token_bytes = 2 × b_kv × L × H_kv × D`

`total_bytes = per_token_bytes × T × batch`

| 符号    | 含义                       | 单位  | Qwen3-0.6B 示例 |
| ------- | -------------------------- | ----- | --------------- |
| `L`     | Transformer 层数           | 层    | 28              |
| `H_kv`  | KV head 数，不是 Q head 数 | 个    | 8               |
| `D`     | 每个 head 的维度           | 元素  | 128             |
| `b_kv`  | KV 每个元素的字节数        | 字节  | FP16 为 2       |
| `T`     | 每个请求的序列长度         | token | 4096            |
| `batch` | 并发请求数                 | 个    | 1               |

公式开头的 `2` 表示 K 和 V 各存一份。缓存形状是：K、V 各一份 `[L, T, H_kv, D]`，元素总数 `2 × L × T × H_kv × D`，再乘 `b_kv` 得到字节数。GQA、MQA 只改 `H_kv`；MLA 不适用这行公式，见下。

代入 Qwen3-0.6B、FP16、T=4096、batch=1：

`2 × 2 × 28 × 8 × 128 = 114,688 B/token`，即 112 KiB/token；乘以 4096 得 448 MiB。

## 模型阵容

| 模型        | L   | H_q | H_kv | D   | FP16 每 token KV |
| ----------- | --- | --- | ---- | --- | ---------------- |
| Qwen3-0.6B  | 28  | 16  | 8    | 128 | 112 KiB          |
| Qwen2.5-7B  | 28  | 28  | 4    | 128 | 56 KiB           |
| Qwen2.5-72B | 80  | 64  | 8    | 128 | 320 KiB          |

Qwen3 系列没有 7B / 72B，本课程用 Qwen3-0.6B 搭配 Qwen2.5-7B 和 Qwen2.5-72B。Qwen3-0.6B 的 `head_dim` 是 128，不能按 `hidden_size / H_q` 去推。

## 精度与每元素字节

这张表是全课程共用的 KV 精度口径，第 6、7 讲沿用同一套数字。

| 精度            | `b_kv`（字节/元素） | scale 与打包开销                                                             | 注意事项                                                 |
| --------------- | ------------------- | ---------------------------------------------------------------------------- | -------------------------------------------------------- |
| FP16            | 2                   | 无                                                                           | 基线；多数模型的默认 KV 精度                             |
| BF16            | 2                   | 无                                                                           | 动态范围比 FP16 大，尾数更短；容量与 FP16 相同           |
| FP8 (E4M3/E5M2) | 1                   | 每 tensor 或每 128 元素一个 FP32 scale，摊薄后通常 < 0.05 B/元素             | 需要 attention kernel 支持；K/V 量化误差在长上下文更明显 |
| INT8            | 1                   | 每 token 或每 channel 一组 scale/zero point，摊薄后通常 < 0.05 B/元素        | 需要反量化；并非所有后端都支持                           |
| INT4            | 0.5                 | 两个值打包进 1 字节；每组 16/32/64/128 个元素带 scale/zero，实际开销约 +6%–25% | 实际占用常高于 0.5 B/元素；本讲不展开，第 7 讲涉及 FP4 |

表里的字节只算 KV 本身，不含 block 对齐、内存碎片和框架预留。权重的精度可以单独选择，例如 FP8 KV 配 BF16 权重，算容量时分两本账。

## 注意力变体

| 变体 | `H_kv`  | 缓存内容                    | 每 token 公式                                  |
| ---- | ------- | --------------------------- | ---------------------------------------------- |
| MHA  | `= H_q` | 每个 Q head 一份 K/V        | `2 × b_kv × L × H_q × D`                       |
| GQA  | `< H_q` | 一组 Q head 共享一份 K/V    | `2 × b_kv × L × H_kv × D`                      |
| MQA  | `= 1`   | 所有 Q head 共享一份 K/V    | `2 × b_kv × L × D`                             |
| MLA  | 不适用  | 压缩 latent + 共享 RoPE key | `b_kv × L × (kv_lora_rank + qk_rope_head_dim)` |

MLA 不乘 2：它存的不是每个 head 的 K 和 V，而是一份压缩后的 KV latent，加上一份所有 head 共用的 RoPE key。DeepSeek-V3 的 `kv_lora_rank=512`、`qk_rope_head_dim=64`：BF16 下每层每 token 约 `(512 + 64) × 2 = 1,152 B`（约 1.13 KiB），61 层合计约 68.6 KiB/token；若把 latent 和 RoPE 都按 FP8 存，则每层 576 B，61 层合计 35,136 B（约 34.3 KiB/token）。第 6 讲会用它解释 GLM-5.3-Flash 的 11 层 MLA；GLM-5.3-Flash 的 `qk_rope_head_dim=0`，只算 512，不要套 DeepSeek 的 64。

## 三个时延指标

- TTFT（Time To First Token）：请求发出到第一个输出 token 返回的时间，包含排队、prefill 和首步采样。prompt 越长、排队越久，TTFT 越大。
- TPOT（Time Per Output Token）：首 token 之后，平均每个输出 token 的间隔，常用 `(总时延 − TTFT) / (输出 token 数 − 1)`。稳态 decode 下约等于一步 decode 的耗时。
- 吞吐：单位时间系统完成的 token 数或请求数。它和单请求的 TPOT 不是一回事：加大 batch 通常提高吞吐，但会让 TPOT 变长。

## 本讲产出

一页“KV 成本模型”：写清模型、精度、序列长度、并发，给出每 token KV 和显存占用。产出表至少包含每 token KV 字节、总 KV、权重，以及“KV 与权重谁更大 / 是否超预算”。

## 目录

- [课程总览](../README.md)
- [教学大纲](syllabus.md)
- [PPT 提纲](ppt-outline.md)
- [课堂实验](hands-on-exercise.md)
- [课后作业](homework.md)
- [实验环境](lab-environment.md)

## 前后衔接

这是课程起点。第 2 讲会把“一个 token 的账”扩展成“一个请求的 block table”。
