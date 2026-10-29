# 第 1 讲：课后作业

## 作业 1：KV 估算器（MHA / GQA）

自己写一个不依赖第三方库的 Python 脚本，不要直接 import 课程里的 `code/kv_estimator.py`，可以用它来对答案。脚本要支持：

- MHA 与 GQA；
- FP16 与 FP8；
- 两个预设：Qwen3-0.6B（`L=28`、`H_q=16`、`H_kv=8`、`D=128`、`params=0.6B`）和 Qwen2.5-7B（`L=28`、`H_q=28`、`H_kv=4`、`D=128`、`params=7.6B`）。

输出每 token KV 与 `T × batch` 的总量表，并写一段“权重和 KV 谁先爆”的判断。所有容量数字注明模型、精度、`T` 和 `batch`。

公式：

`per_token_bytes = 2 × b_kv × L × H_kv × D`

`total_bytes = per_token_bytes × T × batch`

| 符号    | 含义                       |
| ------- | -------------------------- |
| `L`     | 层数                       |
| `H_kv`  | KV head 数，不是 Q head 数 |
| `D`     | 每个 head 的维度           |
| `b_kv`  | 每元素字节                 |
| `T`     | 每请求序列长度             |
| `batch` | 并发请求数                 |

核对点：Qwen3-0.6B 的 FP16 每 token KV 应为 112 KiB，Qwen2.5-7B 应为 56 KiB。如果 Qwen3-0.6B 算出 56 KiB，多半是把 `head_dim` 当成了 `hidden_size / H_q`；它的 `head_dim` 实际是 128。

## 不在本讲范围

MLA 的 latent + RoPE 口径放在第 6 讲；INT4/FP4 的 scale 与打包开销在本讲只给 0.5 B/元素的下界，精确口径见第 7 讲；GLM-5.3-Flash 的 KDA state 也在第 6 讲。本讲不布置这些题，也不计分。

## 巩固题

1. decode 在什么条件下受带宽限制？batch 增大、上下文较短时，它什么时候可能转向 compute-bound？用“算术强度 = FLOPs / 读取字节”解释，并说明为什么序列变长通常会把 decode 推回 memory-bound。
2. 同一模型把 KV 精度从 BF16 降到 FP8，哪些量会变，哪些量不变？分别列出每 token KV、总 KV、权重，以及在固定 KV 显存下能多放多少 token；block/page 留到第 2 讲。
3. TTFT 和 TPOT 分别受什么影响？各举一个长 prompt 短输出、短 prompt 长输出的例子。
4. 读 `09_inference_system/kv_cache/01_concepts/basic/kv_cache_basics.md`，列出文中一个隐含假设，并说明它在长上下文下是否仍成立。示例：它假设 attention 是全注意力，长上下文下是否仍成立？

## 提交

脚本与 Markdown 报告放在独立目录；报告不超过两页。
