# 第 1 讲：算清一个 token 的账

35 分钟（5 分钟安装与自检 + 30 分钟填表和讨论），目标是把 KV 公式落到一张可检查的表上。优先使用本讲自带的 `code/kv_estimator.py`，它只依赖 Python 标准库。

本讲统一使用两行公式：

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

## 准备

```bash
cd 10_ai_related_course/kv_cache_course/01-kv-cache-basics
python3 code/kv_estimator.py --self-check
python3 code/kv_estimator.py
```

`--self-check` 至少应包含三个预设的 FP16 每 token KV；脚本还会打印 FP8 和总量自检：

| 模型        | 精度 | 每 token KV |
| ----------- | ---- | ----------- |
| Qwen3-0.6B  | FP16 | 112 KiB     |
| Qwen2.5-7B  | FP16 | 56 KiB      |
| Qwen2.5-72B | FP16 | 320 KiB     |
| Qwen3-0.6B  | FP8  | 56 KiB      |
| Qwen3-0.6B  | FP16 | T=4096、batch=8 时 3.50 GiB |

如果前三个数对不上，先停下来检查模型 config，不要继续填表。

## 步骤

1. 跑默认表，确认三个模型的 `L`、`H_kv`、`D` 和每 token KV。默认值是 `T=4096`、`batch=1`、`dtype=fp16`。
2. 固定 `T=4096`，分别取 `batch=1/8/32`，用 `--model`、`--batch`、`--dtype` 生成总量表。至少覆盖 FP16 与 FP8 两种精度。
3. 对 Qwen2.5-72B 单独做预算判断：
   - 单卡 40 GiB：权重约 145 GB（135 GiB），装不下。脚本的并发上限列显示“—”，预算状态列显示“权重（超出预算）”；报告可写“无定义，权重先爆”。
   - 8×80 GiB、`--tp 8`：每卡权重约 18.2 GB（16.9 GiB），按 `--mem-util 0.9` 估算，`T=4096`、FP16 KV 时并发约 350；FP8 KV 约 700。`--mem-util 0.9` 已留 10% 打包余量，但未逐项拆分激活、CUDA context、碎片和框架开销，只作量级参考。
4. 把结果填进产出表，写一句判断：在当前 GPU 预算下，先限制并发的是权重、KV，还是其他显存项。

常用命令（覆盖 3 个模型 × 2 种精度 × 3 个 batch）：

```bash
for model in qwen3-0.6b qwen2.5-7b; do
  for dtype in fp16 fp8; do
    for batch in 1 8 32; do
      python3 code/kv_estimator.py --model "$model" --dtype "$dtype" --seq 4096 --batch "$batch"
    done
  done
done
python3 code/kv_estimator.py --model qwen3-0.6b --attention mha --dtype fp16
python3 code/kv_estimator.py --model qwen2.5-72b --gpu-gib 40
python3 code/kv_estimator.py --model qwen2.5-72b --gpu-gib 80 --tp 8 --seq 4096
```

`--attention mha` 把预设的 `H_kv` 临时改成 `H_q`，用于对比 MHA 的账，不是真实模型配置；默认 `native` 使用各模型的 GQA 配置。

## 产出

下表由学员填写，第一行是示例。先填 Qwen3-0.6B 和 Qwen2.5-7B，再补 Qwen2.5-72B；72B 的权重和“KV 与权重谁更大 / 预算状态”按上面的预算说明处理。

| 模型       | L   | H_kv | D   | 精度 | T    | batch | 每 token KV 字节   | 总 KV   | 权重（FP16） | KV 与权重谁更大 |
| ---------- | --- | ---- | --- | ---- | ---- | ----- | ------------------ | ------- | ------------ | --------------- |
| Qwen3-0.6B | 28  | 8    | 128 | FP16 | 4096 | 8     | 114,688（112 KiB） | 3.5 GiB | 1.12 GiB     | KV              |
|            |     |      |     |      |      |       |                    |         |              |                 |
|            |     |      |     |      |      |       |                    |         |              |                 |

“KV 与权重谁更大”按当前 `T`、`batch` 下的 KV 总量与 FP16 权重比较；给了 `--gpu-gib` 时，脚本输出改为“每卡总 KV / 每卡权重 / 并发上限 / 预算状态”，其中预算状态才表示谁超出显存。72B 的 3.5 GiB 例子是 `T=4096、batch=8`，不要和 32K/batch=1 的 3.5 GiB 混为同一个账。

至少填 Qwen3-0.6B 与 Qwen2.5-7B 的 2 模型 × 2 精度 × 3 batch = 12 行；Qwen2.5-72B 另填一行预算判断。

表后写一段判断：这组负载下，先限制并发的是权重、KV，还是其他显存项。数字后面注明模型、精度、`T` 和 `batch`。
