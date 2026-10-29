# 第 1 讲：实验环境

本讲只需要 CPU 和 Python 3.10+，不要求 GPU，也不要求安装 PyTorch、flash-attn 或 triton。第 2 讲起受 nano-vllm 限制，统一按 Python 3.10–3.12 准备。

共用环境、依赖版本和代码基线见 [课程实验环境](../lab-environment.md)。如果根目录的环境说明与本讲冲突，以本讲的 CPU-only 路径为准。

## 本讲材料

- `code/kv_estimator.py`：本讲主计算器，只用标准库。
- `09_inference_system/kv_cache/01_concepts/basic/kv_cache_calc.py`：仓库里的参考计算器，可核对 DeepSeek-V3 MLA 等预设；它没有 Qwen3-0.6B / Qwen2.5-7B / Qwen2.5-72B 预设，Qwen 三档用本讲 `code/kv_estimator.py`。
- `~/Project/study/ai-infra-course/06-kvcache-optimization/code/calculate_qwen3_memory.py`：讲师环境的外部旧版参考，Qwen3-0.6B 默认 `head_dim=64`，会算出 56 KiB；本讲以 `head_dim=128`、112 KiB 为准。学生没有该仓库时可跳过。

## 运行

```bash
cd 10_ai_related_course/kv_cache_course/01-kv-cache-basics
python3 code/kv_estimator.py --self-check
python3 code/kv_estimator.py --model qwen3-0.6b --seq 4096 --batch 8 --dtype fp16
```

Python 3.10 以上即可；脚本没有第三方依赖。若系统里的 `python3` 低于 3.10，换一个解释器，不要改脚本语法。

## 口径提醒

- 每 token KV 按 `2 × b_kv × L × H_kv × D`，开头的 `2` 是 K/V 两份。
- 权重按 FP16 参数估算，和 KV 分开记账。
- 参考计算器如果和本讲数字不同，先核对模型 config 的 `num_key_value_heads`、`head_dim` 和精度，不要取平均。
