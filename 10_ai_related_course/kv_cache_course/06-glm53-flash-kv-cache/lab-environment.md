# 第 6 讲：实验环境

本讲的无 GPU 部分只要求 Python 3 和一份 SGLang 源码；有 GPU 部分再准备可运行的 SGLang 环境。

共用环境、依赖版本和代码基线见 [课程实验环境](../lab-environment.md)。

## 无 GPU 必做

| 项目 | 要求 |
| --- | --- |
| Python | 3.10+，只用标准库 |
| 源码 | `~/Project/ai-infra/sglang`，HEAD `f4de6abee6` |
| 预读 | [三本账：图解 GLM-5.3-Flash 的 KV Cache 布局](../../../09_inference_system/sglang/glm53-flash-kv-layout.md) |
| 复算脚本 | [code/calc_glm53_kv.py](code/calc_glm53_kv.py) |
| 示例日志 | [code/sample-startup.log](code/sample-startup.log) |

## 有 GPU 选做

- 能启动 `zai-org/GLM-5.3-Flash` 的 SGLang 环境，GPU 数量按权重和显存调整。
- 启动日志里能看到 `max_total_num_tokens=` 和 `Mamba Cache is allocated.`。
- 显存不足时，先降 `--mem-fraction-static` 或减少 `--max-running-requests`。

## 口径

- 文章口径：单 rank、attn_tp=1、scaled FP8、64-token page 预留、无 MTP。
- MB = 10^6 B，MiB = 2^20 B。147.6 MB ≈ 140.8 MiB；738 MB ≈ 703.8 MiB（精确字节 703.9 MiB）。
- 启动日志的 `GB` 在 `memory_pool.py:107` 是 `1024^3` B，和文章的 MB 不是一个单位。
- 官方 cookbook 的 TP4/TP2 recipe 用 bf16 KV，TP4 还开 EAGLE MTP；不要直接套文章的单 rank scaled FP8 无 MTP 数字。
