# 第 4 讲：实验环境

本讲以读源码为主。vLLM 只读源码即可完成大部分练习；跑模型是可选项，需要 GPU。没有 GPU 时，用 [code/README.md](code/README.md) 的示例日志和字段说明完成 block table 与 prefix 统计部分。

共用环境、依赖版本和代码基线见 [课程实验环境](../lab-environment.md)。

## 本讲额外要求

- vLLM 位于 `~/Project/ai-infra/vLLM`，HEAD `8b365ff949`；
- 只读源码，不要在课程仓库里复制 vLLM 源码；
- 源码阅读不需要安装 vLLM，也不需要 GPU；
- 可选跑模型时，按 vLLM 官方安装说明准备环境，并记录实际版本；
- instrumentation 片段放在 [code/README.md](code/README.md)，只用于本地 checkout 的临时修改，实验后还原；
- 没有 GPU 时，直接用示例日志理解 `block_table` 和 `prefix_cache_stats` 的字段，不要求执行 vLLM。

## 两条实验轨

| 轨道 | 需要什么 | 能完成什么 |
| --- | --- | --- |
| CPU / 只读源码 | 文本编辑器、Python 3.10+（可选，用于跑日志解析脚本） | 对象图、函数级阅读、映射表、示例日志字段解释 |
| GPU / 跑小模型 | CUDA、vLLM 环境、一个小模型 | 打印真实 block table、prefix 统计、观察 prefill 到 decode 的块增长 |

GPU 轨的启动参数建议：

```bash
export VLLM_LOG_STATS_INTERVAL=1
# 在 Python 里创建 LLM 时打开统计
# llm = LLM(model="...", enforce_eager=True, disable_log_stats=False)
```

`disable_log_stats` 默认是 `True`，不打开时 `make_prefix_cache_stats()` 返回 `None`。
