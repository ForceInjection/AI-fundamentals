# 第 8 讲：实验环境

容量规划可以只用 CPU 和计算器完成；需要用启动日志验证时，准备可用的 vLLM / SGLang 环境。共用环境、依赖版本和代码基线见 [课程实验环境](../lab-environment.md)。

## 本讲额外要求

- 至少准备一个模型的 config 和一份真实或推演的启动日志；
- 记录每个容量数字的来源（config、日志、公式或估算）；
- 如果使用 GPU 实测，记录实际卡型、显存和启动参数；
- 三条实验轨只需选一条，不要求三个引擎都跑。

## 自带材料

| 文件 | 用途 | 性质 |
| --- | --- | --- |
| `code/capacity_plan.py` | 三条轨的算例表与断言 | 可执行脚本，纯标准库 |
| `code/sample_sglang_startup.log` | SGLang 启动日志字段示例 | **推演样例**，只演示格式 |
| `code/sample_vllm_startup.log` | vLLM 启动日志字段示例 | **推演样例**，只演示格式 |

推演日志不是测量结果。它的作用是让你知道该看哪些字段、字段之间怎么对上；不能拿它证明你的容量表正确。验证必须用你自己机器上的真实日志。

## 运行

```bash
cd 10_ai_related_course/kv_cache_course/08-kv-capacity-planning
python3 code/capacity_plan.py all
```

输出包括：

- GLM-5.3-Flash 在 8K / 32K / 128K 下的 token 池、KDA 池与 page 对齐；
- DeepSeek-V4.1-Flash 的 global 池、SWA 池与占比；
- nano-vllm 的 block 池算例；
- 作业参考算例：平均驻留与峰值 64×128K 的两组数。

脚本末尾有断言，参考算例对不上会直接报错。

## 需要核对源码时

- SGLang：`python/sglang/srt/mem_cache/kv_cache_configurator.py` 的 `_apply_token_constraints`、`resolve_max_num_reqs`；
- vLLM：`vllm/config/cache.py` 的 `gpu_memory_utilization`、`kv_cache_memory_bytes`；
- nano-vllm：`nanovllm/config.py` 的 `max_num_batched_tokens`、`max_num_seqs`、`max_model_len`、`kvcache_block_size`。

路径里的 `nanovllm` 是包名，展示名统一写 nano-vllm。
