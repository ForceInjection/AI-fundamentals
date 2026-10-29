# 第 8 讲：容量规划课堂实验（兼结课实验）

40 分钟。目标是把前七讲的账合成一张可执行的容量表。实验只要求选一个模型加一个引擎，不要求三个引擎都填同一套账。

## 选一条轨

| 轨 | 模型 + 引擎 | 要算的账 | 适合 |
| --- | --- | --- | --- |
| A | GLM-5.3-Flash + SGLang | token 池、KDA state 池、page 开销 | 想练 request state 池和双预算 |
| B | DeepSeek-V4.1-Flash + vLLM | global token 池、SWA state 池 | 想练长上下文下的两段账 |
| C | Qwen3-0.6B + nano-vllm | token / block 池 | 只想练 token 池 |

## 步骤

1. 定假设（5 分钟）。写下并发请求数、上下文长度分布、精度、page/block size、每 rank 的 HBM 预算。假设不写全，后面的数没有意义。
2. 算 token 池（10 分钟）。按 `ceil(len/page) × page × bytes_per_token` 算单请求，再乘并发。
3. 算 request state 池（10 分钟）。A 轨用 `5 × 147.6 MB`；B 轨用 `40 × 128 × 528 B`；C 轨没有这一项。
4. 对日志（10 分钟）。打开 `code/sample_sglang_startup.log` 或 `code/sample_vllm_startup.log`，找到 `max_total_num_tokens`、`max_mamba_cache_size`、`max_running_requests`（或 vLLM 的 `GPU KV cache size`），先对照字段含义，再按你自己的并发和长度换算。样例按 64×128K 假设，和你的输入不一定相同。
5. 写结论（5 分钟）。指出先爆的账本，给一条调整建议。

## 输入假设

- A 轨：GLM-5.3-Flash，scaled FP8 7,260 B/token，KDA fp32 147.6 MB/slot，默认 5 slots/request，page 64，并发 32，上下文 128K。
- B 轨：DeepSeek-V4.1-Flash，890 B/token global，SWA 2.7 MB/request，并发 64，上下文 128K。
- C 轨：Qwen3-0.6B，`2 × 28 × 8 × 128 × 2 = 114,688 B/token`，block 256，并发 8，上下文 4K。

## 产出

一页容量表。下表由学员填写，第一行是示例：

| 项目 | 数值 | 公式 / 命令 |
| --- | --- | --- |
| 示例：A 轨 token 池（单请求 128K） | 951.6 MB | 7,260 × 131,072 |
| token 池（你的负载） | 待填 | 待填 |
| request state 池 | 待填 | 待填 |
| page / block 开销 | 待填 | 待填 |
| 每 rank HBM | 待填 | 待填 |
| DRAM 按对象 | 待填 | 待填 |
| 先爆的账本 | 待填 | 待填 |

选型理由写清：为什么选这个引擎、PD/offload/prefix cache 是否启用、验证方式。

## 日志字段说明

SGLang 启动时会打印：

```text
max_total_num_tokens=..., chunked_prefill_size=..., max_prefill_tokens=..., max_running_requests=..., context_len=..., available_gpu_mem=... GB
Mamba Cache is allocated. max_mamba_cache_size: ..., conv_state size: ...GB, ssm_state size: ...GB
```

vLLM 启动时会打印：

```text
GPU KV cache size: 8,388,608 tokens, Maximum concurrency for 131,072 tokens per request: 64.00x
```

`code/` 下的两份日志是**推演样例**，只用来演示字段格式和核对方法，不是真实测量，不能当作验证依据。要验证容量表，必须用你自己机器上的真实启动日志。

## 对账脚本

```bash
python3 code/capacity_plan.py glm
python3 code/capacity_plan.py deepseek
python3 code/capacity_plan.py nano-vllm
```

脚本输出三条轨的算例表，并带断言检查关键数字。它只做算术，不替代真实日志。
