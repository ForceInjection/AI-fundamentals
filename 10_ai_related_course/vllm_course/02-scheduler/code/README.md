# 第 2 讲实验脚本

| 文件                  | 用途                                                | 对应实验 | 依赖                  |
| --------------------- | --------------------------------------------------- | -------- | --------------------- |
| `mixed_load_probe.py` | 长短交错负载的 TTFT 对照探针（流式，异步并发）       | 实验 1-2 | openai 客户端（随 vLLM 安装） |

## mixed_load_probe.py

```bash
# 终端 A:起服务(两种配置各起一次,对照)
vllm serve Qwen/Qwen3-4B                                   # 配置 A:默认
vllm serve Qwen/Qwen3-4B --long-prefill-token-threshold 100000   # 配置 B:抑制切块
# 终端 B:探针(服务端配置切换后重跑)
python3 mixed_load_probe.py --host localhost --port 8000 --model Qwen/Qwen3-4B
```

输出：每条请求（label / TTFT / 总耗时）明细 + 按长短分组的 TTFT 统计 + 总墙钟。
脚本与`--long-prefill-token-threshold`的耦合在服务端，探针不变。

参数：`--long-tokens`（长 prompt 近似 token 数，默认 2000；显存紧张时降低）、
`--model`（与服务的模型名一致）。

注意：TTFT 数字受机器负载与网络影响，对照结论看同机两组的相对差异，不与
他机数字比较；所有数字带配置口径（模型变体 + threshold + 并发数）。
