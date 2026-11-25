# 第 10 讲实验脚本

| 文件             | 用途                                             | 对应实验 | 依赖     |
| ---------------- | ------------------------------------------------ | -------- | -------- |
| `ops_inspect.py` | 运维面巡检：端点四件 + /ping 陷阱 + 认证差异 + metrics 按族盘点 | 实验 1   | 仅标准库 |

另复用第 2 讲 `mixed_load_probe.py` 与 `vllm bench serve`（实验 2）。

## `ops_inspect`.py

```bash
vllm serve Qwen/Qwen3-1.7B --api-key test123 --enable-server-load-tracking
python3 ops_inspect.py --api-key test123
```

输出：端点巡检（health/metrics/version/load + /ping 陷阱 404）、认证差异验证、
`vllm:` 指标按五族盘点(调度/吞吐/缓存/延迟/外挂，未配项标注缺省原因，
如 `spec_decode` 未开启、offload 未配置)。

巡检项与第 1 讲 `process_topology.py`（进程视角）互为补充：一个看进程，
一个看 HTTP 面。
