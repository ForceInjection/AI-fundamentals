# 第 1 讲实验脚本

| 文件                  | 用途                               | 对应实验 | 依赖         |
| --------------------- | ---------------------------------- | -------- | ------------ |
| `process_topology.py` | 观察并归类 vLLM 进程拓扑，对照进程公式 `A + DP + N` | 实验 1   | 仅 Python 标准库 |

## `process_topology`.py

```bash
# 终端 A:起服务(需先装好 vLLM)
vllm serve Qwen/Qwen3-0.6B --enforce-eager
# 终端 B:归类进程
python3 process_topology.py
```

输出：PID / 角色 / 判据 表 + 按角色计数。归类基于命令行与进程 title 特征
(`EngineCore`、`WorkerProc`、`vllm serve` 等，对应 `run_engine_core`、
`MultiprocExecutor` 与 launchers 入口)。平台上进程 title 对 ps 的可见性
不同，归类不出的行会标 UNKNOWN 并给出原始 cmdline，人工判断即可。

无 GPU 也可运行（只读进程表，不触碰模型）；没有 vLLM 进程时输出提示并退出码 1。
